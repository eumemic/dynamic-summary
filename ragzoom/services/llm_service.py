"""LLM service for handling OpenAI API interactions."""

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from openai import AsyncOpenAI

from ragzoom.config import IndexConfig, SecretStr
from ragzoom.contracts.embedding_model import EmbeddingResult, EmbeddingUsageInfo
from ragzoom.services.summary_utils import SummaryResult
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - typing helper only
    from openai import AsyncOpenAI as OpenAIAsyncType

    from ragzoom.services.summarizer import Summarizer
else:  # pragma: no cover - runtime alias for test stubs
    OpenAIAsyncType = AsyncOpenAI


def _get_embedding_dimension(model_id: str) -> int:
    try:
        from ragzoom.model_info import ModelInfo

        return ModelInfo().get_embedding_dimensions(model_id)
    except Exception:
        return 8


def _build_test_openai_client(model_id: str) -> "OpenAIAsyncType":
    """Return a lightweight AsyncOpenAI stand-in for pytest runs.

    The stub avoids network calls while preserving realistic vector shapes so
    downstream components (e.g., Chroma adapters) see non-zero embeddings that
    match the configured model dimensionality.
    """

    dim = _get_embedding_dimension(model_id)

    class _StubEmbeddings:
        def __init__(self) -> None:
            unit = [0.0] * dim
            if dim > 0:
                unit[0] = 1.0
            self._vector = unit

        async def create(self, *, input: object, **kwargs: object) -> object:
            if isinstance(input, str):
                texts: list[str] = [input]
            else:
                texts = list(cast(Sequence[str], input))

            class _Item:
                def __init__(self, embedding: list[float]) -> None:
                    self.embedding = embedding

            class _EmbeddingUsage:
                def __init__(self, total_tokens: int) -> None:
                    self.total_tokens = total_tokens

            class _Resp:
                def __init__(self, items: list[_Item], total_tokens: int) -> None:
                    self.data = items
                    self.usage = _EmbeddingUsage(total_tokens)

            # Estimate ~1 token per 4 chars for stub
            total_tokens = sum(len(t) // 4 + 1 for t in texts)
            return _Resp([_Item(list(self._vector)) for _ in texts], total_tokens)

    class _StubCompletions:
        async def create(self, **kwargs: object) -> object:
            class _Msg:
                def __init__(self) -> None:
                    self.content = "summary"

            class _Choice:
                def __init__(self) -> None:
                    self.message = _Msg()

            class _Usage:
                def __init__(self) -> None:
                    self.prompt_tokens = 0
                    self.completion_tokens = 0
                    self.total_tokens = 0
                    self.prompt_tokens_details = {"cached_tokens": 0}

            class _Resp:
                def __init__(self) -> None:
                    self.choices = [_Choice()]
                    self.usage = _Usage()

            return _Resp()

    class _StubChat:
        def __init__(self) -> None:
            self.completions = _StubCompletions()

    class _StubClient:
        def __init__(self) -> None:
            self.embeddings = _StubEmbeddings()
            self.chat = _StubChat()

    return cast("OpenAIAsyncType", _StubClient())


class LLMService:
    """Service for handling all LLM operations including embeddings and summarization."""

    def __init__(  # jscpd:ignore-start
        self,
        config: IndexConfig,
        api_key: str | SecretStr = "",
        max_concurrent: int = 30,
    ):
        """Initialize LLM service.

        Args:
            config: Index configuration
            api_key: OpenAI API key as SecretStr or string (if not provided, reads from OPENAI_API_KEY env)
            max_concurrent: Maximum concurrent API requests
        """  # jscpd:ignore-end
        self.config = config
        self._max_parallel_api_calls = max(1, max_concurrent)

        # Get API key from parameter or environment
        from ragzoom.config import ensure_secret_str

        actual_key = ensure_secret_str(api_key, "LLMService")

        # OpenAI lists an 8K token hard limit for embedding requests and allows
        # up to 1K items per call. Use conservative defaults so callers stay
        # inside those guardrails, while tests may override them.
        self._embedding_batch_token_limit = 8000
        self._provider_max_embedding_batch_size = 1000

        if os.environ.get("PYTEST_CURRENT_TEST"):
            self.client = _build_test_openai_client(self.config.embedding_model)
        else:
            # Set a 120-second timeout to prevent indefinite hanging on API calls
            self.client = AsyncOpenAI(api_key=actual_key, timeout=120.0)

        # Lazy-init summarizer; track client/config to detect when tests replace them
        self._cached_summarizer: Summarizer | None = None
        self._summarizer_client: OpenAIAsyncType | None = None
        self._summarizer_config: IndexConfig | None = None

    @property
    def _summarizer(self) -> "Summarizer":
        """Lazy-initialize the summarizer with current client and config.

        Re-creates if client or config has changed (enables test mocking).
        """
        from ragzoom.adapters.openai_chat_model import OpenAIChatModel
        from ragzoom.services.summarizer import Summarizer

        # Check if we need to rebuild (client or config changed, or first access)
        needs_rebuild = (
            self._cached_summarizer is None
            or self._summarizer_client is not self.client
            or self._summarizer_config is not self.config
        )
        if needs_rebuild:
            chat_model = OpenAIChatModel(self.client, self.config.summary_model)
            self._cached_summarizer = Summarizer(chat_model, self.config)
            self._summarizer_client = self.client
            self._summarizer_config = self.config

        # At this point _cached_summarizer is guaranteed to be set
        assert self._cached_summarizer is not None
        return self._cached_summarizer

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI."""
        try:
            # Check token count before embedding
            tokens = tokenizer.encode(text)
            token_count = len(tokens)

            # Hard limit at 8000 tokens to leave margin for API overhead
            if token_count > 8000:
                logger.error(
                    f"Text exceeds embedding token limit: {token_count} tokens "
                    f"(limit: 8000). First 200 chars: {text[:200]}..."
                )
                raise ValueError(
                    f"Text too large for embedding: {token_count} tokens exceeds "
                    f"limit of 8000. This is likely due to summary size growth "
                    f"at higher tree levels."
                )

            response = await self.client.embeddings.create(
                model=self.config.embedding_model,
                input=text,
                # Let OpenAI API determine dimensions - no need for hardcoded values
            )
            return response.data[0].embedding
        except Exception as e:
            from ragzoom.error_utils import preserve_exception_chain
            from ragzoom.exceptions import LLMError

            llm_error = LLMError(
                operation="get_embedding",
                model=self.config.embedding_model,
                message=f"Failed to get embedding: {e}",
                text_length=len(text),
            )
            raise preserve_exception_chain(llm_error, e)

    def _prepare_embedding_batches(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches respecting token and item limits.

        Args:
            texts: List of texts to batch

        Returns:
            List of batches, each batch is a list of texts

        Raises:
            ValueError: If any text is empty or exceeds token limit
        """
        token_limit = getattr(self, "_embedding_batch_token_limit", None)
        max_items = getattr(self, "_provider_max_embedding_batch_size", 1000)

        # Safety net: Check for empty strings that could cause API errors
        for i, text in enumerate(texts):
            if not text or not text.strip():
                logger.error(
                    f"Empty text at index {i} in embedding batch - this will cause API errors"
                )
                raise ValueError(
                    f"Empty text at index {i} in embedding batch. This should be filtered by the caller."
                )

        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for idx, text in enumerate(texts):
            token_count = tokenizer.count_tokens(text)

            if token_limit is not None and token_count > token_limit:
                raise ValueError(
                    f"Item {idx} exceeds embedding token limit: {token_count} tokens "
                    f"(limit: {token_limit})."
                )

            would_exceed_tokens = (
                token_limit is not None
                and current_batch
                and current_tokens + token_count > token_limit
            )

            if current_batch and (
                would_exceed_tokens or len(current_batch) >= max_items
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += token_count

            if len(current_batch) >= max_items:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts in batches to respect API limits."""
        if not texts:
            return []

        batches = self._prepare_embedding_batches(texts)
        results: list[list[float]] = []

        try:
            for batch_idx, batch in enumerate(batches, start=1):
                response = await self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=batch,
                )
                results.extend(data.embedding for data in response.data)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Processed embedding batch %s size=%s",
                        batch_idx,
                        len(batch),
                    )
        except Exception as e:
            from ragzoom.error_utils import preserve_exception_chain
            from ragzoom.exceptions import LLMError

            llm_error = LLMError(
                operation="get_batch_embeddings",
                model=self.config.embedding_model,
                message=f"Failed to get batch embeddings: {e}",
                batch_size=len(texts),
            )
            raise preserve_exception_chain(llm_error, e)

        return results

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Public wrapper to obtain embeddings for a batch of texts."""

        return await self._get_embeddings_batch(texts)

    async def embed_texts_with_usage(self, texts: list[str]) -> EmbeddingResult:
        """Get embeddings with usage information from the API response.

        Unlike embed_texts(), this method returns the actual token count
        reported by the embedding API, which is needed for accurate cost
        calculation.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult with embeddings and aggregated usage info
        """
        if not texts:
            return {
                "embeddings": [],
                "usage": {"total_tokens": 0, "model": self.config.embedding_model},
            }

        batches = self._prepare_embedding_batches(texts)
        results: list[list[float]] = []
        total_tokens = 0

        try:
            for batch in batches:
                response = await self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=batch,
                )
                results.extend(data.embedding for data in response.data)
                if response.usage:
                    total_tokens += response.usage.total_tokens
        except Exception as e:
            from ragzoom.error_utils import preserve_exception_chain
            from ragzoom.exceptions import LLMError

            llm_error = LLMError(
                operation="get_batch_embeddings_with_usage",
                model=self.config.embedding_model,
                message=f"Failed to get batch embeddings: {e}",
                batch_size=len(texts),
            )
            raise preserve_exception_chain(llm_error, e)

        usage: EmbeddingUsageInfo = {
            "total_tokens": total_tokens,
            "model": self.config.embedding_model,
        }
        return {"embeddings": results, "usage": usage}

    async def _summarize_text(
        self,
        text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: TelemetryCollector | None = None,
        prev_context: str | None = None,
        text_tokens: int | None = None,
    ) -> SummaryResult:
        """Summarize text to approximately the target token count."""
        return await self._summarizer.summarize(
            text,
            target_tokens,
            prev_context=prev_context,
            parent_id=parent_id,
            reporter=reporter,
            text_tokens=text_tokens,
        )

    async def _contextualize_text(
        self,
        preceding_context: str,
        target_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: TelemetryCollector | None = None,
    ) -> SummaryResult:
        """Generate contextualizing summary of preceding context for target text.

        Unlike _summarize_text which compresses preserving all information,
        this extracts only background information relevant to understanding
        the target text.

        Returns:
            SummaryResult containing the context summary, retry count, token count,
            and accumulated usage across all LLM attempts for cost calculation.
        """
        return await self._summarizer.contextualize(
            preceding_context,
            target_text,
            target_tokens,
            parent_id=parent_id,
            reporter=reporter,
        )
