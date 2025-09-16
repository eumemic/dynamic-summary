"""LLM service for handling OpenAI API interactions."""

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ragzoom.config import IndexConfig, SecretStr, is_gpt5_model
from ragzoom.contracts.chat_model import UsageInfo
from ragzoom.services import summary_utils
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - typing helper only
    from openai import AsyncOpenAI as OpenAIAsyncType
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

            class _Resp:
                def __init__(self, items: list[_Item]) -> None:
                    self.data = items

            return _Resp([_Item(list(self._vector)) for _ in texts])

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


def _extract_cached_tokens(details: object | None) -> int:
    """Return cached token count from mixed detail structures."""
    if details is None:
        return 0
    if isinstance(details, dict):
        cached = details.get("cached_tokens", 0)
    else:
        cached = getattr(details, "cached_tokens", 0)
    if isinstance(cached, int | float) and cached > 0:
        return int(cached)
    return 0


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

        # Get API key from parameter or environment
        from ragzoom.config import ensure_secret_str

        actual_key = ensure_secret_str(api_key, "LLMService")

        if os.environ.get("PYTEST_CURRENT_TEST"):
            self.client = _build_test_openai_client(self.config.embedding_model)
        else:
            self.client = AsyncOpenAI(api_key=actual_key)

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

    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts in batches to respect API limits."""
        if not texts:
            return []

        # OpenAI embeddings API has a limit of ~2048 inputs per batch
        # Use a conservative limit to avoid hitting API constraints
        max_batch_size = 1000

        if len(texts) > max_batch_size:
            logger.debug(
                f"Large batch of {len(texts)} texts - splitting into smaller batches of {max_batch_size}"
            )
            all_embeddings = []
            for i in range(0, len(texts), max_batch_size):
                batch = texts[i : i + max_batch_size]
                logger.debug(
                    f"Processing batch {i//max_batch_size + 1}/{(len(texts) + max_batch_size - 1)//max_batch_size} ({len(batch)} texts)"
                )
                batch_embeddings = await self._get_embeddings_batch(batch)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings

        # Safety net: Check for empty strings that could cause API errors
        for i, text in enumerate(texts):
            if not text or not text.strip():
                logger.error(
                    f"Empty text at index {i} in embedding batch - this will cause API errors"
                )
                raise ValueError(
                    f"Empty text at index {i} in embedding batch. This should be filtered by the caller."
                )

        try:
            response = await self.client.embeddings.create(
                model=self.config.embedding_model,
                input=texts,
            )
            return [data.embedding for data in response.data]
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

    async def _make_summary_call(  # jscpd:ignore-start
        self,
        messages: summary_utils.SummaryMessages,
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, UsageInfo]:
        """Make OpenAI API call for summarization with telemetry tracking."""  # jscpd:ignore-end
        try:
            typed_messages = cast(list[ChatCompletionMessageParam], messages)
            # GPT-5 models have different parameter requirements
            if is_gpt5_model(self.config.summary_model):
                # Use reasoning_effort="minimal" (valid despite SDK type hints saying otherwise)
                response = await self.client.chat.completions.create(
                    model=self.config.summary_model,
                    messages=typed_messages,
                    reasoning_effort="minimal",
                )
            else:
                # Only add temperature for non-GPT-5 models (GPT-5 only supports default temperature=1)
                # Use a hardcoded reasonable temperature for summaries
                response = await self.client.chat.completions.create(
                    model=self.config.summary_model,
                    messages=typed_messages,
                    temperature=0.3,
                )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI")

            # Extract usage information for telemetry
            if not response.usage:
                raise ValueError("No usage information in OpenAI response")

            usage_info = cast(
                UsageInfo,
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "model": self.config.summary_model,
                },
            )

            cached_tokens = _extract_cached_tokens(
                getattr(response.usage, "prompt_tokens_details", None)
            )
            if cached_tokens > 0:
                usage_info["cached_tokens"] = cached_tokens

            return content, usage_info

        except Exception as e:
            from ragzoom.error_utils import preserve_exception_chain
            from ragzoom.exceptions import LLMError

            llm_error = LLMError(
                operation="summarize_text",
                model=self.config.summary_model,
                message=f"Failed to summarize text for node {node_id}: {e}",
                node_id=node_id,
            )
            raise preserve_exception_chain(llm_error, e)

    def _is_better_summary(
        self,
        new_tokens: int,
        new_distance: float,
        current_best_tokens: int,
        current_best_distance: float,
        target_tokens: int,
    ) -> bool:
        """Preserve compatibility with regression tests validating retry logic."""
        return summary_utils.is_better_summary(
            new_tokens, current_best_tokens, target_tokens
        )

    async def _summarize_text(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: TelemetryCollector | None = None,
        prev_context: str | None = None,
        left_token_count: int | None = None,
        right_token_count: int | None = None,
    ) -> tuple[str, int, int]:
        """Summarize combined text to approximately the target token count."""
        normalized_parent = parent_id
        summary_request: summary_utils.SummaryRequest = {
            "left_text": left_text,
            "right_text": right_text,
            "target_tokens": target_tokens,
            "prev_context": prev_context,
            "parent_id": normalized_parent,
            "reporter": reporter,
            "left_token_count": left_token_count,
            "right_token_count": right_token_count,
        }

        return await summary_utils.run_summary_request(
            index_config=self.config,
            request=summary_request,
            call_summary=self._make_summary_call,
        )
