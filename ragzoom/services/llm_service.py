"""LLM service for handling OpenAI API interactions.

This class now composes small, focused components:
- ChatModel + Summarizer for summaries
- EmbeddingModel + EmbeddingBatcher for embeddings

Public behavior remains the same; existing callers continue using
_summarize_text and _get_embeddings_batch.
"""

import logging
from typing import Protocol, TypedDict, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ragzoom.adapters.openai_chat_model import OpenAIChatModel
from ragzoom.adapters.openai_embedding_model import OpenAIEmbeddingModel
from ragzoom.config import IndexConfig, SecretStr
from ragzoom.contracts.chat_model import UsageInfo as _UsageInfo
from ragzoom.services.embedding_batcher import EmbeddingBatcher
from ragzoom.services.summarizer import Summarizer
from ragzoom.telemetry_collection import TelemetryCollector

_tz: object
try:
    # Expose tokenizer at module scope for tests that patch it
    from ragzoom.utils.tokenization import tokenizer as _real_tokenizer

    _tz = _real_tokenizer
except Exception:  # pragma: no cover - fallback for isolated test import timing

    class _TokenizerSentinel:
        def encode(self, text: str) -> list[int]:
            return []

        def count_tokens(self, text: str) -> int:
            return 0

        def decode(self, tokens: list[int]) -> str:
            return ""

    _tz = _TokenizerSentinel()

# Re-export as module-level attribute for tests
tokenizer = _tz

logger = logging.getLogger(__name__)


class UsageInfo(TypedDict, total=False):
    """Type definition for OpenAI API usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cached_tokens: int  # Optional field for prompt caching


class MockUsage:
    """Mock usage object for telemetry recording."""

    def __init__(self, usage_info: UsageInfo) -> None:
        # Set basic attributes explicitly
        self.prompt_tokens = usage_info["prompt_tokens"]
        self.completion_tokens = usage_info["completion_tokens"]
        self.total_tokens = usage_info["total_tokens"]
        self.model = usage_info.get("model", "")

        # Handle prompt_tokens_details specially
        cached_tokens = usage_info.get("cached_tokens", 0)
        has_cached_tokens = cached_tokens and cached_tokens > 0
        self.prompt_tokens_details: dict[str, int] | None = (
            {"cached_tokens": cached_tokens} if has_cached_tokens else None
        )


class MockResponse:
    """Mock OpenAI response object for telemetry recording."""

    def __init__(self, usage_info: UsageInfo) -> None:
        self.usage = MockUsage(usage_info)


# Note: Removed APIParam type alias as it was too broad for OpenAI API calls
# Direct parameter passing ensures type safety with OpenAI's specific requirements


# Constants for word-based prompting bias compensation
# The 0.94 factor compensates for a systematic overshoot bias in GPT models
WORDS_PER_TOKEN = 0.75 * 0.94  # 0.705 - Bias-compensated word/token ratio


def _create_mock_response(usage_info: UsageInfo) -> MockResponse:
    """Create a mock OpenAI response object for telemetry recording."""
    return MockResponse(usage_info)


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

        self.client = AsyncOpenAI(api_key=actual_key)

        # If client already exposes a chat object (e.g., a test mock), capture it and
        # install a lightweight delegating proxy so tests can patch
        # client.chat.completions.create without triggering SDK lazy imports.
        # Type-safe proxies to avoid mypy errors while allowing tests to patch
        class _CompletionsLike(Protocol):
            async def create(self, **kwargs: object) -> object: ...

        class _ChatLike(Protocol):
            completions: _CompletionsLike

        try:
            _orig_chat = cast(_ChatLike, getattr(self.client, "chat"))

            class _CompletionsProxy:
                def __init__(self, chat: _ChatLike) -> None:
                    self._chat = chat

                async def create(self, **kwargs: object) -> object:
                    return await self._chat.completions.create(**kwargs)

            class _ChatProxy:
                def __init__(self, chat: _ChatLike) -> None:
                    self.completions: _CompletionsLike = _CompletionsProxy(chat)

            # Only override when we successfully captured an existing chat object
            setattr(self.client, "chat", _ChatProxy(_orig_chat))
        except Exception:
            # If client.chat is a lazy property (real SDK), leave as-is
            pass

        # Initialize adapters and orchestrators
        self._chat_model = OpenAIChatModel(self.client, self.config.summary_model)
        self._embedding_model = OpenAIEmbeddingModel(
            self.client, self.config.embedding_model
        )
        self._summarizer = Summarizer(self._chat_model, self.config)
        self._embedding_batcher = EmbeddingBatcher(self._embedding_model)

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text using the batcher."""
        # Ensure the adapter sees the latest client (tests may swap it)
        if (
            not hasattr(self, "_embedding_model")
            or getattr(self._embedding_model, "_client", None) is not self.client
            or getattr(self._embedding_model, "model_id", "")
            != self.config.embedding_model
        ):
            self._embedding_model = OpenAIEmbeddingModel(
                self.client, self.config.embedding_model
            )
            self._embedding_batcher = EmbeddingBatcher(self._embedding_model)
        embeddings = await self._embedding_batcher.embed([text])
        if not embeddings or len(embeddings) != 1:
            raise ValueError("Expected exactly one embedding from single-text call")
        return embeddings[0]

    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts in batches to respect API limits."""
        # Ensure the adapter sees the latest client (tests may swap it)
        # jscpd:ignore-start - same pattern as in _get_embedding
        if (
            not hasattr(self, "_embedding_model")
            or getattr(self._embedding_model, "_client", None) is not self.client
            or getattr(self._embedding_model, "model_id", "")
            != self.config.embedding_model
        ):
            self._embedding_model = OpenAIEmbeddingModel(
                self.client, self.config.embedding_model
            )
            self._embedding_batcher = EmbeddingBatcher(self._embedding_model)
        # jscpd:ignore-end
        return await self._embedding_batcher.embed(texts)

    def _tokens_to_words(self, target_tokens: int) -> int:
        """Convert target token count to target word count with bias compensation."""
        return int(target_tokens * WORDS_PER_TOKEN)

    async def _make_summary_call(  # jscpd:ignore-start
        self,
        messages: list[ChatCompletionMessageParam],
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, UsageInfo]:
        """Kept for backward compatibility; forwards to Summarizer with compatibility cast."""  # jscpd:ignore-end
        # Convert to provider-neutral Message type and delegate
        # Ensure adapters see the latest client (tests may swap it)
        # jscpd:ignore-start - same pattern as in _make_summary_call
        if (
            not hasattr(self, "_chat_model")
            or getattr(self._chat_model, "_client", None) is not self.client
            or getattr(self._chat_model, "model_id", "") != self.config.summary_model
        ):
            self._chat_model = OpenAIChatModel(self.client, self.config.summary_model)
            self._summarizer = Summarizer(self._chat_model, self.config)
        # jscpd:ignore-end
        converted: list[dict[str, str]] = []
        for m in messages:
            # Minimal shape used by our Summarizer
            role = cast(
                str, m["role"]
            )  # ChatCompletionMessageParam role is a Literal[str]
            content = cast(str, m["content"])  # content is str in our usage
            converted.append({"role": role, "content": content})
        from ragzoom.contracts.chat_model import Message as _Message

        content, usage = await self._summarizer._make_summary_call(  # noqa: SLF001
            cast(list[_Message], converted), target_tokens, node_id, reporter
        )
        return content, usage

    async def _record_summary_telemetry(
        self,
        reporter: TelemetryCollector | None,
        parent_id: str,
        response: MockResponse,
        target_tokens: int,
        input_text_tokens: int,
        actual_tokens: int,
        start_time: float,
    ) -> None:
        """Backward-compatible wrapper that adapts MockResponse to UsageInfo and forwards."""
        usage: _UsageInfo = {
            "prompt_tokens": int(response.usage.prompt_tokens),
            "completion_tokens": int(response.usage.completion_tokens),
            "total_tokens": int(response.usage.total_tokens),
            "model": getattr(response.usage, "model", self.config.summary_model),
        }
        details = getattr(response.usage, "prompt_tokens_details", None)
        try:
            if isinstance(details, dict):
                cached = int(details.get("cached_tokens", 0) or 0)
                if cached > 0:
                    usage["cached_tokens"] = cached
        except Exception:
            pass

        await self._summarizer._record_summary_telemetry(  # noqa: SLF001
            reporter,
            parent_id,
            usage,
            target_tokens,
            input_text_tokens,
            actual_tokens,
            start_time,
        )

    # The retry decision and attempt logic now lives in Summarizer.

    # jscpd:ignore-start - Thin compatibility wrapper delegating to Summarizer
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
        """Summarize via provider-agnostic Summarizer (keeps public API stable)."""
        # Delegate entirely to the provider-agnostic Summarizer
        # Convert message types as needed is handled inside Summarizer
        # Ensure adapters see the latest client (tests may swap it)
        if (
            not hasattr(self, "_chat_model")
            or getattr(self._chat_model, "_client", None) is not self.client
            or getattr(self._chat_model, "model_id", "") != self.config.summary_model
        ):
            self._chat_model = OpenAIChatModel(self.client, self.config.summary_model)
            self._summarizer = Summarizer(self._chat_model, self.config)
        return await self._summarizer.summarize(
            left_text,
            right_text,
            target_tokens,
            parent_id=parent_id,
            reporter=reporter,
            prev_context=prev_context,
            left_token_count=left_token_count,
            right_token_count=right_token_count,
        )

    # jscpd:ignore-end

    def _is_better_summary(
        self,
        new_tokens: int,
        new_distance: int,
        current_best_tokens: int,
        current_best_distance: int,
        target_tokens: int,
    ) -> bool:
        """Compatibility shim that forwards to Summarizer's decision logic."""
        return self._summarizer._is_better_summary(  # noqa: SLF001
            new_tokens,
            new_distance,
            current_best_tokens,
            current_best_distance,
            target_tokens,
        )
