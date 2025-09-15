"""LLM service for handling OpenAI API interactions.

This class now composes small, focused components:
- ChatModel + Summarizer for summaries
- EmbeddingModel + EmbeddingBatcher for embeddings

Public behavior remains the same; existing callers continue using
_summarize_text and _get_embeddings_batch.
"""

import logging
import os
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

from openai.types.chat import ChatCompletionMessageParam

from ragzoom.adapters.openai_chat_model import OpenAIChatModel
from ragzoom.adapters.openai_embedding_model import OpenAIEmbeddingModel
from ragzoom.config import IndexConfig, SecretStr
from ragzoom.contracts.chat_model import UsageInfo as _UsageInfo
from ragzoom.services.embedding_batcher import EmbeddingBatcher
from ragzoom.services.summarizer import Summarizer
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer as _rz_tokenizer

tokenizer: object | None  # module-level patch point for tests; assigned below
tokenizer = _rz_tokenizer
logger = logging.getLogger(__name__)


class _AsyncClientCtor(Protocol):
    def __call__(self, *, api_key: str, **kwargs: object) -> object: ...


# Expose a module-level AsyncOpenAI symbol for tests to patch.
# The real import happens lazily in __init__ when this is None.
AsyncOpenAI: _AsyncClientCtor | None = None
if TYPE_CHECKING:
    from openai import AsyncOpenAI as OpenAIAsyncType


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


class _CompletionsLike(Protocol):
    @property
    def create(self) -> object: ...


class _ChatLike(Protocol):
    @property
    def completions(self) -> _CompletionsLike: ...


class _EmbeddingsLike(Protocol):
    @property
    def create(self) -> object: ...


class _ClientLike(Protocol):
    @property
    def chat(self) -> _ChatLike: ...

    @property
    def embeddings(self) -> _EmbeddingsLike: ...


class LLMService:
    """Service for handling all LLM operations including embeddings and summarization."""

    # Patchable client facade; always present for typing and tests
    client: "OpenAIAsyncType | _ClientLike"

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
        # Store API key; defer OpenAI client construction until actually needed
        self._api_key: str = actual_key
        # Provide a patchable stub in tests so they can do
        # patch.object(llm_service.client.chat.completions, "create", ...)
        if os.environ.get("PYTEST_CURRENT_TEST") and AsyncOpenAI is None:

            class _StubEmbeddings:
                async def create(self, **kwargs: object) -> object:
                    input_texts = kwargs.get("input", [])
                    if isinstance(input_texts, str):
                        input_list = [input_texts]
                    elif isinstance(input_texts, list):
                        input_list = input_texts
                    else:
                        input_list = [str(input_texts)]

                    class _Item:
                        def __init__(self) -> None:
                            self.embedding = [0.0] * 8

                    class _Resp:
                        def __init__(self, n: int) -> None:
                            self.data = [_Item() for _ in range(n)]

                    return _Resp(len(input_list))

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
                            self.prompt_tokens_details = None

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
                    self.chat = _StubChat()
                    self.embeddings = _StubEmbeddings()

            self.client = _StubClient()
        else:
            # Install a lazy proxy that defers to a real client on first call
            svc = self

            class _LazyEmbeddings:
                async def _create(self, **kwargs: object) -> object:
                    real = svc._get_client()
                    return await getattr(getattr(real, "embeddings"), "create")(
                        **kwargs
                    )

                @property
                def create(self) -> object:
                    return self._create

            class _LazyCompletions:
                async def _create(self, **kwargs: object) -> object:
                    real = svc._get_client()
                    return await getattr(
                        getattr(getattr(real, "chat"), "completions"), "create"
                    )(**kwargs)

                @property
                def create(self) -> object:
                    return self._create

            class _LazyChat:
                def __init__(self) -> None:
                    self._completions = _LazyCompletions()

                @property
                def completions(self) -> _CompletionsLike:
                    return self._completions

            class _LazyClient:
                def __init__(self) -> None:
                    self._chat = _LazyChat()
                    self._embeddings = _LazyEmbeddings()

                @property
                def chat(self) -> _ChatLike:
                    return self._chat

                @property
                def embeddings(self) -> _EmbeddingsLike:
                    return self._embeddings

            self.client = _LazyClient()

        # Adapters and orchestrators are lazy
        self._chat_model: OpenAIChatModel | None = None
        self._embedding_model: OpenAIEmbeddingModel | None = None
        self._summarizer: Summarizer | None = None
        self._embedding_batcher: EmbeddingBatcher | None = None
        # Real client cache (lazy)
        self._real_client: OpenAIAsyncType | None = None

    def _get_client(self) -> "OpenAIAsyncType":
        """Return the OpenAI client, building it on first use.

        If tests set `self.client` directly, that instance is returned without
        importing the SDK.
        """
        if self._real_client is not None:
            return self._real_client
        global AsyncOpenAI
        if AsyncOpenAI is None:
            from openai import AsyncOpenAI as _AsyncOpenAI

            real_client: OpenAIAsyncType = _AsyncOpenAI(api_key=self._api_key)
        else:
            from typing import cast as _cast

            ctor = AsyncOpenAI
            real_client = _cast("OpenAIAsyncType", ctor(api_key=self._api_key))
        self._real_client = real_client
        return real_client

    def _ensure_embedding_adapter_current(self) -> None:
        """Refresh embedding adapter if client or model changed."""
        # Prefer an injected test client if present; otherwise use the real client
        client_for_adapter = self.client or self._get_client()
        if (
            self._embedding_model is None
            or getattr(self._embedding_model, "_client", None) is not client_for_adapter
            or getattr(self._embedding_model, "model_id", "")
            != self.config.embedding_model
        ):
            from typing import cast as _cast

            self._embedding_model = OpenAIEmbeddingModel(
                _cast("OpenAIAsyncType", client_for_adapter),
                self.config.embedding_model,
            )
            self._embedding_batcher = EmbeddingBatcher(self._embedding_model)

    def _ensure_chat_adapter_current(self) -> None:
        """Refresh chat adapter if client or model changed."""
        client_for_adapter = self.client or self._get_client()
        if (
            self._chat_model is None
            or getattr(self._chat_model, "_client", None) is not client_for_adapter
            or getattr(self._chat_model, "model_id", "") != self.config.summary_model
        ):
            from typing import cast as _cast

            self._chat_model = OpenAIChatModel(
                _cast("OpenAIAsyncType", client_for_adapter),
                self.config.summary_model,
            )
            self._summarizer = Summarizer(self._chat_model, self.config)

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text using the batcher."""
        # Ensure the adapter sees the latest client (tests may swap it)
        self._ensure_embedding_adapter_current()
        if self._embedding_batcher is None:
            self._ensure_embedding_adapter_current()
        embeddings = await self._embedding_batcher.embed([text])  # type: ignore[union-attr]
        if not embeddings or len(embeddings) != 1:
            raise ValueError("Expected exactly one embedding from single-text call")
        return embeddings[0]

    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts in batches to respect API limits."""
        # Ensure the adapter sees the latest client (tests may swap it)
        self._ensure_embedding_adapter_current()
        if self._embedding_batcher is None:
            self._ensure_embedding_adapter_current()
        return await self._embedding_batcher.embed(texts)  # type: ignore[union-attr]

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
        self._ensure_chat_adapter_current()
        converted: list[dict[str, str]] = []
        for m in messages:
            # Minimal shape used by our Summarizer
            role = cast(
                str, m["role"]
            )  # ChatCompletionMessageParam role is a Literal[str]
            content = cast(str, m["content"])  # content is str in our usage
            converted.append({"role": role, "content": content})
        from ragzoom.contracts.chat_model import Message as _Message

        if self._summarizer is None:
            self._ensure_chat_adapter_current()
        content, usage = await self._summarizer._make_summary_call(  # type: ignore[union-attr]  # noqa: SLF001
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

        if self._summarizer is None:
            self._ensure_chat_adapter_current()
        await self._summarizer._record_summary_telemetry(  # type: ignore[union-attr]  # noqa: SLF001
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
        # Ensure summarizer is initialized and current
        self._ensure_chat_adapter_current()
        return await self._summarizer.summarize(  # type: ignore[union-attr]
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
        if self._summarizer is None:
            self._ensure_chat_adapter_current()
        return self._summarizer._is_better_summary(  # type: ignore[union-attr]  # noqa: SLF001
            new_tokens,
            new_distance,
            current_best_tokens,
            current_best_distance,
            target_tokens,
        )
