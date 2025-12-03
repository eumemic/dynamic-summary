"""Tests verifying retry conversation handling via the runtime harness."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typing_extensions import TypedDict

from ragzoom.config import IndexConfig
from ragzoom.telemetry_collection import TelemetryCollector
from tests.conftest import IndexerRuntimeHarness
from tests.vector_index_stubs import RecordingVectorIndex


class OpenAIMockParams(TypedDict, total=False):
    messages: list[dict[str, object]]
    model: str
    temperature: float
    max_tokens: int


def _configure_runtime(
    harness: IndexerRuntimeHarness,
    config: IndexConfig,
    vector_index: RecordingVectorIndex,
) -> None:
    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = (
        harness.runtime._append_executor._splitter.__class__(config)
    )
    harness.worker_coordinator._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config
    vector_factory = lambda _model: vector_index  # noqa: E731
    harness.runtime._vector_index_factory = vector_factory
    harness.worker_coordinator._vector_index_factory = vector_factory


def create_test_reporter(config: IndexConfig) -> TelemetryCollector:
    reporter = TelemetryCollector(
        document_id="test_doc", source_tokens=1000, config=config
    )
    for node_id in ("test", "test_node"):
        reporter.track_node_created(node_id, height=1)
    return reporter


@contextmanager
def patched_tokenizers(
    encode_fn: Callable[[str], list[int]],
    count_fn: Callable[[str], int] | None = None,
) -> Iterator[None]:
    if count_fn is None:

        def _default_count(text: str) -> int:
            return len(encode_fn(text))

        count_fn = _default_count

    modules = (
        "ragzoom.services.summary_utils.tokenizer",
        "ragzoom.services.llm_service.tokenizer",
    )
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(patch(f"{module}.encode", side_effect=encode_fn))
            stack.enter_context(patch(f"{module}.count_tokens", side_effect=count_fn))
        yield


@pytest.mark.asyncio
async def test_retry_maintains_conversation_history(
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    config = IndexConfig.load(
        retry_threshold=0.2,
        max_retries=3,
        target_chunk_tokens=100,
    )
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, config, vector_index)

    api_calls: list[OpenAIMockParams] = []

    async def mock_create(**kwargs: OpenAIMockParams) -> object:
        copied = cast(OpenAIMockParams, deepcopy(kwargs))
        api_calls.append(copied)
        messages = copied.get("messages", [])
        if len(api_calls) == 1:
            return _response("A" * 150, 1350, 150, 0)
        if len(api_calls) == 2:
            assert len(messages) == 6
            assert messages[4]["role"] == "assistant"
            assert messages[4]["content"] == "A" * 150
            assert messages[5]["role"] == "user"
            return _response("B" * 95, 1500, 95, 1200)
        return _response("", 0, 0, 0)

    indexer_runtime_harness.llm_service.client = AsyncMock()
    indexer_runtime_harness.llm_service.client.chat.completions.create = mock_create

    reporter = create_test_reporter(config)

    with patched_tokenizers(lambda text: [0] * len(text)):
        summary, retry_count, _ = (
            await indexer_runtime_harness.llm_service._summarize_text(
                left_text="Left text" * 10,
                right_text="Right text" * 10,
                target_tokens=100,
                parent_id="test_node",
                reporter=reporter,
            )
        )

    assert len(api_calls) == 2
    assert retry_count == 1
    assert summary == "B" * 95

    first_messages = api_calls[0]["messages"]
    assert len(first_messages) == 4


@pytest.mark.asyncio
async def test_retry_preserves_original_context(
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    config = IndexConfig.load(
        retry_threshold=0.1,
        max_retries=1,
        target_chunk_tokens=100,
    )
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, config, vector_index)

    api_calls: list[OpenAIMockParams] = []
    original_text = "This is the original text that needs summarising. " * 5

    async def mock_create(**kwargs: OpenAIMockParams) -> object:
        copied = cast(OpenAIMockParams, deepcopy(kwargs))
        api_calls.append(copied)
        messages = copied.get("messages", [])
        if len(api_calls) == 1:
            user_content = " ".join(
                cast(str, msg["content"]) for msg in messages if msg["role"] == "user"
            )
            assert original_text[:30].strip() in user_content
            return _response("A" * 150, 150, 10, 0)
        if len(api_calls) == 2:
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            assert len(user_messages) == 3
            assert original_text[:30].strip() in cast(str, user_messages[0]["content"])
            return _response("B" * 100, 100, 20, 800)
        return _response("", 0, 0, 0)

    indexer_runtime_harness.llm_service.client = AsyncMock()
    indexer_runtime_harness.llm_service.client.chat.completions.create = mock_create

    with patched_tokenizers(lambda text: [0] * len(text)):
        summary, _, _ = await indexer_runtime_harness.llm_service._summarize_text(
            left_text=original_text[: len(original_text) // 2],
            right_text=original_text[len(original_text) // 2 :],
            target_tokens=100,
            parent_id="test",
        )

    assert len(api_calls) == 2
    assert summary == "B" * 100


@pytest.mark.asyncio
async def test_multiple_retries_build_conversation(
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    config = IndexConfig.load(
        retry_threshold=0.1,
        max_retries=3,
        target_chunk_tokens=100,
    )
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, config, vector_index)

    api_calls: list[OpenAIMockParams] = []

    async def mock_create(**kwargs: OpenAIMockParams) -> object:
        copied = cast(OpenAIMockParams, deepcopy(kwargs))
        api_calls.append(copied)
        messages = copied.get("messages", [])
        call_num = len(api_calls)
        if call_num == 1:
            return _response("A" * 150, 1000, 150, 0)
        if call_num == 2:
            assert len(messages) == 6
            return _response("B" * 130, 1200, 130, 1000)
        if call_num == 3:
            assert len(messages) == 8
            return _response("C" * 105, 1400, 105, 1200)
        return _response("", 0, 0, 0)

    indexer_runtime_harness.llm_service.client = AsyncMock()
    indexer_runtime_harness.llm_service.client.chat.completions.create = mock_create

    with patched_tokenizers(lambda text: [0] * len(text)):
        summary, retry_count, _ = (
            await indexer_runtime_harness.llm_service._summarize_text(
                left_text="Test content" * 10,
                right_text="More content" * 10,
                target_tokens=100,
                parent_id="test",
            )
        )

    assert len(api_calls) == 3
    assert retry_count == 2
    assert summary == "C" * 105

    final_messages = api_calls[-1]["messages"]
    assert len(final_messages) == 8


@pytest.mark.asyncio
async def test_no_retry_when_within_threshold(
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    config = IndexConfig.load(
        retry_threshold=0.2,
        target_chunk_tokens=100,
    )
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, config, vector_index)

    api_calls: list[OpenAIMockParams] = []

    async def mock_create(**kwargs: OpenAIMockParams) -> object:
        api_calls.append(cast(OpenAIMockParams, deepcopy(kwargs)))
        return _response("A" * 105, 1000, 105, 0)

    indexer_runtime_harness.llm_service.client = AsyncMock()
    indexer_runtime_harness.llm_service.client.chat.completions.create = mock_create

    with patched_tokenizers(lambda text: [0] * len(text)):
        summary, retry_count, token_count = (
            await indexer_runtime_harness.llm_service._summarize_text(
                left_text="Test content" * 10,
                right_text="More content" * 10,
                target_tokens=100,
                parent_id="test",
            )
        )

    assert len(api_calls) == 1
    assert retry_count == 0
    assert summary == "A" * 105
    assert token_count == 105


@pytest.mark.asyncio
async def test_accept_retry_within_threshold_immediately(
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    config = IndexConfig.load(
        retry_threshold=0.2,
        max_retries=3,
        target_chunk_tokens=100,
    )
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, config, vector_index)

    api_calls: list[OpenAIMockParams] = []

    async def mock_create(**kwargs: OpenAIMockParams) -> object:
        api_calls.append(cast(OpenAIMockParams, deepcopy(kwargs)))
        if len(api_calls) == 1:
            return _response("A" * 130, 1000, 130, 0)
        if len(api_calls) == 2:
            return _response("B" * 115, 1200, 115, 1000)
        pytest.fail("Should stop after second attempt")

    indexer_runtime_harness.llm_service.client = AsyncMock()
    indexer_runtime_harness.llm_service.client.chat.completions.create = mock_create

    with patched_tokenizers(lambda text: [0] * len(text)):
        summary, retry_count, token_count = (
            await indexer_runtime_harness.llm_service._summarize_text(
                left_text="Test content" * 10,
                right_text="More content" * 10,
                target_tokens=100,
                parent_id="test",
            )
        )

    assert len(api_calls) == 2
    assert retry_count == 1
    assert summary == "B" * 115
    assert token_count == 115


@pytest.mark.asyncio
async def test_passthrough_for_text_under_target(
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    config = IndexConfig.load(target_chunk_tokens=100)
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, config, vector_index)

    api_calls: list[OpenAIMockParams] = []

    async def mock_create(**kwargs: OpenAIMockParams) -> object:
        api_calls.append(cast(OpenAIMockParams, deepcopy(kwargs)))
        pytest.fail("Should not call LLM for passthrough")

    indexer_runtime_harness.llm_service.client = AsyncMock()
    indexer_runtime_harness.llm_service.client.chat.completions.create = mock_create

    reporter = create_test_reporter(config)

    with patched_tokenizers(
        lambda text: [0] * min(len(text), 50), lambda text: min(len(text), 50)
    ):
        summary, retry_count, token_count = (
            await indexer_runtime_harness.llm_service._summarize_text(
                left_text="Short",
                right_text="Text",
                target_tokens=100,
                parent_id="test",
                reporter=reporter,
            )
        )

    assert len(api_calls) == 0
    assert retry_count == 0
    assert summary == "Short Text"
    assert token_count == len("Short Text")

    data = reporter.get_telemetry_data("test_doc", config.target_chunk_tokens)
    attempts = data["nodes"][0]["summary_attempts"]
    assert attempts[0]["model"] == "passthrough"


def _response(
    content: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int
) -> object:
    class Usage(dict[str, object]):
        def __getattr__(self, item: str) -> object:  # pragma: no cover - simple helper
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cached_tokens=cached_tokens,
        model="mock-model",
    )
    usage["prompt_tokens_details"] = (
        {"cached_tokens": cached_tokens} if cached_tokens else None
    )

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = usage
    return response
