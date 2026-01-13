"""Telemetry regression tests verifying cached token accounting."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from typing_extensions import TypedDict

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.splitter import TextSplitter
from ragzoom.telemetry_collection import TelemetryCollector
from tests.conftest import IndexerRuntimeHarness
from tests.vector_index_stubs import RecordingVectorIndex


class OpenAIMockParams(TypedDict, total=False):
    """Type for OpenAI API parameters in test mocks."""

    messages: list[dict[str, str | object]]
    model: str
    temperature: float
    max_tokens: int


def create_test_reporter(config: IndexConfig) -> TelemetryCollector:
    """Create a telemetry collector pre-seeded with a test node."""

    reporter = TelemetryCollector(
        document_id="test_doc", source_tokens=1000, config=config
    )
    for node_id in ["test", "test_node"]:
        reporter.track_node_created(node_id, height=1)
    return reporter


class MockOpenAIResponseWithCache:
    """Mock OpenAI response with usage and cached token metadata."""

    def __init__(
        self,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        audio_tokens: int = 0,
    ) -> None:
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        if cached_tokens > 0:
            self.usage.prompt_tokens_details = {
                "cached_tokens": cached_tokens,
                "audio_tokens": audio_tokens,
            }
        else:
            self.usage.prompt_tokens_details = None


@contextmanager
def patched_tokenizers(
    encode_fn: Callable[[str], list[int]],
    count_fn: Callable[[str], int] | None = None,
) -> Iterator[None]:
    """Patch tokenizer helpers used by summary utilities and LLM service."""

    if count_fn is None:

        def _default_count(text: str) -> int:
            return len(encode_fn(text))

        count_fn = _default_count

    with ExitStack() as stack:
        for module in (
            "ragzoom.services.summary_utils.tokenizer",
            "ragzoom.services.llm_service.tokenizer",
        ):
            stack.enter_context(patch(f"{module}.encode", side_effect=encode_fn))
            stack.enter_context(patch(f"{module}.count_tokens", side_effect=count_fn))
        yield


@pytest.mark.asyncio
async def test_cached_tokens_recorded_in_telemetry(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    index_config = IndexConfig.load(
        retry_threshold=0.1,
        max_retries=1,
        target_chunk_tokens=100,
    )
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, index_config, vector_index)

    doc_store = storage_backend.for_document("telemetry-doc")
    doc_store.set_metadata(
        file_path="telemetry.txt",
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
    )

    reporter = create_test_reporter(index_config)
    api_calls: list[OpenAIMockParams] = []

    async def mock_create(**kwargs: OpenAIMockParams) -> MockOpenAIResponseWithCache:
        api_calls.append(cast(OpenAIMockParams, kwargs))
        call_num = len(api_calls)
        if call_num == 1:
            return MockOpenAIResponseWithCache(
                content="A" * 150,
                prompt_tokens=1350,
                completion_tokens=150,
                cached_tokens=0,
            )
        return MockOpenAIResponseWithCache(
            content="B" * 100,
            prompt_tokens=1500,
            completion_tokens=100,
            cached_tokens=1200,
        )

    indexer_runtime_harness.llm_service.client = MagicMock()
    indexer_runtime_harness.llm_service.client.chat.completions.create = mock_create

    def encode(text: str) -> list[int]:
        if "Test content" in text and "More content" in text:
            return [0] * 200
        return [0] * 120

    with patched_tokenizers(encode):
        await indexer_runtime_harness.llm_service._summarize_text(
            "Test content that needs to be long enough to trigger summarization" * 2
            + " "
            + "More content that also needs to be sufficiently long" * 2,
            100,
            parent_id="test_node",
            reporter=reporter,
        )

    chunk_tokens = (
        index_config.target_chunk_tokens
        if index_config.target_chunk_tokens is not None
        else index_config.target_embedding_context_tokens
    )
    data = reporter.get_telemetry_data("test_doc", chunk_tokens)
    test_node = next(n for n in data["nodes"] if n["node_id"] == "test_node")
    attempts = test_node["summary_attempts"]
    assert len(attempts) == 2
    assert attempts[0].get("cached_tokens", 0) == 0
    assert attempts[1]["cached_tokens"] == 1200


@pytest.mark.asyncio
async def test_backward_compatibility_without_cached_tokens(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    index_config = IndexConfig.load(target_chunk_tokens=100)
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, index_config, vector_index)

    doc_store = storage_backend.for_document("telemetry-doc")
    doc_store.set_metadata(
        file_path="telemetry.txt",
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
    )

    reporter = create_test_reporter(index_config)

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Summary"))]
    response.usage = MagicMock(
        prompt_tokens=1000, completion_tokens=100, total_tokens=1100
    )
    response.usage.prompt_tokens_details = None

    async def mock_create(**_kwargs: OpenAIMockParams) -> MagicMock:
        return response

    indexer_runtime_harness.llm_service.client = MagicMock()
    indexer_runtime_harness.llm_service.client.chat.completions.create = mock_create

    def encode(text: str) -> list[int]:
        if "Test content" in text and "More content" in text:
            return [0] * 200
        if "Test content" in text:
            return [0] * 120
        if "More content" in text:
            return [0] * 120
        return [0] * 100

    with patched_tokenizers(encode):
        await indexer_runtime_harness.llm_service._summarize_text(
            "Test content that needs to be long enough to trigger summarization" * 2
            + " "
            + "More content that also needs to be sufficiently long" * 2,
            100,
            parent_id="test_node",
            reporter=reporter,
        )

    chunk_tokens = (
        index_config.target_chunk_tokens
        if index_config.target_chunk_tokens is not None
        else index_config.target_embedding_context_tokens
    )
    data = reporter.get_telemetry_data("test_doc", chunk_tokens)
    test_node = next(n for n in data["nodes"] if n["node_id"] == "test_node")
    attempts = test_node["summary_attempts"]
    assert len(attempts) == 1
    assert attempts[0].get("cached_tokens", 0) == 0


@pytest.mark.asyncio
async def test_cached_tokens_across_multiple_retries(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    index_config = IndexConfig.load(
        retry_threshold=0.05,
        max_retries=3,
        target_chunk_tokens=100,
    )
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, index_config, vector_index)

    doc_store = storage_backend.for_document("telemetry-doc")
    doc_store.set_metadata(
        file_path="telemetry.txt",
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
    )

    reporter = create_test_reporter(index_config)
    api_calls: list[OpenAIMockParams] = []

    async def mock_create(**kwargs: OpenAIMockParams) -> MockOpenAIResponseWithCache:
        api_calls.append(cast(OpenAIMockParams, kwargs))
        call_num = len(api_calls)
        if call_num == 1:
            return MockOpenAIResponseWithCache("A" * 150, 1350, 150, 0)
        if call_num == 2:
            return MockOpenAIResponseWithCache("B" * 130, 1500, 130, 1200)
        if call_num == 3:
            return MockOpenAIResponseWithCache("C" * 102, 1650, 102, 1400)
        return MockOpenAIResponseWithCache("", 0, 0)

    indexer_runtime_harness.llm_service.client = MagicMock()
    indexer_runtime_harness.llm_service.client.chat.completions.create = mock_create

    with patched_tokenizers(lambda text: [0] * len(text)):
        await indexer_runtime_harness.llm_service._summarize_text(
            "Test content that needs to be long enough to trigger summarization" * 2
            + " "
            + "More content that also needs to be sufficiently long" * 2,
            100,
            parent_id="test_node",
            reporter=reporter,
        )

    chunk_tokens = (
        index_config.target_chunk_tokens
        if index_config.target_chunk_tokens is not None
        else index_config.target_embedding_context_tokens
    )
    data = reporter.get_telemetry_data("test_doc", chunk_tokens)
    test_node = next(n for n in data["nodes"] if n["node_id"] == "test_node")
    attempts = test_node["summary_attempts"]
    assert len(attempts) == 3
    assert attempts[0].get("cached_tokens", 0) == 0
    assert attempts[1]["cached_tokens"] == 1200
    assert attempts[2]["cached_tokens"] == 1400
    cache_rates = [
        a.get("cached_tokens", 0) / a["prompt_tokens"] if a["prompt_tokens"] else 0
        for a in attempts
    ]
    assert cache_rates[0] == 0
    assert cache_rates[1] > 0.7
    assert cache_rates[2] > cache_rates[1]


@pytest.mark.asyncio
async def test_passthrough_summary_has_no_cached_tokens(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    index_config = IndexConfig.load(target_chunk_tokens=100)
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, index_config, vector_index)

    doc_store = storage_backend.for_document("telemetry-doc")
    doc_store.set_metadata(
        file_path="telemetry.txt",
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
    )

    reporter = create_test_reporter(index_config)

    async def mock_create(**_kwargs: OpenAIMockParams) -> MockOpenAIResponseWithCache:
        pytest.fail("Should not call API for passthrough")

    indexer_runtime_harness.llm_service.client = MagicMock()
    indexer_runtime_harness.llm_service.client.chat.completions.create = mock_create

    with patched_tokenizers(lambda _text: [0] * 50, lambda _text: 50):
        await indexer_runtime_harness.llm_service._summarize_text(
            "Short Text",
            100,
            parent_id="test_node",
            reporter=reporter,
        )

    chunk_tokens = (
        index_config.target_chunk_tokens
        if index_config.target_chunk_tokens is not None
        else index_config.target_embedding_context_tokens
    )
    data = reporter.get_telemetry_data("test_doc", chunk_tokens)
    test_node = next(n for n in data["nodes"] if n["node_id"] == "test_node")
    attempts = test_node["summary_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["model"] == "passthrough"
    assert attempts[0].get("cached_tokens", 0) == 0


@pytest.mark.asyncio
async def test_cached_tokens_with_high_cache_rate(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    index_config = IndexConfig.load(
        retry_threshold=0.1,
        max_retries=1,
        target_chunk_tokens=100,
    )
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, index_config, vector_index)

    doc_store = storage_backend.for_document("telemetry-doc")
    doc_store.set_metadata(
        file_path="telemetry.txt",
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
    )

    reporter = create_test_reporter(index_config)

    async def mock_create(**kwargs: OpenAIMockParams) -> MockOpenAIResponseWithCache:
        messages = cast(list[dict[str, object]], kwargs.get("messages", []))
        if len(messages) == 4:
            return MockOpenAIResponseWithCache("A" * 120, 2000, 120, 0)
        return MockOpenAIResponseWithCache("B" * 100, 2200, 100, 2090)

    indexer_runtime_harness.llm_service.client = MagicMock()
    indexer_runtime_harness.llm_service.client.chat.completions.create = mock_create

    with patched_tokenizers(lambda text: [0] * len(text)):
        await indexer_runtime_harness.llm_service._summarize_text(
            "Long content " * 50 + " " + "More content " * 50,
            100,
            parent_id="test_node",
            reporter=reporter,
        )

    chunk_tokens = (
        index_config.target_chunk_tokens
        if index_config.target_chunk_tokens is not None
        else index_config.target_embedding_context_tokens
    )
    data = reporter.get_telemetry_data("test_doc", chunk_tokens)
    test_node = next(n for n in data["nodes"] if n["node_id"] == "test_node")
    attempts = test_node["summary_attempts"]
    assert len(attempts) == 2
    cache_rate = attempts[1]["cached_tokens"] / attempts[1]["prompt_tokens"]
    assert cache_rate >= 0.95


def _configure_runtime(
    harness: IndexerRuntimeHarness,
    config: IndexConfig,
    vector_index: RecordingVectorIndex,
) -> None:
    """Apply config overrides to the runtime harness for telemetry tests."""

    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = TextSplitter(config)
    harness.indexing_engine._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config
    vector_factory = lambda _model: vector_index  # noqa: E731
    harness.runtime._vector_index_factory = vector_factory
    harness.indexing_engine._vector_index_factory = vector_factory
