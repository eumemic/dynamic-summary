"""Tests for Phase 4 dynamic summary target calculation in _summarize_pair."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.server.indexing_engine import get_summary_target
from ragzoom.services.summary_utils import AccumulatedUsage, SummaryResult
from tests.conftest import IndexerRuntimeHarness
from tests.vector_index_stubs import RecordingVectorIndex


def _create_mock_async_openai_client() -> AsyncMock:
    """Create a mock async OpenAI client for IndexingEngine."""
    from types import SimpleNamespace

    mock_client = AsyncMock()

    async def async_mock_embeddings(*args: object, **kwargs: object) -> object:
        from typing import cast

        input_texts = cast(list[str] | str, kwargs.get("input", []))
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        embedding_value = [0.1] * 1536
        num_items = len(input_texts)
        return type(
            "MockResponse",
            (),
            {
                "data": [
                    type("MockEmbedding", (), {"embedding": embedding_value})()
                    for _ in input_texts
                ],
                "usage": SimpleNamespace(
                    prompt_tokens=num_items * 10, total_tokens=num_items * 10
                ),
            },
        )()

    mock_client.embeddings.create = async_mock_embeddings
    return mock_client


@pytest.mark.asyncio
@pytest.mark.slow_threshold(10.0)  # Complex indexing operation needs more time on CI
async def test_summarize_pair_uses_dynamic_target_when_none(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    """Test _summarize_pair uses dynamic target when target_chunk_tokens is None."""
    # Configure with target_chunk_tokens=None to activate client-managed chunking
    index_config = IndexConfig.load(
        target_chunk_tokens=None,
        target_embedding_context_tokens=200,
    )
    vector_index = RecordingVectorIndex()

    # Configure runtime with the new config
    harness = indexer_runtime_harness
    harness.runtime._index_config = index_config
    harness.runtime._append_executor._config = index_config
    harness.indexing_engine._index_config = index_config
    harness.llm_service.config = index_config
    harness.telemetry_manager._index_config = index_config

    def vector_factory(_model: str) -> RecordingVectorIndex:
        return vector_index

    harness.runtime._vector_index_factory = vector_factory
    harness.indexing_engine._vector_index_factory = vector_factory
    harness.indexing_engine._openai_client = _create_mock_async_openai_client()

    document_id = "test-dynamic-target"
    storage_backend.clear_document(document_id)
    store = storage_backend.for_document(document_id)
    store.set_metadata(
        file_path="test.txt",
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
    )

    # Track calls to _summarize_text to verify target_tokens calculation
    captured_targets: list[int] = []

    async def capture_summary(*args: object, **kwargs: object) -> SummaryResult:
        """Capture target_tokens for verification."""
        # Args: (text, target_tokens, ...)
        from typing import cast

        if len(args) >= 2:
            captured_targets.append(cast(int, args[1]))
        return SummaryResult(
            summary="test summary",
            retry_count=0,
            summary_tokens=50,
            usage=AccumulatedUsage(prompt_tokens=100, completion_tokens=50),
        )

    async def embed_side_effect(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    embed_mock = AsyncMock(side_effect=embed_side_effect)
    summary_mock = AsyncMock(side_effect=capture_summary)

    harness.llm_service.client = _create_mock_async_openai_client()

    with (
        patch.object(harness.llm_service, "_summarize_text", new=summary_mock),
        patch.object(harness.llm_service, "embed_texts", new=embed_mock),
    ):
        # Append text that will create leaves and trigger summarization
        # With client-managed chunking, each line becomes a leaf
        text = "\n\n".join([f"Turn {i}: Some content here" * 50 for i in range(4)])

        await harness.clear(document_id)
        await harness.append(
            document_id, text, replace_existing=True, file_path="test.txt"
        )
        await harness.wait_for_idle(document_id)

    # Verify that summarization was called
    assert len(captured_targets) > 0, "Expected at least one summary call"

    # Verify that dynamic targets were computed (not fixed values)
    # When target_chunk_tokens=None, we should NOT see the fixed 200 value
    # (target_embedding_context_tokens) being used for inner node summarization

    # Verify dynamic targets are being calculated
    # target=0 is valid (signals passthrough when below 50-token floor)
    # target>0 means dynamic calculation is working
    for target in captured_targets:
        # Dynamic targets should be computed based on span size
        # They should be >= 0 (0 means passthrough)
        assert target >= 0, f"Target should be non-negative, got {target}"

    # The key test: when target_chunk_tokens=None, we should be calling
    # get_summary_target(), which means we won't see the fixed 200 value
    # (target_embedding_context_tokens) for ALL calls
    if 200 in captured_targets:
        # Check if ALL targets are 200 (old behavior) or if some vary (new behavior)
        assert not all(
            t == 200 for t in captured_targets
        ), "Should use dynamic targets, not fixed 200 for all calls"


@pytest.mark.asyncio
@pytest.mark.slow_threshold(10.0)  # Complex indexing operation needs more time on CI
async def test_summarize_pair_uses_fixed_target_when_set(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    """Test _summarize_pair uses fixed target when target_chunk_tokens is an int."""
    # Configure with fixed target_chunk_tokens
    fixed_target = 150
    index_config = IndexConfig.load(
        target_chunk_tokens=fixed_target,
        target_embedding_context_tokens=200,
    )
    vector_index = RecordingVectorIndex()

    # Configure runtime with the new config
    harness = indexer_runtime_harness
    harness.runtime._index_config = index_config
    harness.runtime._append_executor._config = index_config
    harness.indexing_engine._index_config = index_config
    harness.llm_service.config = index_config
    harness.telemetry_manager._index_config = index_config

    def vector_factory(_model: str) -> RecordingVectorIndex:
        return vector_index

    harness.runtime._vector_index_factory = vector_factory
    harness.indexing_engine._vector_index_factory = vector_factory
    harness.indexing_engine._openai_client = _create_mock_async_openai_client()

    document_id = "test-fixed-target"
    storage_backend.clear_document(document_id)
    store = storage_backend.for_document(document_id)
    store.set_metadata(
        file_path="test.txt",
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
    )

    # Track calls to _summarize_text to verify target_tokens
    captured_targets: list[int] = []

    async def capture_summary(*args: object, **kwargs: object) -> SummaryResult:
        """Capture target_tokens for verification."""
        # Args: (text, target_tokens, ...)
        from typing import cast

        if len(args) >= 2:
            captured_targets.append(cast(int, args[1]))
        return SummaryResult(
            summary="test summary",
            retry_count=0,
            summary_tokens=50,
            usage=AccumulatedUsage(prompt_tokens=100, completion_tokens=50),
        )

    async def embed_side_effect(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    embed_mock = AsyncMock(side_effect=embed_side_effect)
    summary_mock = AsyncMock(side_effect=capture_summary)

    harness.llm_service.client = _create_mock_async_openai_client()

    with (
        patch.object(harness.llm_service, "_summarize_text", new=summary_mock),
        patch.object(harness.llm_service, "embed_texts", new=embed_mock),
    ):
        # Append text that will trigger summarization
        text = "\n\n".join([f"Chunk {i}: Some content here" * 20 for i in range(6)])

        await harness.clear(document_id)
        await harness.append(
            document_id, text, replace_existing=True, file_path="test.txt"
        )
        await harness.wait_for_idle(document_id)

    # Verify that summarization was called
    assert len(captured_targets) > 0, "Expected at least one summary call"

    # Verify that ALL calls used the fixed target
    for target in captured_targets:
        assert (
            target == fixed_target
        ), f"Expected fixed target {fixed_target}, got {target}"


@pytest.mark.asyncio
async def test_get_summary_target_integration(
    storage_backend: StorageBackend,
) -> None:
    """Test get_summary_target function with realistic values."""
    # Test case 1: 8000 chars = 2000 tokens at 4 chars/token
    # Height 1: 2000 / 2^1 = 1000 tokens
    target = get_summary_target(8000, 1, 4.0)
    assert target == 1000

    # Test case 2: Height 2: 2000 / 2^2 = 500 tokens
    target = get_summary_target(8000, 2, 4.0)
    assert target == 500

    # Test case 3: Below floor (< 50 tokens) returns 0
    target = get_summary_target(100, 1, 4.0)
    assert target == 0  # 100 chars = 25 tokens, / 2 = 12.5 < 50

    # Test case 4: At floor boundary
    target = get_summary_target(400, 1, 4.0)
    assert target == 50  # 400 chars = 100 tokens, / 2 = 50

    # Test case 5: Different chars_per_token ratio
    target = get_summary_target(10000, 1, 5.0)
    assert target == 1000  # 10000 chars = 2000 tokens at 5 chars/token, / 2 = 1000
