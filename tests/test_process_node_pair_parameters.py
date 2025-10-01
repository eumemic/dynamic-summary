"""Regression tests for summary parameter propagation in the runtime pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.splitter import TextSplitter
from tests.conftest import IndexerRuntimeHarness
from tests.vector_index_stubs import RecordingVectorIndex


def _configure_runtime(
    harness: IndexerRuntimeHarness,
    config: IndexConfig,
    vector_index: RecordingVectorIndex,
) -> None:
    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = TextSplitter(config)
    harness.worker_coordinator._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config
    harness.runtime._vector_index_factory = lambda _model: vector_index
    harness.worker_coordinator._vector_index_factory = lambda _doc_id: vector_index


def _document_text(chunks: int) -> str:
    return "\n\n".join(
        f"Chunk {i}: This is content for chunk number {i}. " * 6 for i in range(chunks)
    )


@pytest.mark.asyncio
async def test_worker_coordinator_passes_token_counts(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    index_config = IndexConfig.load(
        target_chunk_tokens=120,
        preceding_context_tokens=40,
    )
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, index_config, vector_index)

    document_id = "summary-params"
    storage_backend.clear_document(document_id)
    store = storage_backend.for_document(document_id)
    store.set_metadata(
        file_path="summary-params.txt",
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
    )

    summary_mock = AsyncMock(return_value=("summary", 0, 100))

    async def embed_side_effect(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    embed_mock = AsyncMock(side_effect=embed_side_effect)

    indexer_runtime_harness.llm_service.client = AsyncMock()
    with (
        patch.object(
            indexer_runtime_harness.llm_service,
            "_summarize_text",
            new=summary_mock,
        ),
        patch.object(
            indexer_runtime_harness.llm_service,
            "embed_texts",
            new=embed_mock,
        ),
    ):
        await indexer_runtime_harness.clear(document_id)
        await indexer_runtime_harness.append(
            document_id,
            _document_text(6),
            replace_existing=True,
            file_path="summary-params.txt",
        )
        await indexer_runtime_harness.wait_for_idle(document_id)

    assert summary_mock.await_count > 0
    for call in summary_mock.await_args_list:
        args, kwargs = call
        left_text, right_text, target_tokens = args
        assert target_tokens == index_config.target_chunk_tokens
        assert isinstance(kwargs.get("left_token_count"), int)
        assert isinstance(kwargs.get("right_token_count"), int)
        assert kwargs["parent_id"] is not None
        assert "reporter" in kwargs


@pytest.mark.asyncio
async def test_prev_context_present_when_preceding_neighbor_exists(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    index_config = IndexConfig.load(
        target_chunk_tokens=90,
        preceding_context_tokens=45,
    )
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, index_config, vector_index)

    document_id = "summary-prev-context"
    storage_backend.clear_document(document_id)
    store = storage_backend.for_document(document_id)
    store.set_metadata(
        file_path="summary-prev-context.txt",
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
    )

    summary_mock = AsyncMock(return_value=("summary", 0, 80))

    async def embed_side_effect(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    embed_mock = AsyncMock(side_effect=embed_side_effect)

    indexer_runtime_harness.llm_service.client = AsyncMock()
    with (
        patch.object(
            indexer_runtime_harness.llm_service,
            "_summarize_text",
            new=summary_mock,
        ),
        patch.object(
            indexer_runtime_harness.llm_service,
            "embed_texts",
            new=embed_mock,
        ),
    ):
        await indexer_runtime_harness.clear(document_id)
        await indexer_runtime_harness.append(
            document_id,
            _document_text(8),
            replace_existing=True,
            file_path="summary-prev-context.txt",
        )
        await indexer_runtime_harness.wait_for_idle(document_id)

    contexts = [
        kwargs.get("prev_context") for _, kwargs in summary_mock.await_args_list
    ]
    assert None in contexts
    assert any(isinstance(ctx, str) and ctx.strip() for ctx in contexts)
    for ctx in contexts:
        if ctx is not None:
            assert isinstance(ctx, str) and ctx.strip()
