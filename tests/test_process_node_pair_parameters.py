"""Regression tests for summary parameter propagation in the runtime pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.services.summary_utils import AccumulatedUsage, SummaryResult
from ragzoom.splitter import TextSplitter
from tests.conftest import IndexerRuntimeHarness
from tests.vector_index_stubs import RecordingVectorIndex


def _create_mock_sync_openai_client() -> MagicMock:
    """Create a mock sync OpenAI client for IndexingEngine's retriever."""
    from types import SimpleNamespace

    mock_client = MagicMock()

    def sync_mock_embeddings(*args: object, **kwargs: object) -> object:
        from typing import cast

        input_texts = cast(list[str] | str, kwargs.get("input", []))
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        embedding_value = [0.1] * 1536
        num_items = len(input_texts)
        return MagicMock(
            data=[MagicMock(embedding=embedding_value) for _ in input_texts],
            usage=SimpleNamespace(
                prompt_tokens=num_items * 10, total_tokens=num_items * 10
            ),
        )

    mock_client.embeddings.create = sync_mock_embeddings
    return mock_client


def _create_mock_async_openai_client() -> AsyncMock:
    """Create a mock async OpenAI client for IndexingEngine's retriever."""
    from types import SimpleNamespace

    mock_client = AsyncMock()

    async def async_mock_embeddings(*args: object, **kwargs: object) -> object:
        from typing import cast

        input_texts = cast(list[str] | str, kwargs.get("input", []))
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        embedding_value = [0.1] * 1536
        num_items = len(input_texts)
        return MagicMock(
            data=[MagicMock(embedding=embedding_value) for _ in input_texts],
            usage=SimpleNamespace(
                prompt_tokens=num_items * 10, total_tokens=num_items * 10
            ),
        )

    mock_client.embeddings.create = async_mock_embeddings
    return mock_client


def _configure_runtime(
    harness: IndexerRuntimeHarness,
    config: IndexConfig,
    vector_index: RecordingVectorIndex,
) -> None:
    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = TextSplitter(config)
    harness.indexing_engine._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config
    vector_factory = lambda _model: vector_index  # noqa: E731
    harness.runtime._vector_index_factory = vector_factory
    harness.indexing_engine._vector_index_factory = vector_factory
    # Mock the sync OpenAI client used by IndexingEngine's retriever
    harness.indexing_engine._openai_client = _create_mock_sync_openai_client()


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

    summary_mock = AsyncMock(
        return_value=SummaryResult(
            summary="summary",
            retry_count=0,
            summary_tokens=100,
            usage=AccumulatedUsage(prompt_tokens=50, completion_tokens=100),
        )
    )

    async def embed_side_effect(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    embed_mock = AsyncMock(side_effect=embed_side_effect)

    indexer_runtime_harness.llm_service.client = _create_mock_async_openai_client()
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
        # New unified API: text, target_tokens (instead of left_text, right_text, target_tokens)
        text, target_tokens = args
        assert isinstance(text, str)
        assert target_tokens == index_config.target_chunk_tokens
        # Token count is now passed as text_tokens (optional, may be None or int)
        assert kwargs.get("text_tokens") is None or isinstance(
            kwargs.get("text_tokens"), int
        )
        assert kwargs["parent_id"] is not None
        assert "reporter" in kwargs


@pytest.mark.asyncio
async def test_prev_context_present_when_preceding_neighbor_exists(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    index_config = IndexConfig.load(
        target_chunk_tokens=90,
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

    summary_mock = AsyncMock(
        return_value=SummaryResult(
            summary="summary",
            retry_count=0,
            summary_tokens=80,
            usage=AccumulatedUsage(prompt_tokens=50, completion_tokens=80),
        )
    )

    async def embed_side_effect(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    embed_mock = AsyncMock(side_effect=embed_side_effect)

    indexer_runtime_harness.llm_service.client = _create_mock_async_openai_client()
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
    # With the new unified API:
    # - Leaf context summarization calls pass prev_context=None
    # - Inner node pair summarization calls pass prev_context as a string:
    #   - Empty string ("") for first node (span_start=0)
    #   - Non-empty string for nodes with preceding context
    assert (
        None in contexts or "" in contexts
    )  # At least one call without preceding context
    assert any(
        isinstance(ctx, str) and ctx.strip() for ctx in contexts
    )  # Some calls have context
    for ctx in contexts:
        # Contexts should be strings or None (for leaf context summarization)
        assert ctx is None or isinstance(ctx, str)
        # Non-empty string contexts should have content
        if isinstance(ctx, str) and ctx != "":
            assert ctx.strip()


@pytest.mark.asyncio
async def test_worker_uses_document_custom_prompt(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    """Verify worker summarization uses document's stored custom prompt.

    Spec: specs/custom-prompt-config.md § Implementation
    Test: Integration test verifying custom prompt is used in actual LLM calls
    """
    index_config = IndexConfig.load(
        target_chunk_tokens=120,
    )
    vector_index = RecordingVectorIndex()
    _configure_runtime(indexer_runtime_harness, index_config, vector_index)

    document_id = "doc-custom-prompt"
    custom_prompt = "You are a legal document summarizer. Preserve exact terminology."

    storage_backend.clear_document(document_id)

    summary_mock = AsyncMock(
        return_value=SummaryResult(
            summary="summarized legal text",
            retry_count=0,
            summary_tokens=100,
            usage=AccumulatedUsage(prompt_tokens=50, completion_tokens=100),
        )
    )

    async def embed_side_effect(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    embed_mock = AsyncMock(side_effect=embed_side_effect)

    indexer_runtime_harness.llm_service.client = _create_mock_async_openai_client()
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
        # Append with custom prompt - this stores the prompt with the document
        await indexer_runtime_harness.append(
            document_id,
            _document_text(6),  # Enough chunks to trigger summarization
            replace_existing=True,
            file_path="legal-doc.txt",
            summary_system_prompt=custom_prompt,
        )
        await indexer_runtime_harness.wait_for_idle(document_id)

    # Verify summarize was called with the document's custom guidance
    assert summary_mock.await_count > 0, "Summarization should have been called"

    # Check all summarization calls received the custom guidance
    # Note: LLM service uses summarization_guidance (new name), but harness uses
    # summary_system_prompt (old name) until CLI flag rename work item is done
    for call in summary_mock.await_args_list:
        _, kwargs = call
        actual_guidance = kwargs.get("summarization_guidance")
        assert actual_guidance == custom_prompt, (
            f"Expected document custom guidance '{custom_prompt}', "
            f"got '{actual_guidance}'"
        )
