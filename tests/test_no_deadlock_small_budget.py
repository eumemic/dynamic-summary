"""Test that IndexingEngine doesn't deadlock with small budget values.

This tests the fix for issue #287 where the old WorkerCoordinator would
deadlock with small preceding context budget values.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from tests.chunk_size_regression_harness import configure_runtime
from tests.conftest import IndexerRuntimeHarness


@pytest.mark.asyncio
@pytest.mark.slow_threshold(6.0)
async def test_no_deadlock_with_small_budget(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    """Test that indexing completes without deadlock even with small B and K values.

    This specifically tests the scenario that caused deadlock in the old
    WorkerCoordinator when preceding context budget was small (like 400).

    The system is designed to handle "forests" (multiple roots) when odd numbers
    of nodes prevent perfect pairing. This is by design - the DP and greedy
    tiling algorithms handle forests correctly.
    """
    # Configure with small budget values that could trigger deadlock
    index_config = IndexConfig.load(
        target_chunk_tokens=200,
        context_lag_tokens=400,  # K=400 - small lag
        embedding_batch_size=2,
    )

    configure_runtime(indexer_runtime_harness, index_config)

    # Set up async mock for LLM service
    mock_async_client = AsyncMock()

    async def mock_embeddings(*args: object, **kwargs: object) -> object:
        input_texts = cast(list[str] | str, kwargs.get("input", []))
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        return MagicMock(data=[MagicMock(embedding=[0.1] * 1536) for _ in input_texts])

    async def mock_chat(*args: object, **kwargs: object) -> object:
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="Summary text."))],
            usage=MagicMock(prompt_tokens=100, completion_tokens=50),
        )

    mock_async_client.embeddings.create = mock_embeddings
    mock_async_client.chat.completions.create = mock_chat
    indexer_runtime_harness.llm_service.client = mock_async_client

    # Create a document large enough to trigger multiple tree levels
    chunk_text = "This is test content that should create multiple chunks. " * 40
    large_document = " ".join([chunk_text for _ in range(10)])

    document_id = "deadlock-test"
    storage_backend.clear_document(document_id)
    doc_store = storage_backend.for_document(document_id)
    doc_store.set_metadata(
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )

    # This should complete without timeout/deadlock - the key assertion!
    await indexer_runtime_harness.append(
        document_id,
        large_document,
        replace_existing=True,
        file_path="deadlock-test.txt",
    )

    # Wait for idle with a reasonable timeout (if this hangs, the test will fail)
    await asyncio.wait_for(
        indexer_runtime_harness.wait_for_idle(document_id),
        timeout=5.0,
    )

    # Get all nodes to verify tree integrity
    nodes = cast(Sequence[TreeNode], doc_store.nodes.get_all())
    roots = doc_store.nodes.get_root_nodes(document_id)

    # Verify no corrupted spans (span_end >= span_start for all nodes)
    corrupt_spans = []
    for node in nodes:
        span_start = int(getattr(node, "span_start", 0))
        span_end = int(getattr(node, "span_end", 0))
        height = int(getattr(node, "height", 0))
        if span_end < span_start:
            corrupt_spans.append(
                f"Node {node.id[:8]}: span_end ({span_end}) < span_start ({span_start})"
            )
        elif span_start == span_end and height > 0:
            corrupt_spans.append(
                f"Node {node.id[:8]}: zero-width span at height {height}"
            )
    assert len(corrupt_spans) == 0, "Corrupted spans found:\n" + "\n".join(
        corrupt_spans
    )

    # Verify roots are sorted by span (no wraparound issues)
    sorted_roots = sorted(roots, key=lambda n: int(getattr(n, "span_start", 0)))
    for i in range(len(sorted_roots) - 1):
        current_end = int(getattr(sorted_roots[i], "span_end", 0))
        next_start = int(getattr(sorted_roots[i + 1], "span_start", 0))
        assert current_end <= next_start, (
            f"Root spans overlap: {sorted_roots[i].id[:8]} ends at {current_end}, "
            f"{sorted_roots[i + 1].id[:8]} starts at {next_start}"
        )

    # Verify some nodes were created (not empty tree)
    assert len(nodes) >= 1, "No nodes created"
    assert len(roots) >= 1, "No roots found"

    # Verify engine is idle with no pending work
    engine = indexer_runtime_harness.indexing_engine
    assert len(engine._active_jobs) == 0, "Engine has active jobs after wait_for_idle"
    assert document_id not in engine._active_documents, "Document still marked active"
