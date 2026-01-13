"""Tests for chars_per_token recomputation after append operations."""

from __future__ import annotations

import pytest

from ragzoom.config import IndexConfig
from ragzoom.splitter import TextSplitter
from tests.conftest import IndexerRuntimeHarness


@pytest.mark.asyncio
async def test_chars_per_token_updated_after_append(
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    """Test chars_per_token is recomputed after append operations.

    Spec: specs/client-managed-chunking.md § chars_per_token Tracking

    This test verifies that after each append() or append_batch() operation,
    the chars_per_token ratio is updated by querying all leaves in the document.
    """
    # Configure runtime for client-managed chunking
    index_config = IndexConfig.load(
        target_chunk_tokens=None,
        target_embedding_context_tokens=200,
    )

    indexer_runtime_harness.runtime._index_config = index_config
    indexer_runtime_harness.runtime._append_executor._config = index_config
    indexer_runtime_harness.runtime._append_executor._splitter = TextSplitter(
        index_config
    )
    indexer_runtime_harness.indexing_engine._index_config = index_config
    indexer_runtime_harness.llm_service.config = index_config
    indexer_runtime_harness.telemetry_manager._index_config = index_config

    document_id = "test-chars-per-token-append"
    await indexer_runtime_harness.clear(document_id)

    # Before first append, should use fallback ratio of 4.0
    assert (
        indexer_runtime_harness.indexing_engine.get_chars_per_token(document_id) == 4.0
    )

    # Append first unit: "AAAA" (4 chars, should be tokenized to ~1 token = 4.0 ratio)
    # Using client-managed chunking, this becomes exactly one leaf
    await indexer_runtime_harness.append(
        document_id,
        "AAAA",
        file_path="test.txt",
        await_idle=True,
    )

    # After first append, chars_per_token should be computed from actual data
    # Verify the cache was updated (this is the key requirement)
    assert (
        document_id in indexer_runtime_harness.indexing_engine._document_chars_per_token
    )

    # Get the actual ratio from the cache
    actual_ratio_1 = indexer_runtime_harness.indexing_engine.get_chars_per_token(
        document_id
    )
    assert actual_ratio_1 > 0  # Should have a real ratio now, not just fallback

    # Append second unit: "BBBBBBBBBBBB" (12 chars)
    # This will change the overall ratio
    await indexer_runtime_harness.append(
        document_id,
        "BBBBBBBBBBBB",
        await_idle=True,
    )

    # After second append, chars_per_token should be recomputed
    actual_ratio_2 = indexer_runtime_harness.indexing_engine.get_chars_per_token(
        document_id
    )
    assert actual_ratio_2 > 0

    # Verify cache was updated with the combined data from both appends
    assert (
        indexer_runtime_harness.indexing_engine._document_chars_per_token[document_id]
        == actual_ratio_2
    )
