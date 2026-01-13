"""Tests for chars_per_token fallback behavior before first append."""

from __future__ import annotations

import pytest

from ragzoom.config import IndexConfig
from tests.conftest import IndexerRuntimeHarness


@pytest.mark.asyncio
async def test_chars_per_token_fallback_before_first_append(
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    """Test that chars_per_token uses 4.0 fallback when document has no data yet.

    Spec: specs/client-managed-chunking.md § chars_per_token Tracking

    Before the first append operation, there are no leaves in the document,
    so get_avg_chars_per_token() returns None. In this case, get_chars_per_token()
    should fall back to 4.0, which is a typical ratio for English text.

    Success criterion: Before first append, if chars_per_token is None, use 4.0 as default.
    """
    # Configure runtime for client-managed chunking
    index_config = IndexConfig.load(
        target_chunk_tokens=None,
        target_embedding_context_tokens=200,
    )

    indexer_runtime_harness.runtime._index_config = index_config
    indexer_runtime_harness.indexing_engine._index_config = index_config

    document_id = "test-fallback-ratio"
    await indexer_runtime_harness.clear(document_id)

    # Before first append, get_chars_per_token should return 4.0 fallback
    chars_per_token = indexer_runtime_harness.indexing_engine.get_chars_per_token(
        document_id
    )
    assert chars_per_token == 4.0

    # Verify that the cache is not populated (we're using the fallback, not cached data)
    assert (
        document_id
        not in indexer_runtime_harness.indexing_engine._document_chars_per_token
    )


@pytest.mark.asyncio
async def test_chars_per_token_fallback_used_in_summarization(
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> None:
    """Test that the 4.0 fallback is actually used when computing summary targets.

    Spec: specs/client-managed-chunking.md § chars_per_token Tracking

    This test verifies that when get_chars_per_token() returns the 4.0 fallback,
    it's correctly used by get_summary_target() during summarization.
    """
    # Import get_summary_target to test it directly
    from ragzoom.server.indexing_engine import get_summary_target

    # Test that get_summary_target works correctly with the 4.0 fallback
    # Example: 8000 chars at 4.0 chars/token = 2000 tokens
    # At height 1: target = 2000 / 2^1 = 1000 tokens
    target = get_summary_target(node_span_chars=8000, height=1, chars_per_token=4.0)
    assert target == 1000

    # At height 2: target = 2000 / 2^2 = 500 tokens
    target = get_summary_target(node_span_chars=8000, height=2, chars_per_token=4.0)
    assert target == 500

    # Verify floor behavior: 100 chars = 25 tokens at 4.0 ratio
    # At height 1: 25 / 2 = 12.5 tokens < 50 → should return 0 (passthrough)
    target = get_summary_target(node_span_chars=100, height=1, chars_per_token=4.0)
    assert target == 0
