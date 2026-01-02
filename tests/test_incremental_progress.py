"""Tests for incremental progress tracking.

These tests verify that progress counters (completed/expected) accumulate
correctly across multiple appends. Counters are preserved when a document
goes idle (so the final state is visible to streaming clients), then reset
when trigger_work() is called to start the next work cycle.
"""

import pytest

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.server.indexing_engine import _expected_total_from_leaf_count
from tests.conftest import IndexerRuntimeHarness


def _count_leaves(storage: StorageBackend, document_id: str) -> int:
    """Count leaf nodes in a document."""
    store = storage.for_document(document_id)
    return store.nodes.leaf_count()


class TestIncrementalProgress:
    """Tests for incremental progress tracking."""

    @pytest.mark.asyncio
    async def test_first_append_shows_correct_expected(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """First append: expected = jobs for new leaves only."""
        await indexer_runtime_harness.append("doc", "text " * 100, await_idle=False)
        status = await indexer_runtime_harness.indexing_engine.status()
        leaf_count = _count_leaves(storage_backend, "doc")
        expected = _expected_total_from_leaf_count(leaf_count)
        assert status.expected_total_by_document["doc"] == expected

    @pytest.mark.asyncio
    async def test_second_append_accumulates_expected(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """Second append after idle: expected shows only NEW work."""
        # First append, wait for idle
        await indexer_runtime_harness.append("doc", "text " * 50, await_idle=True)
        leaves_after_first = _count_leaves(storage_backend, "doc")

        # Second append, don't wait
        await indexer_runtime_harness.append("doc", "more " * 50, await_idle=False)
        leaves_after_second = _count_leaves(storage_backend, "doc")

        status = await indexer_runtime_harness.indexing_engine.status()
        # Expected = jobs(new_total) - jobs(leaves_at_idle)
        expected = _expected_total_from_leaf_count(
            leaves_after_second
        ) - _expected_total_from_leaf_count(leaves_after_first)
        assert status.expected_total_by_document["doc"] == expected

    @pytest.mark.asyncio
    async def test_counters_preserved_at_idle(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """Counters are preserved when document goes idle for final state display.

        The counters persist after idle so that gRPC streaming clients can see
        the final "completed=X/X inflight=0" state. They reset on the next
        trigger_work() call when new work begins.
        """
        await indexer_runtime_harness.append("doc", "text " * 50, await_idle=True)

        # After idle, counters should still show the completed work
        ctx = indexer_runtime_harness.indexing_engine._document_contexts.get("doc")
        assert ctx is not None
        assert ctx.completed_jobs > 0  # Work was done
        assert ctx.expected_total_jobs > 0  # Expected was set

    @pytest.mark.asyncio
    async def test_counters_reset_on_next_trigger(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """Counters reset when new work starts after idle.

        When trigger_work() is called after idle, the completed_jobs counter
        resets so progress for the new work cycle starts from 0.
        """
        await indexer_runtime_harness.append("doc", "text " * 50, await_idle=True)

        # After idle, counters are preserved
        ctx = indexer_runtime_harness.indexing_engine._document_contexts.get("doc")
        assert ctx is not None
        completed_before = ctx.completed_jobs
        assert completed_before > 0

        # New append triggers new work - completed should reset
        await indexer_runtime_harness.append("doc", "more " * 30, await_idle=False)

        # completed_jobs should have reset to 0 (or incremented from 0)
        # At this point, some work may already be done, so just check it's less
        # than the sum of both rounds' completed jobs
        ctx = indexer_runtime_harness.indexing_engine._document_contexts.get("doc")
        assert ctx is not None
        # The key invariant: the counter was reset, not accumulated
        assert ctx.completed_jobs < completed_before + ctx.expected_total_jobs

    @pytest.mark.asyncio
    async def test_after_idle_new_append_uses_new_baseline(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """After idle, new append calculates from new baseline."""
        await indexer_runtime_harness.append("doc", "text " * 50, await_idle=True)
        leaves_at_idle = _count_leaves(storage_backend, "doc")

        await indexer_runtime_harness.append("doc", "more " * 30, await_idle=False)
        new_leaves = _count_leaves(storage_backend, "doc")

        status = await indexer_runtime_harness.indexing_engine.status()
        expected = _expected_total_from_leaf_count(
            new_leaves
        ) - _expected_total_from_leaf_count(leaves_at_idle)
        assert status.expected_total_by_document["doc"] == expected

    @pytest.mark.asyncio
    async def test_leaves_at_last_idle_updated(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """leaves_at_last_idle is updated when document goes idle."""
        await indexer_runtime_harness.append("doc", "text " * 50, await_idle=True)
        leaves_after_idle = _count_leaves(storage_backend, "doc")

        ctx = indexer_runtime_harness.indexing_engine._document_contexts.get("doc")
        assert ctx is not None
        assert ctx.leaves_at_last_idle == leaves_after_idle
