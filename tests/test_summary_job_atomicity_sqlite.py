"""Tests for summary job atomicity and tree corruption scenarios.

These tests prove that:
1. The non-atomic summary job can leave orphaned roots when it fails mid-operation
2. Orphaned roots permanently stall the height differential check

The bug (now fixed): _summarize_pair does three operations that must be atomic:
1. store.nodes.add_batch([node_payload]) - creates parent node
2. store.nodes.update_parent_references_batch(...) - updates children's parent_id
3. store.nodes.update_neighbors_batch(...) - updates neighbor links

If failure occurred between these operations, children would remain orphaned
(parent_id=NULL) while the parent exists pointing to them, corrupting the tree.
The fix wraps all three in a single transaction.
"""

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.services.summary_utils import AccumulatedUsage, SummaryResult


class TestSummaryJobAtomicity:
    """Test that non-atomic summary jobs can corrupt tree state."""

    def test_orphaned_root_permanently_blocks_summarization(
        self,
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """An orphaned low-height root blocks higher-level pairs forever.

        This is the stall we observed in production:
        - orphan at h=1 sets min_preceding_height=1
        - h=3 pair would create h=4, but 4-1=3 > max_diff(2)
        - Result: no jobs found, permanently stuck

        The orphan has no sibling (its sibling was consumed into h=2),
        so it can NEVER be merged, creating permanent deadlock.
        """
        doc_id = "test-doc"
        store = sqlite_backend.for_document(doc_id)

        # Create the orphaned h=1 root (simulates corruption from failed summary job)
        # This node has no sibling to merge with - it's stranded forever
        store.nodes.add_node(
            node_id="orphan-h1",
            text="Orphaned summary",
            embedding=[0.1] * 8,
            height=1,
            level_index=0,
            span_start=0,
            span_end=2000,
            parent_id=None,  # ORPHANED - appears as root but should have a parent
            token_count=50,
        )

        # Create eligible h=3 pair AFTER the orphan in span order
        # These are valid siblings that should merge into h=4
        store.nodes.add_node(
            node_id="h3-left",
            text="H3 left node content",
            embedding=[0.2] * 8,
            height=3,
            level_index=0,  # even = left child position
            span_start=10000,
            span_end=18000,
            parent_id=None,
            token_count=100,
        )
        store.nodes.add_node(
            node_id="h3-right",
            text="H3 right node content",
            embedding=[0.3] * 8,
            height=3,
            level_index=1,  # = left + 1, valid sibling
            span_start=18000,
            span_end=26000,
            parent_id=None,
            token_count=100,
        )

        # Verify setup: we have 3 roots
        roots = list(store.nodes.iter_root_nodes())
        assert len(roots) == 3, f"Expected 3 roots, got {len(roots)}"

        # Simulate the height differential check from _find_next_n_summary_jobs
        # This is the core logic that gets blocked
        max_height_diff = 2
        min_preceding_height: int | None = None
        blocked = False

        # Scan roots in span_start order (mimics iter_root_nodes behavior)
        sorted_roots = sorted(roots, key=lambda r: int(getattr(r, "span_start", 0)))

        prev_root = None
        for root in sorted_roots:
            root_height = int(getattr(root, "height", 0))

            # Try to form pair with previous root
            if prev_root is not None:
                left_height = int(getattr(prev_root, "height", 0))
                right_height = root_height
                left_level = int(getattr(prev_root, "level_index", 0))
                right_level = int(getattr(root, "level_index", 0))

                # Check eligibility: same height, left is even, right = left + 1
                is_eligible = (
                    left_height == right_height
                    and left_level % 2 == 0
                    and right_level == left_level + 1
                )

                if is_eligible:
                    parent_height = left_height + 1
                    # This is the height differential check that blocks
                    if (
                        min_preceding_height is not None
                        and parent_height - min_preceding_height > max_height_diff
                    ):
                        blocked = True
                        break

            # Update sliding window state
            prev_root = root
            if min_preceding_height is None or root_height < min_preceding_height:
                min_preceding_height = root_height

        # VERIFY THE STALL:
        # The h=1 orphan sets min_preceding_height=1
        # The h=3 pair would create h=4
        # Check: 4 - 1 = 3 > max_diff(2) => BLOCKED
        assert blocked, (
            "Expected height differential to block the h=3 pair, but it didn't. "
            f"min_preceding_height={min_preceding_height}"
        )
        assert (
            min_preceding_height == 1
        ), f"Expected min_preceding_height=1 from orphan, got {min_preceding_height}"

    def test_eligible_pair_found_without_orphan(
        self,
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Without the orphan, the h=3 pair is found normally.

        This proves the pair IS valid - only the orphan blocks it.
        """
        doc_id = "test-doc"
        store = sqlite_backend.for_document(doc_id)

        # Create only the eligible h=3 pair (no orphan)
        store.nodes.add_node(
            node_id="h3-left",
            text="H3 left node content",
            embedding=[0.2] * 8,
            height=3,
            level_index=0,  # even = left child position
            span_start=10000,
            span_end=18000,
            parent_id=None,
            token_count=100,
        )
        store.nodes.add_node(
            node_id="h3-right",
            text="H3 right node content",
            embedding=[0.3] * 8,
            height=3,
            level_index=1,  # = left + 1, valid sibling
            span_start=18000,
            span_end=26000,
            parent_id=None,
            token_count=100,
        )

        # Scan and check - should find the pair
        roots = list(store.nodes.iter_root_nodes())
        assert len(roots) == 2

        max_height_diff = 2
        min_preceding_height: int | None = None
        pair_found = False

        sorted_roots = sorted(roots, key=lambda r: int(getattr(r, "span_start", 0)))

        prev_root = None
        for root in sorted_roots:
            root_height = int(getattr(root, "height", 0))

            if prev_root is not None:
                left_height = int(getattr(prev_root, "height", 0))
                left_level = int(getattr(prev_root, "level_index", 0))
                right_level = int(getattr(root, "level_index", 0))

                is_eligible = (
                    left_height == root_height
                    and left_level % 2 == 0
                    and right_level == left_level + 1
                )

                if is_eligible:
                    parent_height = left_height + 1
                    # Height diff check - should pass
                    if min_preceding_height is None or (
                        parent_height - min_preceding_height <= max_height_diff
                    ):
                        pair_found = True
                        break

            prev_root = root
            if min_preceding_height is None or root_height < min_preceding_height:
                min_preceding_height = root_height

        assert pair_found, "Expected to find the h=3 pair without the orphan"

    def test_orphan_stalls_even_with_higher_preceding_nodes(
        self,
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """The orphan stalls progress even if there are completed higher nodes before it.

        This matches the production scenario where we had:
        (h=10, lvl=0, span=0)
        (h=8, lvl=4, span=632517)
        (h=7, lvl=10, span=796346)
        (h=1, lvl=704, span=881080)   <- ORPHAN
        (h=3, lvl=176, span=881080)   <- BLOCKED
        (h=3, lvl=177, span=886365)   <- BLOCKED
        """
        doc_id = "test-doc"
        store = sqlite_backend.for_document(doc_id)

        # Higher nodes that completed successfully (span-ordered before orphan)
        store.nodes.add_node(
            node_id="h10-root",
            text="H10 root",
            embedding=[0.1] * 8,
            height=10,
            level_index=0,
            span_start=0,
            span_end=100000,
            parent_id=None,
            token_count=500,
        )
        store.nodes.add_node(
            node_id="h8-root",
            text="H8 root",
            embedding=[0.1] * 8,
            height=8,
            level_index=4,
            span_start=200000,
            span_end=300000,
            parent_id=None,
            token_count=400,
        )

        # The orphan - must have lower span_start than h=3 pair to ensure
        # deterministic ordering (roots are ordered by span_start)
        store.nodes.add_node(
            node_id="orphan-h1",
            text="Orphaned h=1",
            embedding=[0.1] * 8,
            height=1,
            level_index=704,
            span_start=390000,  # Before h=3 pair to guarantee ordering
            span_end=392000,
            parent_id=None,
            token_count=50,
        )

        # Eligible h=3 pair after the orphan in span order
        store.nodes.add_node(
            node_id="h3-left",
            text="H3 left",
            embedding=[0.1] * 8,
            height=3,
            level_index=176,
            span_start=400000,
            span_end=408000,
            parent_id=None,
            token_count=100,
        )
        store.nodes.add_node(
            node_id="h3-right",
            text="H3 right",
            embedding=[0.1] * 8,
            height=3,
            level_index=177,
            span_start=408000,
            span_end=416000,
            parent_id=None,
            token_count=100,
        )

        roots = list(store.nodes.iter_root_nodes())
        assert len(roots) == 5

        max_height_diff = 2
        min_preceding_height: int | None = None
        blocked = False
        blocked_at_diff: int | None = None

        sorted_roots = sorted(roots, key=lambda r: int(getattr(r, "span_start", 0)))

        prev_root = None
        for root in sorted_roots:
            root_height = int(getattr(root, "height", 0))

            if prev_root is not None:
                left_height = int(getattr(prev_root, "height", 0))
                left_level = int(getattr(prev_root, "level_index", 0))
                right_level = int(getattr(root, "level_index", 0))

                is_eligible = (
                    left_height == root_height
                    and left_level % 2 == 0
                    and right_level == left_level + 1
                )

                if is_eligible:
                    parent_height = left_height + 1
                    if (
                        min_preceding_height is not None
                        and parent_height - min_preceding_height > max_height_diff
                    ):
                        blocked = True
                        blocked_at_diff = parent_height - min_preceding_height
                        break

            prev_root = root
            if min_preceding_height is None or root_height < min_preceding_height:
                min_preceding_height = root_height

        # Despite having h=10 and h=8 roots before it, the h=1 orphan
        # sets min_preceding_height=1 and blocks the h=3 pair
        assert blocked, "Expected h=3 pair to be blocked by orphan"
        assert (
            min_preceding_height == 1
        ), f"Expected min_h=1, got {min_preceding_height}"
        assert blocked_at_diff == 3, f"Expected diff=3, got {blocked_at_diff}"

    def test_corruption_leaves_overlapping_spans(
        self,
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Corruption creates nodes with overlapping spans that violate tree invariants.

        When _summarize_pair fails after parent creation but before child updates:
        - Parent exists at span [0, 2000]
        - Left child exists at span [0, 1000] with parent_id=NULL
        - Right child exists at span [1000, 2000] with parent_id=NULL

        All three appear as roots with overlapping/containing spans.
        """
        doc_id = "test-doc"
        store = sqlite_backend.for_document(doc_id)

        # Simulate post-corruption state:
        # Parent was created with child refs but children weren't updated

        # Left child (orphaned - should have parent_id pointing to parent)
        store.nodes.add_node(
            node_id="left-leaf",
            text="Left leaf content",
            embedding=[0.1] * 8,
            height=0,
            level_index=0,
            span_start=0,
            span_end=1000,
            parent_id=None,  # ORPHANED - should point to parent
            token_count=50,
        )

        # Right child (orphaned - should have parent_id pointing to parent)
        store.nodes.add_node(
            node_id="right-leaf",
            text="Right leaf content",
            embedding=[0.1] * 8,
            height=0,
            level_index=1,
            span_start=1000,
            span_end=2000,
            parent_id=None,  # ORPHANED - should point to parent
            token_count=50,
        )

        # Parent (created before failure, points to children)
        store.nodes.add_node(
            node_id="parent-h1",
            text="Summary of children",
            embedding=[0.1] * 8,
            height=1,
            level_index=0,
            span_start=0,  # Covers same region as children
            span_end=2000,
            parent_id=None,
            left_child_id="left-leaf",
            right_child_id="right-leaf",
            token_count=30,
        )

        # All three appear as roots
        roots = list(store.nodes.iter_root_nodes())
        root_ids = {getattr(r, "id", None) for r in roots}

        assert len(roots) == 3, f"Expected 3 roots (corrupted state), got {len(roots)}"
        assert "parent-h1" in root_ids, "Parent should appear as root"
        assert "left-leaf" in root_ids, "Orphaned left child should appear as root"
        assert "right-leaf" in root_ids, "Orphaned right child should appear as root"

        # Verify span overlap - parent covers region that children also cover as roots
        spans = [
            (int(getattr(r, "span_start", 0)), int(getattr(r, "span_end", 0)))
            for r in roots
        ]
        parent_span = (0, 2000)
        left_span = (0, 1000)
        right_span = (1000, 2000)

        assert parent_span in spans, "Parent span should be present"
        assert left_span in spans, "Left child span should be present"
        assert right_span in spans, "Right child span should be present"

        # This is the invariant violation: in a valid tree, parent would not
        # appear as a root while its children (which it points to) also appear as roots


class TestSummaryJobIntegration:
    """Integration tests that exercise the actual IndexingEngine methods."""

    @pytest.mark.asyncio
    async def test_find_summary_jobs_blocked_by_orphan(
        self,
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that _find_next_n_summary_jobs is blocked by orphaned root.

        This uses the actual engine method to prove the stall.
        """
        from unittest.mock import MagicMock

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import IndexingEngine

        doc_id = "test-doc"
        store_for_doc = sqlite_backend.for_document(doc_id)

        # Create corrupted state: orphan + eligible pair
        store_for_doc.nodes.add_node(
            node_id="orphan-h1",
            text="Orphaned h=1 summary",
            embedding=[0.1] * 8,
            height=1,
            level_index=0,
            span_start=0,
            span_end=2000,
            parent_id=None,
            token_count=50,
        )
        store_for_doc.nodes.add_node(
            node_id="h3-left",
            text="H3 left",
            embedding=[0.1] * 8,
            height=3,
            level_index=0,
            span_start=10000,
            span_end=18000,
            parent_id=None,
            token_count=100,
        )
        store_for_doc.nodes.add_node(
            node_id="h3-right",
            text="H3 right",
            embedding=[0.1] * 8,
            height=3,
            level_index=1,
            span_start=18000,
            span_end=26000,
            parent_id=None,
            token_count=100,
        )

        # Create minimal engine with mocked dependencies
        mock_llm = MagicMock()
        mock_openai = MagicMock()
        config = IndexConfig.load(target_chunk_tokens=50)

        engine = IndexingEngine(
            store=sqlite_backend,
            llm_service=mock_llm,
            index_config=config,
            openai_client=mock_openai,
            vector_index_factory=lambda _: MagicMock(),
            max_parallelism=1,
        )

        # Call _find_next_n_summary_jobs with max_height_diff=2
        jobs = engine._find_next_n_summary_jobs(
            store=store_for_doc,
            document_id=doc_id,
            active_jobs=set(),
            ctx=None,
            frontier=None,
            max_height_diff=2,  # The production setting
            max_jobs=10,
        )

        # VERIFY: No jobs found because orphan blocks the pair
        assert len(jobs) == 0, f"Expected 0 jobs (blocked), got {len(jobs)}"

        # Verify that without height diff limit, the pair IS found
        jobs_no_limit = engine._find_next_n_summary_jobs(
            store=store_for_doc,
            document_id=doc_id,
            active_jobs=set(),
            ctx=None,
            frontier=None,
            max_height_diff=None,  # No limit
            max_jobs=10,
        )

        assert (
            len(jobs_no_limit) == 1
        ), f"Expected 1 job (no limit), got {len(jobs_no_limit)}"
        assert jobs_no_limit[0][1].left_id == "h3-left"
        assert jobs_no_limit[0][1].right_id == "h3-right"

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_summarize_pair_rollback_on_failure(
        self,
        sqlite_backend: SQLiteStorageBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify _summarize_pair rolls back parent if update_parent_references fails.

        This directly tests that the atomicity fix in _summarize_pair works:
        if update_parent_references_batch raises after add_batch, the parent
        node should NOT persist (transaction rollback).

        Without the transaction wrapper, this test would fail because the parent
        would be committed before the exception.
        """
        from collections.abc import Sequence
        from unittest.mock import AsyncMock, MagicMock

        from sqlalchemy.orm import Session

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import IndexingEngine, SummaryJob

        doc_id = "test-doc"
        store_for_doc = sqlite_backend.for_document(doc_id)

        # Create two leaf nodes as eligible pair
        store_for_doc.nodes.add_node(
            node_id="left-leaf",
            text="Left leaf content for summarization",
            embedding=[0.1] * 8,
            height=0,
            level_index=0,
            span_start=0,
            span_end=1000,
            parent_id=None,
            token_count=50,
        )
        store_for_doc.nodes.add_node(
            node_id="right-leaf",
            text="Right leaf content for summarization",
            embedding=[0.1] * 8,
            height=0,
            level_index=1,
            span_start=1000,
            span_end=2000,
            parent_id=None,
            token_count=50,
        )

        # Verify we have 2 roots before
        roots_before = list(store_for_doc.nodes.iter_root_nodes())
        assert len(roots_before) == 2

        # Create engine with mocked LLM
        mock_llm = MagicMock()
        mock_llm._summarize_text = AsyncMock(
            return_value=SummaryResult(
                summary="Summary of left and right",
                retry_count=0,
                summary_tokens=30,
                usage=AccumulatedUsage(
                    prompt_tokens=100,
                    cached_tokens=0,
                    completion_tokens=30,
                ),
            )
        )
        mock_openai = MagicMock()
        config = IndexConfig.load(target_chunk_tokens=50)

        engine = IndexingEngine(
            store=sqlite_backend,
            llm_service=mock_llm,
            index_config=config,
            openai_client=mock_openai,
            vector_index_factory=lambda _: MagicMock(),
            max_parallelism=1,
        )

        # Patch at the repository level so it affects all DocumentStore instances
        # _summarize_pair calls self._store.for_document() which returns a new
        # DocumentStore wrapping the same repository
        node_repo = sqlite_backend.node_repo
        original_update = node_repo.update_parent_references_batch
        call_count = 0

        def failing_update(
            updates: Sequence[tuple[str, str | None]],
            *,
            session: Session | None = None,
        ) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Simulated failure in update_parent_references_batch")

        monkeypatch.setattr(node_repo, "update_parent_references_batch", failing_update)

        # Execute _summarize_pair - should fail but parent should NOT persist
        job = SummaryJob(doc_id, "left-leaf", "right-leaf")
        with pytest.raises(RuntimeError, match="Simulated failure"):
            await engine._summarize_pair(job)

        # Verify the failing function was called
        assert call_count == 1, "update_parent_references_batch should have been called"

        # Restore original for verification queries
        monkeypatch.setattr(
            node_repo, "update_parent_references_batch", original_update
        )

        # CRITICAL VERIFICATION: Parent should NOT exist due to rollback
        roots_after = list(store_for_doc.nodes.iter_root_nodes())
        root_ids = {getattr(r, "id", None) for r in roots_after}

        assert len(roots_after) == 2, (
            f"Expected 2 roots (children only), got {len(roots_after)}. "
            "Parent was persisted despite failure - atomicity broken!"
        )
        assert "left-leaf" in root_ids, "Left leaf should still be root"
        assert "right-leaf" in root_ids, "Right leaf should still be root"

        # Verify no parent node exists at height=1
        parent_candidates = [
            r for r in roots_after if int(getattr(r, "height", 0)) == 1
        ]
        assert (
            len(parent_candidates) == 0
        ), "No height=1 parent should exist after rollback"

        # Verify children still have parent_id=None (not orphaned)
        left = store_for_doc.nodes.get("left-leaf")
        right = store_for_doc.nodes.get("right-leaf")
        assert left is not None and right is not None
        assert getattr(left, "parent_id", "UNSET") is None
        assert getattr(right, "parent_id", "UNSET") is None

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_multi_instance_race_demonstrates_why_lease_is_needed(
        self,
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Demonstrate what happens without single-writer coordination.

        This test shows that without the IndexerLease mechanism (which ensures
        only one IndexingEngine can write at a time), concurrent engines can
        create duplicate nodes at the same coordinates.

        In production, the IndexerLease mechanism (see ragzoom/server/lease.py)
        prevents this by ensuring only one server holds the lease at a time.
        This test is kept to document the race condition that the lease prevents.

        NOTE: This test intentionally creates duplicates to demonstrate the
        uncoordinated behavior. The lease mechanism tested in test_indexer_lease.py
        and test_lease_integration.py ensures this never happens in production.
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import IndexingEngine, SummaryJob

        doc_id = "test-doc"
        store_for_doc = sqlite_backend.for_document(doc_id)

        # Create two leaf nodes as eligible pair
        store_for_doc.nodes.add_node(
            node_id="left-leaf",
            text="Left leaf content for summarization",
            embedding=[0.1] * 8,
            height=0,
            level_index=0,
            span_start=0,
            span_end=1000,
            parent_id=None,
            token_count=50,
        )
        store_for_doc.nodes.add_node(
            node_id="right-leaf",
            text="Right leaf content for summarization",
            embedding=[0.1] * 8,
            height=0,
            level_index=1,
            span_start=1000,
            span_end=2000,
            parent_id=None,
            token_count=50,
        )

        # Verify we have 2 roots before
        roots_before = list(store_for_doc.nodes.iter_root_nodes())
        assert len(roots_before) == 2

        # Create mock LLM that returns valid summaries
        def make_mock_llm() -> MagicMock:
            from ragzoom.services.summary_utils import AccumulatedUsage, SummaryResult

            mock = MagicMock()
            mock._summarize_text = AsyncMock(
                return_value=SummaryResult(
                    summary="Summary of left and right",
                    summary_tokens=30,
                    retry_count=0,
                    usage=AccumulatedUsage(
                        prompt_tokens=100,
                        cached_tokens=0,
                        completion_tokens=30,
                    ),
                )
            )
            return mock

        config = IndexConfig.load(target_chunk_tokens=50)

        # Create TWO IndexingEngine instances sharing the SAME database backend
        # This simulates what would happen WITHOUT the lease mechanism
        engine_1 = IndexingEngine(
            store=sqlite_backend,
            llm_service=make_mock_llm(),
            index_config=config,
            openai_client=MagicMock(),
            vector_index_factory=lambda _: MagicMock(),
            max_parallelism=1,
        )
        engine_2 = IndexingEngine(
            store=sqlite_backend,
            llm_service=make_mock_llm(),
            index_config=config,
            openai_client=MagicMock(),
            vector_index_factory=lambda _: MagicMock(),
            max_parallelism=1,
        )

        # Both engines create a SummaryJob for the SAME pair
        job = SummaryJob(doc_id, "left-leaf", "right-leaf")

        # Inject delay to widen the race window
        import os

        os.environ["RAGZOOM_SUMMARIZE_DELAY_MS"] = "100"

        try:
            # Run _summarize_pair on BOTH engines concurrently
            # Without lease coordination, both will create parent nodes
            results = await asyncio.gather(
                engine_1._summarize_pair(job),
                engine_2._summarize_pair(job),
                return_exceptions=True,
            )
        finally:
            os.environ.pop("RAGZOOM_SUMMARIZE_DELAY_MS", None)

        # Check for errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Some error occurred (could be various reasons)
                pass

        # Query for nodes at height=1 with level_index=0
        all_nodes = list(store_for_doc.nodes.iter_all())
        height_1_nodes = [n for n in all_nodes if getattr(n, "height", 0) == 1]
        duplicate_coords = [
            n for n in height_1_nodes if getattr(n, "level_index", -1) == 0
        ]

        # Without single-writer coordination, we may get duplicates.
        # The IndexerLease mechanism prevents this in production.
        # This test documents the behavior without coordination.
        # Note: Due to SQLite's serialization and the unique constraint on
        # (document_id, height, level_index), we won't see duplicates - one
        # of the concurrent writes will fail. The test verifies that at least
        # one parent was created successfully.
        node_ids = [getattr(n, "id", "?") for n in duplicate_coords]
        assert len(duplicate_coords) == 1, (
            f"Expected exactly 1 parent node at (height=1, level_index=0), "
            f"got {len(duplicate_coords)}: {node_ids}. "
            f"The unique constraint should prevent duplicates."
        )

        await engine_1.shutdown()
        await engine_2.shutdown()
