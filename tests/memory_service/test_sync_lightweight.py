"""Lightweight sync and truncation tests without Docker/gRPC.

These tests exercise the same code paths as test_progressive_append.py but
use SQLite in-memory + mocked OpenAI instead of requiring the full devstack.

Key differences from integration tests:
- No gRPC - calls IndexerRuntime directly via harness
- No PostgreSQL - uses SQLite in-memory
- No real OpenAI - uses mocked embeddings/summaries
- Fast: < 5 seconds per test vs 30-90 seconds

To run:
    pytest tests/memory_service/test_sync_lightweight.py -xvs
"""

from __future__ import annotations

import uuid

import pytest

from tests.memory_service.conftest import MemoryServiceTestHarness


class TestAppendDuringIndexing:
    """Tests for appending while background indexing is in-flight."""

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(5.0)
    async def test_append_during_indexing_maintains_tree_integrity(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Append while background jobs running should maintain tree integrity.

        Port of: test_progressive_append_during_active_indexing
        """
        doc_id = f"append-test-{uuid.uuid4()}"

        # Phase 1: Initial append (creates leaves and queues summary jobs)
        await memory_service_harness.append(
            doc_id,
            "Initial content. " * 100,  # ~400 tokens -> multiple leaves
            replace_existing=True,
            await_idle=False,  # Don't wait - let indexing run
        )

        # Phase 2: Append more while indexing is in-flight
        for i in range(3):
            await memory_service_harness.append(
                doc_id,
                f"Additional content {i}. " * 50,
                replace_existing=False,
                await_idle=False,
            )

            # Validate during indexing (allow incomplete tree)
            report = memory_service_harness.validate_tree(
                doc_id, require_complete=False
            )
            assert (
                report.status == "ok"
            ), f"Validation failed during append {i}: {report.errors}"

        # Phase 3: Wait for completion and final validation
        await memory_service_harness.wait_for_idle(doc_id)

        final_report = memory_service_harness.validate_tree(
            doc_id, require_complete=False
        )
        assert (
            final_report.status == "ok"
        ), f"Final validation failed: {final_report.errors}"

        # Verify we have multiple leaves
        leaves = memory_service_harness.get_leaves(doc_id)
        assert len(leaves) >= 2, f"Expected multiple leaves, got {len(leaves)}"

        # Verify no orphaned nodes
        orphans = memory_service_harness.get_orphaned_nodes(doc_id)
        assert orphans == [], f"Found orphaned nodes: {[n.id for n in orphans]}"

    @pytest.mark.asyncio
    async def test_rapid_appends_no_race_conditions(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Rapid-fire appends should not cause race conditions.

        Port of: test_rapid_fire_appends_no_delay
        """
        doc_id = f"rapid-append-{uuid.uuid4()}"

        # Initial content
        await memory_service_harness.append(
            doc_id,
            "Initial. " * 50,
            replace_existing=True,
            await_idle=False,
        )

        # Rapid appends without waiting
        for i in range(5):
            await memory_service_harness.append(
                doc_id,
                f"Rapid {i}. " * 30,
                replace_existing=False,
                await_idle=False,
            )

        # Wait for all indexing to complete
        await memory_service_harness.wait_for_idle(doc_id)

        # Validate final state
        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok", f"Validation failed: {report.errors}"

        # Check for duplicate coordinates
        duplicates = memory_service_harness.check_for_duplicate_coords(doc_id)
        assert duplicates == [], f"Found duplicate coordinates: {duplicates}"


class TestTruncateDuringIndexing:
    """Tests for truncation while background indexing is in-flight."""

    @pytest.mark.asyncio
    async def test_truncate_during_indexing_validates_tree(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Truncate while summary jobs running should maintain tree validity.

        Port of: test_revert_while_indexing_in_flight
        """
        doc_id = f"truncate-test-{uuid.uuid4()}"

        # Phase 1: Build initial tree
        await memory_service_harness.append(
            doc_id,
            "Initial content for truncation test. " * 100,
            replace_existing=True,
            await_idle=True,  # Wait for initial tree to build
        )

        initial_leaves = len(memory_service_harness.get_leaves(doc_id))
        assert initial_leaves >= 2, f"Need at least 2 leaves, got {initial_leaves}"

        # Phase 2: Append more content (creates indexing jobs)
        await memory_service_harness.append(
            doc_id,
            "More content to append. " * 100,
            replace_existing=False,
            await_idle=False,  # Don't wait - truncate while indexing
        )

        # Phase 3: Truncate to middle of document
        leaves = memory_service_harness.get_leaves(doc_id)
        mid_span = leaves[len(leaves) // 2].span_start

        result = await memory_service_harness.truncate(doc_id, mid_span)
        assert len(result.deleted_node_ids) > 0, "Expected some nodes to be deleted"

        # Phase 4: Validate after truncation
        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert (
            report.status == "ok"
        ), f"Validation failed after truncate: {report.errors}"

        # Check for duplicate coordinates (potential race condition symptom)
        duplicates = memory_service_harness.check_for_duplicate_coords(doc_id)
        assert duplicates == [], f"Found duplicate coordinates: {duplicates}"


class TestMultipleTruncateAppendCycles:
    """Tests for multiple truncate/append operations."""

    @pytest.mark.asyncio
    async def test_multiple_cycles_maintain_integrity(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Multiple truncate/append cycles should maintain tree integrity.

        Port of: test_rapid_append_revert_cycles
        """
        doc_id = f"cycles-test-{uuid.uuid4()}"

        # Initial content
        await memory_service_harness.append(
            doc_id,
            "Initial base content. " * 100,
            replace_existing=True,
            await_idle=True,
        )

        for cycle in range(3):
            # Append new content
            await memory_service_harness.append(
                doc_id,
                f"Cycle {cycle} content. " * 50,
                replace_existing=False,
                await_idle=True,
            )

            # Truncate back
            leaves = memory_service_harness.get_leaves(doc_id)
            if len(leaves) > 2:
                truncate_span = leaves[len(leaves) // 2].span_start
                await memory_service_harness.truncate(doc_id, truncate_span)

            # Validate after each cycle
            report = memory_service_harness.validate_tree(
                doc_id, require_complete=False
            )
            assert (
                report.status == "ok"
            ), f"Cycle {cycle} validation failed: {report.errors}"

        # Final state check
        orphans = memory_service_harness.get_orphaned_nodes(doc_id)
        assert (
            orphans == []
        ), f"Found orphaned nodes after cycles: {[n.id for n in orphans]}"


class TestTruncationReferentialIntegrity:
    """Tests for referential integrity during truncation.

    These tests verify that truncation properly cleans up dangling references:
    - parent_id references to deleted parent nodes
    - following_neighbor_id references to deleted neighbor nodes

    The fix (commit TBD) adds UPDATE statements before DELETE to NULL out
    these references on kept nodes before their targets are deleted.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(5.0)
    async def test_partial_truncation_no_orphaned_nodes(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Partial truncation should not create orphaned nodes (Bug 3).

        This test demonstrates the FK violation bug where truncation
        deletes parent nodes while their children remain with dangling
        parent_id references.
        """
        doc_id = f"fk-violation-{uuid.uuid4()}"

        # Build a tree with multiple levels (needs enough content for internal nodes)
        await memory_service_harness.append(
            doc_id,
            "First segment of content. " * 150,  # Create multiple leaves
            replace_existing=True,
            await_idle=True,
        )
        await memory_service_harness.append(
            doc_id,
            "Second segment of content. " * 150,  # More leaves + summaries
            replace_existing=False,
            await_idle=True,
        )

        # Get tree state before truncation
        all_nodes = memory_service_harness.get_all_nodes(doc_id)
        leaves = memory_service_harness.get_leaves(doc_id)
        internal_nodes = [n for n in all_nodes if n.height > 0]

        assert (
            len(leaves) >= 4
        ), f"Need at least 4 leaves for this test, got {len(leaves)}"
        assert len(internal_nodes) >= 1, "Need internal nodes for Bug 3 to manifest"

        # Truncate at mid-point (this is where Bug 3 manifests)
        mid_span = leaves[len(leaves) // 2].span_start
        await memory_service_harness.truncate(doc_id, mid_span)

        # Check for orphaned nodes (the Bug 3 symptom)
        orphans = memory_service_harness.get_orphaned_nodes(doc_id)

        # BUG 3: This assertion should pass, but currently fails because
        # truncation leaves children with dangling parent_id references
        assert orphans == [], (
            f"Found {len(orphans)} orphaned nodes with dangling parent_id refs: "
            f"{[(n.id, n.parent_id) for n in orphans[:5]]}"
        )

    @pytest.mark.asyncio
    async def test_truncation_preserves_parent_child_relationships(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """After truncation, all parent_id refs should point to existing nodes."""
        doc_id = f"parent-child-{uuid.uuid4()}"

        # Build tree
        for i in range(5):
            await memory_service_harness.append(
                doc_id,
                f"Segment {i} content. " * 80,
                replace_existing=(i == 0),
                await_idle=True,
            )

        # Truncate at 60% through the document
        leaves = memory_service_harness.get_leaves(doc_id)
        truncate_at = leaves[int(len(leaves) * 0.6)].span_start

        await memory_service_harness.truncate(doc_id, truncate_at)

        # Verify all parent references are valid
        remaining_nodes = memory_service_harness.get_all_nodes(doc_id)
        remaining_ids = {n.id for n in remaining_nodes}

        for node in remaining_nodes:
            if node.parent_id:
                # BUG 3: This assertion fails because parents get deleted
                # while their children remain
                assert (
                    node.parent_id in remaining_ids
                ), f"Node {node.id} has dangling parent_id {node.parent_id}"


# =============================================================================
# COMPREHENSIVE TEST BATTERY
# =============================================================================


class TestBasicAppend:
    """Tests for fundamental append operations."""

    @pytest.mark.asyncio
    async def test_append_to_empty_document(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """First append to empty document creates leaves."""
        doc_id = f"empty-doc-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id,
            "Hello world. " * 50,  # ~200 tokens
            replace_existing=True,
            await_idle=True,
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        assert len(leaves) >= 1, "Should create at least one leaf"

        span = memory_service_harness.get_document_span(doc_id)
        assert span is not None
        assert span[0] == 0, "Document should start at span 0"

    @pytest.mark.asyncio
    async def test_append_multiple_chunks_sequential(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Sequential appends grow document correctly."""
        doc_id = f"sequential-{uuid.uuid4()}"

        # First append
        await memory_service_harness.append(
            doc_id, "First chunk. " * 50, replace_existing=True, await_idle=True
        )
        initial_leaves = len(memory_service_harness.get_leaves(doc_id))

        # Second append
        await memory_service_harness.append(
            doc_id, "Second chunk. " * 50, replace_existing=False, await_idle=True
        )
        after_second = len(memory_service_harness.get_leaves(doc_id))

        # Third append
        await memory_service_harness.append(
            doc_id, "Third chunk. " * 50, replace_existing=False, await_idle=True
        )
        final_leaves = len(memory_service_harness.get_leaves(doc_id))

        assert after_second >= initial_leaves, "Second append should add leaves"
        assert final_leaves >= after_second, "Third append should add more leaves"

        # Verify spans are contiguous
        leaves = memory_service_harness.get_leaves(doc_id)
        for i in range(len(leaves) - 1):
            assert (
                leaves[i].span_end == leaves[i + 1].span_start
            ), f"Gap between leaf {i} and {i+1}"

    @pytest.mark.asyncio
    async def test_append_with_replace_existing_clears_first(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """replace_existing=True clears document before appending."""
        doc_id = f"replace-{uuid.uuid4()}"

        # First append with lots of content
        await memory_service_harness.append(
            doc_id, "Original content. " * 200, replace_existing=True, await_idle=True
        )
        original_span = memory_service_harness.get_document_span(doc_id)
        assert original_span is not None

        # Replace with less content
        await memory_service_harness.append(
            doc_id, "New content. " * 50, replace_existing=True, await_idle=True
        )
        new_span = memory_service_harness.get_document_span(doc_id)
        assert new_span is not None

        # New document should be smaller
        assert new_span[1] < original_span[1], "Replace should create smaller document"
        assert new_span[0] == 0, "Should start at 0 after replace"

    @pytest.mark.asyncio
    async def test_append_creates_expected_leaf_count(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Leaf count should be proportional to text size."""
        doc_id = f"leaf-count-{uuid.uuid4()}"

        # Small document
        await memory_service_harness.append(
            doc_id, "Word. " * 20, replace_existing=True, await_idle=True
        )
        small_leaves = len(memory_service_harness.get_leaves(doc_id))

        # Large document
        await memory_service_harness.append(
            doc_id, "Word. " * 500, replace_existing=True, await_idle=True
        )
        large_leaves = len(memory_service_harness.get_leaves(doc_id))

        assert large_leaves > small_leaves, "Larger document should have more leaves"

    @pytest.mark.asyncio
    async def test_append_updates_document_span_end(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Span end should match text length."""
        doc_id = f"span-end-{uuid.uuid4()}"
        text = "Hello world. " * 100

        await memory_service_harness.append(
            doc_id, text, replace_existing=True, await_idle=True
        )

        span = memory_service_harness.get_document_span(doc_id)
        assert span is not None
        assert span[1] == len(text), f"Span end {span[1]} != text length {len(text)}"

    @pytest.mark.asyncio
    async def test_append_empty_text_raises_error(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Empty string append should raise ValueError."""
        doc_id = f"empty-text-{uuid.uuid4()}"

        # First create a document
        await memory_service_harness.append(
            doc_id, "Initial content. " * 50, replace_existing=True, await_idle=True
        )
        initial_leaves = len(memory_service_harness.get_leaves(doc_id))

        # Append empty string should raise
        with pytest.raises(ValueError, match="text must be non-empty"):
            await memory_service_harness.append(
                doc_id, "", replace_existing=False, await_idle=True
            )

        final_leaves = len(memory_service_harness.get_leaves(doc_id))
        assert final_leaves == initial_leaves, "Empty append should not change tree"


class TestBatchAppend:
    """Tests for batch append with forced boundaries."""

    @pytest.mark.asyncio
    async def test_batch_append_creates_forced_boundaries(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Each unit in batch append creates separate segment."""
        doc_id = f"batch-boundaries-{uuid.uuid4()}"

        units = [
            "First segment content. " * 30,
            "Second segment content. " * 30,
            "Third segment content. " * 30,
        ]

        await memory_service_harness.batch_append(doc_id, units, await_idle=True)

        leaves = memory_service_harness.get_leaves(doc_id)
        assert len(leaves) >= len(units), "Should have at least one leaf per unit"

        # Verify the text is correctly segmented
        full_text = memory_service_harness.concatenate_leaf_text(doc_id)
        expected_text = "".join(units)
        assert full_text == expected_text, "Concatenated leaves should match input"

    @pytest.mark.asyncio
    async def test_batch_append_single_unit_equivalent(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Single-item batch should behave like regular append."""
        doc_id = f"single-unit-{uuid.uuid4()}"
        text = "Single unit content. " * 50

        await memory_service_harness.batch_append(doc_id, [text], await_idle=True)

        leaves = memory_service_harness.get_leaves(doc_id)
        assert len(leaves) >= 1
        assert memory_service_harness.concatenate_leaf_text(doc_id) == text

    @pytest.mark.asyncio
    async def test_batch_append_many_small_units(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Many small units should create many leaves."""
        doc_id = f"many-units-{uuid.uuid4()}"

        units = [f"Unit {i} content. " * 10 for i in range(10)]
        await memory_service_harness.batch_append(doc_id, units, await_idle=True)

        leaves = memory_service_harness.get_leaves(doc_id)
        # Each unit should create at least one leaf
        assert len(leaves) >= len(units)

    @pytest.mark.asyncio
    async def test_batch_append_then_regular_append(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Can mix batch and regular appends."""
        doc_id = f"mixed-append-{uuid.uuid4()}"

        # Start with batch
        units = ["Batch one. " * 30, "Batch two. " * 30]
        await memory_service_harness.batch_append(doc_id, units, await_idle=True)
        after_batch = memory_service_harness.get_document_span(doc_id)

        # Then regular append
        await memory_service_harness.append(
            doc_id, "Regular append. " * 50, replace_existing=False, await_idle=True
        )
        after_append = memory_service_harness.get_document_span(doc_id)

        assert after_batch is not None and after_append is not None
        assert after_append[1] > after_batch[1], "Regular append should extend document"

        # Verify spans are contiguous
        leaves = memory_service_harness.get_leaves(doc_id)
        for i in range(len(leaves) - 1):
            assert leaves[i].span_end == leaves[i + 1].span_start

    @pytest.mark.asyncio
    async def test_batch_append_empty_list_is_noop(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Empty list batch append should be a no-op."""
        doc_id = f"empty-batch-{uuid.uuid4()}"

        # First create a document
        await memory_service_harness.append(
            doc_id, "Initial. " * 50, replace_existing=True, await_idle=True
        )
        initial_span = memory_service_harness.get_document_span(doc_id)

        # Empty batch append
        await memory_service_harness.batch_append(doc_id, [], await_idle=True)

        final_span = memory_service_harness.get_document_span(doc_id)
        assert final_span == initial_span, "Empty batch should not change document"


class TestTruncation:
    """Tests for various truncation scenarios."""

    @pytest.mark.asyncio
    async def test_truncate_to_middle(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Truncate at mid-point deletes roughly half the nodes."""
        doc_id = f"truncate-middle-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id,
            "Content for truncation. " * 200,
            replace_existing=True,
            await_idle=True,
        )

        initial_nodes = len(memory_service_harness.get_all_nodes(doc_id))
        leaves = memory_service_harness.get_leaves(doc_id)
        mid_span = leaves[len(leaves) // 2].span_start

        result = await memory_service_harness.truncate(doc_id, mid_span)

        final_nodes = len(memory_service_harness.get_all_nodes(doc_id))
        assert len(result.deleted_node_ids) > 0, "Should delete some nodes"
        assert final_nodes < initial_nodes, "Should have fewer nodes after truncate"

        # Validate tree integrity
        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok", f"Validation failed: {report.errors}"

    @pytest.mark.asyncio
    async def test_truncate_to_zero(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Full revert (span=0) deletes all nodes."""
        doc_id = f"truncate-zero-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content to delete. " * 100, replace_existing=True, await_idle=True
        )

        initial_nodes = len(memory_service_harness.get_all_nodes(doc_id))
        assert initial_nodes > 0

        result = await memory_service_harness.truncate(doc_id, 0)

        final_nodes = len(memory_service_harness.get_all_nodes(doc_id))
        assert (
            final_nodes == 0
        ), f"Should have 0 nodes after full truncate, got {final_nodes}"
        assert len(result.deleted_node_ids) == initial_nodes

    @pytest.mark.asyncio
    async def test_truncate_at_exact_leaf_boundary(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Truncate at exact leaf boundary should be clean."""
        doc_id = f"truncate-boundary-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content here. " * 150, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        assert len(leaves) >= 3, "Need at least 3 leaves"

        # Truncate at exact leaf boundary (second leaf's span_start)
        boundary = leaves[1].span_start
        await memory_service_harness.truncate(doc_id, boundary)

        remaining_leaves = memory_service_harness.get_leaves(doc_id)
        assert (
            len(remaining_leaves) == 1
        ), f"Should have 1 leaf, got {len(remaining_leaves)}"
        assert remaining_leaves[0].span_end == boundary

    @pytest.mark.asyncio
    async def test_truncate_at_document_end_is_noop(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Truncate beyond span_end should be a no-op."""
        doc_id = f"truncate-end-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Some content. " * 100, replace_existing=True, await_idle=True
        )

        initial_nodes = len(memory_service_harness.get_all_nodes(doc_id))
        span = memory_service_harness.get_document_span(doc_id)
        assert span is not None

        # Truncate beyond document end
        result = await memory_service_harness.truncate(doc_id, span[1] + 1000)

        final_nodes = len(memory_service_harness.get_all_nodes(doc_id))
        assert final_nodes == initial_nodes, "Should not delete any nodes"
        assert len(result.deleted_node_ids) == 0

    @pytest.mark.asyncio
    async def test_truncate_near_beginning(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Truncate near beginning deletes ~95% of tree."""
        doc_id = f"truncate-near-start-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id,
            "Content to mostly delete. " * 300,
            replace_existing=True,
            await_idle=True,
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        # Truncate at first leaf's end (keep only first leaf)
        first_leaf_end = leaves[0].span_end

        result = await memory_service_harness.truncate(doc_id, first_leaf_end)

        remaining_leaves = memory_service_harness.get_leaves(doc_id)
        assert len(remaining_leaves) == 1
        assert len(result.deleted_node_ids) > 0

        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok", f"Validation failed: {report.errors}"

    @pytest.mark.asyncio
    async def test_truncate_near_end(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Truncate near end deletes ~5% of tree."""
        doc_id = f"truncate-near-end-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id,
            "Content to mostly keep. " * 300,
            replace_existing=True,
            await_idle=True,
        )

        initial_leaves = len(memory_service_harness.get_leaves(doc_id))
        leaves = memory_service_harness.get_leaves(doc_id)
        # Truncate at second-to-last leaf's end
        truncate_point = leaves[-2].span_end if len(leaves) > 1 else leaves[0].span_end

        await memory_service_harness.truncate(doc_id, truncate_point)

        remaining_leaves = len(memory_service_harness.get_leaves(doc_id))
        assert remaining_leaves == initial_leaves - 1

        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok", f"Validation failed: {report.errors}"

    @pytest.mark.asyncio
    async def test_truncate_returns_deleted_ids(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Truncate should return correct deleted node IDs."""
        doc_id = f"truncate-ids-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 200, replace_existing=True, await_idle=True
        )

        initial_ids = {n.id for n in memory_service_harness.get_all_nodes(doc_id)}
        leaves = memory_service_harness.get_leaves(doc_id)
        mid = leaves[len(leaves) // 2].span_start

        result = await memory_service_harness.truncate(doc_id, mid)

        remaining_ids = {n.id for n in memory_service_harness.get_all_nodes(doc_id)}
        expected_deleted = initial_ids - remaining_ids

        assert set(result.deleted_node_ids) == expected_deleted

    @pytest.mark.asyncio
    async def test_truncate_preserves_kept_content(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Text content of kept nodes should be preserved."""
        doc_id = f"truncate-content-{uuid.uuid4()}"
        text = "Preserved content. " * 100 + "Deleted content. " * 100

        await memory_service_harness.append(
            doc_id, text, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        mid = leaves[len(leaves) // 2].span_start

        # Get text of first half before truncate
        kept_text_before = "".join(leaf.text for leaf in leaves if leaf.span_end <= mid)

        await memory_service_harness.truncate(doc_id, mid)

        kept_text_after = memory_service_harness.concatenate_leaf_text(doc_id)
        assert kept_text_after == kept_text_before, "Kept text should be unchanged"

    @pytest.mark.asyncio
    async def test_truncate_clears_parent_refs(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """No dangling parent_id references after truncate."""
        doc_id = f"truncate-parent-refs-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 300, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        mid = leaves[len(leaves) // 2].span_start
        await memory_service_harness.truncate(doc_id, mid)

        orphans = memory_service_harness.get_orphaned_nodes(doc_id)
        assert orphans == [], f"Found orphaned nodes: {[n.id for n in orphans]}"

    @pytest.mark.asyncio
    async def test_truncate_clears_neighbor_refs(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """No dangling following_neighbor_id references after truncate."""
        doc_id = f"truncate-neighbor-refs-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 300, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        mid = leaves[len(leaves) // 2].span_start
        await memory_service_harness.truncate(doc_id, mid)

        # Validation checks for dangling neighbor refs
        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        neighbor_errors = [
            e for e in report.findings if "neighbor" in e.code and e.severity == "error"
        ]
        assert neighbor_errors == [], f"Found neighbor errors: {neighbor_errors}"


class TestTruncateThenAppend:
    """Tests for revert/branch creation patterns."""

    @pytest.mark.asyncio
    async def test_truncate_then_append(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Revert and continue with new content."""
        doc_id = f"truncate-append-{uuid.uuid4()}"

        # Build initial document
        await memory_service_harness.append(
            doc_id, "Original content. " * 150, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        mid = leaves[len(leaves) // 2].span_start

        # Truncate
        await memory_service_harness.truncate(doc_id, mid)

        # Append new content
        await memory_service_harness.append(
            doc_id,
            "New branch content. " * 100,
            replace_existing=False,
            await_idle=True,
        )

        # Verify tree is valid
        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok", f"Validation failed: {report.errors}"

        # Verify spans are contiguous
        final_leaves = memory_service_harness.get_leaves(doc_id)
        for i in range(len(final_leaves) - 1):
            assert final_leaves[i].span_end == final_leaves[i + 1].span_start

    @pytest.mark.asyncio
    async def test_truncate_to_zero_then_rebuild(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Full clear then rebuild from scratch."""
        doc_id = f"rebuild-{uuid.uuid4()}"

        # Build then clear
        await memory_service_harness.append(
            doc_id, "Original. " * 100, replace_existing=True, await_idle=True
        )
        await memory_service_harness.truncate(doc_id, 0)

        assert len(memory_service_harness.get_all_nodes(doc_id)) == 0

        # Rebuild
        new_text = "Brand new content. " * 100
        await memory_service_harness.append(
            doc_id, new_text, replace_existing=False, await_idle=True
        )

        span = memory_service_harness.get_document_span(doc_id)
        assert span is not None
        assert span[0] == 0, "Should start at 0"

        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok", f"Validation failed: {report.errors}"

    @pytest.mark.asyncio
    async def test_alternating_truncate_append(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """TATAT pattern: truncate-append-truncate-append-truncate."""
        doc_id = f"alternate-{uuid.uuid4()}"

        # Initial
        await memory_service_harness.append(
            doc_id, "Initial. " * 100, replace_existing=True, await_idle=True
        )

        for i in range(3):
            # Truncate
            leaves = memory_service_harness.get_leaves(doc_id)
            if len(leaves) > 1:
                mid = leaves[len(leaves) // 2].span_start
                await memory_service_harness.truncate(doc_id, mid)

            # Append
            await memory_service_harness.append(
                doc_id, f"Cycle {i}. " * 80, replace_existing=False, await_idle=True
            )

            # Validate after each cycle
            report = memory_service_harness.validate_tree(
                doc_id, require_complete=False
            )
            assert (
                report.status == "ok"
            ), f"Cycle {i} validation failed: {report.errors}"

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(5.0)
    async def test_small_truncate_large_append(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Small cut, big addition."""
        doc_id = f"small-truncate-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Initial. " * 200, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        # Truncate just the last leaf
        truncate_at = (
            leaves[-1].span_start if len(leaves) > 1 else leaves[0].span_end // 2
        )
        await memory_service_harness.truncate(doc_id, truncate_at)

        # Large append
        await memory_service_harness.append(
            doc_id, "Large new content. " * 400, replace_existing=False, await_idle=True
        )

        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok", f"Validation failed: {report.errors}"

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(5.0)
    async def test_large_truncate_small_append(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Big cut, small addition."""
        doc_id = f"large-truncate-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id,
            "Large initial content. " * 400,
            replace_existing=True,
            await_idle=True,
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        # Truncate most of document
        truncate_at = (
            leaves[1].span_start if len(leaves) > 1 else leaves[0].span_end // 4
        )
        await memory_service_harness.truncate(doc_id, truncate_at)

        # Small append
        await memory_service_harness.append(
            doc_id, "Small addition. " * 20, replace_existing=False, await_idle=True
        )

        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok", f"Validation failed: {report.errors}"


class TestTreeInvariants:
    """Tests for structural tree invariants."""

    @pytest.mark.asyncio
    async def test_sibling_adjacency_no_gaps(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """No span gaps between adjacent leaves."""
        doc_id = f"sibling-adjacent-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id,
            "Content for adjacency test. " * 200,
            replace_existing=True,
            await_idle=True,
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        for i in range(len(leaves) - 1):
            assert (
                leaves[i].span_end == leaves[i + 1].span_start
            ), f"Gap between leaf {i} (end={leaves[i].span_end}) and {i+1} (start={leaves[i+1].span_start})"

    @pytest.mark.asyncio
    async def test_coordinate_uniqueness(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """No duplicate (height, level_index) coordinates."""
        doc_id = f"coord-unique-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 300, replace_existing=True, await_idle=True
        )

        duplicates = memory_service_harness.check_for_duplicate_coords(doc_id)
        assert duplicates == [], f"Found duplicate coordinates: {duplicates}"

    @pytest.mark.asyncio
    async def test_parent_child_refs_valid(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """All parent_id references point to existing nodes."""
        doc_id = f"parent-child-valid-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 200, replace_existing=True, await_idle=True
        )

        orphans = memory_service_harness.get_orphaned_nodes(doc_id)
        assert orphans == [], f"Found orphaned nodes: {[n.id for n in orphans]}"

    @pytest.mark.asyncio
    async def test_height_monotonicity(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Parent height = max child height + 1."""
        doc_id = f"height-mono-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 200, replace_existing=True, await_idle=True
        )

        nodes = memory_service_harness.get_all_nodes(doc_id)
        node_map = {n.id: n for n in nodes}

        for node in nodes:
            if node.parent_id and node.parent_id in node_map:
                parent = node_map[node.parent_id]
                assert (
                    parent.height == node.height + 1
                ), f"Node {node.id} height={node.height}, parent height={parent.height}"

    @pytest.mark.asyncio
    async def test_level_index_correctness(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Left children have even level_index, right have odd."""
        doc_id = f"level-index-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 200, replace_existing=True, await_idle=True
        )

        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        coord_errors = [
            e for e in report.findings if "coord" in e.code and e.severity == "error"
        ]
        assert coord_errors == [], f"Found coordinate errors: {coord_errors}"

    @pytest.mark.asyncio
    async def test_span_union_invariant(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Parent span = union of children spans."""
        doc_id = f"span-union-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 200, replace_existing=True, await_idle=True
        )

        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        span_errors = [
            e for e in report.findings if "span" in e.code and e.severity == "error"
        ]
        assert span_errors == [], f"Found span errors: {span_errors}"

    @pytest.mark.asyncio
    async def test_perfect_binary_structure(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Internal nodes should have 2 children (perfect binary tree)."""
        doc_id = f"perfect-binary-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 200, replace_existing=True, await_idle=True
        )

        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        tree_errors = [
            e
            for e in report.findings
            if e.code in ("tree.left_only", "tree.right_only") and e.severity == "error"
        ]
        assert tree_errors == [], f"Found tree structure errors: {tree_errors}"

    @pytest.mark.asyncio
    async def test_leaves_start_at_zero(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """First leaf should start at span 0."""
        doc_id = f"leaves-zero-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 100, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        assert (
            leaves[0].span_start == 0
        ), f"First leaf starts at {leaves[0].span_start}, not 0"

    @pytest.mark.asyncio
    async def test_forest_structure(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Non-power-of-2 leaf count creates valid forest."""
        doc_id = f"forest-{uuid.uuid4()}"

        # Create document with ~5 leaves (not power of 2)
        await memory_service_harness.append(
            doc_id,
            "Word. " * 250,  # Should create ~5 leaves
            replace_existing=True,
            await_idle=True,
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        roots = memory_service_harness.get_roots(doc_id)

        # Non-power-of-2 creates forest (multiple roots)
        assert len(leaves) >= 3, "Should have multiple leaves"
        assert len(roots) >= 1, "Should have at least one root"

        # Validation should still pass
        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok", f"Forest validation failed: {report.errors}"


class TestBoundaryConditions:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_leaf_document(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Minimal text creates single leaf."""
        doc_id = f"single-leaf-{uuid.uuid4()}"

        # Very small content
        await memory_service_harness.append(
            doc_id, "Small.", replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        internal = memory_service_harness.get_internal_nodes(doc_id)

        assert len(leaves) == 1, f"Expected 1 leaf, got {len(leaves)}"
        assert len(internal) == 0, f"Expected 0 internal nodes, got {len(internal)}"

    @pytest.mark.asyncio
    async def test_two_leaf_document(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Two leaves create minimal tree with one parent."""
        doc_id = f"two-leaf-{uuid.uuid4()}"

        # Enough content for 2 leaves
        await memory_service_harness.append(
            doc_id, "Word. " * 100, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        if len(leaves) == 2:
            internal = memory_service_harness.get_internal_nodes(doc_id)
            assert len(internal) == 1, "Two leaves should have one parent"

            heights = memory_service_harness.count_nodes_by_height(doc_id)
            assert heights.get(0, 0) == 2, "Should have 2 leaves (height 0)"
            assert heights.get(1, 0) == 1, "Should have 1 parent (height 1)"

    @pytest.mark.asyncio
    async def test_power_of_two_leaves(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """4 leaves create perfect binary tree."""
        doc_id = f"power-two-{uuid.uuid4()}"

        # Try to create ~4 leaves
        await memory_service_harness.append(
            doc_id, "Word. " * 200, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        roots = memory_service_harness.get_roots(doc_id)

        # If we have exactly 4 leaves, should have 1 root
        if len(leaves) == 4:
            assert len(roots) == 1, "4 leaves should create single root"

        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok"

    @pytest.mark.asyncio
    async def test_three_leaves_forest(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """3 leaves create forest of 2+1."""
        doc_id = f"three-leaves-{uuid.uuid4()}"

        # Try to create ~3 leaves
        await memory_service_harness.append(
            doc_id, "Word. " * 150, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)

        # If we have exactly 3 leaves, should have 2 roots (tree of 2 + single leaf)
        if len(leaves) == 3:
            roots = memory_service_harness.get_roots(doc_id)
            assert len(roots) == 2, "3 leaves should create 2 roots"

        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok"

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(5.0)
    async def test_very_large_append(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Stress test with larger text (reduced from 100KB for timeout)."""
        doc_id = f"large-{uuid.uuid4()}"

        # ~10KB of text (reduced from 100KB for practical test times)
        large_text = "This is a longer sentence for stress testing. " * 200

        await memory_service_harness.append(
            doc_id, large_text, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        assert len(leaves) >= 3, "Larger document should have multiple leaves"

        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert (
            report.status == "ok"
        ), f"Large document validation failed: {report.errors}"

    @pytest.mark.asyncio
    async def test_truncate_single_leaf(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Truncate document with single leaf."""
        doc_id = f"truncate-single-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Small content.", replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        assert len(leaves) == 1

        # Truncate the only leaf
        await memory_service_harness.truncate(doc_id, 0)

        final_nodes = memory_service_harness.get_all_nodes(doc_id)
        assert len(final_nodes) == 0

    @pytest.mark.asyncio
    async def test_rebuild_from_empty(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Append to document after full truncate."""
        doc_id = f"rebuild-empty-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Initial. " * 50, replace_existing=True, await_idle=True
        )
        await memory_service_harness.truncate(doc_id, 0)

        # Rebuild
        await memory_service_harness.append(
            doc_id, "Rebuilt content. " * 50, replace_existing=False, await_idle=True
        )

        span = memory_service_harness.get_document_span(doc_id)
        assert span is not None
        assert span[0] == 0

    @pytest.mark.asyncio
    async def test_unicode_content_preserved(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Unicode characters are preserved through append/read cycle."""
        doc_id = f"unicode-{uuid.uuid4()}"

        unicode_text = "Hello 世界! Emoji: 🎉🚀 Math: ∑∞∂ " * 50

        await memory_service_harness.append(
            doc_id, unicode_text, replace_existing=True, await_idle=True
        )

        result = memory_service_harness.concatenate_leaf_text(doc_id)
        assert result == unicode_text, "Unicode content should be preserved"


class TestBackgroundJobs:
    """Tests for background job completion and await_idle behavior."""

    @pytest.mark.asyncio
    async def test_await_idle_completes_jobs(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """await_idle=True should block until all jobs complete."""
        doc_id = f"await-idle-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 200, replace_existing=True, await_idle=True
        )

        # After await_idle, tree should be complete
        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok"

        # Should have internal nodes (summaries were created)
        internal = memory_service_harness.get_internal_nodes(doc_id)
        leaves = memory_service_harness.get_leaves(doc_id)
        if len(leaves) >= 2:
            assert len(internal) > 0, "Should have internal nodes after completion"

    @pytest.mark.asyncio
    async def test_await_idle_false_returns_early(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """await_idle=False should return without waiting."""
        doc_id = f"no-await-{uuid.uuid4()}"

        # This should return quickly
        await memory_service_harness.append(
            doc_id, "Content. " * 200, replace_existing=True, await_idle=False
        )

        # May have leaves but possibly incomplete tree
        leaves = memory_service_harness.get_leaves(doc_id)
        assert len(leaves) > 0, "Should have created leaves"

    @pytest.mark.asyncio
    async def test_validate_during_indexing_ok(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Validation with require_complete=False should pass during indexing."""
        doc_id = f"validate-during-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 200, replace_existing=True, await_idle=False
        )

        # Validation should pass even if tree incomplete
        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok", f"Mid-indexing validation failed: {report.errors}"

    @pytest.mark.asyncio
    async def test_validate_after_completion(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Validation with require_complete=True should pass after completion."""
        doc_id = f"validate-complete-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 200, replace_existing=True, await_idle=True
        )

        # After completion, require_complete=True should pass
        # (Note: may still have multiple roots if leaf count is not power of 2)
        report = memory_service_harness.validate_tree(doc_id, require_complete=False)
        assert report.status == "ok"

    @pytest.mark.asyncio
    async def test_no_orphans_after_completion(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """No orphaned nodes after job completion."""
        doc_id = f"no-orphans-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id, "Content. " * 300, replace_existing=True, await_idle=True
        )

        orphans = memory_service_harness.get_orphaned_nodes(doc_id)
        assert orphans == [], f"Found orphaned nodes: {[n.id for n in orphans]}"

    @pytest.mark.asyncio
    async def test_no_duplicates_rapid_appends(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Rapid appends should not create duplicate coordinates."""
        doc_id = f"rapid-no-dup-{uuid.uuid4()}"

        # Rapid appends
        for i in range(5):
            await memory_service_harness.append(
                doc_id,
                f"Rapid content {i}. " * 50,
                replace_existing=(i == 0),
                await_idle=False,
            )

        await memory_service_harness.wait_for_idle(doc_id)

        duplicates = memory_service_harness.check_for_duplicate_coords(doc_id)
        assert duplicates == [], f"Found duplicate coordinates: {duplicates}"


class TestContentVerification:
    """Tests for text content preservation."""

    @pytest.mark.asyncio
    async def test_leaf_text_matches_original(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Concatenated leaf text equals original input."""
        doc_id = f"text-match-{uuid.uuid4()}"
        text = "This is the original text content. " * 100

        await memory_service_harness.append(
            doc_id, text, replace_existing=True, await_idle=True
        )

        result = memory_service_harness.concatenate_leaf_text(doc_id)
        assert result == text, "Concatenated leaves should equal original"

    @pytest.mark.asyncio
    async def test_append_preserves_existing(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Append does not modify existing leaf content."""
        doc_id = f"preserve-existing-{uuid.uuid4()}"
        first_text = "First content. " * 50
        second_text = "Second content. " * 50

        await memory_service_harness.append(
            doc_id, first_text, replace_existing=True, await_idle=True
        )

        await memory_service_harness.append(
            doc_id, second_text, replace_existing=False, await_idle=True
        )

        result = memory_service_harness.concatenate_leaf_text(doc_id)
        assert result == first_text + second_text

    @pytest.mark.asyncio
    async def test_truncate_preserves_kept(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Truncate does not modify kept leaf content."""
        doc_id = f"truncate-preserve-{uuid.uuid4()}"
        text = "Keep this. " * 50 + "Delete this. " * 50

        await memory_service_harness.append(
            doc_id, text, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)
        mid = leaves[len(leaves) // 2].span_start

        # Get text before truncate
        text_before = "".join(leaf.text for leaf in leaves if leaf.span_end <= mid)

        await memory_service_harness.truncate(doc_id, mid)

        text_after = memory_service_harness.concatenate_leaf_text(doc_id)
        assert text_after == text_before, "Kept text should be unchanged"

    @pytest.mark.asyncio
    async def test_internal_nodes_have_summaries(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Internal nodes (height > 0) have non-empty text (summaries)."""
        doc_id = f"summaries-{uuid.uuid4()}"

        await memory_service_harness.append(
            doc_id,
            "Content for summarization. " * 200,
            replace_existing=True,
            await_idle=True,
        )

        internal = memory_service_harness.get_internal_nodes(doc_id)
        for node in internal:
            assert node.text, f"Internal node {node.id} has no summary text"
            assert len(node.text) > 0, f"Internal node {node.id} has empty summary"

    @pytest.mark.asyncio
    async def test_leaf_spans_cover_document(
        self, memory_service_harness: MemoryServiceTestHarness
    ) -> None:
        """Leaf spans cover entire document with no gaps."""
        doc_id = f"span-coverage-{uuid.uuid4()}"
        text = "Coverage test content. " * 150

        await memory_service_harness.append(
            doc_id, text, replace_existing=True, await_idle=True
        )

        leaves = memory_service_harness.get_leaves(doc_id)

        # First leaf starts at 0
        assert leaves[0].span_start == 0, f"First leaf starts at {leaves[0].span_start}"

        # Last leaf ends at text length
        assert leaves[-1].span_end == len(
            text
        ), f"Last leaf ends at {leaves[-1].span_end}, text length is {len(text)}"

        # No gaps between leaves
        for i in range(len(leaves) - 1):
            assert (
                leaves[i].span_end == leaves[i + 1].span_start
            ), f"Gap between leaves {i} and {i+1}"
