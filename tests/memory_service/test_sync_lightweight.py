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
