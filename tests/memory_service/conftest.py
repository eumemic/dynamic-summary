"""Fixtures for lightweight memory_service tests without Docker/gRPC.

This module provides a test harness that wraps IndexerRuntimeHarness to enable
fast testing of memory_service sync and truncation logic using SQLite + mocked
OpenAI, without requiring the full devstack.

Usage:
    pytest tests/memory_service/test_sync_lightweight.py -xvs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.indexing.runtime import TruncateResult
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.validation.tree import ValidationReport, validate_document

if TYPE_CHECKING:
    from tests.conftest import IndexerRuntimeHarness


@dataclass
class MemoryServiceTestHarness:
    """Lightweight test harness for memory_service without Docker/gRPC.

    Wraps IndexerRuntimeHarness to provide a memory-service-compatible interface
    for testing sync, truncation, and tree integrity without external services.

    Example:
        async def test_something(memory_service_harness):
            await memory_service_harness.append("doc-1", "some text")
            await memory_service_harness.truncate("doc-1", span_start=50)
            orphans = memory_service_harness.get_orphaned_nodes("doc-1")
            assert orphans == []
    """

    storage_backend: StorageBackend
    indexer_harness: IndexerRuntimeHarness

    async def truncate(self, document_id: str, span_start: int) -> TruncateResult:
        """Truncate document from span position.

        This exercises the same code path as gRPC truncation, including
        delete_nodes_from_span which has Bug 3 (FK violation).
        """
        return await self.indexer_harness.truncate(document_id, span_start)

    async def append(
        self,
        document_id: str,
        text: str,
        *,
        replace_existing: bool = False,
        await_idle: bool = True,
    ) -> IndexingResult:
        """Append text to document."""
        return await self.indexer_harness.append(
            document_id,
            text,
            replace_existing=replace_existing,
            await_idle=await_idle,
        )

    async def batch_append(
        self,
        document_id: str,
        units: list[str],
        *,
        await_idle: bool = True,
    ) -> IndexingResult:
        """Batch append multiple text units."""
        session = self.indexer_harness.runtime.get_session(document_id)
        result = await session.batch_append_text(units)
        if await_idle:
            await self.indexer_harness.wait_for_idle(document_id)
        return result

    async def wait_for_idle(self, document_id: str | None = None) -> None:
        """Wait for background indexing to complete."""
        await self.indexer_harness.wait_for_idle(document_id)

    def get_all_nodes(self, document_id: str) -> list[TreeNode]:
        """Get all nodes for a document."""
        doc_store = self.storage_backend.for_document(document_id)
        return list(doc_store.nodes.get_all())

    def get_leaves(self, document_id: str) -> list[TreeNode]:
        """Get leaf nodes (height=0) for a document, sorted by span_start."""
        nodes = self.get_all_nodes(document_id)
        leaves = [n for n in nodes if n.height == 0]
        return sorted(leaves, key=lambda n: n.span_start)

    def get_orphaned_nodes(self, document_id: str) -> list[TreeNode]:
        """Find nodes with parent_id pointing to non-existent nodes.

        This is the symptom of Bug 3 - truncation deletes parent nodes while
        leaving children with dangling parent_id references.
        """
        nodes = self.get_all_nodes(document_id)
        node_ids = {n.id for n in nodes}
        return [n for n in nodes if n.parent_id and n.parent_id not in node_ids]

    def check_for_duplicate_coords(
        self, document_id: str
    ) -> list[tuple[int, int, int]]:
        """Check for duplicate (height, level_index) coordinates.

        Returns list of (height, level_index, count) for any duplicates found.
        """
        from collections import Counter

        nodes = self.get_all_nodes(document_id)
        coord_counts = Counter((n.height, n.level_index) for n in nodes)
        return [
            (height, level_index, count)
            for (height, level_index), count in coord_counts.items()
            if count > 1
        ]

    def validate_tree(
        self, document_id: str, *, require_complete: bool = False
    ) -> ValidationReport:
        """Run tree validation and return report."""
        return validate_document(
            document_id=document_id,
            store=self.storage_backend,
            require_complete=require_complete,
        )

    def get_internal_nodes(self, document_id: str) -> list[TreeNode]:
        """Get internal nodes (height > 0) for a document."""
        nodes = self.get_all_nodes(document_id)
        return [n for n in nodes if n.height > 0]

    def get_roots(self, document_id: str) -> list[TreeNode]:
        """Get root nodes (nodes without parents in the current tree)."""
        nodes = self.get_all_nodes(document_id)
        node_ids = {n.id for n in nodes}
        return [n for n in nodes if n.parent_id is None or n.parent_id not in node_ids]

    def get_document_span(self, document_id: str) -> tuple[int, int] | None:
        """Get (span_start, span_end) covering entire document.

        Returns None if document has no leaves.
        """
        leaves = self.get_leaves(document_id)
        if not leaves:
            return None
        return (leaves[0].span_start, leaves[-1].span_end)

    def concatenate_leaf_text(self, document_id: str) -> str:
        """Get concatenated text of all leaves in span order."""
        leaves = self.get_leaves(document_id)
        return "".join(leaf.text for leaf in leaves)

    def count_nodes_by_height(self, document_id: str) -> dict[int, int]:
        """Return count of nodes at each height level."""
        from collections import Counter

        nodes = self.get_all_nodes(document_id)
        return dict(Counter(n.height for n in nodes))


@pytest.fixture
async def memory_service_harness(
    storage_backend: StorageBackend,
    indexer_runtime_harness: IndexerRuntimeHarness,
) -> MemoryServiceTestHarness:
    """Provide lightweight test harness for memory_service tests.

    This fixture combines the storage_backend and indexer_runtime_harness
    fixtures from the main conftest.py to provide a convenient interface
    for testing memory_service sync and truncation logic.

    No Docker or gRPC required - uses SQLite in-memory + mocked OpenAI.
    """
    return MemoryServiceTestHarness(
        storage_backend=storage_backend,
        indexer_harness=indexer_runtime_harness,
    )
