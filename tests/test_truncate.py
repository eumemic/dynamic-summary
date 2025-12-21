"""Tests for document truncation (delete nodes from span position onward)."""

from __future__ import annotations

import pytest

from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.indexing import TruncateResult
from tests.conftest import IndexerRuntimeHarness


class TestDeleteNodesFromSpan:
    """Tests for StorageBackend.delete_nodes_from_span."""

    def test_deletes_nodes_at_and_after_span(
        self, storage_backend: StorageBackend
    ) -> None:
        """Should delete all nodes with span_start >= given position."""
        doc_id = "test-truncate"
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Create nodes at different span positions
        nodes: list[NodeDataDict] = [
            {
                "node_id": "node-0-100",
                "text": "First chunk",
                "span_start": 0,
                "span_end": 100,
                "document_id": doc_id,
                "token_count": 10,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "node-100-200",
                "text": "Second chunk",
                "span_start": 100,
                "span_end": 200,
                "document_id": doc_id,
                "token_count": 10,
                "height": 0,
                "level_index": 1,
            },
            {
                "node_id": "node-200-300",
                "text": "Third chunk",
                "span_start": 200,
                "span_end": 300,
                "document_id": doc_id,
                "token_count": 10,
                "height": 0,
                "level_index": 2,
            },
        ]
        doc_store.nodes.add_batch(nodes)

        # Delete from span 100 onward
        deleted_ids = storage_backend.delete_nodes_from_span(doc_id, span_start=100)

        # Should have deleted 2 nodes (100-200 and 200-300)
        assert len(deleted_ids) == 2
        assert "node-100-200" in deleted_ids
        assert "node-200-300" in deleted_ids

        # First node should still exist
        assert doc_store.nodes.get_node("node-0-100") is not None

        # Deleted nodes should be gone
        assert doc_store.nodes.get_node("node-100-200") is None
        assert doc_store.nodes.get_node("node-200-300") is None

    def test_returns_empty_list_when_nothing_to_delete(
        self, storage_backend: StorageBackend
    ) -> None:
        """Should return empty list when no nodes match criteria."""
        doc_id = "test-truncate-empty"
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        nodes: list[NodeDataDict] = [
            {
                "node_id": "node-0-100",
                "text": "Only chunk",
                "span_start": 0,
                "span_end": 100,
                "document_id": doc_id,
                "token_count": 10,
                "height": 0,
                "level_index": 0,
            },
        ]
        doc_store.nodes.add_batch(nodes)

        # Try to delete from span 200 (beyond all nodes)
        deleted_ids = storage_backend.delete_nodes_from_span(doc_id, span_start=200)

        assert deleted_ids == []
        assert doc_store.nodes.get_node("node-0-100") is not None

    def test_deletes_all_nodes_from_span_zero(
        self, storage_backend: StorageBackend
    ) -> None:
        """Span 0 should delete all nodes (equivalent to clear)."""
        doc_id = "test-truncate-all"
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        nodes: list[NodeDataDict] = [
            {
                "node_id": "node-1",
                "text": "First",
                "span_start": 0,
                "span_end": 100,
                "document_id": doc_id,
                "token_count": 10,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "node-2",
                "text": "Second",
                "span_start": 100,
                "span_end": 200,
                "document_id": doc_id,
                "token_count": 10,
                "height": 0,
                "level_index": 1,
            },
        ]
        doc_store.nodes.add_batch(nodes)

        deleted_ids = storage_backend.delete_nodes_from_span(doc_id, span_start=0)

        assert len(deleted_ids) == 2
        assert doc_store.nodes.get_node("node-1") is None
        assert doc_store.nodes.get_node("node-2") is None

    def test_handles_nonexistent_document(
        self, storage_backend: StorageBackend
    ) -> None:
        """Should return empty list for nonexistent document."""
        deleted_ids = storage_backend.delete_nodes_from_span(
            "nonexistent-doc", span_start=0
        )

        assert deleted_ids == []

    def test_deletes_internal_nodes_too(self, storage_backend: StorageBackend) -> None:
        """Should delete both leaf and internal nodes."""
        doc_id = "test-truncate-tree"
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Create a tree: two leaves and one internal node
        nodes: list[NodeDataDict] = [
            {
                "node_id": "leaf-0-100",
                "text": "Leaf 1",
                "span_start": 0,
                "span_end": 100,
                "document_id": doc_id,
                "token_count": 10,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf-100-200",
                "text": "Leaf 2",
                "span_start": 100,
                "span_end": 200,
                "document_id": doc_id,
                "token_count": 10,
                "height": 0,
                "level_index": 1,
            },
            {
                "node_id": "internal-0-200",
                "text": "Summary of both",
                "span_start": 0,
                "span_end": 200,
                "document_id": doc_id,
                "token_count": 5,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leaf-0-100",
                "right_child_id": "leaf-100-200",
            },
        ]
        doc_store.nodes.add_batch(nodes)

        # Delete from span 100 - should get leaf-100-200 but NOT internal-0-200
        # (internal has span_start=0)
        deleted_ids = storage_backend.delete_nodes_from_span(doc_id, span_start=100)

        assert len(deleted_ids) == 1
        assert "leaf-100-200" in deleted_ids

        # Internal node should still exist (span_start=0 < 100)
        assert doc_store.nodes.get_node("internal-0-200") is not None

    def test_only_affects_specified_document(
        self, storage_backend: StorageBackend
    ) -> None:
        """Should not delete nodes from other documents."""
        doc1_id = "test-truncate-doc1"
        doc2_id = "test-truncate-doc2"

        for doc_id in [doc1_id, doc2_id]:
            doc_store = storage_backend.for_document(doc_id)
            doc_store.set_metadata(
                file_path=f"{doc_id}.txt",
                embedding_model="text-embedding-3-small",
                summary_model="gpt-4o-mini",
            )
            doc_store.nodes.add_batch(
                [
                    {
                        "node_id": f"{doc_id}-node",
                        "text": "Content",
                        "span_start": 0,
                        "span_end": 100,
                        "document_id": doc_id,
                        "token_count": 10,
                        "height": 0,
                        "level_index": 0,
                    }
                ]
            )

        # Delete from doc1 only
        deleted_ids = storage_backend.delete_nodes_from_span(doc1_id, span_start=0)

        assert len(deleted_ids) == 1
        assert f"{doc1_id}-node" in deleted_ids

        # doc2's node should still exist
        doc2_store = storage_backend.for_document(doc2_id)
        assert doc2_store.nodes.get_node(f"{doc2_id}-node") is not None


class TestRuntimeTruncate:
    """Tests for DocumentIndexSession.truncate_from_span."""

    @pytest.mark.asyncio
    async def test_truncate_deletes_nodes_and_returns_result(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """Truncate should delete nodes and return TruncateResult."""
        doc_id = "test-runtime-truncate"

        # Index a document
        await indexer_runtime_harness.append(
            doc_id,
            "First part. " * 20 + "\n\n" + "Second part. " * 20,
            replace_existing=True,
            await_idle=True,
        )

        doc_store = storage_backend.for_document(doc_id)
        leaves_before = list(doc_store.nodes.get_leaves())
        assert len(leaves_before) >= 2, "Need at least 2 leaves for meaningful test"

        # Find a span_start in the middle
        leaves_sorted = sorted(leaves_before, key=lambda n: n.span_start)
        mid_leaf = leaves_sorted[len(leaves_sorted) // 2]
        truncate_point = mid_leaf.span_start

        # Truncate from the midpoint
        result = await indexer_runtime_harness.truncate(doc_id, truncate_point)

        assert isinstance(result, TruncateResult)
        assert result.document_id == doc_id
        assert result.span_start == truncate_point
        assert len(result.deleted_node_ids) > 0

        # Verify nodes were actually deleted
        leaves_after = list(doc_store.nodes.get_leaves())
        assert len(leaves_after) < len(leaves_before)

        # All remaining leaves should have span_start < truncate_point
        for leaf in leaves_after:
            assert leaf.span_start < truncate_point

    @pytest.mark.asyncio
    async def test_truncate_returns_empty_when_nothing_to_delete(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """Truncate past all nodes should return empty list."""
        doc_id = "test-truncate-empty"

        await indexer_runtime_harness.append(
            doc_id,
            "Short content.",
            replace_existing=True,
            await_idle=True,
        )

        doc_store = storage_backend.for_document(doc_id)
        leaves = list(doc_store.nodes.get_leaves())
        max_span_end = max(leaf.span_end for leaf in leaves)

        # Truncate beyond all content
        result = await indexer_runtime_harness.truncate(doc_id, max_span_end + 1000)

        assert result.deleted_node_ids == []
        # Original nodes should still exist
        assert len(list(doc_store.nodes.get_leaves())) == len(leaves)

    @pytest.mark.asyncio
    async def test_truncate_from_zero_deletes_all(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """Truncate from span 0 should delete all nodes."""
        doc_id = "test-truncate-all"

        await indexer_runtime_harness.append(
            doc_id,
            "Some content here.",
            replace_existing=True,
            await_idle=True,
        )

        doc_store = storage_backend.for_document(doc_id)
        leaves_before = list(doc_store.nodes.get_leaves())
        assert len(leaves_before) > 0

        result = await indexer_runtime_harness.truncate(doc_id, span_start=0)

        assert len(result.deleted_node_ids) > 0
        assert list(doc_store.nodes.get_leaves()) == []

    @pytest.mark.asyncio
    async def test_truncate_nonexistent_document(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Truncate on nonexistent document should return empty result."""
        result = await indexer_runtime_harness.truncate("nonexistent-doc", span_start=0)

        assert result.deleted_node_ids == []
        assert result.document_id == "nonexistent-doc"
        assert result.span_start == 0
