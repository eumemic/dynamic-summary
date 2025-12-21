"""Tests for document truncation (delete nodes from span position onward)."""

from __future__ import annotations

from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend


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
