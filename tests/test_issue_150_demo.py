"""Demonstration test for issue #150 using backend + DocumentStore.

This version avoids leaking DB sessions and uses the StorageBackend
and per-document DocumentStore APIs exclusively.
"""

import numpy as np
from numpy.typing import NDArray

from ragzoom.contracts.storage_backend import StorageBackend


class TestIssue150Demonstration:
    """Demonstrate the exact usage pattern requested in issue #150."""

    def test_usage_pattern_from_issue(self, storage_backend: StorageBackend) -> None:
        """Test the exact usage pattern described in issue #150."""
        # This demonstrates the usage pattern from the issue description:
        #
        # # Atomic multi-operation sequence
        # with store.transaction() as session:
        #     store.delete_document_nodes(doc_id, session=session)
        #     store.add_document(doc_data, session=session)
        #     store.nodes.add_nodes_batch(nodes, session=session)
        #     # All commit together or all rollback

        # Setup: Create initial document with nodes
        doc_id = "demo-doc"
        doc_store = storage_backend.for_document(doc_id)
        # Create initial document record
        doc_store.set_metadata(
            file_path="demo.txt",
            content_hash="old-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        old_nodes_data: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "old-node",
                "text": "Old content",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 11,
                "document_id": doc_id,
                "token_count": 2,
            }
        ]
        doc_store.nodes.add_batch(old_nodes_data)

        # Verify initial state
        assert doc_store.get_metadata() is not None
        assert doc_store.nodes.get_node("old-node") is not None

        # Demonstrate atomic multi-operation sequence from issue #150
        new_doc_data: dict[str, str | int] = {
            "document_id": doc_id,
            "file_path": "demo.txt",
            "content_hash": "new-hash",
            "chunk_count": 1,
            "embedding_model": "text-embedding-3-small",
            "summary_model": "gpt-4o-mini",
        }

        new_nodes_data: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "new-node",
                "text": "New content",
                "embedding": [0.2] * 1536,
                "span_start": 0,
                "span_end": 11,
                "document_id": doc_id,
                "token_count": 2,
            }
        ]

        # ATOMIC OPERATION: All operations commit together or all rollback
        with doc_store.transaction() as session:
            # Delete old nodes
            deleted_count = doc_store.clear_document(doc_id, session=session)
            assert deleted_count == 1

            # Update document metadata (using add since update isn't implemented yet)
            # In a real scenario, this would be update_document()
            doc_store.set_metadata(
                file_path=str(new_doc_data["file_path"]),
                content_hash=str(new_doc_data["content_hash"]),
                chunk_count=int(new_doc_data["chunk_count"]),
                embedding_model=str(new_doc_data["embedding_model"]),
                summary_model=str(new_doc_data["summary_model"]),
            )

            # Add new nodes
            new_nodes = doc_store.nodes.add_batch(new_nodes_data, session=session)
            assert len(new_nodes) == 1

            # All operations are part of the same transaction
            # If any operation fails, everything rolls back
            # If we reach here, everything commits atomically

        # Verify final state: atomic replacement succeeded
        final_doc = doc_store.get_metadata()
        assert final_doc is not None
        assert final_doc.content_hash == "new-hash"  # Document updated

        assert doc_store.nodes.get_node("old-node") is None  # Old node deleted
        assert doc_store.nodes.get_node("new-node") is not None  # New node added

    def test_backward_compatibility_demonstration(
        self, storage_backend: StorageBackend
    ) -> None:
        """Demonstrate that existing code works unchanged (backward compatibility)."""
        # Existing code that doesn't use transactions continues to work

        # Add document without session (existing API)
        doc_store = storage_backend.for_document("backward-compat-doc")
        doc_store.set_metadata(
            file_path="test.txt",
            content_hash="test-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        assert doc_store.document_id == "backward-compat-doc"

        # Add nodes without session (existing API)
        nodes_data: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "compat-node",
                "text": "Test content",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 12,
                "document_id": "backward-compat-doc",
                "token_count": 3,
            }
        ]
        nodes = doc_store.nodes.add_batch(nodes_data)  # No session parameter
        assert len(nodes) == 1

        # Delete nodes without session (existing API)
        deleted_count = storage_backend.clear_document(
            "backward-compat-doc"
        )  # No session
        assert deleted_count == 1

        # All existing APIs work exactly as before
        assert doc_store.nodes.get_node("compat-node") is None
