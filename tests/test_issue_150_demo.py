"""Demonstration test for issue #150 atomic multi-operation functionality."""


class TestIssue150Demonstration:
    """Demonstrate the exact usage pattern requested in issue #150."""

    def test_usage_pattern_from_issue(self, store):
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
        store.add_document(
            document_id=doc_id,
            file_path="demo.txt",
            content_hash="old-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        old_nodes_data = [
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
        store.nodes.add_nodes_batch(old_nodes_data)

        # Verify initial state
        assert store.get_document_by_id(doc_id) is not None
        assert store.nodes.get_node("old-node") is not None

        # Demonstrate atomic multi-operation sequence from issue #150
        new_doc_data = {
            "document_id": doc_id,
            "file_path": "demo.txt",
            "content_hash": "new-hash",
            "chunk_count": 1,
            "embedding_model": "text-embedding-3-small",
            "summary_model": "gpt-4o-mini",
        }

        new_nodes_data = [
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
        with store.transaction() as session:
            # Delete old nodes
            deleted_count = store.delete_document_nodes(doc_id, session=session)
            assert deleted_count == 1

            # Update document metadata (using add since update isn't implemented yet)
            # In a real scenario, this would be update_document()
            store.add_document(**new_doc_data, session=session)

            # Add new nodes
            new_nodes = store.nodes.add_nodes_batch(new_nodes_data, session=session)
            assert len(new_nodes) == 1

            # All operations are part of the same transaction
            # If any operation fails, everything rolls back
            # If we reach here, everything commits atomically

        # Verify final state: atomic replacement succeeded
        final_doc = store.get_document_by_id(doc_id)
        assert final_doc is not None
        assert final_doc.content_hash == "new-hash"  # Document updated

        assert store.nodes.get_node("old-node") is None  # Old node deleted
        assert store.nodes.get_node("new-node") is not None  # New node added

    def test_backward_compatibility_demonstration(self, store):
        """Demonstrate that existing code works unchanged (backward compatibility)."""
        # Existing code that doesn't use transactions continues to work

        # Add document without session (existing API)
        doc = store.add_document(
            document_id="backward-compat-doc",
            file_path="test.txt",
            content_hash="test-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
            # No session parameter - uses existing behavior
        )
        assert doc.id == "backward-compat-doc"

        # Add nodes without session (existing API)
        nodes_data = [
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
        nodes = store.nodes.add_nodes_batch(nodes_data)  # No session parameter
        assert len(nodes) == 1

        # Delete nodes without session (existing API)
        deleted_count = store.delete_document_nodes("backward-compat-doc")  # No session
        assert deleted_count == 1

        # All existing APIs work exactly as before
        assert store.nodes.get_node("compat-node") is None
