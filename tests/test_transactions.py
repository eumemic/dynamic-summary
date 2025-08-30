"""Test transactional operations for the Store class."""

from typing import Any

import pytest


class TestTransactionContext:
    """Test the transaction context manager."""

    def test_transaction_context_manager_success(self, store: Any) -> None:
        """Test successful transaction commits all operations."""
        # Use transaction to add document and nodes atomically
        doc_id = "test-doc"
        nodes_data = [
            {
                "node_id": "node-1",
                "text": "Test content",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 12,
                "document_id": doc_id,
                "token_count": 3,
            }
        ]

        with store.transaction() as session:
            # Add document
            doc_store = store.add_document(
                document_id=doc_id,
                file_path="test.txt",
                content_hash="test-hash",
                chunk_count=1,
                embedding_model="text-embedding-3-small",
                summary_model="gpt-4o-mini",
                session=session,
            )

            # Add nodes
            nodes = store.nodes.add_nodes_batch(nodes_data, session=session)

            # Both should be available within the transaction
            assert doc_store.document_id == doc_id
            assert len(nodes) == 1
            assert nodes[0].id == "node-1"

        # Verify both operations were committed
        persisted_doc = store.get_document_by_id(doc_id)
        persisted_node = store.nodes.get_node("node-1")

        assert persisted_doc is not None
        assert persisted_doc.id == doc_id
        assert persisted_node is not None
        assert persisted_node.id == "node-1"

    @pytest.mark.integration
    def test_transaction_context_manager_rollback(self, store: Any) -> None:
        """Test failed transaction rolls back all operations."""
        if hasattr(store, "__class__") and "Mock" in store.__class__.__name__:
            pytest.skip("Mock store doesn't support true rollback behavior")

        doc_id = "test-doc-rollback"

        # Simulate a transaction that fails
        with pytest.raises(ValueError, match="Simulated error"):
            with store.transaction() as session:
                # Add document
                store.add_document(
                    document_id=doc_id,
                    file_path="test.txt",
                    content_hash="test-hash",
                    chunk_count=1,
                    embedding_model="text-embedding-3-small",
                    summary_model="gpt-4o-mini",
                    session=session,
                )

                # Simulate error before commit
                raise ValueError("Simulated error")

        # Verify nothing was committed
        persisted_doc = store.get_document_by_id(doc_id)
        assert persisted_doc is None

    def test_transaction_with_parent_references(self, store: Any) -> None:
        """Test transaction with parent reference updates."""
        doc_id = "test-doc-parents"

        # Create nodes in transaction
        nodes_data = [
            {
                "node_id": "leaf-1",
                "text": "Leaf 1",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 6,
                "document_id": doc_id,
                "token_count": 2,
                "height": 0,
            },
            {
                "node_id": "leaf-2",
                "text": "Leaf 2",
                "embedding": [0.2] * 1536,
                "span_start": 7,
                "span_end": 13,
                "document_id": doc_id,
                "token_count": 2,
                "height": 0,
            },
            {
                "node_id": "parent-1",
                "text": "Parent of leaves",
                "embedding": [0.3] * 1536,
                "span_start": 0,
                "span_end": 13,
                "document_id": doc_id,
                "token_count": 4,
                "height": 1,
                "left_child_id": "leaf-1",
                "right_child_id": "leaf-2",
            },
        ]

        with store.transaction() as session:
            # Add document
            store.add_document(
                document_id=doc_id,
                file_path="test.txt",
                content_hash="test-hash",
                chunk_count=2,
                embedding_model="text-embedding-3-small",
                summary_model="gpt-4o-mini",
                session=session,
            )

            # Add nodes
            store.nodes.add_nodes_batch(nodes_data, session=session)

            # Update parent references
            parent_updates = [
                ("leaf-1", "parent-1"),
                ("leaf-2", "parent-1"),
            ]
            store.update_parent_references_batch(parent_updates, session=session)

        # Verify all operations were committed
        leaf1 = store.nodes.get_node("leaf-1")
        leaf2 = store.nodes.get_node("leaf-2")
        parent = store.nodes.get_node("parent-1")

        assert leaf1 is not None
        assert leaf1.parent_id == "parent-1"
        assert leaf2 is not None
        assert leaf2.parent_id == "parent-1"
        assert parent is not None
        assert parent.left_child_id == "leaf-1"
        assert parent.right_child_id == "leaf-2"


class TestBackwardCompatibility:
    """Test that existing code still works without transactions."""

    def test_add_document_without_session(self, store: Any) -> None:
        """Test add_document works without session parameter."""
        doc_store = store.add_document(
            document_id="test-doc-no-session",
            file_path="test.txt",
            content_hash="test-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        assert doc_store.document_id == "test-doc-no-session"

        # Verify it was persisted
        persisted_doc = store.get_document_by_id("test-doc-no-session")
        assert persisted_doc is not None

    def test_add_nodes_batch_without_session(self, store: Any) -> None:
        """Test add_nodes_batch works without session parameter."""
        nodes_data = [
            {
                "node_id": "node-no-session",
                "text": "Test content",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 12,
                "document_id": None,
                "token_count": 3,
            }
        ]

        nodes = store.nodes.add_nodes_batch(nodes_data)

        assert len(nodes) == 1
        assert nodes[0].id == "node-no-session"

        # Verify it was persisted
        persisted_node = store.nodes.get_node("node-no-session")
        assert persisted_node is not None

    def test_delete_document_nodes_without_session(self, store: Any) -> None:
        """Test clear_document works without session parameter."""
        # First add a document with nodes
        doc_id = "test-doc-delete"
        store.add_document(
            document_id=doc_id,
            file_path="test.txt",
            content_hash="test-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        nodes_data = [
            {
                "node_id": "node-to-delete",
                "text": "Test content",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 12,
                "document_id": doc_id,
                "token_count": 3,
            }
        ]
        store.nodes.add_nodes_batch(nodes_data)

        # Delete nodes without session
        deleted_count = store.clear_document(doc_id)

        assert deleted_count == 1

        # Verify node was deleted
        persisted_node = store.nodes.get_node("node-to-delete")
        assert persisted_node is None


class TestAtomicReindexing:
    """Test atomic re-indexing scenario from issue #150."""

    def test_atomic_reindexing_success(self, store: Any) -> None:
        """Test successful atomic re-indexing of a document."""
        doc_id = "test-doc-reindex"

        # First, index the document with initial content
        store.add_document(
            document_id=doc_id,
            file_path="test.txt",
            content_hash="old-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        old_nodes_data = [
            {
                "node_id": "old-node-1",
                "text": "Old content",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 11,
                "document_id": doc_id,
                "token_count": 2,
            }
        ]
        store.nodes.add_nodes_batch(old_nodes_data)

        # Verify old content exists
        assert store.nodes.get_node("old-node-1") is not None

        # Now atomically re-index with new content
        new_nodes_data = [
            {
                "node_id": "new-node-1",
                "text": "New content",
                "embedding": [0.2] * 1536,
                "span_start": 0,
                "span_end": 11,
                "document_id": doc_id,
                "token_count": 2,
            }
        ]

        with store.transaction() as session:
            # Delete old nodes
            deleted_count = store.clear_document(doc_id, session=session)
            assert deleted_count == 1

            # Add new nodes
            new_nodes = store.nodes.add_nodes_batch(new_nodes_data, session=session)
            assert len(new_nodes) == 1

        # Verify atomic operation: old gone, new present
        assert store.nodes.get_node("old-node-1") is None
        assert store.nodes.get_node("new-node-1") is not None

    @pytest.mark.integration
    def test_atomic_reindexing_rollback(self, store: Any) -> None:
        """Test atomic re-indexing rolls back on failure."""
        if hasattr(store, "__class__") and "Mock" in store.__class__.__name__:
            pytest.skip("Mock store doesn't support true rollback behavior")
        doc_id = "test-doc-reindex-fail"

        # First, index the document with initial content
        store.add_document(
            document_id=doc_id,
            file_path="test.txt",
            content_hash="old-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        old_nodes_data = [
            {
                "node_id": "old-node-fail",
                "text": "Old content",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 11,
                "document_id": doc_id,
                "token_count": 2,
            }
        ]
        store.nodes.add_nodes_batch(old_nodes_data)

        # Verify old content exists
        old_node = store.nodes.get_node("old-node-fail")
        assert old_node is not None

        # Attempt atomic re-index that fails
        with pytest.raises(ValueError, match="Simulated reindex failure"):
            with store.transaction() as session:
                # Delete old nodes
                store.clear_document(doc_id, session=session)

                # Simulate failure before adding new nodes
                raise ValueError("Simulated reindex failure")

        # Verify old content is still there (rollback succeeded)
        persisted_old_node = store.nodes.get_node("old-node-fail")
        assert persisted_old_node is not None
        assert persisted_old_node.text == "Old content"


class TestTransactionSafety:
    """Test transaction safety improvements addressing code review feedback."""

    def test_repository_method_rollback_on_exception(self, store: Any) -> None:
        """Test that repository methods properly rollback on exceptions when managing their own session."""
        # Add initial data
        store.add_document(
            document_id="rollback-test-doc",
            file_path="test.txt",
            content_hash="hash123",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # First, add a node with correct dimension to establish expected dimension
        store.node_repo.add_node(
            node_id="valid-node",
            text="Valid content",
            embedding=[0.1] * 1536,  # Standard dimension
            span_start=0,
            span_end=13,
            document_id="rollback-test-doc",
            token_count=2,
        )

        # Test rollback by triggering an exception DURING the operation
        # Use wrong embedding dimension to cause validation error
        from ragzoom.exceptions import InvalidOperationError

        # Real store raises InvalidOperationError, mock store raises ValueError
        with pytest.raises((InvalidOperationError, ValueError), match="dimension"):
            store.node_repo.add_nodes_batch(
                [
                    {
                        "node_id": "test-node-exception",
                        "text": "Test content",
                        "embedding": [0.1]
                        * 100,  # Wrong dimension - will cause exception
                        "span_start": 0,
                        "span_end": 12,
                        "document_id": "rollback-test-doc",
                        "token_count": 3,
                    }
                ]
            )

        # Verify the node was not persisted due to rollback
        node = store.nodes.get_node("test-node-exception")
        assert node is None

        # Verify the valid node is still there
        valid_node = store.nodes.get_node("valid-node")
        assert valid_node is not None

    def test_nested_transaction_prevention(self, store: Any) -> None:
        """Test that nested transactions are properly prevented."""
        with store.transaction():
            # Attempt to start nested transaction should fail
            with pytest.raises(
                RuntimeError, match="Nested transactions are not supported"
            ):
                with store.transaction():
                    pass

    def test_mock_store_rollback_simulation(self, mock_store: Any) -> None:
        """Test that mock store properly simulates rollback behavior."""
        # Add initial data
        mock_store.add_document(
            document_id="mock-rollback-test",
            file_path="test.txt",
            content_hash="hash123",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Verify document exists
        assert "mock-rollback-test" in mock_store.documents

        # Test rollback simulation
        with pytest.raises(ValueError, match="Simulated failure"):
            with mock_store.transaction() as session:
                # Delete the document
                mock_store.clear_document("mock-rollback-test", session=session)

                # Verify document is gone during transaction
                # (This behavior may vary based on mock implementation)

                # Simulate failure
                raise ValueError("Simulated failure")

        # Verify document was restored after rollback
        assert "mock-rollback-test" in mock_store.documents

    def test_mock_store_nested_transaction_prevention(self, mock_store: Any) -> None:
        """Test that mock store prevents nested transactions like real store."""
        with mock_store.transaction():
            # Attempt to start nested transaction should fail
            with pytest.raises(
                RuntimeError, match="Nested transactions are not supported"
            ):
                with mock_store.transaction():
                    pass

    def test_session_scope_context_manager(self, store: Any) -> None:
        """Test the new _session_scope context manager in BaseRepository."""
        # Only test with real store (mock store doesn't have repository structure)
        if hasattr(store, "doc_repo"):
            # This tests the new safe session handling
            doc_repo = store.doc_repo

            # Test successful operation
            with doc_repo._session_scope() as session:
                # The session should be managed properly
                assert session is not None

            # Test exception handling - this should trigger rollback
            with pytest.raises(ValueError, match="Test exception"):
                with doc_repo._session_scope() as session:
                    # Simulate an operation that fails
                    raise ValueError("Test exception")

            # If we get here, the exception was properly handled and session cleaned up
        else:
            # For mock store, just test that it has the required transaction behavior
            assert hasattr(store, "transaction")
            assert hasattr(store, "_active_transaction")
