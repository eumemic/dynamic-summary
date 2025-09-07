"""SQLite-based tests for transactional operations.

Using the real in-memory SQLite backend
for testing transactional operations and atomic behavior.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.document_store import DocumentStore


@pytest.mark.usefixtures("sqlite_backend")
class TestTransactionContextSQLite:
    """Test transaction-like behavior with SQLite backend."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("test-doc")

    def test_transaction_context_manager_success(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test successful transaction commits all operations."""
        # Use transaction to add document and nodes atomically
        doc_id = "test-doc"
        nodes_data: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "node-1",
                "text": "Test content",
                "span_start": 0,
                "span_end": 12,
                "document_id": doc_id,
                "token_count": 3,
                "height": 0,
                "path": "",
            }
        ]

        # Use raw SQLAlchemy session for transaction management
        with sqlite_backend.db.SessionLocal() as session:
            try:
                # Add document
                sqlite_backend.doc_repo.add_document(
                    document_id=doc_id,
                    file_path="test.txt",
                    content_hash="test-hash",
                    chunk_count=1,
                    embedding_model="text-embedding-3-small",
                    summary_model="gpt-4o-mini",
                    session=session,
                )

                # Add nodes
                doc_store.nodes.add_batch(nodes_data, session=session)
                session.commit()
            except Exception:
                session.rollback()
                raise

        # Verify both operations were committed
        persisted_doc = sqlite_backend.doc_repo.get_document_by_id(doc_id)
        persisted_node = doc_store.nodes.get_node("node-1")

        assert persisted_doc is not None
        assert persisted_doc.id == doc_id
        assert persisted_node is not None
        assert persisted_node.id == "node-1"

    def test_transaction_context_manager_rollback(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test failed transaction rolls back all operations."""
        doc_id = "test-doc-rollback"

        # Simulate a transaction that fails
        with pytest.raises(ValueError, match="Simulated error"):
            with sqlite_backend.db.SessionLocal() as session:
                try:
                    # Add document
                    sqlite_backend.doc_repo.add_document(
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
                except Exception:
                    session.rollback()
                    raise

        # Verify nothing was committed
        persisted_doc = sqlite_backend.doc_repo.get_document_by_id(doc_id)
        assert persisted_doc is None

    def test_transaction_with_parent_references(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test transaction with parent reference updates."""
        doc_id = "test-doc-parents"

        # Create nodes in transaction
        nodes_data: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "leaf-1",
                "text": "Leaf 1",
                "span_start": 0,
                "span_end": 6,
                "document_id": doc_id,
                "token_count": 2,
                "height": 0,
                "path": "0",
            },
            {
                "node_id": "leaf-2",
                "text": "Leaf 2",
                "span_start": 7,
                "span_end": 13,
                "document_id": doc_id,
                "token_count": 2,
                "height": 0,
                "path": "1",
            },
            {
                "node_id": "parent-1",
                "text": "Parent of leaves",
                "span_start": 0,
                "span_end": 13,
                "document_id": doc_id,
                "token_count": 4,
                "height": 1,
                "path": "",
                "left_child_id": "leaf-1",
                "right_child_id": "leaf-2",
            },
        ]

        # Use raw SQLAlchemy session for transaction management
        with sqlite_backend.db.SessionLocal() as session:
            try:
                # Add document
                sqlite_backend.doc_repo.add_document(
                    document_id=doc_id,
                    file_path="test.txt",
                    content_hash="test-hash",
                    chunk_count=2,
                    embedding_model="text-embedding-3-small",
                    summary_model="gpt-4o-mini",
                    session=session,
                )

                # Add nodes
                doc_store.nodes.add_batch(nodes_data, session=session)

                # Update parent references
                parent_updates = [
                    ("leaf-1", "parent-1"),
                    ("leaf-2", "parent-1"),
                ]
                doc_store.nodes.update_parent_references_batch(
                    parent_updates, session=session
                )
                session.commit()
            except Exception:
                session.rollback()
                raise

        # Verify all operations were committed
        leaf1 = doc_store.nodes.get_node("leaf-1")
        leaf2 = doc_store.nodes.get_node("leaf-2")
        parent = doc_store.nodes.get_node("parent-1")

        assert leaf1 is not None
        assert leaf1.parent_id == "parent-1"
        assert leaf2 is not None
        assert leaf2.parent_id == "parent-1"
        assert parent is not None
        assert parent.left_child_id == "leaf-1"
        assert parent.right_child_id == "leaf-2"


@pytest.mark.usefixtures("sqlite_backend")
class TestBackwardCompatibilitySQLite:
    """Test that existing code still works without transactions."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("test-doc-compat")

    def test_add_document_without_session(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test add_document works without session parameter."""
        doc_id = "test-doc-no-session"
        sqlite_backend.doc_repo.add_document(
            document_id=doc_id,
            file_path="test.txt",
            content_hash="test-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Verify it was persisted
        persisted_doc = sqlite_backend.doc_repo.get_document_by_id(doc_id)
        assert persisted_doc is not None
        assert persisted_doc.id == doc_id

    def test_add_nodes_batch_without_session(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test add_nodes_batch works without session parameter."""
        nodes_data: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "node-no-session",
                "text": "Test content",
                "span_start": 0,
                "span_end": 12,
                "document_id": "test-doc-compat",
                "token_count": 3,
                "height": 0,
                "path": "",
            }
        ]

        doc_store.nodes.add_batch(nodes_data)

        # Verify it was persisted
        persisted_node = doc_store.nodes.get_node("node-no-session")
        assert persisted_node is not None
        assert persisted_node.id == "node-no-session"

    def test_delete_document_nodes_without_session(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test clear_document works without session parameter."""
        # First add a document with nodes
        doc_id = "test-doc-delete"
        sqlite_backend.doc_repo.add_document(
            document_id=doc_id,
            file_path="test.txt",
            content_hash="test-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        nodes_data: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "node-to-delete",
                "text": "Test content",
                "span_start": 0,
                "span_end": 12,
                "document_id": doc_id,
                "token_count": 3,
                "height": 0,
                "path": "",
            }
        ]
        doc_store = sqlite_backend.for_document(doc_id)
        doc_store.nodes.add_batch(nodes_data)

        # Delete nodes without session
        deleted_count = sqlite_backend.doc_repo.clear_document(doc_id)

        assert deleted_count == 1

        # Verify node was deleted
        persisted_node = doc_store.nodes.get_node("node-to-delete")
        assert persisted_node is None


@pytest.mark.usefixtures("sqlite_backend")
class TestAtomicReindexingSQLite:
    """Test atomic re-indexing scenario from issue #150."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("test-doc-reindex")

    def test_atomic_reindexing_success(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test successful atomic re-indexing of a document."""
        doc_id = "test-doc-reindex"

        # First, index the document with initial content
        sqlite_backend.doc_repo.add_document(
            document_id=doc_id,
            file_path="test.txt",
            content_hash="old-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        old_nodes_data: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "old-node-1",
                "text": "Old content",
                "span_start": 0,
                "span_end": 11,
                "document_id": doc_id,
                "token_count": 2,
                "height": 0,
                "path": "",
            }
        ]
        doc_store.nodes.add_batch(old_nodes_data)

        # Verify old content exists
        assert doc_store.nodes.get_node("old-node-1") is not None

        # Now atomically re-index with new content
        new_nodes_data: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "new-node-1",
                "text": "New content",
                "span_start": 0,
                "span_end": 11,
                "document_id": doc_id,
                "token_count": 2,
                "height": 0,
                "path": "",
            }
        ]

        # Use raw SQLAlchemy session for transaction management
        with sqlite_backend.db.SessionLocal() as session:
            try:
                # Delete old nodes
                deleted_count = sqlite_backend.doc_repo.clear_document(
                    doc_id, session=session
                )
                assert deleted_count == 1

                # Add new nodes
                doc_store.nodes.add_batch(new_nodes_data, session=session)
                session.commit()
            except Exception:
                session.rollback()
                raise

        # Verify atomic operation: old gone, new present
        assert doc_store.nodes.get_node("old-node-1") is None
        assert doc_store.nodes.get_node("new-node-1") is not None

    def test_atomic_reindexing_rollback(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test atomic re-indexing rolls back on failure."""
        doc_id = "test-doc-reindex-fail"

        # First, index the document with initial content
        sqlite_backend.doc_repo.add_document(
            document_id=doc_id,
            file_path="test.txt",
            content_hash="old-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        old_nodes_data: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "old-node-fail",
                "text": "Old content",
                "span_start": 0,
                "span_end": 11,
                "document_id": doc_id,
                "token_count": 2,
                "height": 0,
                "path": "",
            }
        ]
        doc_store.nodes.add_batch(old_nodes_data)

        # Verify old content exists
        old_node = doc_store.nodes.get_node("old-node-fail")
        assert old_node is not None

        # Attempt atomic re-index that fails
        with pytest.raises(ValueError, match="Simulated reindex failure"):
            with sqlite_backend.db.SessionLocal() as session:
                try:
                    # Delete old nodes
                    sqlite_backend.doc_repo.clear_document(doc_id, session=session)

                    # Simulate failure before adding new nodes
                    raise ValueError("Simulated reindex failure")
                except Exception:
                    session.rollback()
                    raise

        # Verify old content is still there (rollback succeeded)
        persisted_old_node = doc_store.nodes.get_node("old-node-fail")
        assert persisted_old_node is not None
        assert persisted_old_node.text == "Old content"


@pytest.mark.usefixtures("sqlite_backend")
class TestTransactionSafetySQLite:
    """Test transaction safety improvements addressing code review feedback."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("rollback-test-doc")

    def test_repository_method_rollback_on_exception(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test that operations rollback properly on exceptions."""
        # Add initial document
        doc_id = "rollback-test-doc"
        sqlite_backend.doc_repo.add_document(
            document_id=doc_id,
            file_path="test.txt",
            content_hash="hash123",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # First, add a valid node
        valid_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "valid-node",
                "text": "Valid content",
                "span_start": 0,
                "span_end": 13,
                "document_id": doc_id,
                "token_count": 2,
                "height": 0,
                "path": "",
            }
        ]
        doc_store.nodes.add_batch(valid_nodes)

        # Test rollback by triggering an exception in a transaction
        with pytest.raises(ValueError, match="Simulated exception"):
            with sqlite_backend.db.SessionLocal() as session:
                try:
                    # Add a test node
                    invalid_nodes: list[
                        dict[
                            str,
                            str
                            | int
                            | float
                            | bool
                            | list[float]
                            | NDArray[np.float64]
                            | None,
                        ]
                    ] = [
                        {
                            "node_id": "test-node-exception",
                            "text": "Test content",
                            "span_start": 0,
                            "span_end": 12,
                            "document_id": doc_id,
                            "token_count": 3,
                            "height": 0,
                            "path": "",
                        }
                    ]
                    doc_store.nodes.add_batch(invalid_nodes, session=session)

                    # Simulate error before commit
                    raise ValueError("Simulated exception")
                except Exception:
                    session.rollback()
                    raise

        # Verify the node was not persisted due to rollback
        node = doc_store.nodes.get_node("test-node-exception")
        assert node is None

        # Verify the valid node is still there
        valid_node = doc_store.nodes.get_node("valid-node")
        assert valid_node is not None

    def test_nested_session_handling(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test that nested sessions work properly (SQLite doesn't prevent nesting)."""
        # SQLite backend doesn't have the same transaction prevention as StoreManager
        # but we can test that nested sessions work
        with sqlite_backend.db.SessionLocal() as session1:
            with sqlite_backend.db.SessionLocal() as session2:
                # Both sessions should work independently
                assert session1 is not session2

    def test_sqlite_rollback_simulation(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test that SQLite backend properly handles rollback behavior."""
        # Add initial data
        doc_id = "sqlite-rollback-test"
        sqlite_backend.doc_repo.add_document(
            document_id=doc_id,
            file_path="test.txt",
            content_hash="hash123",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Verify document exists
        doc = sqlite_backend.doc_repo.get_document_by_id(doc_id)
        assert doc is not None

        # Test rollback simulation
        with pytest.raises(ValueError, match="Simulated failure"):
            with sqlite_backend.db.SessionLocal() as session:
                try:
                    # Delete the document
                    sqlite_backend.doc_repo.clear_document(doc_id, session=session)

                    # Simulate failure
                    raise ValueError("Simulated failure")
                except Exception:
                    session.rollback()
                    raise

        # Verify document was restored after rollback
        doc = sqlite_backend.doc_repo.get_document_by_id(doc_id)
        assert doc is not None

    def test_session_context_manager(
        self, doc_store: DocumentStore, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Test the session context manager from database manager."""
        # Test successful operation
        for session in sqlite_backend.db.session():
            # The session should be managed properly
            assert session is not None
            break  # Only test the first (and only) session

        # Test exception handling - session should be cleaned up properly
        with pytest.raises(ValueError, match="Test exception"):
            for session in sqlite_backend.db.session():
                # Simulate an operation that fails
                raise ValueError("Test exception")

        # If we get here, the exception was properly handled and session cleaned up
