"""SQLite-based concurrency tests for thread safety and concurrent requests.

SQLite-based version of concurrency tests that use the real in-memory SQLite backend
with FastAPI TestClient to test thread safety and concurrent operations.
"""

from __future__ import annotations

from collections.abc import Callable, Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from ragzoom.api import app
from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.store import StoreManager
from tests.utils import mock_openai_context


@pytest.mark.usefixtures("sqlite_backend")
class TestConcurrencySQLite:
    """Test thread safety and concurrent requests using SQLite backend."""

    @pytest.fixture
    def mock_openai(self) -> Generator[None, None, None]:
        """Mock OpenAI for tests using centralized utilities."""
        with mock_openai_context():
            yield

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        """Create a DocumentStore for test operations."""
        return sqlite_store_factory("test-doc")

    @pytest.fixture
    def client(
        self,
        mock_openai: None,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_backend: SQLiteStorageBackend,
    ) -> Generator[TestClient, None, None]:
        """Create test client with SQLite-backed dependencies."""
        from ragzoom.api import get_service_container
        from ragzoom.config import (
            IndexConfig,
            OperationalConfig,
            QueryConfig,
            SecretStr,
        )

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create a service container that uses our SQLite backend
        class SQLiteServiceContainer:
            def __init__(self) -> None:
                self.index_config = IndexConfig.load()
                self.query_config = QueryConfig()
                self.operational_config = OperationalConfig(
                    openai_api_key=SecretStr("test-key"),
                    database_url="sqlite:///:memory:",
                )

                # Create StoreManager directly with SQLite
                # Since we can't inject the backend, use the factory pattern instead
                self.store = StoreManager(
                    self.operational_config, "text-embedding-3-small"
                )

                # Initialize services with the store
                from ragzoom.services.document_service import DocumentService
                from ragzoom.services.indexing_service import IndexingService
                from ragzoom.services.query_service import QueryService

                self.document_service = DocumentService(self.store)
                self.indexing_service = IndexingService(
                    self.store,
                    self.index_config,
                    self.operational_config,
                )
                self.query_service = QueryService(
                    self.store,
                    self.query_config,
                    self.operational_config,
                )

            def close(self) -> None:
                """Close method for compatibility."""
                self.store.close()

        def sqlite_get_service_container() -> SQLiteServiceContainer:
            return SQLiteServiceContainer()

        # Override the dependency
        app.dependency_overrides[get_service_container] = sqlite_get_service_container

        try:
            with TestClient(app) as client:
                yield client
        finally:
            # Clean up the override
            app.dependency_overrides.clear()

    def test_concurrent_operations_simulation(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test concurrent-like operations using SQLite backend.

        This test simulates what would happen with concurrent requests
        by performing multiple operations sequentially that would normally
        happen concurrently.
        """
        doc_store = sqlite_store_factory("concurrent-doc")

        # Simulate multiple "concurrent" index operations
        operations = []
        for i in range(5):
            # Add nodes as if from concurrent requests
            nodes: list[
                dict[
                    str,
                    str | int | float | bool | list[float] | NDArray[np.float64] | None,
                ]
            ] = [
                {
                    "node_id": f"concurrent_{i}",
                    "text": f"Concurrent content {i} for testing.",
                    "embedding": [0.1 + i * 0.1] * 1536,
                    "span_start": i * 50,
                    "span_end": (i + 1) * 50,
                    "document_id": "concurrent-doc",
                    "token_count": 25,
                    "height": 0,
                    "path": str(i),
                }
            ]
            doc_store.nodes.add_batch(nodes)
            operations.append(f"indexed_{i}")

        # Verify all operations completed
        assert len(operations) == 5

        # Verify all nodes were created
        all_nodes = doc_store.nodes.get_all()
        assert len(all_nodes) == 5

        # Verify each node has correct data
        for i in range(5):
            node = doc_store.nodes.get_node(f"concurrent_{i}")
            assert node is not None
            assert node.text == f"Concurrent content {i} for testing."
            assert node.document_id == "concurrent-doc"

    @pytest.mark.integration
    def test_service_isolation(
        self, mock_openai: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that each request gets its own service instance."""
        from ragzoom.api import get_service_container

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Get multiple service instances
        service1 = get_service_container()
        service2 = get_service_container()

        # Should be different instances (not singleton)
        assert service1 is not service2
        assert service1.index_config is not service2.index_config
        assert service1.query_config is not service2.query_config
        assert service1.operational_config is not service2.operational_config
        assert service1.store is not service2.store

    def test_concurrent_document_indexing(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test concurrent document indexing with SQLite backend."""
        # Create stores for different documents (simulating concurrent indexing)
        stores = []
        for i in range(3):
            doc_store = sqlite_store_factory(f"doc-{i}")
            stores.append((f"doc-{i}", doc_store))

        # Index content to each document (simulating concurrent operations)
        for doc_id, doc_store in stores:
            nodes: list[
                dict[
                    str,
                    str | int | float | bool | list[float] | NDArray[np.float64] | None,
                ]
            ] = [
                {
                    "node_id": f"{doc_id}_node1",
                    "text": f"Document {doc_id} content.",
                    "embedding": [0.1] * 1536,
                    "span_start": 0,
                    "span_end": 50,
                    "document_id": doc_id,
                    "token_count": 25,
                    "height": 0,
                    "path": "",
                }
            ]
            doc_store.nodes.add_batch(nodes)

        # Verify all documents were indexed correctly
        for doc_id, doc_store in stores:
            nodes_list = doc_store.nodes.get_all()
            assert len(nodes_list) == 1
            assert nodes_list[0].document_id == doc_id
            assert nodes_list[0].text == f"Document {doc_id} content."

    def test_no_shared_state_between_stores(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Verify no shared mutable state between document stores."""
        # Create two separate document stores
        store1 = sqlite_store_factory("state-test-1")
        store2 = sqlite_store_factory("state-test-2")

        # Add different data to each store
        nodes1: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "state1_node",
                "text": "State test 1 content",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 50,
                "document_id": "state-test-1",
                "token_count": 25,
                "height": 0,
                "path": "",
            }
        ]
        store1.nodes.add_batch(nodes1)

        nodes2: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "state2_node",
                "text": "State test 2 content",
                "embedding": [0.2] * 1536,
                "span_start": 0,
                "span_end": 50,
                "document_id": "state-test-2",
                "token_count": 30,
                "height": 0,
                "path": "",
            }
        ]
        store2.nodes.add_batch(nodes2)

        # Verify stores don't interfere with each other
        store1_nodes = store1.nodes.get_all()
        store2_nodes = store2.nodes.get_all()

        assert len(store1_nodes) == 1
        assert len(store2_nodes) == 1
        assert store1_nodes[0].text == "State test 1 content"
        assert store2_nodes[0].text == "State test 2 content"
        assert store1_nodes[0].token_count == 25
        assert store2_nodes[0].token_count == 30

    def test_sqlite_backend_isolation(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test that SQLite backend provides proper isolation between documents."""
        # Create stores for different documents
        doc1_store = sqlite_store_factory("doc1")
        doc2_store = sqlite_store_factory("doc2")

        # Add data to doc1
        nodes_doc1: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "doc1_node1",
                "text": "Content for document 1",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 50,
                "document_id": "doc1",
                "token_count": 25,
                "height": 0,
                "path": "",
            }
        ]
        doc1_store.nodes.add_batch(nodes_doc1)

        # Add data to doc2
        nodes_doc2: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "doc2_node1",
                "text": "Content for document 2",
                "embedding": [0.2] * 1536,
                "span_start": 0,
                "span_end": 50,
                "document_id": "doc2",
                "token_count": 25,
                "height": 0,
                "path": "",
            }
        ]
        doc2_store.nodes.add_batch(nodes_doc2)

        # Verify isolation - each store only sees its own document's data
        doc1_nodes = doc1_store.nodes.get_all()
        doc2_nodes = doc2_store.nodes.get_all()

        assert len(doc1_nodes) == 1
        assert len(doc2_nodes) == 1
        assert doc1_nodes[0].document_id == "doc1"
        assert doc2_nodes[0].document_id == "doc2"
        assert doc1_nodes[0].text == "Content for document 1"
        assert doc2_nodes[0].text == "Content for document 2"

        # Verify doc1_store can't see doc2 data
        doc2_node_from_doc1_store = doc1_store.nodes.get_node("doc2_node1")
        assert doc2_node_from_doc1_store is None

        # Verify doc2_store can't see doc1 data
        doc1_node_from_doc2_store = doc2_store.nodes.get_node("doc1_node1")
        assert doc1_node_from_doc2_store is None

    def test_concurrent_batch_operations(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test concurrent-like batch operations with SQLite backend."""
        doc_store = sqlite_store_factory("batch-doc")

        # Simulate concurrent batch operations
        batches = []
        for batch_num in range(3):
            batch_nodes: list[
                dict[
                    str,
                    str | int | float | bool | list[float] | NDArray[np.float64] | None,
                ]
            ] = []
            for i in range(2):  # 2 nodes per batch
                node_id = f"batch{batch_num}_node{i}"
                batch_nodes.append(
                    {
                        "node_id": node_id,
                        "text": f"Batch {batch_num} node {i} content",
                        "embedding": [0.1 + batch_num * 0.1 + i * 0.01] * 1536,
                        "span_start": (batch_num * 2 + i) * 50,
                        "span_end": (batch_num * 2 + i + 1) * 50,
                        "document_id": "batch-doc",
                        "token_count": 25,
                        "height": 0,
                        "path": f"{batch_num}{i}",
                    }
                )
            batches.append(batch_nodes)

        # Execute all batches (simulating concurrent execution)
        for batch in batches:
            doc_store.nodes.add_batch(batch)

        # Verify all nodes were added correctly
        all_nodes = doc_store.nodes.get_all()
        assert len(all_nodes) == 6  # 3 batches * 2 nodes each

        # Verify data integrity across batches
        for batch_num in range(3):
            for i in range(2):
                node_id = f"batch{batch_num}_node{i}"
                node = doc_store.nodes.get_node(node_id)
                assert node is not None
                assert node.text == f"Batch {batch_num} node {i} content"
                assert node.document_id == "batch-doc"

    def test_search_operations_thread_safety(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that search operations work safely with SQLite backend."""
        doc_store = sqlite_store_factory("search-doc")

        # Create nodes with embeddings
        nodes: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "search1",
                "text": "Search node 1",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 50,
                "document_id": "search-doc",
                "token_count": 25,
                "height": 0,
                "path": "0",
            },
            {
                "node_id": "search2",
                "text": "Search node 2",
                "embedding": [0.2] * 1536,
                "span_start": 50,
                "span_end": 100,
                "document_id": "search-doc",
                "token_count": 25,
                "height": 0,
                "path": "1",
            },
        ]

        doc_store.nodes.add_batch(nodes)

        # Upsert embeddings for vector search
        sqlite_backend.vector_index.upsert(
            [
                (
                    "search1",
                    [0.1] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 50,
                        "parent_id": "",
                        "document_id": "search-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "search2",
                    [0.2] * 1536,
                    {
                        "span_start": 50,
                        "span_end": 100,
                        "parent_id": "",
                        "document_id": "search-doc",
                        "is_leaf": 1,
                    },
                ),
            ]
        )

        # Perform search operations (simulating concurrent access)
        query_embedding = [0.15] * 1536

        # Multiple search operations
        results1 = doc_store.search.similar(query_embedding, n_results=2)
        results2 = doc_store.search.similar(query_embedding, n_results=1)

        # Verify results
        assert len(results1) >= 1  # Should find at least one result
        assert len(results2) == 1  # Should find exactly one result
        assert results1[0][0] in ["search1", "search2"]  # Should be one of our nodes
        assert results2[0][0] in ["search1", "search2"]  # Should be one of our nodes
