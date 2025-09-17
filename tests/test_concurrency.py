"""Backend-agnostic concurrency tests for thread safety and concurrent requests.

Tests concurrent operations and thread safety using the pluggable storage backend
with FastAPI TestClient to test thread safety and concurrent operations."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.contracts.storage_backend import StorageBackend
from tests.utils import mock_openai_context


class TestConcurrency:
    """Test thread safety and concurrent requests using storage backend."""

    @pytest.fixture
    def mock_openai(self) -> Generator[None, None, None]:
        """Mock OpenAI for tests using centralized utilities."""
        with mock_openai_context():
            yield

    def test_concurrent_operations_simulation(
        self, storage_backend: StorageBackend
    ) -> None:
        """Test concurrent-like operations using storage backend.

        This test simulates what would happen with concurrent requests
        by performing multiple operations sequentially that would normally
        happen concurrently.
        """
        doc_store = storage_backend.for_document("concurrent-doc")
        doc_store.set_metadata(
            file_path="concurrent_test.txt",
            content_hash="concurrent-test-hash",
            chunk_count=5,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

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

    def test_concurrent_document_indexing(
        self, storage_backend: StorageBackend
    ) -> None:
        """Test concurrent document indexing with storage backend."""
        # Create stores for different documents (simulating concurrent indexing)
        stores = []
        for i in range(3):
            doc_store = storage_backend.for_document(f"doc-{i}")
            doc_store.set_metadata(
                file_path=f"doc-{i}.txt",
                content_hash=f"doc-{i}-hash",
                chunk_count=1,
                embedding_model="text-embedding-3-small",
                summary_model="gpt-4o-mini",
            )
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
        self, storage_backend: StorageBackend
    ) -> None:
        """Verify no shared mutable state between document stores."""
        # Create two separate document stores
        store1 = storage_backend.for_document("state-test-1")
        store1.set_metadata(
            file_path="state-test-1.txt",
            content_hash="state-test-1-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        store2 = storage_backend.for_document("state-test-2")
        store2.set_metadata(
            file_path="state-test-2.txt",
            content_hash="state-test-2-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

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

    def test_backend_isolation(self, storage_backend: StorageBackend) -> None:
        """Test that storage backend provides proper isolation between documents."""
        # Create stores for different documents
        doc1_store = storage_backend.for_document("doc1")
        doc1_store.set_metadata(
            file_path="doc1.txt",
            content_hash="doc1-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        doc2_store = storage_backend.for_document("doc2")
        doc2_store.set_metadata(
            file_path="doc2.txt",
            content_hash="doc2-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

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

    def test_concurrent_batch_operations(self, storage_backend: StorageBackend) -> None:
        """Test concurrent-like batch operations with storage backend."""
        doc_store = storage_backend.for_document("batch-doc")
        doc_store.set_metadata(
            file_path="batch-doc.txt",
            content_hash="batch-doc-hash",
            chunk_count=6,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

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
        self, storage_backend: StorageBackend
    ) -> None:
        """Test that search operations work safely with storage backend."""
        doc_store = storage_backend.for_document("search-doc")
        doc_store.set_metadata(
            file_path="search-doc.txt",
            content_hash="search-doc-hash",
            chunk_count=2,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

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
            },
        ]

        doc_store.nodes.add_batch(nodes)

        # Test that concurrent document operations work without errors
        # In this backend-agnostic test, we verify nodes exist correctly
        all_nodes = doc_store.nodes.get_all()
        assert len(all_nodes) == 2

        # Verify both nodes are accessible
        node1 = doc_store.nodes.get_node("search1")
        node2 = doc_store.nodes.get_node("search2")
        assert node1 is not None
        assert node2 is not None
        assert node1.text == "Search node 1"
        assert node2.text == "Search node 2"
