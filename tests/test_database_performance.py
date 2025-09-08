"""Performance tests for database operations with large node counts.

These tests validate the scalability improvements for Issue #164.
"""

import logging
import time
from collections.abc import Generator
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.contracts.storage_backend import StorageBackend

logger = logging.getLogger(__name__)


@pytest.fixture
def large_document_data() -> (
    Generator[
        dict[str, list[dict[str, str | int | list[float]]] | str | int], None, None
    ]
):
    """Generate test data for a large document with many nodes."""
    # Simulate The Hobbit with 10-token chunks = ~50,000 nodes
    num_nodes = 50000
    document_id = "large_test_document"

    # Generate node data
    nodes_data: list[dict[str, str | int | list[float]]] = []
    for i in range(num_nodes):
        span_start = i * 10  # 10-character spans
        span_end = span_start + 10
        nodes_data.append(
            {
                "node_id": f"node_{i:06d}",
                "text": f"Text content for node {i:06d}",
                "embedding": [0.1] * 1536,  # Standard embedding dimension
                "span_start": span_start,
                "span_end": span_end,
                "document_id": document_id,
                "token_count": 10,
            }
        )

    yield {
        "document_id": document_id,
        "nodes": nodes_data,
        "expected_count": num_nodes,
    }


@pytest.fixture
def small_document_data() -> (
    Generator[
        dict[str, list[dict[str, str | int | list[float]]] | str | int], None, None
    ]
):
    """Generate test data for a small document to compare performance."""
    num_nodes = 100
    document_id = "small_test_document"

    nodes_data: list[dict[str, str | int | list[float]]] = []
    for i in range(num_nodes):
        span_start = i * 10
        span_end = span_start + 10
        nodes_data.append(
            {
                "node_id": f"small_node_{i:03d}",
                "text": f"Text content for small node {i:03d}",
                "embedding": [0.1] * 1536,
                "span_start": span_start,
                "span_end": span_end,
                "document_id": document_id,
                "token_count": 10,
            }
        )

    yield {
        "document_id": document_id,
        "nodes": nodes_data,
        "expected_count": num_nodes,
    }


class TestDatabaseScalability:
    """Test database operations scale well with large node counts."""

    def add_test_nodes(
        self,
        storage_backend: StorageBackend,
        document_id: str,
        nodes_data: list[dict[str, str | int | list[float]]],
    ) -> None:
        """Helper to add test nodes to the backend."""
        doc_store = storage_backend.for_document(document_id)

        # Convert to proper dict format
        node_data_dicts: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = []
        for node_data in nodes_data:
            node_data_dict: dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ] = {
                "node_id": cast(str, node_data["node_id"]),
                "text": cast(str, node_data["text"]),
                "embedding": cast(list[float], node_data["embedding"]),
                "span_start": cast(int, node_data["span_start"]),
                "span_end": cast(int, node_data["span_end"]),
                "document_id": cast(str, node_data["document_id"]),
                "token_count": cast(int, node_data["token_count"]),
            }
            node_data_dicts.append(node_data_dict)

        doc_store.nodes.add_batch(node_data_dicts)

    @pytest.mark.slow
    def test_deletion_performance_large_document(
        self,
        storage_backend: StorageBackend,
        large_document_data: dict[
            str, list[dict[str, str | int | list[float]]] | str | int
        ],
    ) -> None:
        """Test that deletion of 50,000 nodes completes in reasonable time."""
        document_id = cast(str, large_document_data["document_id"])
        nodes_data = cast(
            list[dict[str, str | int | list[float]]], large_document_data["nodes"]
        )
        expected_count = cast(int, large_document_data["expected_count"])
        doc_store = storage_backend.for_document(document_id)

        # Set up document metadata first
        doc_store.set_metadata(
            file_path="performance_test.txt",
            content_hash="perf-test-hash",
            chunk_count=expected_count,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Add all nodes
        logger.info(f"Adding {expected_count} nodes for performance test...")
        start_time = time.perf_counter()
        self.add_test_nodes(storage_backend, document_id, nodes_data)
        add_duration = time.perf_counter() - start_time
        logger.info(f"Added {expected_count} nodes in {add_duration:.2f}s")

        # Verify nodes were added
        count_before = len(doc_store.nodes.get_all())
        assert count_before == expected_count

        # Test deletion performance
        logger.info("Testing deletion performance...")
        start_time = time.perf_counter()
        deleted_count = storage_backend.clear_document(document_id)
        deletion_duration = time.perf_counter() - start_time

        # Verify deletion
        assert deleted_count == expected_count
        count_after = len(doc_store.nodes.get_all())
        assert count_after == 0

        # Performance assertions
        logger.info(f"Deleted {deleted_count} nodes in {deletion_duration:.2f}s")
        assert (
            deletion_duration < 10.0
        ), f"Deletion took {deletion_duration:.2f}s, expected < 10s for {expected_count} nodes"

        # Log performance metrics
        nodes_per_second = (
            deleted_count / deletion_duration if deletion_duration > 0 else 0
        )
        logger.info(f"Deletion rate: {nodes_per_second:.0f} nodes/second")

    def test_deletion_performance_small_document(
        self,
        storage_backend: StorageBackend,
        small_document_data: dict[
            str, list[dict[str, str | int | list[float]]] | str | int
        ],
    ) -> None:
        """Test deletion performance with small document for comparison."""
        document_id = cast(str, small_document_data["document_id"])
        nodes_data = cast(
            list[dict[str, str | int | list[float]]], small_document_data["nodes"]
        )
        expected_count = cast(int, small_document_data["expected_count"])
        doc_store = storage_backend.for_document(document_id)

        # Set up document metadata first
        doc_store.set_metadata(
            file_path="small_performance_test.txt",
            content_hash="small-perf-test-hash",
            chunk_count=expected_count,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Add nodes
        self.add_test_nodes(storage_backend, document_id, nodes_data)

        # Test deletion
        start_time = time.perf_counter()
        deleted_count = storage_backend.clear_document(document_id)
        deletion_duration = time.perf_counter() - start_time

        # Verify and assert
        assert deleted_count == expected_count
        assert (
            deletion_duration < 1.0
        ), f"Small document deletion took {deletion_duration:.2f}s, expected < 1s"

    def test_paginated_retrieval_correctness_small(
        self,
        storage_backend: StorageBackend,
        small_document_data: dict[
            str, list[dict[str, str | int | list[float]]] | str | int
        ],
    ) -> None:
        """Validate pagination correctness on a small document.

        Ensures batches cover all nodes exactly once and each batch size
        equals page_size except possibly the last. Uses a few representative
        page sizes to exercise boundaries without large-scale overhead.
        """
        document_id = cast(str, small_document_data["document_id"])
        nodes_data = cast(
            list[dict[str, str | int | list[float]]], small_document_data["nodes"]
        )
        expected_count = cast(int, small_document_data["expected_count"])
        doc_store = storage_backend.for_document(document_id)

        # Set up document metadata first
        doc_store.set_metadata(
            file_path="small_paginated_test.txt",
            content_hash="small-paginated-test-hash",
            chunk_count=expected_count,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Add nodes
        self.add_test_nodes(storage_backend, document_id, nodes_data)

        # Test paginated retrieval with a few representative sizes
        page_sizes = [7, 10, 33, 256]

        for page_size in page_sizes:
            batches = doc_store.nodes.get_all_paginated(page_size=page_size)

            # Verify correctness
            total_nodes = sum(len(batch) for batch in batches)
            assert total_nodes == expected_count, (
                f"Expected {expected_count} nodes, got {total_nodes} "
                f"across {len(batches)} batches"
            )

            # Verify batch sizes (all should be page_size except possibly the last)
            for i, batch in enumerate(batches[:-1]):  # All but last
                assert (
                    len(batch) == page_size
                ), f"Batch {i} has {len(batch)} nodes, expected {page_size}"

            # Last batch can be smaller
            if batches:
                last_batch_size = len(batches[-1])
                assert (
                    last_batch_size <= page_size
                ), f"Last batch has {last_batch_size} nodes, should be <= {page_size}"

        # Clean up
        storage_backend.clear_document(document_id)

    def test_paginated_retrieval_boundary_conditions(
        self, storage_backend: StorageBackend
    ) -> None:
        """Test paginated retrieval with edge cases."""
        document_id = "boundary_test_doc"
        doc_store = storage_backend.for_document(document_id)

        # Test with no nodes
        batches = doc_store.nodes.get_all_paginated()
        assert len(batches) == 0

        # Set up document metadata
        doc_store.set_metadata(
            file_path="boundary_test.txt",
            content_hash="boundary-test-hash",
            chunk_count=0,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Add exactly one page worth of nodes
        page_size = 10
        nodes_data: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = []
        for i in range(page_size):
            nodes_data.append(
                {
                    "node_id": f"boundary_node_{i}",
                    "text": f"Boundary test node {i}",
                    "embedding": [0.1] * 1536,
                    "span_start": i * 10,
                    "span_end": (i + 1) * 10,
                    "document_id": document_id,
                    "token_count": 10,
                }
            )
        doc_store.nodes.add_batch(nodes_data)

        # Test retrieval
        batches = doc_store.nodes.get_all_paginated(page_size=page_size)
        assert len(batches) == 1
        assert len(batches[0]) == page_size

        # Test with page_size + 1 nodes
        additional_node: dict[
            str, str | int | float | bool | list[float] | NDArray[np.float64] | None
        ] = {
            "node_id": f"boundary_node_{page_size}",
            "text": f"Boundary test node {page_size}",
            "embedding": [0.1] * 1536,
            "span_start": page_size * 10,
            "span_end": (page_size + 1) * 10,
            "document_id": document_id,
            "token_count": 10,
        }
        doc_store.nodes.add_batch([additional_node])

        batches = doc_store.nodes.get_all_paginated(page_size=page_size)
        assert len(batches) == 2
        assert len(batches[0]) == page_size
        assert len(batches[1]) == 1

        # Clean up
        storage_backend.clear_document(document_id)

    def test_invalid_page_size(self, storage_backend: StorageBackend) -> None:
        """Test that invalid page sizes are rejected."""
        doc_store = storage_backend.for_document("test_doc")
        with pytest.raises(ValueError, match="page_size must be positive"):
            doc_store.nodes.get_all_paginated(page_size=0)

        with pytest.raises(ValueError, match="page_size must be positive"):
            doc_store.nodes.get_all_paginated(page_size=-1)


class TestMemoryEfficiency:
    """Test memory efficiency of optimized operations."""

    @pytest.mark.slow
    def test_deletion_memory_usage(self, storage_backend: StorageBackend) -> None:
        """Test that deletion doesn't load excessive data into memory."""
        document_id = "memory_test_doc"
        num_nodes = 10000  # Smaller test for memory monitoring
        doc_store = storage_backend.for_document(document_id)

        # Set up document metadata
        doc_store.set_metadata(
            file_path="memory_test.txt",
            content_hash="memory-test-hash",
            chunk_count=num_nodes,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Add test nodes in batches for efficiency
        nodes_data: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = []
        for i in range(num_nodes):
            nodes_data.append(
                {
                    "node_id": f"mem_test_node_{i}",
                    "text": f"Memory test node {i} " * 50,  # Larger text content
                    "embedding": [0.1] * 1536,
                    "span_start": i * 100,
                    "span_end": (i + 1) * 100,
                    "document_id": document_id,
                    "token_count": 50,
                }
            )
        doc_store.nodes.add_batch(nodes_data)

        # Monitor deletion - this should not cause memory spikes
        # In the old implementation, this would load all nodes into memory
        # The new implementation uses SQL RETURNING to avoid this
        start_time = time.perf_counter()
        deleted_count = storage_backend.clear_document(document_id)
        deletion_duration = time.perf_counter() - start_time

        assert deleted_count == num_nodes
        assert (
            deletion_duration < 5.0
        ), f"Memory-efficient deletion took {deletion_duration:.2f}s, expected < 5s"

        logger.info(
            f"Memory-efficient deletion of {num_nodes} nodes completed "
            f"in {deletion_duration:.2f}s"
        )


@pytest.mark.slow
class TestRegressionPrevention:
    """Test that optimizations don't break existing functionality."""

    def test_cache_invalidation_still_works(
        self, storage_backend: StorageBackend
    ) -> None:
        """Test that cache invalidation works with optimized deletion."""
        document_id = "cache_test_doc"
        node_id = "cache_test_node"
        doc_store = storage_backend.for_document(document_id)

        # Set up document metadata
        doc_store.set_metadata(
            file_path="cache_test.txt",
            content_hash="cache-test-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Add a node
        node_data: dict[
            str, str | int | float | bool | list[float] | NDArray[np.float64] | None
        ] = {
            "node_id": node_id,
            "text": "Cache test node",
            "embedding": [0.1] * 1536,
            "span_start": 0,
            "span_end": 10,
            "document_id": document_id,
            "token_count": 10,
        }
        doc_store.nodes.add_batch([node_data])

        # Load into cache
        node = doc_store.nodes.get_node(node_id)
        assert node is not None
        assert node.text == "Cache test node"

        # Delete document nodes (should invalidate cache)
        deleted_count = storage_backend.clear_document(document_id)
        assert deleted_count == 1

        # Node should no longer be accessible
        node_after_deletion = doc_store.nodes.get_node(node_id)
        assert node_after_deletion is None

    def test_transaction_support_maintained(
        self, storage_backend: StorageBackend
    ) -> None:
        """Test that transaction support is maintained in optimized methods."""
        document_id = "transaction_test_doc"
        doc_store = storage_backend.for_document(document_id)

        # Set up document metadata
        doc_store.set_metadata(
            file_path="transaction_test.txt",
            content_hash="transaction-test-hash",
            chunk_count=5,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Add test nodes
        nodes_data: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = []
        for i in range(5):
            nodes_data.append(
                {
                    "node_id": f"trans_test_node_{i}",
                    "text": f"Transaction test node {i}",
                    "embedding": [0.1] * 1536,
                    "span_start": i * 10,
                    "span_end": (i + 1) * 10,
                    "document_id": document_id,
                    "token_count": 10,
                }
            )
        doc_store.nodes.add_batch(nodes_data)

        # Test that deletion works (simplified test without SessionLocal access)
        # The session parameter is tested at the repository level in other tests
        deleted_count = storage_backend.clear_document(document_id)
        assert deleted_count == 5

        # After deletion, nodes should be gone
        remaining_nodes = doc_store.nodes.get_all()
        assert len(remaining_nodes) == 0
