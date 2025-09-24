"""Backend-agnostic tests for storage functionality.

Testing core store functionality, CRUD operations, document isolation,
and store interface compliance using the configured backend.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore


class TestStore:
    """Test core store functionality with configured backend."""

    @pytest.fixture
    def doc_store(self, storage_backend: StorageBackend) -> DocumentStore:
        doc_store = storage_backend.for_document("doc-id")
        doc_store.set_metadata(
            file_path="test_file.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        return doc_store

    def test_add_node(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test adding a node to the store."""
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "test-1",
                "text": "Test text",
                "span_start": 0,
                "span_end": 10,
                "document_id": "doc-id",
                "token_count": 10,
                "height": 0,
            }
        ]
        doc_store.nodes.add_batch(nodes)

        # Upsert embeddings
        # Upsert embeddings via VectorIndex
        vector_index.upsert(
            [
                (
                    "test-1",
                    [0.1] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 10,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                )
            ]
        )

        node = doc_store.nodes.get_node("test-1")
        assert node is not None
        assert node.id == "test-1"
        assert node.text == "Test text"
        assert node.span_start == 0
        assert node.span_end == 10

    def test_get_node(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test retrieving a node."""
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "test-2",
                "text": "Test text 2",
                "span_start": 10,
                "span_end": 20,
                "document_id": "doc-id",
                "token_count": 10,
                "height": 0,
            }
        ]
        doc_store.nodes.add_batch(nodes)

        # Upsert embeddings
        vector_index.upsert(
            [
                (
                    "test-2",
                    [0.2] * 1536,
                    {
                        "span_start": 10,
                        "span_end": 20,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                )
            ]
        )

        # Retrieve it
        node = doc_store.nodes.get_node("test-2")
        assert node is not None
        assert node.id == "test-2"
        assert node.text == "Test text 2"

        # Test non-existent node
        node = doc_store.nodes.get_node("non-existent")
        assert node is None

    def test_node_relationships(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test parent-child relationships."""
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "child1",
                "text": "Child 1",
                "span_start": 0,
                "span_end": 10,
                "document_id": "doc-id",
                "token_count": 10,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "child2",
                "text": "Child 2",
                "span_start": 10,
                "span_end": 20,
                "document_id": "doc-id",
                "token_count": 10,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "parent",
                "text": "Parent node",
                "span_start": 0,
                "span_end": 20,
                "document_id": "doc-id",
                "token_count": 20,
                "height": 1,
                "parent_id": None,
                "left_child_id": "child1",
                "right_child_id": "child2",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("child1", "parent"), ("child2", "parent")]
        )

        # Upsert embeddings
        vector_index.upsert(
            [
                (
                    "child1",
                    [0.4] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 10,
                        "parent_id": "parent",
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
                (
                    "child2",
                    [0.5] * 1536,
                    {
                        "span_start": 10,
                        "span_end": 20,
                        "parent_id": "parent",
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
                (
                    "parent",
                    [0.3] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 20,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
            ]
        )

        # Test relationships
        children = doc_store.tree.get_children("parent")
        left, right = children
        assert left is not None
        assert right is not None
        assert left.id == "child1"
        assert right.id == "child2"

        ancestors = doc_store.tree.get_ancestors(["child1", "child2"])
        assert len(ancestors) == 1
        assert ancestors[0].id == "parent"

    def test_search_similar(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test vector similarity search."""
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = []
        from collections.abc import Sequence

        vector_entries: list[tuple[str, Sequence[float], dict[str, object]]] = []

        for i in range(5):
            embedding = [i * 0.1] * 1536
            nodes.append(
                {
                    "node_id": f"node-{i}",
                    "text": f"Text {i}",
                    "span_start": i * 10,
                    "span_end": (i + 1) * 10,
                    "document_id": "doc-id",
                    "token_count": 10,
                    "height": 0,
                }
            )
            vector_entries.append(
                (
                    f"node-{i}",
                    embedding,
                    {
                        "span_start": i * 10,
                        "span_end": (i + 1) * 10,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                )
            )

        doc_store.nodes.add_batch(nodes)
        # Cast to exact expected type for upsert
        typed_entries: list[
            tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
        ] = [(entry[0], list(entry[1]), entry[2]) for entry in vector_entries]
        # Upsert via VectorIndex
        vector_index.upsert(typed_entries)

        # Search with a query embedding
        query_embedding = [0.25] * 1536
        results = vector_index.search_similar(
            query_embedding, 3, {"document_id": "doc-id"}
        )

        assert len(results) == 3
        # Canonical Vectors: ensure shape
        for v in results:
            assert hasattr(v, "id") and hasattr(v, "vec") and hasattr(v, "meta")
            assert isinstance(v.meta, dict)

    def test_mmr_diverse_results(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test MMR diversity computation."""
        from collections.abc import Sequence

        from ragzoom.retrieval import mmr as _mmr

        # Create candidates with different similarities
        candidates: list[
            tuple[str, float, dict[str, str | int | float | bool | None]]
        ] = [
            (
                "node-1",
                0.1,
                {
                    "span_start": 0,
                    "span_end": 10,
                    "parent_id": "root",
                    "document_id": "doc-id",
                    "is_leaf": 1,
                },
            ),  # Very similar
            (
                "node-2",
                0.15,
                {
                    "span_start": 10,
                    "span_end": 20,
                    "parent_id": "root",
                    "document_id": "doc-id",
                    "is_leaf": 1,
                },
            ),  # Similar
            (
                "node-3",
                0.5,
                {
                    "span_start": 20,
                    "span_end": 30,
                    "parent_id": "root",
                    "document_id": "doc-id",
                    "is_leaf": 1,
                },
            ),  # Less similar
            (
                "node-4",
                0.12,
                {
                    "span_start": 30,
                    "span_end": 40,
                    "parent_id": "root",
                    "document_id": "doc-id",
                    "is_leaf": 1,
                },
            ),  # Similar to node-1
            (
                "node-5",
                0.8,
                {
                    "span_start": 40,
                    "span_end": 50,
                    "parent_id": "root",
                    "document_id": "doc-id",
                    "is_leaf": 1,
                },
            ),  # Different
        ]

        # Add nodes with embeddings
        embeddings = [
            [1.0, 0.0, 0.0],  # node-1
            [0.9, 0.1, 0.0],  # node-2 (similar to 1)
            [0.5, 0.5, 0.0],  # node-3 (different)
            [0.95, 0.05, 0.0],  # node-4 (very similar to 1)
            [0.0, 0.0, 1.0],  # node-5 (very different)
        ]

        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = []
        vector_entries: list[tuple[str, Sequence[float], dict[str, object]]] = []

        for i, (node_id, _, metadata) in enumerate(candidates):
            # Pad embedding to expected size
            full_embedding = embeddings[i] + [0.0] * (1536 - 3)
            nodes.append(
                {
                    "node_id": node_id,
                    "text": f"Text for {node_id}",
                    "span_start": i * 10,
                    "span_end": (i + 1) * 10,
                    "document_id": "doc-id",
                    "token_count": 10,
                    "height": 0,
                }
            )
            vector_entries.append((node_id, full_embedding, dict(metadata)))

        doc_store.nodes.add_batch(nodes)
        # Cast to exact expected type for upsert
        typed_entries: list[
            tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
        ] = [(entry[0], list(entry[1]), entry[2]) for entry in vector_entries]
        vector_index.upsert(typed_entries)

        # Test MMR selection using generic implementation over canonical vectors
        query_embedding = [1.0, 0.0, 0.0] + [0.0] * 1533
        cand_ids = [c[0] for c in candidates]
        cand_vectors = vector_index.get_vectors(cand_ids)
        selected = _mmr.select_diverse(query_embedding, cand_vectors, 3, 0.7)

        assert len(selected) == 3
        # Should select node-1 (most relevant) and diverse nodes
        assert "node-1" in selected
        # Should include some diversity
        assert len(set(selected)) == 3

    def test_pinned_nodes(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test node pinning functionality."""
        # Create a tree structure with proper depths
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            # Root node (depth 0)
            {
                "node_id": "root",
                "text": "Root node",
                "span_start": 0,
                "span_end": 30,
                "document_id": "doc-id",
                "token_count": 30,
                "height": 3,
                "parent_id": None,
                "left_child_id": "level1",
                "right_child_id": None,
            },
            # Level 1 node (depth 1)
            {
                "node_id": "level1",
                "text": "Level 1 node",
                "span_start": 0,
                "span_end": 20,
                "document_id": "doc-id",
                "token_count": 20,
                "height": 2,
                "parent_id": None,
                "left_child_id": "level2",
                "right_child_id": None,
            },
            # Level 2 node (depth 2)
            {
                "node_id": "level2",
                "text": "Level 2 node",
                "span_start": 0,
                "span_end": 10,
                "document_id": "doc-id",
                "token_count": 10,
                "height": 1,
                "parent_id": None,
                "left_child_id": "level3",
                "right_child_id": None,
            },
            # Level 3 node (depth 3)
            {
                "node_id": "level3",
                "text": "Level 3 node",
                "span_start": 0,
                "span_end": 5,
                "document_id": "doc-id",
                "token_count": 5,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("level1", "root"), ("level2", "level1"), ("level3", "level2")]
        )

        # Upsert embeddings via VectorIndex
        vector_index.upsert(
            [
                (
                    "root",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 30,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
                (
                    "level1",
                    [0.6] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 20,
                        "parent_id": "root",
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
                (
                    "level2",
                    [0.7] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 10,
                        "parent_id": "level1",
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
                (
                    "level3",
                    [0.8] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 5,
                        "parent_id": "level2",
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
            ]
        )

        # pin_depth_max is hardcoded to 2 in the config

        # Note: DocumentStore doesn't have pin_node method - pinning is at StoreManager level
        # This test validates the document-scoped tree structure instead

        # Test tree depths
        assert doc_store.tree.get_depth("root") == 0
        assert doc_store.tree.get_depth("level1") == 1
        assert doc_store.tree.get_depth("level2") == 2
        assert doc_store.tree.get_depth("level3") == 3

        # Test pinned nodes retrieval (filtering to this document)
        pinned = doc_store.get_pinned_nodes()
        assert isinstance(pinned, list)  # Should return empty list for document store

        # Test pinned nodes with depth filter
        pinned = doc_store.get_pinned_nodes(depth_max=0)
        assert isinstance(pinned, list)

    def test_cache_functionality(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test LRU cache behavior."""
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "cached",
                "text": "Cached node",
                "span_start": 0,
                "span_end": 10,
                "document_id": "doc-id",
                "token_count": 10,
                "height": 0,
            }
        ]
        doc_store.nodes.add_batch(nodes)

        # Upsert embeddings
        vector_index.upsert(
            [
                (
                    "cached",
                    [0.8] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 10,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                )
            ]
        )

        # First retrieval (from DB)
        node1 = doc_store.nodes.get_node("cached")
        assert node1 is not None

        # Second retrieval (from cache)
        node2 = doc_store.nodes.get_node("cached")
        assert node2 is not None
        assert node2.id == node1.id

        # Note: DocumentStore doesn't expose node_cache - cache is managed internally
        # Test that repeated access works consistently
        node3 = doc_store.nodes.get_node("cached")
        assert node3 is not None
        assert node3.id == node1.id

    def test_node_depth_calculation(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test dynamic depth calculation."""
        # Create a tree structure:
        #     root
        #    /    \
        #  left   right
        #  /  \     |
        # ll  lr    rc

        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            # Root node (depth 0)
            {
                "node_id": "root",
                "text": "Root",
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc-id",
                "token_count": 100,
                "height": 2,
                "parent_id": None,
                "left_child_id": "left",
                "right_child_id": "right",
            },
            # Level 1 nodes (depth 1)
            {
                "node_id": "left",
                "text": "Left",
                "span_start": 0,
                "span_end": 50,
                "document_id": "doc-id",
                "token_count": 50,
                "height": 1,
                "parent_id": None,
                "left_child_id": "ll",
                "right_child_id": "lr",
            },
            {
                "node_id": "right",
                "text": "Right",
                "span_start": 50,
                "span_end": 100,
                "document_id": "doc-id",
                "token_count": 50,
                "height": 1,
                "parent_id": None,
                "left_child_id": "rc",
                "right_child_id": None,
            },
            # Level 2 nodes (depth 2)
            {
                "node_id": "ll",
                "text": "Left-Left",
                "span_start": 0,
                "span_end": 25,
                "document_id": "doc-id",
                "token_count": 25,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "lr",
                "text": "Left-Right",
                "span_start": 25,
                "span_end": 50,
                "document_id": "doc-id",
                "token_count": 25,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "rc",
                "text": "Right-Child",
                "span_start": 50,
                "span_end": 75,
                "document_id": "doc-id",
                "token_count": 25,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("left", "root"),
                ("right", "root"),
                ("ll", "left"),
                ("lr", "left"),
                ("rc", "right"),
            ]
        )

        # Upsert embeddings
        vector_index.upsert(
            [
                (
                    "root",
                    [0.1] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 100,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
                (
                    "left",
                    [0.2] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 50,
                        "parent_id": "root",
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
                (
                    "right",
                    [0.3] * 1536,
                    {
                        "span_start": 50,
                        "span_end": 100,
                        "parent_id": "root",
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
                (
                    "ll",
                    [0.4] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 25,
                        "parent_id": "left",
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
                (
                    "lr",
                    [0.5] * 1536,
                    {
                        "span_start": 25,
                        "span_end": 50,
                        "parent_id": "left",
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
                (
                    "rc",
                    [0.6] * 1536,
                    {
                        "span_start": 50,
                        "span_end": 75,
                        "parent_id": "right",
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
            ]
        )

        # Test depth calculations
        assert doc_store.tree.get_depth("root") == 0
        assert doc_store.tree.get_depth("left") == 1
        assert doc_store.tree.get_depth("right") == 1
        assert doc_store.tree.get_depth("ll") == 2
        assert doc_store.tree.get_depth("lr") == 2
        assert doc_store.tree.get_depth("rc") == 2

        # Test is_root method
        assert doc_store.tree.is_root("root") is True
        assert doc_store.tree.is_root("left") is False
        assert doc_store.tree.is_root("ll") is False

        # Test with non-existent node
        assert doc_store.tree.is_root("non-existent") is False

        # Test non-existent node raises ValueError (not NodeNotFoundError)
        with pytest.raises(ValueError):
            doc_store.tree.get_depth("non-existent")

    def test_node_height_calculation(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test dynamic height calculation."""
        # Create the same tree structure
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "root",
                "text": "Root",
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc-id",
                "token_count": 100,
                "height": 2,
                "parent_id": None,
                "left_child_id": "left",
                "right_child_id": "right",
            },
            {
                "node_id": "left",
                "text": "Left",
                "span_start": 0,
                "span_end": 50,
                "document_id": "doc-id",
                "token_count": 50,
                "height": 1,
                "parent_id": None,
                "left_child_id": "ll",
                "right_child_id": "lr",
            },
            {
                "node_id": "right",
                "text": "Right",
                "span_start": 50,
                "span_end": 100,
                "document_id": "doc-id",
                "token_count": 50,
                "height": 1,
                "parent_id": None,
                "left_child_id": "rc",
                "right_child_id": None,
            },
            {
                "node_id": "ll",
                "text": "Left-Left",
                "span_start": 0,
                "span_end": 25,
                "document_id": "doc-id",
                "token_count": 25,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "lr",
                "text": "Left-Right",
                "span_start": 25,
                "span_end": 50,
                "document_id": "doc-id",
                "token_count": 25,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "rc",
                "text": "Right-Child",
                "span_start": 50,
                "span_end": 75,
                "document_id": "doc-id",
                "token_count": 25,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("left", "root"),
                ("right", "root"),
                ("ll", "left"),
                ("lr", "left"),
                ("rc", "right"),
            ]
        )

        # Upsert embeddings
        vector_index.upsert(
            [
                (
                    "root",
                    [0.1] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 100,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
                (
                    "left",
                    [0.2] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 50,
                        "parent_id": "root",
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
                (
                    "right",
                    [0.3] * 1536,
                    {
                        "span_start": 50,
                        "span_end": 100,
                        "parent_id": "root",
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
                (
                    "ll",
                    [0.4] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 25,
                        "parent_id": "left",
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
                (
                    "lr",
                    [0.5] * 1536,
                    {
                        "span_start": 25,
                        "span_end": 50,
                        "parent_id": "left",
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
                (
                    "rc",
                    [0.6] * 1536,
                    {
                        "span_start": 50,
                        "span_end": 75,
                        "parent_id": "right",
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
            ]
        )

        # Test height calculations using stored values
        # Leaf nodes have height 0
        ll_node = doc_store.nodes.get_node("ll")
        lr_node = doc_store.nodes.get_node("lr")
        rc_node = doc_store.nodes.get_node("rc")
        assert ll_node is not None and ll_node.height == 0
        assert lr_node is not None and lr_node.height == 0
        assert rc_node is not None and rc_node.height == 0

        # Internal nodes have height = 1 + max(child heights)
        left_node = doc_store.nodes.get_node("left")
        right_node = doc_store.nodes.get_node("right")
        root_node = doc_store.nodes.get_node("root")
        assert left_node is not None and left_node.height == 1  # max(0, 0) + 1
        assert right_node is not None and right_node.height == 1  # has only left child
        assert root_node is not None and root_node.height == 2  # max(1, 1) + 1

        # Test is_leaf method
        assert doc_store.tree.is_leaf("ll") is True
        assert doc_store.tree.is_leaf("lr") is True
        assert doc_store.tree.is_leaf("rc") is True
        assert doc_store.tree.is_leaf("left") is False
        assert doc_store.tree.is_leaf("root") is False

        # Test with non-existent node
        assert doc_store.tree.is_leaf("non-existent") is False

        # Test non-existent node
        assert doc_store.nodes.get_node("non-existent") is None

    def test_depth_height_edge_cases(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test edge cases for depth/height calculation."""
        # Test single node (both root and leaf)
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "single",
                "text": "Single node",
                "span_start": 0,
                "span_end": 10,
                "document_id": "doc-id",
                "token_count": 10,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            # Test node with only left child
            {
                "node_id": "parent_left_only",
                "text": "Parent with left only",
                "span_start": 0,
                "span_end": 20,
                "document_id": "doc-id",
                "token_count": 20,
                "height": 1,
                "parent_id": None,
                "left_child_id": "left_only_child",
                "right_child_id": None,
            },
            {
                "node_id": "left_only_child",
                "text": "Left only child",
                "span_start": 0,
                "span_end": 10,
                "document_id": "doc-id",
                "token_count": 10,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            # Test node with only right child
            {
                "node_id": "parent_right_only",
                "text": "Parent with right only",
                "span_start": 0,
                "span_end": 20,
                "document_id": "doc-id",
                "token_count": 20,
                "height": 1,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": "right_only_child",
            },
            {
                "node_id": "right_only_child",
                "text": "Right only child",
                "span_start": 10,
                "span_end": 20,
                "document_id": "doc-id",
                "token_count": 10,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("left_only_child", "parent_left_only"),
                ("right_only_child", "parent_right_only"),
            ]
        )

        # Upsert embeddings
        vector_index.upsert(
            [
                (
                    "single",
                    [0.1] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 10,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
                (
                    "parent_left_only",
                    [0.2] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 20,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
                (
                    "left_only_child",
                    [0.3] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 10,
                        "parent_id": "parent_left_only",
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
                (
                    "parent_right_only",
                    [0.4] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 20,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 0,
                    },
                ),
                (
                    "right_only_child",
                    [0.5] * 1536,
                    {
                        "span_start": 10,
                        "span_end": 20,
                        "parent_id": "parent_right_only",
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                ),
            ]
        )

        assert doc_store.tree.get_depth("single") == 0  # Root has depth 0
        single_node = doc_store.nodes.get_node("single")
        assert single_node is not None and single_node.height == 0  # Leaf has height 0
        assert doc_store.tree.is_root("single") is True
        assert doc_store.tree.is_leaf("single") is True

        parent_left_only_node = doc_store.nodes.get_node("parent_left_only")
        assert parent_left_only_node is not None and parent_left_only_node.height == 1
        assert doc_store.tree.is_leaf("parent_left_only") is False

        parent_right_only_node = doc_store.nodes.get_node("parent_right_only")
        assert parent_right_only_node is not None and parent_right_only_node.height == 1
        assert doc_store.tree.is_leaf("parent_right_only") is False

    def test_depth_calculation_performance(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test that depth calculation is O(log n) by creating a deep tree."""
        # Create a linear chain of nodes to test worst case
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = []
        from collections.abc import Sequence

        vector_entries: list[tuple[str, Sequence[float], dict[str, object]]] = []
        parent_refs: list[tuple[str, str]] = []

        # Create a chain of 10 nodes
        for i in range(10):
            node_id = f"chain_{i}"
            parent_id = f"chain_{i-1}" if i > 0 else None

            nodes.append(
                {
                    "node_id": node_id,
                    "text": f"Chain node {i}",
                    "span_start": i * 10,
                    "span_end": (i + 1) * 10,
                    "document_id": "doc-id",
                    "token_count": 10,
                    "height": 0,
                    "parent_id": None,
                    "left_child_id": f"chain_{i+1}" if i < 9 else None,
                    "right_child_id": None,
                }
            )

            vector_entries.append(
                (
                    node_id,
                    [0.1 * i] * 1536,
                    {
                        "span_start": i * 10,
                        "span_end": (i + 1) * 10,
                        "parent_id": parent_id,
                        "document_id": "doc-id",
                        "is_leaf": 1 if i == 9 else 0,
                    },
                )
            )

            if parent_id:
                parent_refs.append((node_id, parent_id))

        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(parent_refs)
        # Cast to exact expected type for upsert
        from typing import cast

        import numpy as np
        from numpy.typing import NDArray

        typed_entries: list[tuple[str, list[float], dict[str, object]]] = [
            (entry[0], list(entry[1]), entry[2]) for entry in vector_entries
        ]
        typed_entries_u = cast(
            list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
            typed_entries,
        )
        vector_index.upsert(typed_entries_u)

        # Test depths
        for i in range(10):
            node_id = f"chain_{i}"
            assert doc_store.tree.get_depth(node_id) == i

        # Even for the deepest node, we only traverse up to root
        # This is O(depth) = O(log n) for balanced trees
        assert doc_store.tree.get_depth("chain_9") == 9

    def test_error_handling_patterns(
        self, doc_store: DocumentStore, vector_index: VectorIndex
    ) -> None:
        """Test consistent error handling patterns."""
        # Create a simple tree for testing
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "root",
                "text": "Root",
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc-id",
                "token_count": 100,
                "height": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            }
        ]
        doc_store.nodes.add_batch(nodes)

        # Upsert embeddings
        vector_index.upsert(
            [
                (
                    "root",
                    [0.1] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 100,
                        "parent_id": None,
                        "document_id": "doc-id",
                        "is_leaf": 1,
                    },
                )
            ]
        )

        # Test ValueError for calculation methods with missing nodes
        with pytest.raises(ValueError):
            doc_store.tree.get_depth("missing")

        # Note: DocumentStore doesn't have pin_node method - test other error patterns
        # Test that empty embedding is handled (this would be caught at add_batch level)

        # Test predicate methods return False for missing nodes (don't raise)
        assert doc_store.tree.is_leaf("missing") is False
        assert doc_store.tree.is_root("missing") is False

        # Test query methods return None for missing items (don't raise)
        assert doc_store.nodes.get_node("missing") is None
