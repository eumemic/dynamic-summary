"""SQLite-based tests for data builders.

SQLite-based tests for TreeNode and Document builder functionality
with the real in-memory SQLite backend.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.document_store import DocumentStore


@pytest.mark.usefixtures("sqlite_backend")
class TestBuildersSQLite:
    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("test-doc")

    def test_tree_node_builder_basic(self, doc_store: DocumentStore) -> None:
        """Test basic TreeNode creation with builder pattern."""
        # Create nodes using the SQLite pattern
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "test-node-1",
                "text": "Test node text",
                "embedding": [],
                "span_start": 0,
                "span_end": 10,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 0,
                "path": "0",
            }
        ]
        doc_store.nodes.add_batch(nodes)

        # Verify node was created
        node = doc_store.nodes.get_node("test-node-1")
        assert node is not None
        assert node.id == "test-node-1"
        assert node.text == "Test node text"
        assert node.span_start == 0
        assert node.span_end == 10
        assert node.token_count == 5
        assert node.height == 0

    def test_tree_node_with_parent_child_relationships(
        self, doc_store: DocumentStore
    ) -> None:
        """Test TreeNode creation with parent-child relationships."""
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "child",
                "text": "Child node",
                "embedding": [],
                "span_start": 0,
                "span_end": 50,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 0,
                "parent_id": "parent",
                "path": "00",
            },
            {
                "node_id": "parent",
                "text": "Parent node",
                "embedding": [],
                "span_start": 0,
                "span_end": 50,
                "document_id": "test-doc",
                "token_count": 20,
                "height": 1,
                "left_child_id": "child",
                "right_child_id": None,
                "path": "0",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch([("child", "parent")])

        # Verify relationships
        child = doc_store.nodes.get_node("child")
        parent = doc_store.nodes.get_node("parent")

        assert child is not None
        assert parent is not None
        assert child.parent_id == "parent"
        assert parent.left_child_id == "child"
        assert parent.right_child_id is None

    def test_tree_node_with_custom_embedding(self, doc_store: DocumentStore) -> None:
        """Test TreeNode with custom embedding vector."""
        custom_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "embedded-node",
                "text": "Node with custom embedding",
                "embedding": custom_embedding,
                "span_start": 0,
                "span_end": 25,
                "document_id": "test-doc",
                "token_count": 15,
                "height": 0,
                "path": "0",
            }
        ]
        doc_store.nodes.add_batch(nodes)

        # Verify node was created (embedding is handled by vector index, not stored in DB)
        node = doc_store.nodes.get_node("embedded-node")
        assert node is not None
        assert node.text == "Node with custom embedding"
        assert node.token_count == 15

    def test_tree_node_with_document_reference(self, doc_store: DocumentStore) -> None:
        """Test TreeNode with specific document reference."""
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "doc-specific-node",
                "text": "Node for specific document",
                "embedding": [],
                "span_start": 0,
                "span_end": 30,
                "document_id": "test-doc",
                "token_count": 12,
                "height": 0,
                "path": "0",
            }
        ]
        doc_store.nodes.add_batch(nodes)

        node = doc_store.nodes.get_node("doc-specific-node")
        assert node is not None
        assert node.document_id == "test-doc"

    def test_tree_node_with_spans_and_tokens(self, doc_store: DocumentStore) -> None:
        """Test TreeNode with custom span and token count."""
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "span-node",
                "text": "Node with custom span and token count",
                "embedding": [],
                "span_start": 100,
                "span_end": 200,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 0,
                "path": "0",
            }
        ]
        doc_store.nodes.add_batch(nodes)

        node = doc_store.nodes.get_node("span-node")
        assert node is not None
        assert node.span_start == 100
        assert node.span_end == 200
        assert node.token_count == 50

    def test_document_builder_basic(self, doc_store: DocumentStore) -> None:
        """Test basic Document creation using document store."""
        # The document is already created by the fixture, verify its properties
        # Access the document through the store's document property
        doc_id = doc_store.document_id
        assert doc_id == "test-doc"

        # Test demonstrates document scoping via the store factory pattern
        # The actual document is handled by the fixture setup
        assert doc_id == "test-doc"

    def test_create_test_tree_nodes(self, doc_store: DocumentStore) -> None:
        """Test creation of a simple test tree structure."""
        # Create a simple 3-node tree
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "node-0",
                "text": "Test text for node 0",
                "embedding": [],
                "span_start": 0,
                "span_end": 10,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 1,
                "left_child_id": "node-1",
                "right_child_id": "node-2",
                "path": "",
            },
            {
                "node_id": "node-1",
                "text": "Test text for node 1",
                "embedding": [],
                "span_start": 10,
                "span_end": 20,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 0,
                "parent_id": "node-0",
                "path": "0",
            },
            {
                "node_id": "node-2",
                "text": "Test text for node 2",
                "embedding": [],
                "span_start": 20,
                "span_end": 30,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 0,
                "parent_id": "node-0",
                "path": "1",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("node-1", "node-0"), ("node-2", "node-0")]
        )

        # Verify tree structure
        root = doc_store.nodes.get_node("node-0")
        left_child = doc_store.nodes.get_node("node-1")
        right_child = doc_store.nodes.get_node("node-2")

        assert root is not None
        assert left_child is not None
        assert right_child is not None

        assert root.left_child_id == "node-1"
        assert root.right_child_id == "node-2"
        assert left_child.parent_id == "node-0"
        assert right_child.parent_id == "node-0"

    def test_single_child_chain_structure(self, doc_store: DocumentStore) -> None:
        """Test creation of single-child chain to avoid duplicates."""
        # Create a chain: root -> middle -> leaf (single children)
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "leaf",
                "text": "Leaf node",
                "embedding": [],
                "span_start": 0,
                "span_end": 20,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 0,
                "parent_id": "middle",
                "path": "00",
            },
            {
                "node_id": "middle",
                "text": "Middle node",
                "embedding": [],
                "span_start": 0,
                "span_end": 20,
                "document_id": "test-doc",
                "token_count": 15,
                "height": 1,
                "left_child_id": "leaf",
                "right_child_id": None,  # Single child - avoid duplicates
                "parent_id": "root",
                "path": "0",
            },
            {
                "node_id": "root",
                "text": "Root node",
                "embedding": [],
                "span_start": 0,
                "span_end": 20,
                "document_id": "test-doc",
                "token_count": 20,
                "height": 2,
                "left_child_id": "middle",
                "right_child_id": None,  # Single child - avoid duplicates
                "path": "",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("leaf", "middle"), ("middle", "root")]
        )

        # Verify chain structure
        root = doc_store.nodes.get_node("root")
        middle = doc_store.nodes.get_node("middle")
        leaf = doc_store.nodes.get_node("leaf")

        assert root is not None
        assert middle is not None
        assert leaf is not None

        assert root.left_child_id == "middle"
        assert root.right_child_id is None
        assert middle.left_child_id == "leaf"
        assert middle.right_child_id is None
        assert leaf.parent_id == "middle"
        assert middle.parent_id == "root"
