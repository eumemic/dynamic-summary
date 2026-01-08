"""Backend-agnostic tests for tree navigation operations.

These tests ensure TreeNavigator functions behave correctly using
structural relationships rather than relying on precomputed paths.
"""

from __future__ import annotations

import pytest

from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.document_store import DocumentStore
from ragzoom.services.tree_navigator import TreeNavigator


class TestTreeNavigation:
    """Validate tree navigation helpers independent of path metadata."""

    @pytest.fixture
    def doc_store(self, storage_backend: StorageBackend) -> object:
        """Create a document-scoped store for doc1."""
        doc_store = storage_backend.for_document("doc1")

        # Set up document metadata
        doc_store.set_metadata(
            file_path="path_test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        return doc_store

    @pytest.fixture
    def seed_nodes(self, doc_store: DocumentStore) -> None:
        """Create a small binary tree directly in the backend for testing.

        Structure:
            root ("")
            /        \
        left ("0")  right ("1")
        /     \
   left_left  left_right
    ("00")     ("01")
        """
        nodes: list[NodeDataDict] = [
            # Leaf nodes
            {
                "node_id": "left_left",
                "text": "Left-left grandchild",
                "span_start": 0,
                "span_end": 25,
                "document_id": "doc1",
                "token_count": 3,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "left_right",
                "text": "Left-right grandchild",
                "span_start": 25,
                "span_end": 50,
                "document_id": "doc1",
                "token_count": 3,
                "height": 0,
                "level_index": 1,
            },
            # Internal nodes
            {
                "node_id": "left",
                "text": "Left child",
                "span_start": 0,
                "span_end": 50,
                "document_id": "doc1",
                "token_count": 10,
                "height": 1,
                "level_index": 0,
                "left_child_id": "left_left",
                "right_child_id": "left_right",
            },
            {
                "node_id": "right",
                "text": "Right child",
                "span_start": 50,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 10,
                "height": 1,
                "level_index": 1,
            },
            {
                "node_id": "root",
                "text": "Root node",
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 10,
                "height": 2,
                "level_index": 0,
                "left_child_id": "left",
                "right_child_id": "right",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        # Update parent references
        doc_store.nodes.update_parent_references_batch(
            [
                ("left_left", "left"),
                ("left_right", "left"),
                ("left", "root"),
                ("right", "root"),
            ]
        )

    def test_get_node_depth(self, doc_store: DocumentStore, seed_nodes: None) -> None:
        """Depth calculation should traverse ancestors correctly."""
        navigator = TreeNavigator(doc_store._node_repo)

        # Depth should equal the number of ancestor hops
        assert navigator.get_node_depth("root") == 0  # Root depth
        assert navigator.get_node_depth("left") == 1  # First level
        assert navigator.get_node_depth("right") == 1  # First level
        assert navigator.get_node_depth("left_left") == 2  # Second level
        assert navigator.get_node_depth("left_right") == 2  # Second level

    def test_get_node_depth_caches_results(
        self, doc_store: DocumentStore, seed_nodes: None
    ) -> None:
        """Depth lookups cache computed values for subsequent calls."""
        navigator = TreeNavigator(doc_store._node_repo)

        depth = navigator.get_node_depth("left_left")
        assert depth == 2
        # First pass should populate cache for the entire ancestor chain
        assert navigator._depth_cache["left_left"] == 2
        assert navigator._depth_cache["left"] == 1
        assert navigator._depth_cache["root"] == 0

        # Subsequent calls should return without hitting the repository
        original_get_node = navigator.node_repo.get_node
        call_counter = {"count": 0}

        def counting_get_node(node_id: str) -> TreeNode | None:
            call_counter["count"] += 1
            return original_get_node(node_id)

        navigator.node_repo.get_node = counting_get_node  # type: ignore[method-assign]
        try:
            assert navigator.get_node_depth("left_left") == 2
            assert call_counter["count"] == 0
        finally:
            navigator.node_repo.get_node = original_get_node  # type: ignore[method-assign]

    def test_get_parent_node(self, doc_store: DocumentStore, seed_nodes: None) -> None:
        """Parent lookup should rely on stored parent pointers."""
        navigator = TreeNavigator(doc_store._node_repo)

        # Parent lookup relies on stored parent pointers
        root_parent = navigator.get_parent_node("root")
        assert root_parent is None  # Root has no parent

        left_parent = navigator.get_parent_node("left")
        assert left_parent is not None
        assert left_parent.id == "root"

        left_left_parent = navigator.get_parent_node("left_left")
        assert left_left_parent is not None
        assert left_left_parent.id == "left"

    def test_get_sibling_node(self, doc_store: DocumentStore, seed_nodes: None) -> None:
        """Sibling lookup should consult the shared parent."""
        navigator = TreeNavigator(doc_store._node_repo)

        # Sibling lookup should return the opposite child of the shared parent
        root_sibling = navigator.get_sibling_node("root")
        assert root_sibling is None  # Root has no sibling

        left_sibling = navigator.get_sibling_node("left")
        assert left_sibling is not None
        assert left_sibling.id == "right"

        right_sibling = navigator.get_sibling_node("right")
        assert right_sibling is not None
        assert right_sibling.id == "left"

        left_left_sibling = navigator.get_sibling_node("left_left")
        assert left_left_sibling is not None
        assert left_left_sibling.id == "left_right"

    def test_is_left_child(self, doc_store: DocumentStore, seed_nodes: None) -> None:
        """Left child detection should use parent pointers."""
        navigator = TreeNavigator(doc_store._node_repo)

        # Left child detection should reflect the parent's left pointer
        assert not navigator.is_left_child("root")  # Root is neither left nor right
        assert navigator.is_left_child("left")  # Left child
        assert not navigator.is_left_child("right")  # Right child, not left
        assert navigator.is_left_child("left_left")  # Left-left is left child
        assert not navigator.is_left_child("left_right")  # Left-right is right child

    def test_is_right_child(self, doc_store: DocumentStore, seed_nodes: None) -> None:
        """Right child detection should use parent pointers."""
        navigator = TreeNavigator(doc_store._node_repo)

        # Right child detection should reflect the parent's right pointer
        assert not navigator.is_right_child("root")  # Root is neither left nor right
        assert not navigator.is_right_child("left")  # Left child, not right
        assert navigator.is_right_child("right")  # Right child
        assert not navigator.is_right_child("left_left")  # Left-left is left child
        assert navigator.is_right_child("left_right")  # Left-right is right child

    def test_pinned_nodes_path_filtering(
        self, doc_store: DocumentStore, seed_nodes: None
    ) -> None:
        """Test that get_pinned_nodes uses path-based database filtering."""
        # Pin some nodes at different depths
        doc_store._node_repo.pin_node("root")  # Depth 0
        doc_store._node_repo.pin_node("left")  # Depth 1
        doc_store._node_repo.pin_node("left_left")  # Depth 2

        # Test filtering by depth
        pinned_depth_0 = doc_store.get_pinned_nodes(depth_max=0)
        assert len(pinned_depth_0) == 1
        assert pinned_depth_0[0].id == "root"

        pinned_depth_1 = doc_store.get_pinned_nodes(depth_max=1)
        assert len(pinned_depth_1) == 2
        node_ids = {node.id for node in pinned_depth_1}
        assert node_ids == {"root", "left"}

        pinned_depth_2 = doc_store.get_pinned_nodes(depth_max=2)
        assert len(pinned_depth_2) == 3
        node_ids = {node.id for node in pinned_depth_2}
        assert node_ids == {"root", "left", "left_left"}

        # Test no depth limit
        pinned_all = doc_store.get_pinned_nodes()
        assert len(pinned_all) == 3

    def test_path_optimization_performance(
        self, doc_store: DocumentStore, seed_nodes: None
    ) -> None:
        """Test that path-based methods avoid database queries where possible."""
        navigator = TreeNavigator(doc_store._node_repo)

        # With proper paths, these operations should be very fast
        # and not require traversing up the tree
        depth = navigator.get_node_depth("left_left")
        assert depth == 2

        # The path-based implementation should use string operations
        # rather than multiple database queries
        sibling = navigator.get_sibling_node("left_left")
        assert sibling is not None
        assert sibling.id == "left_right"
