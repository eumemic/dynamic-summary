"""Tests for left-balanced tree validation."""

import pytest

from ragzoom.validate import validate_tree_is_left_balanced


class TestTreeLeftBalanced:
    """Tests to ensure indexed trees maintain left-balanced property."""

    @pytest.fixture
    def mock_store_with_valid_tree(self):
        """Create a mock store with a valid left-balanced tree."""

        # Using a simple mock instead of the real Store
        class MockNode:
            def __init__(self, id, left_child_id=None, right_child_id=None):
                self.id = id
                self.left_child_id = left_child_id
                self.right_child_id = right_child_id

        class MockStore:
            def __init__(self):
                self.nodes = {}

            def add_node(self, node):
                self.nodes[node.id] = node

            def get_all_nodes_for_document(self, doc_id):
                return list(self.nodes.values())

        store = MockStore()

        # Create a valid left-balanced tree (happens to be full):
        #       root
        #      /    \
        #     P1     P2
        #    /  \   /  \
        #   L1  L2 L3  L4

        store.add_node(MockNode("L1"))
        store.add_node(MockNode("L2"))
        store.add_node(MockNode("L3"))
        store.add_node(MockNode("L4"))
        store.add_node(MockNode("P1", "L1", "L2"))
        store.add_node(MockNode("P2", "L3", "L4"))
        store.add_node(MockNode("root", "P1", "P2"))

        return store

    @pytest.fixture
    def mock_store_with_left_balanced_tree(self):
        """Create a mock store with a valid left-balanced tree (single left child)."""

        class MockNode:
            def __init__(self, id, left_child_id=None, right_child_id=None):
                self.id = id
                self.left_child_id = left_child_id
                self.right_child_id = right_child_id

        class MockStore:
            def __init__(self):
                self.nodes = {}

            def add_node(self, node):
                self.nodes[node.id] = node

            def get_all_nodes_for_document(self, doc_id):
                return list(self.nodes.values())

        store = MockStore()

        # Create a valid left-balanced tree (P2 only has left child):
        #       root
        #      /    \
        #     P1     P2
        #    /  \   /
        #   L1  L2 L3

        store.add_node(MockNode("L1"))
        store.add_node(MockNode("L2"))
        store.add_node(MockNode("L3"))
        store.add_node(MockNode("P1", "L1", "L2"))
        store.add_node(MockNode("P2", "L3", None))  # Valid: only left child
        store.add_node(MockNode("root", "P1", "P2"))

        return store

    def test_full_tree_passes_validation(self, mock_store_with_valid_tree):
        """Test that a full binary tree passes left-balanced validation."""
        result = validate_tree_is_left_balanced(mock_store_with_valid_tree, "test-doc")
        assert result is None  # None means valid

    def test_single_left_child_passes_validation(
        self, mock_store_with_left_balanced_tree
    ):
        """Test that a tree with single left children passes validation."""
        result = validate_tree_is_left_balanced(
            mock_store_with_left_balanced_tree, "test-doc"
        )
        assert result is None  # Valid left-balanced tree

    def test_single_node_tree_passes(self):
        """Test that a tree with just one node (root) is valid."""

        class MockNode:
            def __init__(self, id):
                self.id = id
                self.left_child_id = None
                self.right_child_id = None

        class MockStore:
            def get_all_nodes_for_document(self, doc_id):
                return [MockNode("root")]

        store = MockStore()
        result = validate_tree_is_left_balanced(store, "test-doc")
        assert result is None  # Single node tree is valid

    def test_invalid_child_reference_fails(self):
        """Test that referencing non-existent children fails validation."""

        class MockNode:
            def __init__(self, id, left_child_id=None, right_child_id=None):
                self.id = id
                self.left_child_id = left_child_id
                self.right_child_id = right_child_id

        class MockStore:
            def get_all_nodes_for_document(self, doc_id):
                # Node references children that don't exist
                return [MockNode("root", "missing-left", "missing-right")]

        store = MockStore()
        result = validate_tree_is_left_balanced(store, "test-doc")
        assert result is not None
        assert "Invalid tree" in result
        assert "non-existent" in result

    def test_right_child_without_left_child_fails(self):
        """Test that a node with only a right child fails validation."""

        class MockNode:
            def __init__(self, id, left_child_id=None, right_child_id=None):
                self.id = id
                self.left_child_id = left_child_id
                self.right_child_id = right_child_id

        class MockStore:
            def __init__(self):
                self.nodes = {}

            def add_node(self, node):
                self.nodes[node.id] = node

            def get_all_nodes_for_document(self, doc_id):
                return list(self.nodes.values())

        store = MockStore()

        # Invalid tree with right child but no left child:
        #      root
        #         \
        #          P1
        #         /  \
        #        L1  L2

        store.add_node(MockNode("L1"))
        store.add_node(MockNode("L2"))
        store.add_node(MockNode("P1", "L1", "L2"))
        store.add_node(MockNode("root", None, "P1"))  # Only right child!

        result = validate_tree_is_left_balanced(store, "test-doc")
        assert result is not None
        assert "not left-balanced" in result
        assert "root" in result  # Should identify the problematic node
        assert "right child but no left child" in result
