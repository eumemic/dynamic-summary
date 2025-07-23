"""Tests for tree fullness validation during indexing."""

import pytest

from ragzoom.validate import validate_tree_is_full


class TestTreeFullness:
    """Tests to ensure indexed trees are always full binary trees."""

    @pytest.fixture
    def mock_store_with_full_tree(self):
        """Create a mock store with a valid full tree."""

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

        # Create a full binary tree:
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
    def mock_store_with_incomplete_tree(self):
        """Create a mock store with an incomplete tree (missing child)."""

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

        # Create an incomplete tree (P2 only has left child):
        #       root
        #      /    \
        #     P1     P2
        #    /  \   /
        #   L1  L2 L3

        store.add_node(MockNode("L1"))
        store.add_node(MockNode("L2"))
        store.add_node(MockNode("L3"))
        store.add_node(MockNode("P1", "L1", "L2"))
        store.add_node(MockNode("P2", "L3", None))  # Missing right child!
        store.add_node(MockNode("root", "P1", "P2"))

        return store

    def test_full_tree_passes_validation(self, mock_store_with_full_tree):
        """Test that a full binary tree passes validation."""
        result = validate_tree_is_full(mock_store_with_full_tree, "test-doc")
        assert result is None  # None means valid

    def test_incomplete_tree_fails_validation(self, mock_store_with_incomplete_tree):
        """Test that an incomplete tree fails validation."""
        result = validate_tree_is_full(mock_store_with_incomplete_tree, "test-doc")
        assert result is not None  # Should return error message
        assert "Tree is not full" in result
        assert "P2" in result  # Should identify the problematic node
        assert "missing its right child" in result

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
        result = validate_tree_is_full(store, "test-doc")
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
        result = validate_tree_is_full(store, "test-doc")
        assert result is not None
        assert "Invalid tree" in result
        assert "non-existent" in result

    def test_odd_number_of_leaves(self):
        """Test tree building with odd number of leaves results in a full tree."""

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

        # With 3 leaves, tree should look like:
        #      root
        #     /    \
        #    P1     L3
        #   /  \
        #  L1  L2
        # This is a valid full tree

        store.add_node(MockNode("L1"))
        store.add_node(MockNode("L2"))
        store.add_node(MockNode("L3"))
        store.add_node(MockNode("P1", "L1", "L2"))
        store.add_node(MockNode("root", "P1", "L3"))

        result = validate_tree_is_full(store, "test-doc")
        assert result is None  # Should be valid
