"""Tests for binary tree path utilities."""

from ragzoom.utils.path_utils import (
    calculate_tree_depth,
    generate_leaf_path,
    get_all_ancestor_paths,
    get_depth,
    get_parent_path,
    get_position_at_level,
    get_sibling_path,
    path_exists_in_left_balanced_tree,
)


class TestPathUtilities:
    """Test binary tree path utility functions."""

    def test_get_parent_path(self):
        """Test parent path generation."""
        assert get_parent_path("") == ""  # Root has no parent
        assert get_parent_path("0") == ""  # First level -> root
        assert get_parent_path("1") == ""  # First level -> root
        assert get_parent_path("00") == "0"  # Second level -> first level
        assert get_parent_path("01") == "0"
        assert get_parent_path("10") == "1"
        assert get_parent_path("11") == "1"
        assert get_parent_path("001") == "00"  # Third level -> second level

    def test_get_sibling_path(self):
        """Test sibling path generation."""
        assert get_sibling_path("") is None  # Root has no sibling
        assert get_sibling_path("0") == "1"  # Left child -> right child
        assert get_sibling_path("1") == "0"  # Right child -> left child
        assert get_sibling_path("00") == "01"
        assert get_sibling_path("01") == "00"
        assert get_sibling_path("10") == "11"
        assert get_sibling_path("11") == "10"
        assert get_sibling_path("001") == "000"

    def test_get_depth(self):
        """Test depth calculation."""
        assert get_depth("") == 0  # Root at depth 0
        assert get_depth("0") == 1  # First level
        assert get_depth("1") == 1
        assert get_depth("00") == 2  # Second level
        assert get_depth("11") == 2
        assert get_depth("001") == 3  # Third level

    def test_get_position_at_level(self):
        """Test position calculation at each level."""
        assert get_position_at_level("") == 0  # Root at position 0
        assert get_position_at_level("0") == 0  # First position at level 1
        assert get_position_at_level("1") == 1  # Second position at level 1
        assert get_position_at_level("00") == 0  # First position at level 2
        assert get_position_at_level("01") == 1  # Second position at level 2
        assert get_position_at_level("10") == 2  # Third position at level 2
        assert get_position_at_level("11") == 3  # Fourth position at level 2

    def test_get_all_ancestor_paths(self):
        """Test ancestor path generation."""
        assert get_all_ancestor_paths("") == []  # Root has no ancestors
        assert get_all_ancestor_paths("0") == [""]  # First level -> root
        assert get_all_ancestor_paths("1") == [""]
        assert get_all_ancestor_paths("00") == [
            "0",
            "",
        ]  # Second level -> first -> root
        assert get_all_ancestor_paths("11") == ["1", ""]
        assert get_all_ancestor_paths("001") == ["00", "0", ""]  # Third level paths

    def test_calculate_tree_depth(self):
        """Test tree depth calculation."""
        assert calculate_tree_depth(0) == 0  # No leaves
        assert calculate_tree_depth(1) == 0  # Single leaf (root)
        assert calculate_tree_depth(2) == 1  # Two leaves need depth 1
        assert calculate_tree_depth(3) == 2  # Three leaves need depth 2
        assert calculate_tree_depth(4) == 2  # Four leaves need depth 2
        assert calculate_tree_depth(5) == 3  # Five leaves need depth 3
        assert calculate_tree_depth(8) == 3  # Eight leaves need depth 3
        assert calculate_tree_depth(9) == 4  # Nine leaves need depth 4

    def test_generate_leaf_path(self):
        """Test leaf path generation."""
        # Single leaf (root)
        assert generate_leaf_path(0, 0) == ""

        # Two leaves (depth 1)
        assert generate_leaf_path(0, 1) == "0"
        assert generate_leaf_path(1, 1) == "1"

        # Four leaves (depth 2)
        assert generate_leaf_path(0, 2) == "00"
        assert generate_leaf_path(1, 2) == "01"
        assert generate_leaf_path(2, 2) == "10"
        assert generate_leaf_path(3, 2) == "11"

        # Eight leaves (depth 3)
        assert generate_leaf_path(0, 3) == "000"
        assert generate_leaf_path(7, 3) == "111"

    def test_path_exists_in_left_balanced_tree(self):
        """Test path existence validation for left-balanced trees."""
        # Empty tree
        assert not path_exists_in_left_balanced_tree("", 0)
        assert not path_exists_in_left_balanced_tree("0", 0)

        # Single node tree (root only)
        assert path_exists_in_left_balanced_tree("", 1)
        assert not path_exists_in_left_balanced_tree("0", 1)

        # Two leaf tree (depth 1)
        assert path_exists_in_left_balanced_tree("", 2)  # Root exists
        assert path_exists_in_left_balanced_tree("0", 2)  # Left leaf exists
        assert path_exists_in_left_balanced_tree("1", 2)  # Right leaf exists
        assert not path_exists_in_left_balanced_tree("00", 2)  # Too deep

        # Three leaf tree (left-balanced, depth 2)
        assert path_exists_in_left_balanced_tree("", 3)  # Root exists
        assert path_exists_in_left_balanced_tree(
            "0", 3
        )  # Left internal node (has children)
        assert path_exists_in_left_balanced_tree(
            "1", 3
        )  # Right internal node (has one child)
        assert path_exists_in_left_balanced_tree("00", 3)  # First leaf
        assert path_exists_in_left_balanced_tree("01", 3)  # Second leaf
        assert path_exists_in_left_balanced_tree(
            "10", 3
        )  # Third leaf (right subtree gets one)
        assert not path_exists_in_left_balanced_tree(
            "11", 3
        )  # Fourth position doesn't exist

        # Four leaf tree (complete, depth 2)
        assert path_exists_in_left_balanced_tree("10", 4)  # Third leaf exists
        assert path_exists_in_left_balanced_tree("11", 4)  # Fourth leaf exists

    def test_integration_leaf_to_root_traversal(self):
        """Test complete traversal from leaf to root."""
        # Test with a leaf at path "001" in an 8-leaf tree
        leaf_path = "001"

        # Get all ancestors
        ancestors = get_all_ancestor_paths(leaf_path)
        expected = ["00", "0", ""]
        assert ancestors == expected

        # Verify each ancestor exists in an 8-leaf tree
        for path in ancestors:
            assert path_exists_in_left_balanced_tree(path, 8)

        # Get sibling
        sibling = get_sibling_path(leaf_path)
        assert sibling == "000"
        assert path_exists_in_left_balanced_tree(sibling, 8)

    def test_tree_structure_consistency(self):
        """Test that generated paths form a consistent tree structure."""
        # Generate paths for a 5-leaf tree
        num_leaves = 5
        tree_depth = calculate_tree_depth(num_leaves)
        assert tree_depth == 3

        # Generate all leaf paths
        leaf_paths = [generate_leaf_path(i, tree_depth) for i in range(num_leaves)]
        expected_leaves = ["000", "001", "010", "011", "100"]
        assert leaf_paths == expected_leaves

        # Verify each leaf exists
        for path in leaf_paths:
            assert path_exists_in_left_balanced_tree(path, num_leaves)

        # Verify non-existent leaves don't exist
        assert not path_exists_in_left_balanced_tree("101", num_leaves)
        assert not path_exists_in_left_balanced_tree("110", num_leaves)
        assert not path_exists_in_left_balanced_tree("111", num_leaves)

        # Verify internal nodes exist correctly
        assert path_exists_in_left_balanced_tree("00", num_leaves)  # Has 2 children
        assert path_exists_in_left_balanced_tree(
            "01", num_leaves
        )  # Has 1 child (promoted)
        assert path_exists_in_left_balanced_tree(
            "1", num_leaves
        )  # Has 1 child (the single leaf "100")
        assert path_exists_in_left_balanced_tree(
            "10", num_leaves
        )  # Internal node with one child
