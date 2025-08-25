"""Utilities for working with binary tree path encodings."""

import math


def get_parent_path(path: str) -> str:
    """Get parent path by removing the last bit.

    Args:
        path: Binary path string (e.g., "001")

    Returns:
        Parent path (e.g., "00"), or empty string if already at root
    """
    return path[:-1] if path else ""


def get_sibling_path(path: str) -> str | None:
    """Get sibling path by flipping the last bit.

    Args:
        path: Binary path string (e.g., "001")

    Returns:
        Sibling path (e.g., "000"), or None if this is the root
    """
    if not path:  # Root has no sibling
        return None
    return path[:-1] + str(int(path[-1]) ^ 1)


def get_depth(path: str) -> int:
    """Get node depth (distance from root).

    Args:
        path: Binary path string

    Returns:
        Depth value (0 for root, incrementing by 1 for each level down)
    """
    return len(path)


def get_position_at_level(path: str) -> int:
    """Get the position of this node at its level (0-indexed).

    Args:
        path: Binary path string (e.g., "101")

    Returns:
        Position at level (e.g., 5 for "101" which is binary 101 = 5)
    """
    if not path:  # Root is at position 0
        return 0
    return int(path, 2)


def get_all_ancestor_paths(path: str) -> list[str]:
    """Get all ancestor paths including the root.

    Args:
        path: Binary path string (e.g., "001")

    Returns:
        List of ancestor paths from immediate parent to root (e.g., ["00", "0", ""])
    """
    if not path:  # Root has no ancestors
        return []
    return [path[:i] for i in range(len(path) - 1, -1, -1)]


def calculate_tree_depth(num_leaves: int) -> int:
    """Calculate the depth needed for a complete binary tree with num_leaves leaves.

    Args:
        num_leaves: Number of leaf nodes

    Returns:
        Tree depth (path length for leaf nodes)
    """
    if num_leaves <= 1:
        return 0
    return math.ceil(math.log2(num_leaves))


def generate_leaf_path(leaf_index: int, tree_depth: int) -> str:
    """Generate a binary path for a leaf node at given index.

    Args:
        leaf_index: 0-based index of the leaf
        tree_depth: Total tree depth (path length for leaves)

    Returns:
        Binary path string (e.g., "001" for leaf 1 at depth 3)
    """
    if tree_depth == 0:
        return ""
    return format(leaf_index, f"0{tree_depth}b")


def path_exists_in_left_balanced_tree(path: str, num_leaves: int) -> bool:
    """Check if a path corresponds to an existing node in a left-balanced binary tree.

    In a left-balanced tree with N leaves, not all possible paths exist.
    This function determines which paths are valid based on how the tree builder
    actually constructs trees (leaves filled left-to-right, internal nodes only
    exist if they have children).

    Args:
        path: Binary path string to check
        num_leaves: Total number of leaves in the tree

    Returns:
        True if the path exists in the tree, False otherwise
    """
    if not path:  # Root always exists if there are any nodes
        return num_leaves > 0

    if num_leaves <= 1:
        return path == "" and num_leaves == 1

    tree_depth = calculate_tree_depth(num_leaves)

    # Path longer than tree depth cannot exist
    if len(path) > tree_depth:
        return False

    # For leaf nodes (path length == tree_depth), check position directly
    if len(path) == tree_depth:
        leaf_position = int(path, 2)
        return leaf_position < num_leaves

    # For internal nodes (shorter than tree depth), they exist if they have
    # at least one leaf descendant
    depth_diff = tree_depth - len(path)
    # Calculate the range of leaf positions this internal node would cover
    first_leaf_position = int(path, 2) << depth_diff if path else 0
    # Internal node exists if any leaf in its range exists
    return first_leaf_position < num_leaves
