"""Pure functions for tree skeleton construction for dataflow implementation.

This module provides functions to build the complete tree structure upfront,
with all relationships established but without summaries/embeddings.
The tree is then processed by the dataflow implementation to fill in content.
"""

import math
import uuid
from dataclasses import dataclass, field


@dataclass
class SkeletonNode:
    """Node in the tree skeleton with all relationships but potentially no content."""

    id: str
    height: int
    span_start: int
    span_end: int
    path: str
    document_id: str

    # Text content - None for internal nodes until filled by dataflow
    text: str | None = None

    # Tree relationships
    parent_id: str | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None

    # Neighbor relationships for dataflow dependencies
    preceding_neighbor_id: str | None = None
    following_neighbor_id: str | None = None

    # Placeholders for content to be filled by dataflow
    embedding: list[float] | None = None
    token_count: int = 0


@dataclass
class TreeSkeleton:
    """Container for the complete tree skeleton."""

    lookup: dict[str, SkeletonNode] = field(default_factory=dict)

    def get_root(self) -> SkeletonNode | None:
        """Find and return the root node (node with no parent)."""
        for node in self.lookup.values():
            if node.parent_id is None:
                return node
        return None

    def get_nodes_at_height(self, height: int) -> list[SkeletonNode]:
        """Get all nodes at a specific height."""
        return [node for node in self.lookup.values() if node.height == height]

    def get_nodes_by_height(self) -> dict[int, list[SkeletonNode]]:
        """Get nodes grouped by height, sorted within each level by span_start."""
        nodes_by_height: dict[int, list[SkeletonNode]] = {}
        for node in self.lookup.values():
            if node.height not in nodes_by_height:
                nodes_by_height[node.height] = []
            nodes_by_height[node.height].append(node)

        # Sort nodes within each level by span_start
        for height_nodes in nodes_by_height.values():
            height_nodes.sort(key=lambda n: n.span_start)

        return nodes_by_height


def _generate_node_id() -> str:
    """Generate a unique node ID."""
    return str(uuid.uuid4())


def _calculate_tree_depth(num_leaves: int) -> int:
    """Calculate the depth of the tree for binary path encoding."""
    if num_leaves <= 1:
        return 1
    return math.ceil(math.log2(num_leaves))


def _generate_leaf_path(index: int, tree_depth: int) -> str:
    """Generate binary path for a leaf node."""
    return format(index, f"0{tree_depth}b")


def _derive_parent_path(child_path: str) -> str:
    """Derive parent path by removing last bit from child path."""
    return child_path[:-1] if child_path else ""


def create_leaf_nodes(
    chunks: list[str], document_id: str
) -> tuple[TreeSkeleton, list[SkeletonNode]]:
    """Create leaf nodes from document chunks.

    Args:
        chunks: List of text chunks from the document
        document_id: ID of the document being indexed

    Returns:
        Tuple of (TreeSkeleton containing all nodes, list of leaf nodes)
    """
    if not chunks:
        raise ValueError("No chunks provided")

    skeleton = TreeSkeleton()
    leaves: list[SkeletonNode] = []

    # Calculate tree depth for path generation
    tree_depth = _calculate_tree_depth(len(chunks))

    # Track position in document
    current_pos = 0

    for i, chunk in enumerate(chunks):
        node_id = _generate_node_id()

        leaf = SkeletonNode(
            id=node_id,
            text=chunk,  # Leaves have text
            height=0,  # Leaves are at height 0
            span_start=current_pos,
            span_end=current_pos + len(chunk),
            path=_generate_leaf_path(i, tree_depth),
            document_id=document_id,
            # No parent/children for leaves initially
            parent_id=None,
            left_child_id=None,
            right_child_id=None,
            # Neighbor relationships
            preceding_neighbor_id=leaves[-1].id if leaves else None,
            following_neighbor_id=None,  # Will be set when next leaf is created
        )

        # Update previous leaf's following_neighbor_id
        if leaves:
            leaves[-1].following_neighbor_id = leaf.id

        leaves.append(leaf)
        skeleton.lookup[node_id] = leaf
        current_pos = leaf.span_end

    return skeleton, leaves


def build_internal_nodes(skeleton: TreeSkeleton, leaves: list[SkeletonNode]) -> None:
    """Build internal nodes from leaves bottom-up.

    Modifies skeleton in-place, adding all internal nodes and updating relationships.

    Args:
        skeleton: TreeSkeleton to modify
        leaves: List of leaf nodes to build tree from
    """
    if not leaves:
        return

    # Special case: single leaf is also the root
    if len(leaves) == 1:
        return

    current_level = leaves
    current_height = 1

    while len(current_level) > 1:
        parents: list[SkeletonNode] = []

        i = 0
        while i < len(current_level):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else None

            parent_id = _generate_node_id()

            parent = SkeletonNode(
                id=parent_id,
                text=None,  # To be filled by dataflow
                height=current_height,
                span_start=left.span_start,
                span_end=right.span_end if right else left.span_end,
                path=_derive_parent_path(left.path),
                document_id=left.document_id,
                # Children
                parent_id=None,  # Will be set when grandparent is created
                left_child_id=left.id,
                right_child_id=right.id if right else None,
                # Neighbors
                preceding_neighbor_id=parents[-1].id if parents else None,
                following_neighbor_id=None,  # Will be set when next parent is created
            )

            # Update relationships
            left.parent_id = parent.id
            if right:
                right.parent_id = parent.id

            # Update previous parent's following_neighbor_id
            if parents:
                parents[-1].following_neighbor_id = parent.id

            parents.append(parent)
            skeleton.lookup[parent.id] = parent

            # Move to next pair
            i += 2 if right else 1

        current_level = parents
        current_height += 1
