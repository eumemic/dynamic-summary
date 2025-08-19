"""Service for tree navigation and traversal operations."""

import logging
from typing import TYPE_CHECKING

from ragzoom.models import TreeNode
from ragzoom.repositories.node_repository import NodeRepository

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TreeNavigator:
    """Service for tree navigation and traversal operations."""

    def __init__(self, node_repository: NodeRepository):
        """Initialize tree navigator.

        Args:
            node_repository: Node repository for data access
        """
        self.node_repo = node_repository

    def get_children(self, node_id: str) -> tuple[TreeNode | None, TreeNode | None]:
        """Get left and right children of a node.

        Args:
            node_id: Node identifier

        Returns:
            Tuple of (left_child, right_child), either can be None
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            return None, None

        left = (
            self.node_repo.get_node(node.left_child_id) if node.left_child_id else None
        )
        right = (
            self.node_repo.get_node(node.right_child_id)
            if node.right_child_id
            else None
        )
        return left, right

    def get_ancestors(self, node_ids: list[str]) -> list[TreeNode]:
        """Get all ancestors of given nodes using batch loading for efficiency.

        Args:
            node_ids: List of node identifiers

        Returns:
            List of ancestor TreeNodes
        """
        all_ancestors = set()
        current_level = set(node_ids)

        # Keep going until we've reached all roots
        while current_level:
            # Batch load all nodes at current level
            nodes_at_level = self.node_repo.get_nodes(list(current_level))

            # Collect parent IDs for next level
            next_level = set()
            for node in nodes_at_level:
                if node.parent_id and node.parent_id not in all_ancestors:
                    all_ancestors.add(node.parent_id)
                    next_level.add(node.parent_id)

            # Move up to next level
            current_level = next_level

        # Batch load all ancestors and return
        if all_ancestors:
            return self.node_repo.get_nodes(list(all_ancestors))
        return []

    def get_root_node(self) -> TreeNode | None:
        """Get the root node (node with no parent).

        Returns:
            Root TreeNode if found, None otherwise
        """
        with self.node_repo.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(parent_id=None).first()
            if node:
                session.expunge(node)
            return node

    def get_root_node_for_document(self, document_id: str | None) -> TreeNode | None:
        """Get the root node for a specific document.

        Args:
            document_id: Document identifier (None for global document)

        Returns:
            Root TreeNode for document if found, None otherwise
        """
        with self.node_repo.SessionLocal() as session:
            query = session.query(TreeNode).filter_by(parent_id=None)
            if document_id:
                query = query.filter_by(document_id=document_id)
            node = query.first()
            if node:
                session.expunge(node)
            return node

    def get_node_depth(self, node_id: str) -> int:
        """Calculate depth of a node (distance from root).

        Args:
            node_id: Node identifier

        Returns:
            Depth value (0 for root nodes, incrementing by 1 for each level down)

        Raises:
            ValueError: If node not found
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        depth = 0
        current_id = node.parent_id

        while current_id:
            depth += 1
            parent = self.node_repo.get_node(current_id)
            if not parent:
                break
            current_id = parent.parent_id

        return depth

    def is_leaf_node(self, node_id: str) -> bool:
        """Check if a node is a leaf (has no children).

        Args:
            node_id: Node identifier

        Returns:
            True if node is a leaf, False otherwise

        Raises:
            ValueError: If node not found
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        return not node.left_child_id and not node.right_child_id

    def is_root_node(self, node_id: str) -> bool:
        """Check if a node is a root (has no parent).

        Args:
            node_id: Node identifier

        Returns:
            True if node is a root, False otherwise

        Raises:
            ValueError: If node not found
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        return node.parent_id is None
