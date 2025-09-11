"""Service for tree navigation and traversal operations."""

import logging
from typing import TYPE_CHECKING

from ragzoom.exceptions import NodeNotFoundError
from ragzoom.models import PostgresTreeNode
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

    def get_children(
        self, node_id: str
    ) -> tuple[PostgresTreeNode | None, PostgresTreeNode | None]:
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

    def get_ancestors(self, node_ids: list[str]) -> list[PostgresTreeNode]:
        """Get all ancestors of given nodes using path-based traversal for instant lookup.

        Args:
            node_ids: List of node identifiers

        Returns:
            List of ancestor TreeNodes
        """
        from ragzoom.utils.path_utils import get_all_ancestor_paths

        # Get the nodes to access their paths
        nodes = self.node_repo.get_nodes(node_ids)
        if not nodes:
            return []

        # Use path-based logic to find all ancestors
        ancestor_paths = set()
        for node in nodes:
            ancestor_paths.update(get_all_ancestor_paths(node.path))

        # Single batch fetch of all ancestors by their paths
        if ancestor_paths:
            return self.node_repo.get_nodes_by_paths(list(ancestor_paths))

        return []

    def get_root_node(self) -> PostgresTreeNode | None:
        """Get the root node (node with no parent).

        Returns:
            Root TreeNode if found, None otherwise
        """
        with self.node_repo.SessionLocal() as session:
            node = session.query(PostgresTreeNode).filter_by(parent_id=None).first()
            if node:
                session.expunge(node)
            return node

    def get_root_node_for_document(
        self, document_id: str | None
    ) -> PostgresTreeNode | None:
        """Get the root node for a specific document.

        Args:
            document_id: Document identifier (None for global document)

        Returns:
            Root TreeNode for document if found, None otherwise
        """
        with self.node_repo.SessionLocal() as session:
            query = session.query(PostgresTreeNode).filter_by(parent_id=None)
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
            NodeNotFoundError: If node not found
        """
        from ragzoom.utils.path_utils import get_depth

        node = self.node_repo.get_node(node_id)
        if not node:
            raise NodeNotFoundError(node_id)

        return get_depth(node.path)

    def is_leaf_node(self, node_id: str) -> bool:
        """Check if a node is a leaf (has no children).

        Args:
            node_id: Node identifier

        Returns:
            False if the node does not exist, True if it's a leaf, False if it has children
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            return False

        # Prefer structural check to support multiple model types
        left = getattr(node, "left_child_id", None)
        right = getattr(node, "right_child_id", None)
        return not left and not right

    def is_root_node(self, node_id: str) -> bool:
        """Check if a node is a root (has no parent).

        Args:
            node_id: Node identifier

        Returns:
            False if the node does not exist, True if it's a root, False if it has a parent
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            return False

        # Prefer structural check to support multiple model types
        return getattr(node, "parent_id", None) is None

    def get_parent_node(self, node_id: str) -> PostgresTreeNode | None:
        """Get the parent node using path-based lookup.

        Args:
            node_id: Node identifier

        Returns:
            Parent TreeNode or None if this is root or node not found
        """
        from ragzoom.utils.path_utils import get_parent_path

        node = self.node_repo.get_node(node_id)
        if not node:
            return None

        parent_path = get_parent_path(node.path)
        # For root node (path=""), get_parent_path returns "", but root has no parent
        if node.path == "":  # Root node
            return None

        parents = self.node_repo.get_nodes_by_paths([parent_path])
        return parents[0] if parents else None

    def get_sibling_node(self, node_id: str) -> PostgresTreeNode | None:
        """Get the sibling node using path-based lookup.

        Args:
            node_id: Node identifier

        Returns:
            Sibling TreeNode or None if no sibling or node not found
        """
        from ragzoom.utils.path_utils import get_sibling_path

        node = self.node_repo.get_node(node_id)
        if not node:
            return None

        sibling_path = get_sibling_path(node.path)
        if sibling_path is not None:  # Root has no sibling
            siblings = self.node_repo.get_nodes_by_paths([sibling_path])
            return siblings[0] if siblings else None
        return None

    def is_left_child(self, node_id: str) -> bool:
        """Check if node is a left child of its parent.

        Args:
            node_id: Node identifier

        Returns:
            True if node is a left child, False if right child or root
        """
        node = self.node_repo.get_node(node_id)
        if not node or not node.path:
            return False

        return node.path[-1] == "0"

    def is_right_child(self, node_id: str) -> bool:
        """Check if node is a right child of its parent.

        Args:
            node_id: Node identifier

        Returns:
            True if node is a right child, False if left child or root
        """
        node = self.node_repo.get_node(node_id)
        if not node or not node.path:
            return False

        return node.path[-1] == "1"
