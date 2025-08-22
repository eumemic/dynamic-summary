"""Service for tree navigation and traversal operations."""

import logging
from typing import TYPE_CHECKING

from ragzoom.exceptions import NodeNotFoundError
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
        """Get all ancestors of given nodes using path-based traversal for instant lookup.

        Args:
            node_ids: List of node identifiers

        Returns:
            List of ancestor TreeNodes
        """
        from ragzoom.utils.path_utils import get_all_ancestor_paths

        # First get the nodes to access their paths
        nodes = self.node_repo.get_nodes(node_ids)

        # Check if all nodes have valid path support (path must exist and be meaningful)
        has_path_support = False
        if nodes:
            # All nodes must have path attribute that's not None
            all_have_path = all(
                hasattr(node, "path") and node.path is not None for node in nodes
            )
            if all_have_path:
                # Paths must be meaningful (not all empty strings or root nodes)
                paths = [node.path for node in nodes]
                unique_paths = set(paths)
                # Valid if we have more than one unique path OR at least one non-root path
                has_path_support = len(unique_paths) > 1 or any(
                    path != "" for path in paths
                )

        if has_path_support:
            # Use optimized path-based logic
            ancestor_paths = set()
            for node in nodes:
                ancestor_paths.update(get_all_ancestor_paths(node.path))

            # Single batch fetch of all ancestors by their paths
            if ancestor_paths:
                return self.node_repo.get_nodes_by_paths(list(ancestor_paths))
        else:
            # Fallback to traditional parent traversal for backward compatibility
            ancestors_set = set()
            for node_id in node_ids:
                current = self.node_repo.get_node(node_id)
                while current and current.parent_id:
                    parent = self.node_repo.get_node(current.parent_id)
                    if parent:
                        ancestors_set.add(parent.id)
                        current = parent
                    else:
                        break

            # Return full node objects for ancestors
            if ancestors_set:
                ancestors = []
                for aid in ancestors_set:
                    if aid:
                        ancestor = self.node_repo.get_node(aid)
                        if ancestor:
                            ancestors.append(ancestor)
                return ancestors

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
            NodeNotFoundError: If node not found
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            raise NodeNotFoundError(node_id)

        # Use path field for instant depth calculation if available and meaningful
        # Only use path-based logic if the node has a path that's not None
        # and either it's non-empty OR it's empty but node has no parent (true root)
        if (
            hasattr(node, "path")
            and node.path is not None
            and (node.path != "" or node.parent_id is None)
        ):
            from ragzoom.utils.path_utils import get_depth

            return get_depth(node.path)

        # Fallback to traversal-based calculation for backward compatibility
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
            False if the node does not exist, True if it's a leaf, False if it has children
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            return False

        return not node.left_child_id and not node.right_child_id

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

        return node.parent_id is None

    def get_parent_node(self, node_id: str) -> TreeNode | None:
        """Get the parent node using path-based lookup when possible.

        Args:
            node_id: Node identifier

        Returns:
            Parent TreeNode or None if this is root or node not found
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            return None

        # Use path field for instant parent lookup if available
        if hasattr(node, "path") and node.path is not None:
            from ragzoom.utils.path_utils import get_parent_path

            parent_path = get_parent_path(node.path)
            # For root node (path=""), get_parent_path returns "", but root has no parent
            if node.path == "":  # Root node
                return None
            if parent_path is not None:
                parents = self.node_repo.get_nodes_by_paths([parent_path])
                return parents[0] if parents else None
            return None

        # Fallback to parent_id lookup
        if node.parent_id:
            return self.node_repo.get_node(node.parent_id)
        return None

    def get_sibling_node(self, node_id: str) -> TreeNode | None:
        """Get the sibling node using path-based lookup when possible.

        Args:
            node_id: Node identifier

        Returns:
            Sibling TreeNode or None if no sibling or node not found
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            return None

        # Use path field for instant sibling lookup if available
        if hasattr(node, "path") and node.path is not None:
            from ragzoom.utils.path_utils import get_sibling_path

            sibling_path = get_sibling_path(node.path)
            if sibling_path is not None:  # Root has no sibling
                siblings = self.node_repo.get_nodes_by_paths([sibling_path])
                return siblings[0] if siblings else None
            return None

        # Fallback to parent-child traversal
        if not node.parent_id:
            return None  # Root has no sibling

        parent = self.node_repo.get_node(node.parent_id)
        if not parent:
            return None

        # Return the other child
        if parent.left_child_id == node_id and parent.right_child_id:
            return self.node_repo.get_node(parent.right_child_id)
        elif parent.right_child_id == node_id and parent.left_child_id:
            return self.node_repo.get_node(parent.left_child_id)

        return None

    def is_left_child(self, node_id: str) -> bool:
        """Check if node is a left child of its parent.

        Args:
            node_id: Node identifier

        Returns:
            True if node is a left child, False if right child or root
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            return False

        # Use path field for instant determination if available
        if hasattr(node, "path") and node.path is not None and node.path:
            return node.path[-1] == "0"

        # Fallback to parent lookup
        if not node.parent_id:
            return False  # Root is neither left nor right

        parent = self.node_repo.get_node(node.parent_id)
        return parent is not None and parent.left_child_id == node_id

    def is_right_child(self, node_id: str) -> bool:
        """Check if node is a right child of its parent.

        Args:
            node_id: Node identifier

        Returns:
            True if node is a right child, False if left child or root
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            return False

        # Use path field for instant determination if available
        if hasattr(node, "path") and node.path is not None and node.path:
            return node.path[-1] == "1"

        # Fallback to parent lookup
        if not node.parent_id:
            return False  # Root is neither left nor right

        parent = self.node_repo.get_node(node.parent_id)
        return parent is not None and parent.right_child_id == node_id
