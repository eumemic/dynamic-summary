"""Service for tree navigation and traversal operations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from ragzoom.contracts.node_repository import NodeRepository as NodeRepositoryProtocol
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.exceptions import NodeNotFoundError


class TreeNavigator:
    """Service for tree navigation and traversal operations."""

    def __init__(self, node_repository: NodeRepositoryProtocol):
        """Initialize tree navigator.

        Args:
            node_repository: Node repository for data access
        """
        self.node_repo = node_repository

    def _get_node(self, node_id: str) -> TreeNode | None:
        """Fetch a node with cache awareness."""

        return self.node_repo.get_node(node_id)

    def _get_parent(self, node: TreeNode) -> TreeNode | None:
        parent_id = getattr(node, "parent_id", None)
        if not parent_id:
            return None
        try:
            return self.node_repo.get_node(parent_id)
        except Exception:  # pragma: no cover - repository handles errors
            return None

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

        child_ids = [
            child_id
            for child_id in (node.left_child_id, node.right_child_id)
            if child_id
        ]
        fetched = self._batched_fetch(child_ids)
        left = fetched.get(node.left_child_id) if node.left_child_id else None
        right = fetched.get(node.right_child_id) if node.right_child_id else None
        return left, right

    def _batched_fetch(self, node_ids: Iterable[str]) -> dict[str, TreeNode]:
        """Fetch nodes in batches and return an id -> node mapping."""

        ids = list(node_ids)
        if not ids:
            return {}
        fetched = self.node_repo.get_nodes(ids)
        return {node.id: node for node in fetched}

    def get_ancestors(self, node_ids: list[str]) -> list[TreeNode]:
        """Return ancestors for the supplied nodes using structural traversal."""

        current = self._batched_fetch(node_ids)
        if not current:
            return []

        seen: set[str] = set(current.keys())
        ancestors: list[TreeNode] = []
        frontier: set[str] = {
            cast(str, node.parent_id)
            for node in current.values()
            if getattr(node, "parent_id", None)
        }

        while frontier:
            parents = self._batched_fetch(frontier)
            if not parents:
                break

            frontier = set()
            for parent in parents.values():
                if parent.id in seen:
                    continue
                ancestors.append(parent)
                seen.add(parent.id)
                parent_key = getattr(parent, "parent_id", None)
                if parent_key:
                    frontier.add(cast(str, parent_key))

        return ancestors

    def get_root_node(self) -> TreeNode | None:
        """Get the root node (node with no parent).

        Returns:
            Root TreeNode if found, None otherwise
        """
        roots = self.node_repo.get_root_nodes()
        return roots[0] if roots else None

    def get_root_node_for_document(self, document_id: str | None) -> TreeNode | None:
        """Get the root node for a specific document.

        Args:
            document_id: Document identifier (None for global document)

        Returns:
            Root TreeNode for document if found, None otherwise
        """
        roots = self.node_repo.get_root_nodes(document_id)
        return roots[0] if roots else None

    def get_node_depth(self, node_id: str) -> int:
        """Calculate depth of a node (distance from root).

        Args:
            node_id: Node identifier

        Returns:
            Depth value (0 for root nodes, incrementing by 1 for each level down)

        Raises:
            NodeNotFoundError: If node not found
        """
        node = self._get_node(node_id)
        if not node:
            raise NodeNotFoundError(node_id)

        ancestors = self.get_ancestors([node_id])
        return len(ancestors)

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

    def get_parent_node(self, node_id: str) -> TreeNode | None:
        """Return the parent node using structural relationships.

        Args:
            node_id: Node identifier

        Returns:
            Parent TreeNode or None if this is root or node not found
        """
        node = self._get_node(node_id)
        if not node:
            return None

        return self._get_parent(node)

    def get_sibling_node(self, node_id: str) -> TreeNode | None:
        """Return the sibling node using the shared parent reference.

        Args:
            node_id: Node identifier

        Returns:
            Sibling TreeNode or None if no sibling or node not found
        """
        node = self._get_node(node_id)
        if not node:
            return None
        parent = self._get_parent(node)
        if parent is None:
            return None

        child_ids = [
            child_id
            for child_id in (parent.left_child_id, parent.right_child_id)
            if child_id
        ]
        if not child_ids:
            return None

        children = self._batched_fetch(child_ids)

        if parent.left_child_id == node.id:
            sibling_id = parent.right_child_id
        elif parent.right_child_id == node.id:
            sibling_id = parent.left_child_id
        else:
            sibling_id = None

        if not sibling_id:
            return None

        return children.get(sibling_id)

    def is_left_child(self, node_id: str) -> bool:
        """Check if node is the left child of its parent.

        Args:
            node_id: Node identifier

        Returns:
            True if node is a left child, False if right child or root
        """
        node = self._get_node(node_id)
        if not node:
            return False
        parent = self._get_parent(node)
        if parent is None:
            return False

        return parent.left_child_id == node.id

    def is_right_child(self, node_id: str) -> bool:
        """Check if node is the right child of its parent.

        Args:
            node_id: Node identifier

        Returns:
            True if node is a right child, False if left child or root
        """
        node = self._get_node(node_id)
        if not node:
            return False
        parent = self._get_parent(node)
        if parent is None:
            return False

        return parent.right_child_id == node.id
