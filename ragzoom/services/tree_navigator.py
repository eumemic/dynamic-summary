"""Service for tree navigation and traversal operations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ragzoom.contracts.node_repository import NodeRepository as NodeRepositoryProtocol
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.exceptions import NodeNotFoundError
from ragzoom.tree_coordinate import TreeCoordinate


def _node_to_coordinate(node: TreeNode) -> TreeCoordinate | None:
    """Extract TreeCoordinate from a node, or None if level_index is missing."""
    raw_index = getattr(node, "level_index", None)
    if raw_index is None:
        return None
    return TreeCoordinate(
        document_id=getattr(node, "document_id", None),
        height=int(getattr(node, "height", 0)),
        level_index=int(raw_index),
    )


class TreeNavigator:
    """Service for tree navigation and traversal operations."""

    def __init__(self, node_repository: NodeRepositoryProtocol):
        """Initialize tree navigator.

        Args:
            node_repository: Node repository for data access
        """
        self.node_repo = node_repository
        self._depth_cache: dict[str, int] = {}

    def clear_depth_cache(self, node_ids: list[str]) -> None:
        """Invalidate cached depths for the given node IDs."""

        for node_id in node_ids:
            self._depth_cache.pop(node_id, None)

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

    def _coordinate_for(self, node: TreeNode) -> TreeCoordinate | None:
        return _node_to_coordinate(node)

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

        initial = self._batched_fetch(node_ids)
        if not initial:
            return []

        visited: set[str] = set(initial.keys())
        ancestors: list[TreeNode] = []

        coords_by_doc: defaultdict[str | None, set[TreeCoordinate]] = defaultdict(set)
        pending_coords: defaultdict[str | None, set[tuple[int, int]]] = defaultdict(set)

        for node in initial.values():
            coord = _node_to_coordinate(node)
            if coord is None:
                continue
            parent_coord = coord.parent()
            coords_by_doc[parent_coord.document_id].add(parent_coord)

        while coords_by_doc:
            next_coords: defaultdict[str | None, set[TreeCoordinate]] = defaultdict(set)

            for doc_id, coords in coords_by_doc.items():
                tuples = []
                seen_tuples = pending_coords[doc_id]
                for coord in coords:
                    key = coord.as_tuple()
                    if key in seen_tuples or coord.level_index < 0:
                        continue
                    seen_tuples.add(key)
                    tuples.append(key)

                if not tuples:
                    continue

                fetched = self.node_repo.get_nodes_by_height_levels(doc_id, tuples)
                for node in fetched:
                    if node.id in visited:
                        continue
                    ancestors.append(node)
                    visited.add(node.id)

                    coord = _node_to_coordinate(node)
                    if coord is None:
                        continue
                    next_coords[coord.parent().document_id].add(coord.parent())

            coords_by_doc = next_coords

        return ancestors

    def get_sibling_node(self, node_id: str) -> TreeNode | None:
        """Return the structural sibling of ``node_id`` if it exists."""

        node = self._get_node(node_id)
        if node is None:
            return None
        coord = self._coordinate_for(node)
        if coord is not None:
            try:
                sibling_coord = coord.sibling()
            except ValueError:
                sibling_coord = None
            if sibling_coord is not None:
                fetched = self.node_repo.get_nodes_by_height_levels(
                    sibling_coord.document_id, [sibling_coord.as_tuple()]
                )
                for sibling in fetched:
                    if sibling.id != node.id:
                        return sibling

        parent = self._get_parent(node)
        if parent is None:
            return None
        if parent.left_child_id == node.id:
            sibling_id = parent.right_child_id
        elif parent.right_child_id == node.id:
            sibling_id = parent.left_child_id
        else:
            sibling_id = None
        if not sibling_id:
            return None
        siblings = self._batched_fetch([sibling_id])
        return siblings.get(sibling_id)

    def get_neighbor_node(self, node_id: str, offset: int) -> TreeNode | None:
        """Return a neighbor ``offset`` positions away on the same height."""

        node = self._get_node(node_id)
        if node is None:
            return None
        coord = self._coordinate_for(node)
        if coord is not None:
            try:
                neighbor_coord = coord.neighbor(offset)
            except ValueError:
                neighbor_coord = None
            if neighbor_coord is not None:
                fetched = self.node_repo.get_nodes_by_height_levels(
                    neighbor_coord.document_id, [neighbor_coord.as_tuple()]
                )
                if fetched:
                    return fetched[0]

        neighbor_attr = (
            "preceding_neighbor_id" if offset < 0 else "following_neighbor_id"
        )
        neighbor_id = getattr(node, neighbor_attr, None)
        if not neighbor_id:
            return None
        neighbors = self._batched_fetch([neighbor_id])
        return neighbors.get(neighbor_id)

    def get_preceding_neighbor(self, node_id: str) -> TreeNode | None:
        """Return the immediate preceding neighbor at the same height."""

        return self.get_neighbor_node(node_id, -1)

    def get_following_neighbor(self, node_id: str) -> TreeNode | None:
        """Return the immediate following neighbor at the same height."""

        return self.get_neighbor_node(node_id, 1)

    def get_root_node(self) -> TreeNode | None:
        """Get the root node (node with no parent).

        Returns:
            Root TreeNode if found, None otherwise
        """
        return next(self.node_repo.iter_root_nodes_for_document(None), None)

    def get_root_node_for_document(self, document_id: str | None) -> TreeNode | None:
        """Get the root node for a specific document.

        Args:
            document_id: Document identifier (None for global document)

        Returns:
            Root TreeNode for document if found, None otherwise
        """
        return next(self.node_repo.iter_root_nodes_for_document(document_id), None)

    def get_node_depth(self, node_id: str) -> int:
        """Calculate depth of a node (distance from root).

        Args:
            node_id: Node identifier

        Returns:
            Depth value (0 for root nodes, incrementing by 1 for each level down)

        Raises:
            NodeNotFoundError: If node not found
        """
        cached_depth = self._depth_cache.get(node_id)
        if cached_depth is not None:
            return cached_depth

        node = self._get_node(node_id)
        if not node:
            raise NodeNotFoundError(node_id)

        depth_attr = getattr(node, "depth", None)
        if depth_attr is not None:
            self._depth_cache[node_id] = int(depth_attr)
            return int(depth_attr)

        trail: list[TreeNode] = []
        current = node
        base_depth = 0

        while True:
            cached = getattr(current, "depth", None)
            if cached is not None:
                base_depth = int(cached)
                self._depth_cache[current.id] = base_depth
                break

            parent = self._get_parent(current)
            if parent is None:
                base_depth = 0
                setattr(current, "depth", 0)
                self._depth_cache[current.id] = 0
                break

            trail.append(current)
            current = parent

        depth = base_depth
        for descendant in reversed(trail):
            depth += 1
            setattr(descendant, "depth", depth)
            self._depth_cache[descendant.id] = depth

        result = depth if trail else base_depth
        self._depth_cache[node_id] = result
        return result

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
