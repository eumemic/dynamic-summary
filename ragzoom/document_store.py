"""Document-scoped store that prevents cross-document contamination."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ragzoom.models import TreeNode
from ragzoom.repositories.node_repository import NodeRepository
from ragzoom.services.search_service import SearchService
from ragzoom.services.tree_navigator import TreeNavigator


class DocumentNodeRepository:
    """Node repository automatically scoped to a specific document."""

    def __init__(self, document_id: str | None, node_repo: NodeRepository):
        self.document_id = document_id
        self._repo = node_repo

    def add_node(
        self,
        node_id: str,
        text: str,
        embedding: list[float] | NDArray[np.float64],
        span_start: int,
        span_end: int,
        parent_id: str | None = None,
        left_child_id: str | None = None,
        right_child_id: str | None = None,
        token_count: int = 0,
        height: int = 0,
        is_left_child: bool | None = None,
    ) -> TreeNode:
        """Add a node scoped to this document."""
        return self._repo.add_node(
            node_id=node_id,
            text=text,
            embedding=embedding,
            span_start=span_start,
            span_end=span_end,
            parent_id=parent_id,
            left_child_id=left_child_id,
            right_child_id=right_child_id,
            document_id=self.document_id,
            token_count=token_count,
            height=height,
            is_left_child=is_left_child,
        )

    def add_batch(
        self, nodes_data: list[dict[str, Any]], *, session: Any = None
    ) -> list[TreeNode]:
        """Add multiple nodes to this document in batch."""
        # Ensure all nodes have the document_id set
        for node_data in nodes_data:
            node_data["document_id"] = self.document_id
        return self._repo.add_nodes_batch(nodes_data, session=session)

    def get(self, node_id: str) -> TreeNode | None:
        """Get a node by ID, ensuring it belongs to this document."""
        node = self._repo.get_node(node_id)
        if node and node.document_id == self.document_id:
            return node
        return None

    # Alias for compatibility with repository naming
    def get_node(self, node_id: str) -> TreeNode | None:
        """Alias of get() to match NodeRepository interface methods."""
        return self.get(node_id)

    # Backward-compatible alias to match NodeRepository interface
    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]:
        """Get multiple nodes by IDs, filtered to this document only."""
        nodes = self._repo.get_nodes(node_ids)
        return [node for node in nodes if node.document_id == self.document_id]

    def get_many(self, node_ids: list[str]) -> list[TreeNode]:
        """Get multiple nodes, filtering to this document only."""
        nodes = self._repo.get_nodes(node_ids)
        return [node for node in nodes if node.document_id == self.document_id]

    def get_all(self) -> list[TreeNode]:
        """Get all nodes for this document."""
        return self._repo.get_all_nodes_for_document(self.document_id)

    def get_all_paginated(self, *, page_size: int = 1000) -> list[list[TreeNode]]:
        """Get all nodes for this document in paginated batches."""
        return self._repo.get_all_nodes_for_document_paginated(
            self.document_id, page_size=page_size
        )

    def get_leaves(self) -> list[TreeNode]:
        """Get all leaf nodes for this document."""
        all_leaves = self._repo.get_leaf_nodes()
        return [node for node in all_leaves if node.document_id == self.document_id]

    def update_access(self, node_id: str) -> None:
        """Update access time for a node."""
        # First verify the node belongs to this document
        node = self.get(node_id)
        if node:
            self._repo.update_node_access(node_id)

    # Additional helper used by CoverageBuilder sibling logic
    def get_nodes_by_paths(self, paths: list[str]) -> list[TreeNode]:
        """Get nodes by path values, filtered to this document only."""
        nodes = self._repo.get_nodes_by_paths(paths)
        return [node for node in nodes if node.document_id == self.document_id]

    def update_parent_references_batch(
        self, updates: list[tuple[str, str]], *, session: Any = None
    ) -> None:
        """Update parent references for nodes in this document."""
        # Note: We trust that the caller is only updating nodes from this document
        # as this is typically called during tree construction where document consistency is maintained
        self._repo.update_parent_references_batch(updates, session=session)


class DocumentSearchService:
    """Search service automatically scoped to a specific document."""

    def __init__(self, document_id: str | None, search_service: SearchService):
        self.document_id = document_id
        self._service = search_service

    def similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar nodes within this document only."""
        where = {"document_id": self.document_id} if self.document_id else None
        return self._service.search_similar(query_embedding, n_results, where)

    def mmr_diverse(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        candidates: list[tuple[str, float, dict[str, Any]]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        """Apply MMR to get diverse results from candidates."""
        return self._service.compute_mmr_diverse_results(
            query_embedding, candidates, lambda_param, k
        )


class DocumentTreeNavigator:
    """Tree navigation automatically scoped to a specific document."""

    def __init__(self, document_id: str | None, tree_navigator: TreeNavigator):
        self.document_id = document_id
        self._navigator = tree_navigator

    def get_children(self, node_id: str) -> tuple[TreeNode | None, TreeNode | None]:
        """Get children of a node, verifying document scope."""
        # First verify the parent node belongs to this document
        parent = self._navigator.node_repo.get_node(node_id)
        if not parent or parent.document_id != self.document_id:
            return None, None

        return self._navigator.get_children(node_id)

    def get_ancestors(self, node_ids: list[str]) -> list[TreeNode]:
        """Get ancestors of nodes within this document."""
        # Filter input nodes to this document first
        valid_nodes = []
        for node_id in node_ids:
            node = self._navigator.node_repo.get_node(node_id)
            if node and node.document_id == self.document_id:
                valid_nodes.append(node_id)

        if not valid_nodes:
            return []

        ancestors = self._navigator.get_ancestors(valid_nodes)
        # Filter ancestors to this document (should already be the case, but defensive)
        return [node for node in ancestors if node.document_id == self.document_id]

    def get_root(self) -> TreeNode | None:
        """Get the root node for this document."""
        return self._navigator.get_root_node_for_document(self.document_id)

    def get_depth(self, node_id: str) -> int:
        """Get depth of a node, verifying it belongs to this document."""
        node = self._navigator.node_repo.get_node(node_id)
        if not node or node.document_id != self.document_id:
            raise ValueError(f"Node {node_id} not found in document {self.document_id}")

        return self._navigator.get_node_depth(node_id)

    def is_leaf(self, node_id: str) -> bool:
        """Check if node is a leaf, verifying document scope."""
        node = self._navigator.node_repo.get_node(node_id)
        if not node or node.document_id != self.document_id:
            return False

        return self._navigator.is_leaf_node(node_id)

    def is_root(self, node_id: str) -> bool:
        """Check if node is root, verifying document scope."""
        node = self._navigator.node_repo.get_node(node_id)
        if not node or node.document_id != self.document_id:
            return False

        return self._navigator.is_root_node(node_id)


class DocumentStore:
    """Store scoped to a single document - prevents cross-contamination."""

    def __init__(
        self,
        document_id: str | None,
        node_repo: NodeRepository,
        search_service: SearchService,
        tree_navigator: TreeNavigator,
    ):
        """Initialize document-scoped store.

        Args:
            document_id: Document ID to scope all operations to
            node_repo: Node repository to wrap
            search_service: Search service to wrap
            tree_navigator: Tree navigator to wrap
        """
        self.document_id = document_id

        # Create document-scoped wrappers
        self.nodes = DocumentNodeRepository(document_id, node_repo)
        self.search = DocumentSearchService(document_id, search_service)
        self.tree = DocumentTreeNavigator(document_id, tree_navigator)
