"""Document management service for RagZoom."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode

logger = logging.getLogger(__name__)


@dataclass
class DocumentInfo:
    """Document information with metadata."""

    document_id: str
    file_path: str | None
    indexed_at: datetime
    node_count: int


@dataclass
class SystemStatus:
    """System status information."""

    total_nodes: int
    leaf_nodes: int
    tree_depth: int


@dataclass
class NodeSnapshot:
    """Serialized view of a tree node for API responses."""

    node_id: str
    document_id: str | None
    parent_id: str | None
    left_child_id: str | None
    right_child_id: str | None
    span_start: int
    span_end: int
    text: str
    token_count: int
    height: int
    level_index: int
    preceding_neighbor_id: str | None
    following_neighbor_id: str | None
    is_pinned: bool
    created_at: datetime | None
    preceding_context_summary: str | None


@dataclass
class NodesPage:
    """Paginated result for span queries."""

    nodes: list[NodeSnapshot]
    total_matching: int


class DocumentService:
    """Service for document management operations."""

    def __init__(self, store: StorageBackend):
        """Initialize document service.

        Args:
            store: Store instance for data access
        """
        self.store = store

    def list_documents(self) -> list[DocumentInfo]:
        """List all indexed documents with metadata (backend-agnostic).

        Avoids exposing DB sessions by using Store APIs.
        """
        out: list[DocumentInfo] = []
        for doc in self.store.list_documents():
            ds = self.store.for_document(doc.id)
            # Use efficient count to avoid loading all nodes
            node_count = getattr(ds.nodes, "count", lambda: len(ds.nodes.get_all()))()
            out.append(
                DocumentInfo(
                    document_id=doc.id,
                    file_path=doc.file_path,
                    indexed_at=doc.indexed_at,
                    node_count=node_count,
                )
            )
        return out

    def get_system_status(self) -> SystemStatus:
        """Get system status information without exposing sessions."""
        total_nodes = 0
        leaf_nodes = 0
        tree_depth = 0
        for doc in self.store.list_documents():
            ds = self.store.for_document(doc.id)
            # Use repository-level aggregations when available
            total_nodes += ds.nodes.count()
            leaf_nodes += ds.nodes.leaf_count()
            tree_depth = max(tree_depth, ds.nodes.max_height())

        return SystemStatus(
            total_nodes=total_nodes,
            leaf_nodes=leaf_nodes,
            tree_depth=tree_depth,
        )

    def clear_document(self, document_id: str) -> int:
        """Clear all data for a specific document.

        Args:
            document_id: ID of document to clear

        Returns:
            Number of nodes deleted
        """
        return self.store.clear_document(document_id)

    def clear_all_documents(self) -> int:
        """Clear all documents and nodes in a backend-agnostic way."""
        total = 0
        for doc in self.store.list_documents():
            total += self.store.clear_document(doc.id)
        return total

    def get_nodes_in_span(
        self,
        document_id: str,
        span_start: int,
        span_end: int,
        *,
        limit: int,
        min_height: int | None = None,
    ) -> NodesPage:
        """Return ordered nodes overlapping the requested span."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        if span_end <= span_start:
            raise ValueError("span_end must be greater than span_start")

        store = self.store.for_document(document_id)
        nodes, total = store.get_nodes_in_span(
            span_start,
            span_end,
            limit=limit,
            min_height=min_height,
        )
        snapshots = [self._to_snapshot(node) for node in nodes]
        return NodesPage(nodes=snapshots, total_matching=total)

    def get_nodes_by_ids(
        self,
        document_id: str,
        node_ids: Sequence[str],
    ) -> list[NodeSnapshot]:
        """Return details for a specific set of node IDs."""
        if not node_ids:
            return []
        store = self.store.for_document(document_id)
        nodes = store.nodes.get_many(list(node_ids))
        return [self._to_snapshot(node) for node in nodes]

    @staticmethod
    def _to_snapshot(node: TreeNode) -> NodeSnapshot:
        """Convert a backend node into an API dataclass."""
        return NodeSnapshot(
            node_id=str(getattr(node, "id")),
            document_id=getattr(node, "document_id", None),
            parent_id=getattr(node, "parent_id", None),
            left_child_id=getattr(node, "left_child_id", None),
            right_child_id=getattr(node, "right_child_id", None),
            span_start=int(getattr(node, "span_start", 0)),
            span_end=int(getattr(node, "span_end", 0)),
            text=str(getattr(node, "text", "")),
            token_count=int(getattr(node, "token_count", 0)),
            height=int(getattr(node, "height", 0)),
            level_index=int(getattr(node, "level_index", 0)),
            preceding_neighbor_id=getattr(node, "preceding_neighbor_id", None),
            following_neighbor_id=getattr(node, "following_neighbor_id", None),
            is_pinned=bool(getattr(node, "is_pinned", 0)),
            created_at=getattr(node, "created_at", None),
            preceding_context_summary=getattr(node, "preceding_context_summary", None),
        )
