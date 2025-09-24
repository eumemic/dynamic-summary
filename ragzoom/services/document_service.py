"""Document management service for RagZoom."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ragzoom.contracts.node_repository import NodeRepository as NodeRepositoryProtocol
from ragzoom.contracts.storage_backend import StorageBackend

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
    pinned_nodes: int


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
        pinned_nodes = 0
        for doc in self.store.list_documents():
            ds = self.store.for_document(doc.id)
            # Use repository-level aggregations when available
            total_nodes += ds.nodes.count()
            leaf_nodes += ds.nodes.leaf_count()
            tree_depth = max(tree_depth, ds.nodes.max_height())
            pinned_nodes += ds.nodes.pinned_count()

        return SystemStatus(
            total_nodes=total_nodes,
            leaf_nodes=leaf_nodes,
            tree_depth=tree_depth,
            pinned_nodes=pinned_nodes,
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

    def pin_node(self, node_id: str) -> None:
        """Pin a node to always include it.

        Args:
            node_id: ID of node to pin

        Raises:
            NodeNotFoundError: If node doesn't exist
            InvalidOperationError: If node cannot be pinned
        """
        for doc in self.store.list_documents():
            ds = self.store.for_document(doc.id)
            if ds.nodes.get_node(node_id):
                ds._node_repo.pin_node(node_id)
                return
        raise ValueError(f"Node {node_id} not found")


class HasNodeRepo(Protocol):
    node_repo: NodeRepositoryProtocol
