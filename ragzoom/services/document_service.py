"""Document management service for RagZoom."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.repositories.node_repository import NodeRepository
from ragzoom.store import StoreManager

logger = logging.getLogger(__name__)


@dataclass
class DocumentInfo:
    """Document information with metadata."""

    document_id: str
    file_path: str | None
    indexed_at: datetime
    chunk_count: int
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

    def __init__(self, store: StoreManager | StorageBackend):
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
        repo = cast("HasNodeRepo", self.store).node_repo
        for doc in self.store.list_documents():
            # Compute node count via repository
            node_count = len(repo.get_all_nodes_for_document(doc.id))
            out.append(
                DocumentInfo(
                    document_id=doc.id,
                    file_path=doc.file_path,
                    indexed_at=doc.indexed_at,
                    chunk_count=doc.chunk_count,
                    node_count=node_count,
                )
            )
        return out

    def get_system_status(self) -> SystemStatus:
        """Get system status information without exposing sessions."""
        repo = cast("HasNodeRepo", self.store).node_repo
        # Total nodes across all documents
        all_nodes = len(repo.get_all_nodes_for_document(None))
        # Leaf nodes across all documents
        leaf_nodes = len(repo.get_leaf_nodes())
        # Tree depth: derive from max height among all nodes
        nodes_all = repo.get_all_nodes_for_document(None)
        tree_depth = max((n.height for n in nodes_all), default=0)

        pinned = repo.get_pinned_nodes(None)

        return SystemStatus(
            total_nodes=all_nodes,
            leaf_nodes=leaf_nodes,
            tree_depth=tree_depth,
            pinned_nodes=len(pinned),
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
        repo = cast("HasNodeRepo", self.store).node_repo
        repo.pin_node(node_id)


class HasNodeRepo(Protocol):
    node_repo: NodeRepository
