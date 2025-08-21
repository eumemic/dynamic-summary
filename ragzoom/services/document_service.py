"""Document management service for RagZoom."""

import logging
from dataclasses import dataclass
from datetime import datetime

from ragzoom.store import Document, Store, TreeNode

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

    def __init__(self, store: Store):
        """Initialize document service.

        Args:
            store: Store instance for data access
        """
        self.store = store

    def list_documents(self) -> list[DocumentInfo]:
        """List all indexed documents with metadata.

        Returns:
            List of DocumentInfo objects with document metadata
        """
        documents = []

        with self.store.SessionLocal() as session:
            docs = session.query(Document).all()

            for doc in docs:
                # Get node count for this document
                node_count = (
                    session.query(TreeNode).filter_by(document_id=doc.id).count()
                )

                documents.append(
                    DocumentInfo(
                        document_id=doc.id,
                        file_path=doc.file_path,
                        indexed_at=doc.indexed_at,
                        chunk_count=doc.chunk_count,
                        node_count=node_count,
                    )
                )

        return documents

    def get_system_status(self) -> SystemStatus:
        """Get system status information.

        Returns:
            SystemStatus with node counts and tree information
        """
        with self.store.SessionLocal() as session:
            all_nodes = session.query(TreeNode).count()

        leaf_nodes = self.store.get_leaf_nodes()
        root = self.store.get_root_node()
        pinned = self.store.get_pinned_nodes()

        return SystemStatus(
            total_nodes=all_nodes,
            leaf_nodes=len(leaf_nodes),
            tree_depth=root.height if root else 0,
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
        """Clear all documents and nodes from the database.

        Returns:
            Number of nodes deleted
        """
        with self.store.SessionLocal() as session:
            # Count nodes before deletion
            deleted_count = session.query(TreeNode).count()

            # Delete all nodes and documents
            session.query(TreeNode).delete()
            session.query(Document).delete()
            session.commit()

        # Clear the cache
        self.store.node_cache.clear()
        self.store.cache_order.clear()

        return deleted_count

    def pin_node(self, node_id: str) -> None:
        """Pin a node to always include it.

        Args:
            node_id: ID of node to pin

        Raises:
            NodeNotFoundError: If node doesn't exist
            InvalidOperationError: If node cannot be pinned
        """
        self.store.pin_node(node_id)
