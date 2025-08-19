"""Repository for Document CRUD operations."""

import logging
from typing import TYPE_CHECKING

from ragzoom.models import Document, TreeNode
from ragzoom.services.cache_manager import CacheManager
from ragzoom.storage.database_manager import DatabaseManager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for Document database operations."""

    def __init__(
        self, database_manager: DatabaseManager, cache_manager: CacheManager[TreeNode]
    ):
        """Initialize document repository.

        Args:
            database_manager: Database manager for DB operations
            cache_manager: Cache manager for clearing cached nodes
        """
        self.db_manager = database_manager
        self.cache_manager = cache_manager
        self.SessionLocal = database_manager.SessionLocal

    def get_document_by_path(self, file_path: str) -> Document | None:
        """Get a document by file path.

        Args:
            file_path: Path to the document file

        Returns:
            Document if found, None otherwise
        """
        with self.SessionLocal() as session:
            return session.query(Document).filter_by(file_path=file_path).first()

    def get_document_by_id(self, document_id: str) -> Document | None:
        """Get a document by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document if found, None otherwise
        """
        with self.SessionLocal() as session:
            return session.query(Document).filter_by(id=document_id).first()

    def get_document_embedding_model(self, document_id: str) -> str | None:
        """Get the embedding model used for a specific document.

        Args:
            document_id: Document identifier

        Returns:
            Embedding model name if document found, None otherwise
        """
        doc = self.get_document_by_id(document_id)
        return doc.embedding_model if doc else None

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        content_hash: str,
        chunk_count: int,
        embedding_model: str,
        summary_model: str,
    ) -> Document:
        """Add a document record.

        Args:
            document_id: Unique identifier for the document
            file_path: Optional path to the source file
            content_hash: SHA256 hash of the document content
            chunk_count: Number of chunks in the document
            embedding_model: Name of the embedding model used for indexing
            summary_model: Name of the summarization model used

        Returns:
            Created Document

        Note: Model name validation is performed by the indexing layer
        to ensure they're valid OpenAI models before storage.
        """
        with self.SessionLocal() as session:
            doc = Document(
                id=document_id,
                file_path=file_path,
                content_hash=content_hash,
                chunk_count=chunk_count,
                embedding_model=embedding_model,
                summary_model=summary_model,
            )
            session.add(doc)
            session.commit()
            return doc

    def delete_document_nodes(self, document_id: str) -> int:
        """Delete all nodes associated with a document.

        Args:
            document_id: Document identifier

        Returns:
            Number of nodes deleted
        """
        with self.SessionLocal() as session:
            # Get all nodes for this document
            nodes = session.query(TreeNode).filter_by(document_id=document_id).all()
            node_ids = [n.id for n in nodes]

            # Delete from SQLite
            deleted_count = (
                session.query(TreeNode).filter_by(document_id=document_id).delete()
            )
            session.commit()

            # Delete from Chroma
            if node_ids:
                self.db_manager.collection.delete(ids=node_ids)

            # Clear from cache
            for node_id in node_ids:
                self.cache_manager.remove(node_id)

            return deleted_count

    def clear_document(self, document_id: str) -> int:
        """Clear all data for a document, including orphaned nodes and document record.

        This handles both complete documents and orphaned nodes from interrupted indexing.
        Unlike delete_document_nodes, this also removes the Document record.

        Args:
            document_id: ID of the document to clear

        Returns:
            Number of nodes deleted
        """
        # Delete all nodes with this document_id (handles orphaned nodes from interrupted runs)
        deleted_count = self.delete_document_nodes(document_id)

        # Also delete document record if it exists
        with self.SessionLocal() as session:
            session.query(Document).filter_by(id=document_id).delete()
            session.commit()

        return deleted_count

    def get_document_token_stats(self, document_id: str) -> dict[str, float | int]:
        """Get token statistics for a document using efficient SQL aggregation.

        Args:
            document_id: Document identifier

        Returns:
            Dict with keys: avg_tokens, min_tokens, max_tokens, total_tokens, node_count
        """
        with self.SessionLocal() as session:
            from sqlalchemy import func

            result = (
                session.query(
                    func.avg(TreeNode.token_count).label("avg_tokens"),
                    func.min(TreeNode.token_count).label("min_tokens"),
                    func.max(TreeNode.token_count).label("max_tokens"),
                    func.sum(TreeNode.token_count).label("total_tokens"),
                    func.count(TreeNode.id).label("node_count"),
                )
                .filter(
                    TreeNode.document_id == document_id,
                    TreeNode.token_count.isnot(None),
                )
                .one()
            )

            return {
                "avg_tokens": float(result.avg_tokens) if result.avg_tokens else 0.0,
                "min_tokens": result.min_tokens or 0,
                "max_tokens": result.max_tokens or 0,
                "total_tokens": result.total_tokens or 0,
                "node_count": result.node_count or 0,
            }
