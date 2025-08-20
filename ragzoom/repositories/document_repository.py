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

    def _get_session(self, session=None):
        """Get session for database operations.

        Args:
            session: Optional existing session to use

        Returns:
            Tuple of (session, should_commit) where should_commit indicates
            if this method should handle commit/rollback
        """
        if session is not None:
            return session, False  # Don't commit - caller manages lifecycle
        else:
            return self.SessionLocal(), True  # We manage lifecycle

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
        *,
        session=None,
    ) -> Document:
        """Add a document record.

        Args:
            document_id: Unique identifier for the document
            file_path: Optional path to the source file
            content_hash: SHA256 hash of the document content
            chunk_count: Number of chunks in the document
            embedding_model: Name of the embedding model used for indexing
            summary_model: Name of the summarization model used
            session: Optional database session for transactional operations

        Returns:
            Created Document

        Note: Model name validation is performed by the indexing layer
        to ensure they're valid OpenAI models before storage.
        """
        db_session, should_commit = self._get_session(session)
        try:
            doc = Document(
                id=document_id,
                file_path=file_path,
                content_hash=content_hash,
                chunk_count=chunk_count,
                embedding_model=embedding_model,
                summary_model=summary_model,
            )
            db_session.add(doc)
            if should_commit:
                db_session.commit()
            return doc
        finally:
            if should_commit:
                db_session.close()

    def delete_document_nodes(self, document_id: str, *, session=None) -> int:
        """Delete all nodes associated with a document.

        Args:
            document_id: Document identifier
            session: Optional database session for transactional operations

        Returns:
            Number of nodes deleted
        """
        db_session, should_commit = self._get_session(session)
        try:
            # Get all nodes for this document
            nodes = db_session.query(TreeNode).filter_by(document_id=document_id).all()
            node_ids = [n.id for n in nodes]

            # Delete from PostgreSQL (embeddings are stored in the same table now)
            deleted_count = (
                db_session.query(TreeNode).filter_by(document_id=document_id).delete()
            )
            if should_commit:
                db_session.commit()

            # Clear from cache
            for node_id in node_ids:
                self.cache_manager.remove(node_id)

            return deleted_count
        finally:
            if should_commit:
                db_session.close()

    def clear_document(self, document_id: str, *, session=None) -> int:
        """Clear all data for a document, including orphaned nodes and document record.

        This handles both complete documents and orphaned nodes from interrupted indexing.
        Unlike delete_document_nodes, this also removes the Document record.

        Args:
            document_id: ID of the document to clear
            session: Optional database session for transactional operations

        Returns:
            Number of nodes deleted
        """
        db_session, should_commit = self._get_session(session)
        try:
            # Delete all nodes with this document_id (handles orphaned nodes from interrupted runs)
            deleted_count = self.delete_document_nodes(document_id, session=db_session)

            # Also delete document record if it exists
            db_session.query(Document).filter_by(id=document_id).delete()
            if should_commit:
                db_session.commit()

            return deleted_count
        finally:
            if should_commit:
                db_session.close()

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
