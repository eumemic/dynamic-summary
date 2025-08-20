"""Repository for Document CRUD operations."""

import logging
from typing import TYPE_CHECKING, Optional

from ragzoom.models import Document, TreeNode
from ragzoom.repositories.base_repository import BaseRepository
from ragzoom.services.cache_manager import CacheManager
from ragzoom.storage.database_manager import DatabaseManager

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository):
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
        *,
        session: Optional["Session"] = None,
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
        with self._session_scope(session) as db_session:
            doc = Document(
                id=document_id,
                file_path=file_path,
                content_hash=content_hash,
                chunk_count=chunk_count,
                embedding_model=embedding_model,
                summary_model=summary_model,
            )
            db_session.add(doc)
            return doc

    # jscpd:ignore-start - Similar pattern but different operation from clear_document
    def delete_document_nodes(
        self, document_id: str, *, session: Optional["Session"] = None
    ) -> int:
        """Delete all nodes associated with a document.

        This optimized version avoids loading all nodes into memory by using
        SQL RETURNING to get node IDs only for cache invalidation.

        Args:
            document_id: Document identifier
            session: Optional session for transaction support

        Returns:
            Number of nodes deleted
        """
        # jscpd:ignore-end
        with self._session_scope(session) as db_session:
            # Use a subquery to get node IDs for cache clearing without loading full objects
            # This is much more memory-efficient for large node counts
            from sqlalchemy import text

            # Delete with RETURNING to get node IDs for cache invalidation
            # This avoids loading all nodes into memory first
            result = db_session.execute(
                text(
                    "DELETE FROM tree_nodes WHERE document_id = :document_id RETURNING id"
                ),
                {"document_id": document_id},
            )

            # Get deleted node IDs for cache clearing
            deleted_node_ids = [row[0] for row in result]
            deleted_count = len(deleted_node_ids)

            # Clear from cache efficiently using batch removal
            # This avoids O(n²) performance from individual removals
            if deleted_count > 1000:
                logger.debug(f"Batch clearing {deleted_count} nodes from cache...")
            self.cache_manager.remove_batch(deleted_node_ids)

            return deleted_count

    def clear_document(
        self, document_id: str, *, session: Optional["Session"] = None
    ) -> int:
        """Clear all data for a document, including orphaned nodes and document record.

        This handles both complete documents and orphaned nodes from interrupted indexing.
        Unlike delete_document_nodes, this also removes the Document record.

        Args:
            document_id: ID of the document to clear
            session: Optional session for atomic operations

        Returns:
            Number of nodes deleted
        """
        with self._session_scope(session) as db_session:
            # Delete all nodes with this document_id (handles orphaned nodes from interrupted runs)
            deleted_count = self.delete_document_nodes(document_id, session=db_session)

            # Also delete document record if it exists
            db_session.query(Document).filter_by(id=document_id).delete()

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
