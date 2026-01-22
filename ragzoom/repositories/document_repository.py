"""Repository for Document CRUD operations."""

import logging
from typing import TYPE_CHECKING, Optional

from ragzoom.models import Document, PostgresTreeNode
from ragzoom.repositories.base_repository import BaseRepository
from ragzoom.services.cache_manager import CacheManager
from ragzoom.storage.database_manager import DatabaseManager

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository):
    """Repository for Document database operations."""

    def __init__(
        self,
        database_manager: DatabaseManager,
        cache_manager: CacheManager[PostgresTreeNode],
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

    def get_document_is_temporal(self, document_id: str) -> bool | None:
        """Get the is_temporal flag for a document.

        Args:
            document_id: Document identifier

        Returns:
            True if document is temporal, False if not, None if document not found
        """
        doc = self.get_document_by_id(document_id)
        if doc is None:
            return None
        # Convert int (0/1) to bool
        return bool(doc.is_temporal)

    def set_document_is_temporal(self, document_id: str, *, is_temporal: bool) -> None:
        """Set the is_temporal flag for a document.

        Args:
            document_id: Document identifier
            is_temporal: Whether the document is temporal

        Raises:
            ValueError: If document does not exist
        """
        with self.SessionLocal() as session:
            doc = session.query(Document).filter_by(id=document_id).first()
            if doc is None:
                raise ValueError(f"Document not found: {document_id}")
            doc.is_temporal = 1 if is_temporal else 0
            session.commit()

    def list_documents(self) -> list[Document]:
        """Return all Document rows."""
        with self.SessionLocal() as session:
            return session.query(Document).all()

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        embedding_model: str,
        summary_model: str,
        *,
        session: Optional["Session"] = None,
    ) -> Document:
        """Add a document record.

        Args:
            document_id: Unique identifier for the document
            file_path: Optional path to the source file
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

    def delete_nodes_from_span(
        self, document_id: str, span_start: int, *, session: Optional["Session"] = None
    ) -> list[str]:
        """Delete all nodes whose span extends beyond the given position.

        Used for truncating a document after a conversation revert. Deletes any
        node where span_end > span_start, which includes:
        - Leaf nodes starting at or after the truncation point
        - Internal (summary) nodes whose span covers content beyond the point

        Args:
            document_id: Document identifier
            span_start: Truncation point - delete nodes where span_end > this value
            session: Optional session for transaction support

        Returns:
            List of deleted node IDs (for vector index cleanup)
        """
        with self._session_scope(session) as db_session:
            from sqlalchemy import text

            # Step 1: NULL out parent_id on kept children whose parents will be deleted.
            # This prevents FK violations where children point to deleted parents.
            # A kept child (span_end <= span_start) may have a parent that spans
            # beyond the truncation point (parent.span_end > span_start).
            db_session.execute(
                text(
                    "UPDATE tree_nodes SET parent_id = NULL "
                    "WHERE document_id = :document_id "
                    "AND span_end <= :span_start "
                    "AND parent_id IN ("
                    "    SELECT id FROM tree_nodes "
                    "    WHERE document_id = :document_id AND span_end > :span_start"
                    ")"
                ),
                {"document_id": document_id, "span_start": span_start},
            )

            # Step 2: NULL out following_neighbor_id on kept nodes whose neighbors
            # will be deleted. This prevents dangling neighbor references.
            db_session.execute(
                text(
                    "UPDATE tree_nodes SET following_neighbor_id = NULL "
                    "WHERE document_id = :document_id "
                    "AND span_end <= :span_start "
                    "AND following_neighbor_id IN ("
                    "    SELECT id FROM tree_nodes "
                    "    WHERE document_id = :document_id AND span_end > :span_start"
                    ")"
                ),
                {"document_id": document_id, "span_start": span_start},
            )

            # Step 3: Delete nodes whose span extends beyond the truncation point.
            # This catches both leaves starting after the point AND internal
            # nodes that summarize content beyond the point.
            result = db_session.execute(
                text(
                    "DELETE FROM tree_nodes "
                    "WHERE document_id = :document_id AND span_end > :span_start "
                    "RETURNING id"
                ),
                {"document_id": document_id, "span_start": span_start},
            )

            deleted_node_ids = [str(row[0]) for row in result]

            if deleted_node_ids:
                self.cache_manager.remove_batch(deleted_node_ids)

            return deleted_node_ids

    def get_document_token_stats(self, document_id: str) -> dict[str, float | int]:
        """Get token statistics for a document using efficient SQL aggregation.

        Args:
            document_id: Document identifier

        Returns:
            Dict with keys: avg_tokens, min_tokens, max_tokens, total_tokens, node_count

        Note:
            SQL aggregate functions (MIN, MAX, AVG, SUM) return NULL for empty result
            sets. This is legitimate for documents with no nodes. We convert NULL to 0
            as a reasonable convention for empty-set statistics.
        """
        with self.SessionLocal() as session:
            from sqlalchemy import func

            result = (
                session.query(
                    func.avg(PostgresTreeNode.token_count).label("avg_tokens"),
                    func.min(PostgresTreeNode.token_count).label("min_tokens"),
                    func.max(PostgresTreeNode.token_count).label("max_tokens"),
                    func.sum(PostgresTreeNode.token_count).label("total_tokens"),
                    func.count(PostgresTreeNode.id).label("node_count"),
                )
                .filter(
                    PostgresTreeNode.document_id == document_id,
                    PostgresTreeNode.token_count.isnot(None),
                )
                .one()
            )

            # SQL aggregates return NULL for empty sets (documented above).
            # COUNT() never returns NULL, so no fallback needed.
            return {
                "avg_tokens": (
                    float(result.avg_tokens) if result.avg_tokens is not None else 0.0
                ),
                "min_tokens": result.min_tokens if result.min_tokens is not None else 0,
                "max_tokens": result.max_tokens if result.max_tokens is not None else 0,
                "total_tokens": (
                    result.total_tokens if result.total_tokens is not None else 0
                ),
                "node_count": result.node_count,
            }
