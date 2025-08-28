"""Indexing service for RagZoom document processing."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragzoom.config import IndexConfig, OperationalConfig
from ragzoom.index import TreeBuilder
from ragzoom.store import Store, TreeNode

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result from document indexing operation."""

    document_id: str
    chunks_created: int
    tree_depth: int
    telemetry: dict[str, Any] | None = None


class IndexingService:
    """Service for document indexing operations."""

    def __init__(
        self,
        store: Store,
        index_config: IndexConfig,
        operational_config: OperationalConfig,
    ):
        """Initialize indexing service.

        Args:
            store: Store instance for data access
            index_config: Configuration for indexing
            operational_config: Operational configuration
        """
        self.store = store
        self.index_config = index_config
        self.operational_config = operational_config
        # TreeBuilder will be created per-request with a DocumentStore

    # jscpd:ignore-start - Legitimate sync/async and method pattern duplication
    def index_document(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Index a document from text.

        Args:
            text: Document text to index
            document_id: Optional document ID (defaults to filename if file_path provided)
            file_path: Optional file path for metadata
            show_progress: Whether to show progress bar
            collect_telemetry: Whether to collect telemetry data

        Returns:
            IndexingResult with document stats and optional telemetry
        """
        # Generate document ID if not provided
        if not document_id:
            if file_path:
                document_id = Path(file_path).name
            else:
                raise ValueError("Either document_id or file_path must be provided")

        # Check if document exists and needs re-indexing
        existing_doc = self.store.get_document_by_path(file_path) if file_path else None
        content_hash = self.store.compute_content_hash(text)

        if existing_doc:
            if existing_doc.content_hash == content_hash:
                logger.info(f"Document at {file_path} unchanged, skipping re-indexing")
                return IndexingResult(
                    document_id=existing_doc.id,
                    chunks_created=existing_doc.chunk_count or 0,
                    tree_depth=0,  # We'd need to query for this
                    telemetry=None,
                )
            else:
                logger.info(f"Document at {file_path} has changed, re-indexing...")
                document_id = existing_doc.id

        # Clear existing data for the document
        deleted_count = self.store.clear_document(document_id)
        if deleted_count > 0:
            logger.info(
                f"Cleared existing data for '{document_id}' ({deleted_count} nodes)"
            )

        # Create document with full metadata BEFORE indexing
        # This ensures the document exists with proper metadata before TreeBuilder runs
        self.store.add_document(
            document_id=document_id,
            file_path=file_path,
            content_hash=content_hash,
            chunk_count=0,  # Will be updated after indexing
            embedding_model=self.index_config.embedding_model,
            summary_model=self.index_config.summary_model,
        )

        # Create document-scoped store and TreeBuilder
        document_store = self.store.for_document(document_id)

        tree_builder = TreeBuilder(
            self.index_config,
            document_store,
            api_key=self.operational_config.openai_api_key.get_secret_value(),
            max_concurrent=30,
        )

        # Index with or without telemetry
        if collect_telemetry:
            doc_id, telemetry = tree_builder.add_document_with_telemetry(
                text,
                document_id=document_id,
                show_progress=show_progress,
            )
        else:
            doc_id = tree_builder.add_document(
                text,
                document_id=document_id,
                show_progress=show_progress,
            )
            telemetry = None

        # Get document statistics and update metadata
        with self.store.SessionLocal() as session:
            # Get leaf nodes for this specific document
            doc_leaves = (
                session.query(TreeNode)
                .filter_by(document_id=doc_id)
                .filter(
                    TreeNode.left_child_id.is_(None),
                    TreeNode.right_child_id.is_(None),
                )
                .all()
            )

            # Get root node for this document
            root = (
                session.query(TreeNode)
                .filter_by(document_id=doc_id, parent_id=None)
                .first()
            )

            # Update the document's chunk count now that indexing is complete
            from ragzoom.models import Document

            doc = session.query(Document).filter_by(id=document_id).first()
            if doc:
                doc.chunk_count = len(doc_leaves)
                session.commit()

        tree_height = root.height if root else 0

        return IndexingResult(
            document_id=doc_id,
            chunks_created=len(doc_leaves),
            tree_depth=tree_height,
            telemetry=telemetry,
        )

    def index_from_file(
        self,
        file_path: str,
        document_id: str | None = None,
        show_progress: bool = True,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Index a document from file.

        Args:
            file_path: Path to file to index
            document_id: Optional document ID (defaults to filename)
            show_progress: Whether to show progress bar
            collect_telemetry: Whether to collect telemetry data

        Returns:
            IndexingResult with document stats and optional telemetry

        Raises:
            OSError: If file cannot be read
        """
        # Read file
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")

        # Use filename as document ID if not provided
        if not document_id:
            document_id = path.name

        return self.index_document(
            text,
            document_id=document_id,
            file_path=str(path.absolute()),
            show_progress=show_progress,
            collect_telemetry=collect_telemetry,
        )

    async def index_document_async(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = False,  # Default False for async
    ) -> IndexingResult:
        """Index a document asynchronously.

        Args:
            text: Document text to index
            document_id: Optional document ID
            file_path: Optional file path for metadata
            show_progress: Whether to show progress bar

        Returns:
            IndexingResult with document stats
        """
        # Generate document ID if not provided
        if not document_id:
            if file_path:
                document_id = Path(file_path).name
            else:
                raise ValueError("Either document_id or file_path must be provided")

        # Clear existing data for the document
        deleted_count = self.store.clear_document(document_id)
        if deleted_count > 0:
            logger.info(
                f"Cleared existing data for '{document_id}' ({deleted_count} nodes)"
            )

        # Create document-scoped store and TreeBuilder
        document_store = self.store.for_document(document_id)

        tree_builder = TreeBuilder(
            self.index_config,
            document_store,
            api_key=self.operational_config.openai_api_key.get_secret_value(),
            max_concurrent=30,
        )

        # Index document
        doc_id = await tree_builder.add_document_async(
            text,
            document_id=document_id,
            show_progress=show_progress,
        )

        # Calculate content hash for metadata
        content_hash = self.store.compute_content_hash(text)

        # Get document statistics
        with self.store.SessionLocal() as session:
            # Get leaf nodes for this specific document
            doc_leaves = (
                session.query(TreeNode)
                .filter_by(document_id=doc_id)
                .filter(
                    TreeNode.left_child_id.is_(None),
                    TreeNode.right_child_id.is_(None),
                )
                .all()
            )

            # Get root node for this document
            root = (
                session.query(TreeNode)
                .filter_by(document_id=doc_id, parent_id=None)
                .first()
            )

        tree_height = root.height if root else 0

        # Set complete document metadata after indexing
        document_store.set_metadata(
            file_path=file_path,
            content_hash=content_hash,
            chunk_count=len(doc_leaves),
            embedding_model=self.index_config.embedding_model,
            summary_model=self.index_config.summary_model,
        )

        return IndexingResult(
            document_id=doc_id,
            chunks_created=len(doc_leaves),
            tree_depth=tree_height,
        )

    # jscpd:ignore-end
