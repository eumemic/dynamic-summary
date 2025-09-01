"""Indexing service for RagZoom document processing."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ragzoom.telemetry_types import TelemetryDataDict

from ragzoom.config import IndexConfig, OperationalConfig
from ragzoom.index import TreeBuilder
from ragzoom.models import TreeNode
from ragzoom.store import Store

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result from document indexing operation."""

    document_id: str
    chunks_created: int
    tree_depth: int
    telemetry: Optional["TelemetryDataDict"] = None


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
        import asyncio

        # Simply delegate to async version using asyncio.run
        # This ensures both sync and async paths are ALWAYS identical
        return asyncio.run(
            self.index_document_async(
                text=text,
                document_id=document_id,
                file_path=file_path,
                show_progress=show_progress,
                collect_telemetry=collect_telemetry,
            )
        )

    # jscpd:ignore-start - Legitimate sync wrapper pattern
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

    # jscpd:ignore-end

    async def index_document_async(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = False,  # Default False for async
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Index a document asynchronously.

        Args:
            text: Document text to index
            document_id: Optional document ID
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

        # Compute content hash for metadata
        content_hash = self.store.compute_content_hash(text)

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
            # TreeBuilder's add_document_with_telemetry is sync only, so we need to run it in executor
            import asyncio
            from functools import partial

            func = partial(
                tree_builder.add_document_with_telemetry,
                text,
                show_progress=show_progress,
            )
            loop = asyncio.get_event_loop()
            doc_id, telemetry = await loop.run_in_executor(None, func)
        else:
            doc_id = await tree_builder.add_document_async(
                text,
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
