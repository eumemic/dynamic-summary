"""Indexing service for RagZoom document processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ragzoom.config import IndexConfig, OperationalConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.index import AppendStats, TreeBuilder
from ragzoom.telemetry_types import TelemetryDataDict

if TYPE_CHECKING:
    from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.vector_factory import create_vector_index

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result from document indexing operation."""

    document_id: str
    chunks_created: int
    tree_depth: int
    mutated_nodes: int | None = None
    resummarized_nodes: int | None = None
    new_leaves: int | None = None
    telemetry: TelemetryDataDict | None = None
    telemetry_run_id: str | None = None


class IndexingService:
    """Service for document indexing operations."""

    def __init__(
        self,
        store: StorageBackend,
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

    def _finalize_result(
        self,
        document_id: str,
        telemetry: TelemetryDataDict | None,
        *,
        mutated_nodes: int | None = None,
        resummarized_nodes: int | None = None,
        new_leaves: int | None = None,
    ) -> IndexingResult:
        doc_store_final = self.store.for_document(document_id)
        leaves = doc_store_final.nodes.get_leaves()
        root = doc_store_final.tree.get_root()
        tree_height = root.height if root else 0

        total_leaves = len(leaves)

        if mutated_nodes is None:
            mutated_nodes = doc_store_final.nodes.count()
        if resummarized_nodes is None:
            resummarized_nodes = max(mutated_nodes - total_leaves, 0)
        if new_leaves is None:
            new_leaves = total_leaves

        return IndexingResult(
            document_id=document_id,
            chunks_created=total_leaves,
            tree_depth=tree_height,
            mutated_nodes=mutated_nodes,
            resummarized_nodes=resummarized_nodes,
            new_leaves=new_leaves,
            telemetry=telemetry,
        )

    def _from_append_result(self, append_result: AppendStats) -> IndexingResult:
        """Convert append stats into a finalized indexing result."""

        return self._finalize_result(
            append_result.document_id,
            append_result.telemetry,
            mutated_nodes=append_result.mutated_nodes,
            resummarized_nodes=append_result.resummarized_nodes,
            new_leaves=append_result.new_leaves,
        )

    def _create_tree_builder(self, document_id: str) -> TreeBuilder:
        document_store = self.store.for_document(document_id)
        vector_index = create_vector_index(
            self.operational_config.vector_backend,
            self.operational_config.database_url,
            self.index_config.embedding_model,
        )
        return TreeBuilder(
            self.index_config,
            document_store,
            api_key=self.operational_config.openai_api_key.get_secret_value(),
            max_concurrent=30,
            vector_index=vector_index,
        )

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

        # Acquire per-document lock when supported by backend
        from contextlib import AbstractContextManager, nullcontext

        _lock_fn = getattr(self.store, "lock_document", None)
        cm_any = _lock_fn(document_id) if callable(_lock_fn) else None
        # Some test doubles (unconfigured Mocks) may return non-context managers; coerce to a valid one
        if not (hasattr(cm_any, "__enter__") and hasattr(cm_any, "__exit__")):
            lock_cm = cast(AbstractContextManager[object], nullcontext())
        else:
            lock_cm = cast(AbstractContextManager[object], cm_any)

        with lock_cm:
            # Clear existing data for the document
            deleted_count = self.store.clear_document(document_id)
            if deleted_count > 0:
                logger.info(
                    f"Cleared existing data for '{document_id}' ({deleted_count} nodes)"
                )
            # Also clear vectors for this document to avoid stale candidates
            try:
                vector_index_for_clear = create_vector_index(
                    self.operational_config.vector_backend,
                    self.operational_config.database_url,
                    self.index_config.embedding_model,
                )
                _ = vector_index_for_clear.delete(filter={"document_id": document_id})
            except Exception:
                # Non-fatal; retrieval path defensively filters any stale vectors
                pass

            # Create document with full metadata BEFORE indexing
            # This ensures the document exists with proper metadata before TreeBuilder runs
            self.store.add_document(
                document_id=document_id,
                file_path=file_path,
                embedding_model=self.index_config.embedding_model,
                summary_model=self.index_config.summary_model,
            )

            # Create document-scoped store and TreeBuilder
            tree_builder = self._create_tree_builder(document_id)

            reporter: TelemetryCollector | None = None
            if collect_telemetry:
                from ragzoom.telemetry_collection import TelemetryCollector

                reporter = TelemetryCollector(
                    document_id,
                    tree_builder.tokenizer.count_tokens(text),
                    self.index_config,
                    document_path=file_path,
                )

            append_result = await tree_builder.append_text_async(
                text,
                show_progress=show_progress,
                reporter=reporter,
            )

            return self._from_append_result(append_result)

    def append_to_document(
        self,
        document_id: str,
        new_text: str,
        show_progress: bool = False,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Sync wrapper for append_to_document_async."""

        import asyncio

        return asyncio.run(
            self.append_to_document_async(
                document_id=document_id,
                new_text=new_text,
                show_progress=show_progress,
                collect_telemetry=collect_telemetry,
            )
        )

    async def append_to_document_async(
        self,
        document_id: str,
        new_text: str,
        show_progress: bool = False,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Append new text to an existing document incrementally."""

        if not document_id:
            raise ValueError("document_id is required for append")

        from contextlib import AbstractContextManager, nullcontext

        _lock_fn = getattr(self.store, "lock_document", None)
        cm_any = _lock_fn(document_id) if callable(_lock_fn) else None
        if not (hasattr(cm_any, "__enter__") and hasattr(cm_any, "__exit__")):
            lock_cm = cast(AbstractContextManager[object], nullcontext())
        else:
            lock_cm = cast(AbstractContextManager[object], cm_any)

        with lock_cm:
            doc_record = self.store.get_document_by_id(document_id)
            if doc_record is None:
                self.store.add_document(
                    document_id=document_id,
                    file_path=None,
                    embedding_model=self.index_config.embedding_model,
                    summary_model=self.index_config.summary_model,
                )

            tree_builder = self._create_tree_builder(document_id)

            reporter = None
            if collect_telemetry:
                from ragzoom.telemetry_collection import TelemetryCollector

                token_estimate = tree_builder.tokenizer.count_tokens(new_text)
                reporter = TelemetryCollector(
                    document_id,
                    token_estimate,
                    self.index_config,
                    document_path=None,
                )

            append_result = await tree_builder.append_text_async(
                new_text,
                show_progress=show_progress,
                reporter=reporter,
            )

            return self._from_append_result(append_result)
