"""Tree building and indexing functionality for RagZoom."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import NDArray

from ragzoom.config import IndexConfig, SecretStr
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.dataflow import build_tree_dataflow
from ragzoom.dataflow.core import ProcessingStrategy
from ragzoom.document_store import DocumentStore
from ragzoom.progress import AsyncProgressWrapper, GlobalProgressTracker
from ragzoom.services.llm_service import LLMService
from ragzoom.splitter import TextSplitter
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.telemetry_types import TelemetryDataDict
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


@dataclass
class DocumentPreparationResult:
    """Result from document preparation phase.

    Replaces the confusing 3-tuple return from _prepare_document with clear semantics.
    """

    document_id: str
    content_hash: str
    skip_indexing: bool
    existing_doc_id: str | None = None


@dataclass
class IndexingContext:
    """Context for document indexing operations.

    Groups related parameters to reduce parameter list complexity.
    """

    document_id: str
    content_hash: str
    file_path: str | None
    async_progress: AsyncProgressWrapper | None
    overall_start_time: float
    show_progress: bool
    reporter: TelemetryCollector | None


class TreeBuilder:
    """Tree builder with concurrent processing."""

    def __init__(
        self,
        config: IndexConfig,
        document_store: DocumentStore,
        vector_index: VectorIndex,
        api_key: str | SecretStr = "",
        max_concurrent: int = 30,
    ):
        """Initialize tree builder.

        Args:
            config: Index configuration
            document_store: DocumentStore instance for persistence within a single document
            api_key: OpenAI API key as SecretStr or string (if not provided, reads from OPENAI_API_KEY env)
            max_concurrent: Maximum concurrent API requests
        """
        self.config = config
        self.document_store = document_store
        self.splitter = TextSplitter(config)
        self.vector_index = vector_index
        # Convert string to SecretStr for security
        if isinstance(api_key, str) and not isinstance(api_key, SecretStr):
            api_key = SecretStr(api_key) if api_key else SecretStr("")
        self.llm_service = LLMService(config, api_key, max_concurrent)

        # Backward compatibility: provide access to centralized tokenizer
        self.tokenizer = tokenizer

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        return str(uuid.uuid4())

    def _validate_model_names(self) -> None:
        """Validate that configured model names are in known lists.

        This is a lightweight check that doesn't make API calls.
        Unknown models will log a warning but proceed (to support new models).
        """
        # Known valid embedding models
        valid_embedding_models = {
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        }
        if self.config.embedding_model not in valid_embedding_models:
            logger.warning(
                f"Embedding model '{self.config.embedding_model}' not in known list. "
                f"Will attempt to use it anyway. Known models: {valid_embedding_models}"
            )

        # Known valid summary models
        valid_summary_models = {
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-5",
        }
        if self.config.summary_model not in valid_summary_models:
            logger.warning(
                f"Summary model '{self.config.summary_model}' not in known list. "
                f"Will attempt to use it anyway. Known models: {valid_summary_models}"
            )

    def _calculate_target_tokens(self, text: str) -> int:
        """Calculate target tokens as min of leaf_tokens or half the text size."""
        tokens = tokenizer.encode(text)
        half_size = len(tokens) // 2
        return min(self.config.target_chunk_tokens, half_size)

    async def _summarize_text(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        *,
        prev_context: str | None = None,
        parent_id: str | None = None,
        reporter: TelemetryCollector | None = None,
        left_token_count: int | None = None,
        right_token_count: int | None = None,
    ) -> tuple[str, int, int]:
        """Delegate to LLMService for text summarization."""
        return await self.llm_service._summarize_text(
            left_text,
            right_text,
            target_tokens,
            parent_id=parent_id,
            reporter=reporter,
            prev_context=prev_context,
            left_token_count=left_token_count,
            right_token_count=right_token_count,
        )

    def _update_parent_reference(self, node_id: str, parent_id: str) -> None:
        """Update a node's parent reference."""
        # Use DocumentStore's proper interface instead of direct database access
        self.document_store.update_parent_reference(node_id, parent_id)

    def _create_and_validate_chunks(
        self, text: str, show_progress: bool = True
    ) -> list[str]:
        """Create chunks from text and validate their sizes.

        Args:
            text: The text to split into chunks
            show_progress: Whether to show progress logs

        Returns:
            List of text chunks
        """
        # Split into chunks
        chunks = self.splitter.split_text(text)

        # Log only when progress bar is not active to avoid display issues
        if not show_progress:
            logger.info("Splitting document into chunks...")
            logger.info(f"Split document into {len(chunks)} chunks")

        # Early validation: Check chunk sizes immediately after splitting
        from ragzoom.validate import validate, validate_chunk_sizes

        # Create simple objects with just the fields needed for validation
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_obj = type("ChunkObj", (), {"text": chunk, "id": f"chunk_{i}"})()
            chunk_objects.append(chunk_obj)

        validate(
            lambda: validate_chunk_sizes(
                chunk_objects, self.config.target_chunk_tokens
            ),
            "early chunk size validation",
        )

        return chunks

    def _setup_progress_tracking(
        self, chunk_count: int, show_progress: bool = True
    ) -> tuple[GlobalProgressTracker | None, AsyncProgressWrapper | None]:
        """Setup progress tracking for document indexing.

        Args:
            chunk_count: Number of chunks to process
            show_progress: Whether to show progress bar

        Returns:
            tuple: (progress_tracker, async_progress_wrapper)
        """
        # Create progress tracker early so we can use it for logging
        # Respect global progress configuration to fully suppress bars in tests
        from ragzoom.progress import get_progress_config

        global_cfg = get_progress_config()
        effective_show = show_progress and not global_cfg.disable_bars

        progress = (
            GlobalProgressTracker(
                chunk_count,
                effective_show,
                embedding_batch_size=self.config.embedding_batch_size,
            )
            if effective_show
            else None
        )

        # Create async wrapper for progress (tracker already created above)
        async_progress = AsyncProgressWrapper(progress) if progress else None

        return progress, async_progress

    @overload
    async def _add_document_impl(
        self,
        text: str,
        show_progress: bool = True,
        reporter: None = None,
    ) -> str: ...

    @overload
    async def _add_document_impl(
        self,
        text: str,
        show_progress: bool = True,
        reporter: TelemetryCollector = ...,  # jscpd:ignore-start
    ) -> tuple[str, TelemetryDataDict]: ...  # jscpd:ignore-end

    async def _add_document_impl(
        self,
        text: str,
        show_progress: bool = True,
        reporter: TelemetryCollector | None = None,
    ) -> str | tuple[str, TelemetryDataDict]:
        """Add a document to the tree using dataflow parallelism.

        Returns:
            If reporter is None: document_id
            If reporter is provided: (document_id, metrics)
        """
        # Step 1: Validate models and get document ID from DocumentStore
        self._validate_model_names()
        document_id = self.document_store.document_id
        if not document_id:
            raise ValueError("DocumentStore must have a document_id set")

        # Step 2: Create and validate chunks
        chunks = self._create_and_validate_chunks(text, show_progress)

        # Step 3: Setup progress tracking
        progress, async_progress = self._setup_progress_tracking(
            len(chunks), show_progress
        )

        # Track overall start time
        overall_start_time = time.time()

        try:
            # Step 4: Build complete tree using dataflow
            # This creates all nodes (leaves + internal) and generates all embeddings
            tree_nodes = await build_tree_dataflow(
                chunks=chunks,
                document_id=document_id,
                llm_service=self.llm_service,
                target_tokens=self.config.target_chunk_tokens,
                max_summary_concurrency=30,  # Use default max_concurrent value
                max_embedding_concurrency=10,  # Reasonable default
                embedding_batch_size=self.config.embedding_batch_size,
                processing_strategy=ProcessingStrategy(self.config.processing_strategy),
                reporter=reporter,
                progress=async_progress,
            )

            # Store all nodes using the document store
            doc_store = self.document_store

            # Group nodes by height and insert level by level to respect foreign key constraints
            # Each batch insert is a single SQL statement, and parents must exist before children
            from itertools import groupby

            sorted_nodes = sorted(tree_nodes, key=lambda n: n.height)
            for height, nodes_at_height in groupby(
                sorted_nodes, key=lambda n: n.height
            ):
                nodes_list = list(nodes_at_height)  # Consume the iterator
                logger.debug(f"Inserting {len(nodes_list)} nodes at height {height}")
                # Prepare node data for this height level
                nodes_data: list[
                    dict[
                        str,
                        str
                        | int
                        | float
                        | bool
                        | list[float]
                        | NDArray[np.float64]
                        | None,
                    ]
                ] = []
                for node in nodes_list:
                    node_data: dict[
                        str,
                        str
                        | int
                        | float
                        | bool
                        | list[float]
                        | NDArray[np.float64]
                        | None,
                    ] = {
                        "node_id": node.id,
                        "text": node.text,
                        "document_id": node.document_id,
                        "span_start": node.span_start,
                        "span_end": node.span_end,
                        "parent_id": None,  # All nodes inserted with NULL parent initially
                        "left_child_id": node.left_child_id,
                        "right_child_id": node.right_child_id,
                        "preceding_neighbor_id": node.preceding_neighbor_id,
                        "following_neighbor_id": node.following_neighbor_id,
                        # Embeddings are not stored in SQL; kept separate for VectorIndex
                        "token_count": node.token_count,
                        "height": node.height,
                        "path": node.path,
                    }
                    nodes_data.append(node_data)

                # Insert all nodes at this height level
                doc_store.nodes.add_batch(nodes_data)

            # Now update ALL parent references after all nodes exist
            parent_updates = []
            for node in tree_nodes:
                if (
                    node.parent_id
                ):  # Update parent reference for any node that has a parent
                    parent_updates.append((node.id, node.parent_id))
            if parent_updates:
                doc_store.nodes.update_parent_references_batch(parent_updates)

            # Step 6: Find root node ID
            root_node = max(tree_nodes, key=lambda n: n.height)
            root_id = root_node.id

            # Progress is already tracked within dataflow, no need for final update

            # Final completion logging with total elapsed time
            if root_id:
                total_elapsed = time.time() - overall_start_time
                mins, secs = divmod(int(total_elapsed), 60)
                if not show_progress:
                    logger.info(
                        f"Document indexed successfully: {document_id} [{mins}m {secs}s total elapsed]"
                    )

            # Two-phase apply for backends with external vector index (e.g., SQLite + PythonVectorIndex):
            # After all SQL writes are complete and parent references set, upsert vectors.
            # Upsert vectors into the configured VectorIndex (required dependency)
            upsert_items: list[
                tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
            ] = []
            from typing import cast as _cast

            for n in tree_nodes:
                meta = {
                    "span_start": int(n.span_start),
                    "span_end": int(n.span_end),
                    "parent_id": n.parent_id or "",
                    "document_id": n.document_id or "",
                    "is_leaf": 1 if int(getattr(n, "height", 0)) == 0 else 0,
                }
                emb = getattr(n, "embedding", None)
                if emb is not None:
                    upsert_items.append((n.id, _cast(list[float], emb), meta))
            if upsert_items:
                self.vector_index.upsert(upsert_items)

            # Finalize telemetry if collector was used
            if reporter:
                telemetry: TelemetryDataDict = reporter.finalize()
                return document_id, telemetry

            return document_id
        finally:
            # Always close progress if it exists
            if progress:
                progress.close()

    def add_document(
        self,
        text: str,
        show_progress: bool = True,
    ) -> str:
        """Sync wrapper for add_document."""
        return asyncio.run(self.add_document_async(text, show_progress))

    def add_document_with_telemetry(
        self,
        text: str,
        show_progress: bool = False,
    ) -> tuple[str, TelemetryDataDict]:
        """Add document and return telemetry data. Used for benchmarking.

        This is a convenience method that creates a TelemetryCollector internally
        and returns the collected telemetry data. For production use, add_document() is preferred
        as it doesn't have the overhead of telemetry collection.

        The dual-method pattern ensures:
        - Normal indexing (add_document) has zero telemetry overhead
        - Benchmarking gets detailed telemetry without modifying core logic
        - Internal implementation (_add_document_impl) remains flexible

        Returns:
            Tuple of (document_id, telemetry_dict)
        """
        # Create collector internally with config for pricing
        source_tokens = tokenizer.count_tokens(text)
        collector = TelemetryCollector(
            self.document_store.document_id or "benchmark",
            source_tokens,
            self.config,
            document_path=None,
        )

        # Run indexing with collector - will return (doc_id, telemetry)
        result = asyncio.run(self._add_document_impl(text, show_progress, collector))

        # Extract tuple returned when collector is provided
        # Type checker knows result is a tuple because we passed a collector
        doc_id, telemetry = result
        return doc_id, telemetry

    async def add_document_async(
        self,
        text: str,
        show_progress: bool = True,
    ) -> str:
        """Async version of add_document - called by sync wrapper."""
        result = await self._add_document_impl(text, show_progress)
        # Type checker knows result is a string when no reporter is provided
        return result

    async def _process_node_pair(
        self,
        left_id: str,
        left_text: str,
        right_id: str | None,
        right_text: str | None,
        prev_context: str | None,
        document_id: str | None,
        current_height: int,  # Tree height for this level (children height + 1)
        reporter: TelemetryCollector | None = None,
        left_node: TreeNode | None = None,  # Pre-fetched node data
        right_node: TreeNode | None = None,  # Pre-fetched node data
        doc_store: DocumentStore | None = None,
    ) -> dict[str, object]:
        """Process a single node pair - generate summary and embedding.

        Returns:
            Dictionary containing node data and parent updates to be applied later
        """
        if doc_store is None:
            doc_store = self.document_store

        parent_id = self._generate_node_id()

        # Use pre-fetched nodes if provided, otherwise fetch them
        if left_node is None:
            left_node = doc_store.nodes.get(left_id)
        if right_id and right_node is None:
            right_node = doc_store.nodes.get(right_id)

        if not left_node:
            logger.error(f"Failed to retrieve left child node: {left_id}")
            raise ValueError("Left child node not found in store")
        if right_id and not right_node:
            logger.error(f"Failed to retrieve right child node: {right_id}")
            raise ValueError("Right child node not found in store")

        # Track parent node creation with span from children
        if reporter:
            if right_node:
                parent_span = (left_node.span_start, right_node.span_end)
            else:
                parent_span = (left_node.span_start, left_node.span_end)
            reporter.track_node_created(
                node_id=parent_id,
                height=reporter._current_height + 1,  # Parent is one level higher
                span=parent_span,
            )

        # Use consistent token budget for all heights
        # Target tokens for the summary (guidance for LLM, not hard limit)
        target_tokens = self.config.target_chunk_tokens

        # Generate summary (async) with retry mechanism support
        summary, retry_count, token_count = await self.llm_service._summarize_text(
            left_text,
            right_text or "",  # Pass empty string if no right text
            target_tokens,
            parent_id=parent_id,
            reporter=reporter,
            prev_context=prev_context,
            left_token_count=left_node.token_count,
            right_token_count=right_node.token_count if right_node else 0,
        )

        # Derive parent path from left child path
        from ragzoom.utils.path_utils import get_parent_path

        parent_path = get_parent_path(left_node.path)

        # Embedding will be generated in batch after all summaries are collected
        # This avoids 183 individual API calls for a typical level

        # Return data to be stored later in batch
        return {
            "node_data": {
                "node_id": parent_id,
                "text": summary,
                "embedding": None,  # Will be filled in after batch generation
                "span_start": left_node.span_start,
                "span_end": right_node.span_end if right_node else left_node.span_end,
                "left_child_id": left_id,
                "right_child_id": right_id,  # Can be None
                "document_id": document_id,
                "token_count": token_count,
                "height": current_height,  # Store pre-calculated height
                "path": parent_path,  # Binary path derived from child paths
            },
            "parent_updates": [
                (left_id, parent_id),
                (right_id, parent_id) if right_id else None,
            ],
            "parent_id": parent_id,
            "summary": summary,
            "token_count": token_count,  # Pass token count for telemetry
            # Store validation data for later
            "validation_data": {
                "left_span_start": left_node.span_start,
                "right_span_end": right_node.span_end if right_node else None,
            },
        }
