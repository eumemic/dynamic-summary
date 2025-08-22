"""Tree building and indexing functionality for RagZoom."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast, overload

from ragzoom.config import IndexConfig
from ragzoom.document_store import DocumentStore
from ragzoom.progress import AsyncProgressWrapper, GlobalProgressTracker
from ragzoom.services.llm_service import LLMService
from ragzoom.splitter import TextSplitter
from ragzoom.store import StoreManager, TreeNode
from ragzoom.telemetry_collection import TelemetryCollector
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
        store: StoreManager,
        api_key: str = "",
        max_concurrent: int = 30,
    ):
        """Initialize tree builder.

        Args:
            config: Index configuration
            store: StoreManager instance for persistence
            api_key: OpenAI API key (if not provided, reads from OPENAI_API_KEY env)
            max_concurrent: Maximum concurrent API requests
        """
        self.config = config
        self.store = store
        self.splitter = TextSplitter(config)
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
        with self.store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            node = session.query(TreeNode).filter_by(id=node_id).first()
            if node:
                node.parent_id = parent_id
                session.commit()

                # Invalidate the cache entry for this node since we've updated it
                if node_id in self.store.node_cache:
                    del self.store.node_cache[node_id]
                    if node_id in self.store.cache_order:
                        self.store.cache_order.remove(node_id)

    def _prepare_document(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
    ) -> DocumentPreparationResult:
        """Prepare document for indexing: validate models, check existence, determine ID.

        Returns:
            DocumentPreparationResult with clear semantics for next steps
        """
        # Validate model names to warn about potential issues
        self._validate_model_names()

        # Compute content hash
        content_hash = self.store.compute_content_hash(text)

        # Check if document already exists
        existing_doc = None
        if file_path:
            existing_doc = self.store.get_document_by_path(file_path)
            if existing_doc:
                # Check if content changed
                if existing_doc.content_hash == content_hash:
                    logger.info(
                        f"Document at {file_path} unchanged, skipping re-indexing"
                    )
                    return DocumentPreparationResult(
                        document_id=existing_doc.id,
                        content_hash=content_hash,
                        skip_indexing=True,
                        existing_doc_id=existing_doc.id,
                    )
                else:
                    logger.info(f"Document at {file_path} has changed, re-indexing...")
                    # Delete old nodes
                    deleted = self.store.clear_document(existing_doc.id)
                    logger.info(f"Deleted {deleted} old nodes")
                    document_id = existing_doc.id

        # Determine final document ID
        if not document_id:
            if file_path:
                # Use filename (without path) as document_id
                from pathlib import Path

                document_id = Path(file_path).name
            else:
                document_id = self._generate_node_id()

        return DocumentPreparationResult(
            document_id=document_id,
            content_hash=content_hash,
            skip_indexing=False,
            existing_doc_id=None,
        )

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
        # When progress bar is active, we suppress info logs to avoid disrupting the display
        progress = (
            GlobalProgressTracker(chunk_count, show_progress) if show_progress else None
        )

        # Create async wrapper for progress (tracker already created above)
        async_progress = AsyncProgressWrapper(progress) if progress else None

        return progress, async_progress

    def _prepare_chunk_positions(
        self,
        chunks: list[str],
        text: str,
        reporter: TelemetryCollector | None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Prepare chunk data with positions and validate document coverage.

        Args:
            chunks: List of text chunks to process
            text: Original document text
            reporter: Optional telemetry collector

        Returns:
            tuple: (leaf_ids, chunk_data_with_positions)
        """
        leaf_ids: list[str] = []
        chunk_data: list[dict[str, Any]] = []

        # Prepare all chunk data with character positions
        # Now that splitter handles whitespace gaps, positioning is straightforward
        current_pos = 0
        for i, chunk in enumerate(chunks):
            node_id = self._generate_node_id()

            # Chunks now have complete coverage with no gaps
            chunk_start = current_pos
            chunk_end = chunk_start + len(chunk)

            # Verify this chunk matches the original text
            if text[chunk_start:chunk_end] != chunk:
                # This should not happen with the fixed splitter, but provide fallback
                logger.warning(f"Chunk {i} position mismatch, using find() fallback")
                chunk_start = text.find(chunk, current_pos)
                if chunk_start == -1:
                    logger.error(f"Could not find chunk {i} in text")
                    chunk_start = current_pos
                chunk_end = chunk_start + len(chunk)

            chunk_data.append(
                {
                    "id": node_id,
                    "text": chunk,
                    "span_start": chunk_start,
                    "span_end": chunk_end,
                }
            )

            # Track node creation for telemetry
            if reporter:
                reporter.track_node_created(
                    node_id=node_id,
                    height=0,  # Leaves have height 0
                    span=(chunk_start, chunk_end),
                )
            leaf_ids.append(node_id)

            # Track chunk creation
            if reporter:
                try:
                    chunk_tokens = tokenizer.count_tokens(chunk)
                    reporter.record_chunk_created(node_id, chunk_tokens)
                except Exception as e:
                    logger.warning(
                        f"Failed to record telemetry for chunk creation: {e}"
                    )

            current_pos = chunk_end

        # Early validation: Check document coverage before processing embeddings
        from ragzoom.validate import validate, validate_document_coverage

        # Create node objects for validation using actual chunk data
        leaf_nodes_for_validation = []
        for data in chunk_data:
            node_obj = type(
                "Node",
                (),
                {
                    "id": data["id"],
                    "span_start": data["span_start"],
                    "span_end": data["span_end"],
                    "text": data["text"],
                },
            )()
            leaf_nodes_for_validation.append(node_obj)

        validate(
            lambda: validate_document_coverage(text, leaf_nodes_for_validation),
            "early document coverage check",
        )

        return leaf_ids, chunk_data

    async def _generate_embeddings_batch(
        self,
        chunks: list[str],
        chunk_data: list[dict[str, Any]],
        overall_start_time: float,
        show_progress: bool,
        async_progress: AsyncProgressWrapper | None,
        reporter: TelemetryCollector | None,
    ) -> list[Any]:
        """Generate embeddings for chunks in batches.

        Args:
            chunks: List of text chunks
            chunk_data: Chunk data with positions
            overall_start_time: Start time for elapsed tracking
            show_progress: Whether to show progress logs
            async_progress: Progress tracker
            reporter: Optional telemetry collector

        Returns:
            List of embeddings for all chunks
        """
        # Get embeddings in batches
        batch_size = self.config.embedding_batch_size
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch_texts = [cast(str, d["text"]) for d in chunk_data[i : i + batch_size]]
            batch_end = min(i + batch_size, len(chunks))

            # Show which batch we're processing with cumulative elapsed time
            if not show_progress:
                elapsed = time.time() - overall_start_time
                mins, secs = divmod(int(elapsed), 60)
                logger.info(
                    f"Processing embedding batch: chunks {i+1}-{batch_end} of {len(chunks)} [{mins}m {secs}s elapsed]"
                )

            # Track embedding call with node-level detail
            if reporter:
                node_embeddings = []
                for j in range(i, batch_end):
                    node_id = chunk_data[j]["id"]
                    text = chunk_data[j]["text"]
                    # Cache token count to avoid re-tokenization later
                    if "token_count" not in chunk_data[j]:
                        chunk_data[j]["token_count"] = len(tokenizer.encode(text))
                    token_count = chunk_data[j]["token_count"]
                    node_embeddings.append((node_id, token_count))

                start_time = time.time()

            batch_embeddings = await self.llm_service._get_embeddings_batch(batch_texts)

            if reporter:
                reporter.record_embedding_call_v2(
                    node_embeddings=node_embeddings,
                    batch_size=len(batch_texts),
                    model=self.config.embedding_model,
                    start_time=start_time,
                )
            all_embeddings.extend(batch_embeddings)

            # Update progress for embeddings
            if async_progress:
                await async_progress.update(len(batch_texts))

        return all_embeddings

    def _prepare_leaf_nodes_data(
        self,
        chunk_data: list[dict[str, Any]],
        all_embeddings: list[Any],
        document_id: str,
    ) -> list[dict[str, Any]]:
        """Prepare leaf node data for batch insertion.

        Args:
            chunk_data: Chunk data with positions
            all_embeddings: Embeddings for all chunks
            document_id: Document ID for the nodes

        Returns:
            List of leaf node data ready for database insertion
        """
        from ragzoom.utils.path_utils import calculate_tree_depth, generate_leaf_path

        leaf_nodes_data = []
        preceding_leaf_id = None  # Track preceding leaf for document order

        # Calculate tree depth for path generation
        num_leaves = len(chunk_data)
        tree_depth = calculate_tree_depth(num_leaves)

        for i, (data, embedding) in enumerate(zip(chunk_data, all_embeddings)):
            text = cast(str, data["text"])
            # Use cached token count if available, otherwise compute it
            token_count = data.get("token_count", tokenizer.count_tokens(text))

            # Generate binary path for this leaf
            path = generate_leaf_path(i, tree_depth)

            leaf_nodes_data.append(
                {
                    "node_id": cast(str, data["id"]),
                    "text": text,
                    "embedding": embedding,
                    "span_start": cast(int, data["span_start"]),
                    "span_end": cast(int, data["span_end"]),
                    "document_id": document_id,
                    "token_count": token_count,
                    "preceding_neighbor_id": preceding_leaf_id,
                    "height": 0,  # Leaf nodes have height 0
                    "path": path,  # Binary path encoding position in tree
                }
            )

            # Update preceding ID for next iteration
            preceding_leaf_id = cast(str, data["id"])

        return leaf_nodes_data

    async def _index_chunks(
        self,
        chunks: list[str],
        text: str,
        context: IndexingContext,
    ) -> tuple[list[str], bool]:
        """Index chunks: create embeddings, prepare leaf nodes, and store in database.

        Returns:
            tuple: (leaf_ids, existing_doc_found)
        """
        # Create leaf nodes with batch embeddings
        if not context.show_progress and len(chunks) > 100:
            logger.info("Preparing chunk data...")

        # Prepare chunk positions and validate coverage
        leaf_ids, chunk_data = self._prepare_chunk_positions(
            chunks, text, context.reporter
        )

        # Generate embeddings for all chunks
        all_embeddings = await self._generate_embeddings_batch(
            chunks,
            chunk_data,
            context.overall_start_time,
            context.show_progress,
            context.async_progress,
            context.reporter,
        )

        # Prepare all leaf nodes for batch insertion
        leaf_nodes_data = self._prepare_leaf_nodes_data(
            chunk_data, all_embeddings, context.document_id
        )

        # Check if this is updating an existing document
        existing_doc = context.file_path and self.store.get_document_by_path(
            context.file_path
        )

        # Add document record BEFORE creating nodes (foreign key constraint)
        if not existing_doc:
            self.store.add_document(
                context.document_id,
                context.file_path,
                context.content_hash,
                len(chunks),
                self.config.embedding_model,
                self.config.summary_model,
            )

        # Batch insert all leaf nodes at once using document-scoped store
        if leaf_nodes_data:
            doc_store = self.store.for_document(context.document_id)
            doc_store.nodes.add_batch(leaf_nodes_data)
        else:
            # Update existing document
            with self.store.SessionLocal() as session:
                from ragzoom.store import Document

                doc = session.query(Document).filter_by(id=context.document_id).first()
                if doc:
                    doc.content_hash = context.content_hash
                    doc.chunk_count = len(chunks)
                    doc.indexed_at = datetime.utcnow()
                    doc.embedding_model = self.config.embedding_model
                    doc.summary_model = self.config.summary_model
                    session.commit()

        return leaf_ids, existing_doc is not None

    @overload
    async def _add_document_impl(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
        reporter: None = None,
    ) -> str: ...

    @overload
    async def _add_document_impl(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
        reporter: TelemetryCollector = ...,  # jscpd:ignore-start
    ) -> tuple[str, dict[str, Any]]: ...  # jscpd:ignore-end

    async def _add_document_impl(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
        reporter: TelemetryCollector | None = None,
    ) -> str | tuple[str, dict[str, Any]]:
        """Add a document to the tree, creating leaf nodes.

        Returns:
            If reporter is None: document_id
            If reporter is provided: (document_id, metrics)
        """
        # Step 1: Prepare document (validation, hashing, existence check)
        prep_result = self._prepare_document(text, document_id, file_path)

        # Early return if document is unchanged
        if prep_result.skip_indexing:
            return prep_result.document_id

        # Step 2: Create and validate chunks
        chunks = self._create_and_validate_chunks(text, show_progress)

        # Step 3: Setup progress tracking
        progress, async_progress = self._setup_progress_tracking(
            len(chunks), show_progress
        )

        # Track overall start time for cumulative elapsed time
        overall_start_time = time.time()

        try:
            # Step 4: Index chunks (embeddings + leaf nodes)
            indexing_context = IndexingContext(
                document_id=prep_result.document_id,
                content_hash=prep_result.content_hash,
                file_path=file_path,
                async_progress=async_progress,
                overall_start_time=overall_start_time,
                show_progress=show_progress,
                reporter=reporter,
            )
            leaf_ids, existing_doc_updated = await self._index_chunks(
                chunks=chunks,
                text=text,
                context=indexing_context,
            )

            # Build tree from leaves using document-scoped store
            doc_store = self.store.for_document(prep_result.document_id)
            root_id = await self._build_tree_from_leaves(
                leaf_ids,
                chunks,
                prep_result.document_id,
                async_progress,
                overall_start_time,
                reporter,
                doc_store,
            )

            # Final completion logging with total elapsed time
            if root_id:
                total_elapsed = time.time() - overall_start_time
                mins, secs = divmod(int(total_elapsed), 60)
                if not show_progress:
                    logger.info(
                        f"Document indexed successfully: {prep_result.document_id} [{mins}m {secs}s total elapsed]"
                    )

            # Finalize telemetry if collector was used
            if reporter:
                telemetry = reporter.finalize()
                return prep_result.document_id, telemetry

            return prep_result.document_id
        finally:
            # Always close progress if it exists
            if progress:
                progress.close()

    def add_document(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
    ) -> str:
        """Sync wrapper for add_document."""
        return asyncio.run(
            self.add_document_async(text, document_id, file_path, show_progress)
        )

    def add_document_with_telemetry(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = False,
    ) -> tuple[str, dict[str, Any]]:
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
            document_id or "benchmark",
            source_tokens,
            self.config,
            document_path=file_path,
        )

        # Run indexing with collector - will return (doc_id, telemetry)
        result = asyncio.run(
            self._add_document_impl(
                text, document_id, file_path, show_progress, collector
            )
        )

        # Extract tuple returned when collector is provided
        # Type checker knows result is a tuple because we passed a collector
        doc_id, telemetry = result
        return doc_id, telemetry

    async def add_document_async(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
    ) -> str:
        """Async version of add_document - called by sync wrapper."""
        result = await self._add_document_impl(
            text, document_id, file_path, show_progress
        )
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
    ) -> dict[str, Any]:
        """Process a single node pair - generate summary and embedding.

        Returns:
            Dictionary containing node data and parent updates to be applied later
        """
        if doc_store is None:
            doc_store = self.store.for_document(document_id)

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

    async def _build_tree_from_leaves(
        self,
        leaf_ids: list[str],
        leaf_texts: list[str],
        document_id: str | None = None,
        progress: AsyncProgressWrapper | None = None,
        overall_start_time: float | None = None,
        reporter: TelemetryCollector | None = None,
        doc_store: DocumentStore | None = None,
    ) -> str:
        """Build tree bottom-up from leaf nodes with concurrent processing."""
        if doc_store is None:
            doc_store = self.store.for_document(document_id)

        current_level_ids = leaf_ids
        current_level_texts = leaf_texts
        current_level_nodes = None  # Will be populated after first batch insert

        # Calculate total tree height (distance from root to furthest leaf)
        # Note: This is used for progress tracking estimation

        # Track leaf level
        if reporter:
            try:
                reporter.record_tree_height_complete(0, len(leaf_ids))
            except Exception as e:
                logger.warning(f"Failed to record telemetry for tree height: {e}")

        current_height = 1  # Track height for logging (leaves are at height 0)
        while len(current_level_ids) > 1:
            next_level_ids: list[str] = []
            next_level_texts: list[str] = []
            # Note: current_height will be incremented after processing this height

            # Use pre-stored nodes if available, otherwise fetch from database
            if current_level_nodes is not None:
                # Use nodes from previous batch insert
                nodes_by_id = {node.id: node for node in current_level_nodes}
            else:
                # Pre-fetch all nodes for this level (first iteration with leaf nodes)
                all_nodes = doc_store.nodes.get_many(current_level_ids)
                nodes_by_id = {node.id: node for node in all_nodes}

            # Process pairs concurrently
            tasks = []
            pair_info: list[tuple[int, int | None]] = []

            # Process all nodes in pairs, with the last one having no right child if odd
            i = 0
            while i < len(current_level_ids):
                left_id = current_level_ids[i]
                left_text = current_level_texts[i]

                # Check if we have a right node
                if i + 1 < len(current_level_ids):
                    right_id = current_level_ids[i + 1]
                    right_text = current_level_texts[i + 1]
                    pair_info.append((i, i + 1))
                    i += 2  # Move to next pair
                else:
                    # Odd node - no right child
                    right_id = None
                    right_text = None
                    pair_info.append((i, None))
                    i += 1  # This was the last node

                # Get adjacent context
                prev_context = None
                if pair_info[-1][0] > 0:  # Use the left index from the current pair
                    prev_context, _ = self.splitter.get_adjacent_context(
                        current_level_texts, pair_info[-1][0] - 1
                    )

                # Get pre-fetched nodes
                left_node = nodes_by_id.get(left_id)
                right_node = nodes_by_id.get(right_id) if right_id else None

                # Create async task with pre-fetched nodes
                task = self._process_node_pair(
                    left_id,
                    left_text,
                    right_id,
                    right_text,
                    prev_context,
                    document_id,
                    current_height,
                    reporter,
                    left_node=left_node,
                    right_node=right_node,
                    doc_store=doc_store,
                )
                tasks.append(task)

            # Process all pairs concurrently
            if tasks:
                # Log tree building progress only when no progress bar
                if not (
                    progress and progress.tracker and progress.tracker.show_progress
                ):
                    if overall_start_time:
                        elapsed = time.time() - overall_start_time
                        mins, secs = divmod(int(elapsed), 60)
                        logger.info(
                            f"Building tree height {current_height}: processing {len(tasks)} node pairs [{mins}m {secs}s elapsed]"
                        )
                    else:
                        logger.info(
                            f"Building tree height {current_height}: processing {len(tasks)} node pairs"
                        )

                # Track completion count
                completed_count = 0

                # Wrap each task to update progress when it completes
                async def track_progress(task: Any, task_index: int) -> Any:
                    nonlocal completed_count
                    result = await task

                    # Update progress immediately when this pair completes
                    # For odd nodes (single child), only update by 1
                    if progress:
                        # Check if this is an odd node (has None for right child)
                        # task_index comes from enumerate(tasks) and corresponds to the position
                        # in both the tasks list and pair_info list (created in the same loop).
                        # Even though tasks complete out of order due to parallel execution,
                        # each task's index is captured in its closure when track_progress is created.
                        if (
                            task_index < len(pair_info)
                            and pair_info[task_index][1] is None
                        ):
                            await progress.update(1)  # Single node processed
                        else:
                            await progress.update(2)  # Pair processed

                    # Log batch completion every 10 tasks
                    completed_count += 1
                    if completed_count % 10 == 0 and overall_start_time:
                        if not (
                            progress
                            and progress.tracker
                            and progress.tracker.show_progress
                        ):
                            elapsed = time.time() - overall_start_time
                            mins, secs = divmod(int(elapsed), 60)
                            logger.info(
                                f"  Completed {completed_count}/{len(tasks)} pairs at height {current_height} [{mins}m {secs}s elapsed total]"
                            )

                    return result

                # Create tracked tasks
                tracked_tasks = [
                    track_progress(task, i) for i, task in enumerate(tasks)
                ]

                # Process all tasks concurrently (semaphore already controls parallelism)
                results = await asyncio.gather(*tracked_tasks)

                # Batch generate embeddings for all summaries at this level
                # This avoids individual API calls per node (e.g., 183 calls → 3 batch calls)
                # Extract summaries, warning about any empty ones (shouldn't happen due to verbatim fallback)
                summaries = []
                for i, result in enumerate(results):
                    summary = result["summary"]
                    if not summary or not summary.strip():
                        # This shouldn't happen as _summarize_text has verbatim fallback
                        logger.warning(
                            f"Unexpected empty summary for node {result.get('parent_id', 'unknown')}. "
                            f"This may indicate a bug in the summarization process."
                        )
                        # Use a minimal fallback to avoid embedding API errors
                        summary = "empty"
                        result["summary"] = summary  # Update for consistency
                    summaries.append(summary)

                start_time = time.time()
                embeddings = await self.llm_service._get_embeddings_batch(summaries)

                # Track batch embedding call for telemetry
                if reporter:
                    node_embeddings = []
                    for result in results:
                        # Use the token count from summarization
                        token_count = result.get(
                            "token_count",
                            tokenizer.count_tokens(result["summary"]),
                        )
                        node_embeddings.append((result["parent_id"], token_count))

                    reporter.record_embedding_call_v2(
                        node_embeddings=node_embeddings,
                        batch_size=len(summaries),
                        model=self.config.embedding_model,
                        start_time=start_time,
                    )

                # Update results with the generated embeddings
                # Since we process all results, we can directly zip them
                for result, embedding in zip(results, embeddings):
                    result["node_data"]["embedding"] = embedding

                # Extract data for batch processing
                nodes_to_add = []
                parent_updates = []
                next_level_ids = []
                next_level_texts = []

                # Track preceding node for this level
                preceding_node_id = None

                for result in results:
                    # Add preceding neighbor ID to node data
                    result["node_data"]["preceding_neighbor_id"] = preceding_node_id

                    # Add node data for batch insertion
                    nodes_to_add.append(result["node_data"])

                    # Collect parent updates
                    for update in result["parent_updates"]:
                        if (
                            update is not None
                        ):  # Skip None updates (for nodes without right child)
                            parent_updates.append(update)

                    # Track IDs and texts for next level
                    next_level_ids.append(result["parent_id"])
                    next_level_texts.append(result["summary"])

                    # Update preceding ID for next iteration
                    preceding_node_id = result["parent_id"]

                # Batch store all nodes for this level
                if nodes_to_add:
                    current_level_nodes = doc_store.nodes.add_batch(nodes_to_add)
                else:
                    current_level_nodes = []

                # Batch update all parent references
                if parent_updates:
                    doc_store.nodes.update_parent_references_batch(parent_updates)

            current_level_ids = next_level_ids
            current_level_texts = next_level_texts
            # current_level_nodes is already set from the batch insert above

            # Track tree level completion
            if reporter and current_level_ids:
                try:
                    reporter.record_tree_height_complete(
                        current_height, len(current_level_ids)
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to record telemetry for tree level completion: {e}"
                    )

            current_height += 1

        # Return root node ID
        if current_level_ids:
            if overall_start_time:
                elapsed = time.time() - overall_start_time
                mins, secs = divmod(int(elapsed), 60)
                if not (
                    progress and progress.tracker and progress.tracker.show_progress
                ):
                    logger.info(
                        f"Tree building complete. Root node at height {current_height - 1} with ID: {current_level_ids[0][:8]}... [{mins}m {secs}s elapsed total]"
                    )
            else:
                if not (
                    progress and progress.tracker and progress.tracker.show_progress
                ):
                    logger.info(
                        f"Tree building complete. Root node at height {current_height - 1} with ID: {current_level_ids[0][:8]}..."
                    )
        return current_level_ids[0] if current_level_ids else ""
