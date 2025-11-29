"""Tree building and indexing functionality for RagZoom."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ragzoom.append_patch import AppendPatchBuilder, PatchTracking
from ragzoom.config import IndexConfig, SecretStr
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.dataflow import (
    build_full_document_patch,
    run_tree_patch,
)
from ragzoom.dataflow.core import ProcessingStrategy, TreePatch
from ragzoom.dataflow.domain import DomainNode
from ragzoom.document_store import DocumentStore
from ragzoom.progress import AsyncProgressWrapper, GlobalProgressTracker
from ragzoom.services.llm_service import LLMService
from ragzoom.splitter import TextSplitter
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.telemetry_embeddings import compute_fidelity_for_telemetry
from ragzoom.telemetry_types import TelemetryDataDict
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


@dataclass
class DocumentPreparationResult:
    """Result from document preparation phase.

    Replaces the confusing 3-tuple return from _prepare_document with clear semantics.
    """

    document_id: str
    skip_indexing: bool
    existing_doc_id: str | None = None


@dataclass
class IndexingContext:
    """Context for document indexing operations.

    Groups related parameters to reduce parameter list complexity.
    """

    document_id: str
    file_path: str | None
    async_progress: AsyncProgressWrapper | None
    overall_start_time: float
    show_progress: bool
    reporter: TelemetryCollector | None


@dataclass
class AppendStats:
    """Summary of an incremental append operation."""

    document_id: str
    mutated_nodes: int
    resummarized_nodes: int
    new_leaves: int
    total_leaves: int
    telemetry: TelemetryDataDict | None = None


VectorPayload: TypeAlias = tuple[
    str,
    list[float] | NDArray[np.float64],
    dict[str, object],
]


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
        self._summarize_text = self.llm_service._summarize_text

        # Backward compatibility: provide access to centralized tokenizer
        self.tokenizer = tokenizer

        # Delegate patch building to AppendPatchBuilder
        self._patch_builder = AppendPatchBuilder(
            document_store=self.document_store,
            tokenizer=self.tokenizer,
        )

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        return str(uuid.uuid4())

    def _domain_to_repo_dict(self, node: DomainNode) -> NodeDataDict:
        """Convert DomainNode into repository payload for upsert."""
        return {
            "node_id": node.id,
            "text": node.text,
            "span_start": int(node.span_start),
            "span_end": int(node.span_end),
            "parent_id": node.parent_id,
            "left_child_id": node.left_child_id,
            "right_child_id": node.right_child_id,
            "document_id": node.document_id,
            "token_count": int(node.token_count),
            "height": int(node.height),
            "preceding_neighbor_id": node.preceding_neighbor_id,
            "following_neighbor_id": node.following_neighbor_id,
            "level_index": int(node.level_index),
        }

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

    async def _append_into_empty_document(
        self,
        text: str,
        show_progress: bool,
        reporter: TelemetryCollector | None,
    ) -> AppendStats:
        """Bootstrap a new document using the append machinery."""

        self._validate_model_names()
        document_id = self.document_store.document_id
        if not document_id:
            raise ValueError("DocumentStore must have a document_id set")

        chunks = self._create_and_validate_chunks(text, show_progress)
        progress, async_progress = self._setup_progress_tracking(
            len(chunks), show_progress
        )
        overall_start_time = time.time()

        try:
            patch = build_full_document_patch(
                chunks=chunks,
                document_id=document_id,
                reporter=reporter,
            )
            tree_nodes = await run_tree_patch(
                patch=patch,
                llm_service=self.llm_service,
                target_tokens=self.config.target_chunk_tokens,
                max_summary_concurrency=30,
                max_embedding_concurrency=10,
                embedding_batch_size=self.config.embedding_batch_size,
                processing_strategy=ProcessingStrategy(self.config.processing_strategy),
                reporter=reporter,
                progress=async_progress,
            )

            doc_store = self.document_store
            from itertools import groupby

            sorted_nodes = sorted(tree_nodes, key=lambda n: n.height)
            for height, nodes_at_height in groupby(
                sorted_nodes, key=lambda n: n.height
            ):
                nodes_batch = list(nodes_at_height)
                if not nodes_batch:
                    continue
                logger.debug(
                    "Inserting %d nodes at height %d", len(nodes_batch), height
                )
                payload: list[NodeDataDict] = []
                for node in nodes_batch:
                    payload.append(
                        {
                            "node_id": node.id,
                            "text": node.text,
                            "document_id": node.document_id,
                            "span_start": node.span_start,
                            "span_end": node.span_end,
                            "parent_id": None,
                            "left_child_id": node.left_child_id,
                            "right_child_id": node.right_child_id,
                            "preceding_neighbor_id": node.preceding_neighbor_id,
                            "following_neighbor_id": node.following_neighbor_id,
                            "token_count": node.token_count,
                            "height": node.height,
                            "level_index": node.level_index,
                        }
                    )
                doc_store.nodes.add_batch(payload)

            parent_updates = [
                (node.id, node.parent_id)
                for node in tree_nodes
                if node.parent_id is not None
            ]
            if parent_updates:
                doc_store.nodes.update_parent_references_batch(parent_updates)

            if tree_nodes and not show_progress:
                total_elapsed = time.time() - overall_start_time
                mins, secs = divmod(int(total_elapsed), 60)
                logger.info(
                    "Document indexed successfully: %s [%dm %ds total elapsed]",
                    document_id,
                    mins,
                    secs,
                )

            upsert_items: list[VectorPayload] = []
            for node in tree_nodes:
                embedding = getattr(node, "embedding", None)
                if embedding is None:
                    continue
                upsert_items.append(
                    (
                        node.id,
                        [float(x) for x in embedding],
                        {
                            "span_start": int(node.span_start),
                            "span_end": int(node.span_end),
                            "parent_id": node.parent_id or "",
                            "document_id": node.document_id,
                            "is_leaf": 1 if int(node.height) == 0 else 0,
                            "height": int(node.height),
                            "level_index": int(node.level_index),
                            "coord_version": 1,
                        },
                    )
                )
            if upsert_items:
                self.vector_index.upsert(upsert_items)

            leaf_count = len(doc_store.nodes.get_leaves())
            total_nodes = doc_store.nodes.count()
            resummarized_nodes = max(total_nodes - leaf_count, 0)
            if reporter:
                token_limit = getattr(
                    self.llm_service, "_embedding_batch_token_limit", 8000
                )
                max_items = getattr(
                    self.llm_service, "_provider_max_embedding_batch_size", 1000
                )
                await compute_fidelity_for_telemetry(
                    document_store=self.document_store,
                    collector=reporter,
                    vector_index=self.vector_index,
                    embedder=self.llm_service,
                    token_limit=token_limit,
                    max_batch_items=max_items,
                )
                telemetry_payload = reporter.finalize()
            else:
                telemetry_payload = None

            return AppendStats(
                document_id=document_id,
                mutated_nodes=total_nodes,
                resummarized_nodes=resummarized_nodes,
                new_leaves=leaf_count,
                total_leaves=leaf_count,
                telemetry=telemetry_payload,
            )
        finally:
            if progress:
                progress.close()

    def add_document(
        self,
        text: str,
        show_progress: bool = True,
    ) -> str:
        """Index text into an empty document synchronously."""

        stats = asyncio.run(self.append_text_async(text, show_progress))
        return stats.document_id

    def add_document_with_telemetry(
        self,
        text: str,
        show_progress: bool = False,
    ) -> tuple[str, TelemetryDataDict]:
        """Index text and return telemetry payload for benchmarking."""

        source_tokens = tokenizer.count_tokens(text)
        collector = TelemetryCollector(
            self.document_store.document_id or "benchmark",
            source_tokens,
            self.config,
            document_path=None,
        )
        stats = asyncio.run(
            self.append_text_async(
                text, show_progress=show_progress, reporter=collector
            )
        )
        if stats.telemetry is None:
            raise RuntimeError("Telemetry collection failed")
        return stats.document_id, stats.telemetry

    async def add_document_async(
        self,
        text: str,
        show_progress: bool = True,
    ) -> str:
        """Async helper mirroring :meth:`add_document`."""

        stats = await self.append_text_async(text, show_progress=show_progress)
        return stats.document_id

    def _validate_append_preconditions(self, new_text: str) -> str:
        """Validate preconditions for append operation.

        Returns:
            Validated document_id.

        Raises:
            ValueError: If text is empty or document_id is not set.
            NotImplementedError: If storage backend lacks required methods.
        """
        if not new_text:
            raise ValueError("append_text_async requires non-empty text")

        document_id = self.document_store.document_id
        if not document_id:
            raise ValueError("DocumentStore must have a document_id set")

        nodes_repo = self.document_store.nodes
        if not hasattr(nodes_repo, "upsert_nodes_batch"):
            raise NotImplementedError(
                "Storage backend must implement upsert_nodes_batch() to enable incremental appends"
            )

        return document_id

    def _prepare_vector_upserts(
        self,
        mutated_node_objs: list[DomainNode],
    ) -> tuple[list[VectorPayload], list[str]]:
        """Prepare vector payloads from mutated nodes.

        Returns:
            Tuple of (vector_upserts, vector_node_ids).
        """
        vector_upserts: list[VectorPayload] = []
        vector_node_ids: list[str] = []
        for node in mutated_node_objs:
            if node.embedding is None:
                continue
            meta: dict[str, object] = {
                "span_start": int(node.span_start),
                "span_end": int(node.span_end),
                "parent_id": node.parent_id or "",
                "document_id": node.document_id,
                "is_leaf": 1 if int(node.height) == 0 else 0,
                "height": int(node.height),
                "level_index": int(node.level_index),
                "coord_version": 1,
            }
            vector_node_ids.append(node.id)
            vector_upserts.append((node.id, [float(x) for x in node.embedding], meta))
        return vector_upserts, vector_node_ids

    def _build_neighbor_updates(
        self,
        mutated_node_objs: list[DomainNode],
        tracking: PatchTracking,
    ) -> list[tuple[str, str | None, str | None]]:
        """Build list of neighbor relationship updates.

        Returns:
            List of (node_id, preceding_id, following_id) tuples.
        """
        neighbor_map: dict[str, tuple[str | None, str | None]] = {}
        for node_id, preceding, following in tracking.neighbor_updates:
            neighbor_map[node_id] = (preceding, following)

        for node in mutated_node_objs:
            original = tracking.original_neighbors.get(node.id)
            new_pair = (node.preceding_neighbor_id, node.following_neighbor_id)
            if original != new_pair:
                neighbor_map[node.id] = new_pair

        return [
            (node_id, values[0], values[1]) for node_id, values in neighbor_map.items()
        ]

    def _prepare_append_chunks(
        self,
        right_leaf: TreeNode,
        new_text: str,
    ) -> list[str]:
        """Combine right leaf text with new text and split into chunks.

        Returns:
            List of validated chunks.

        Raises:
            ValueError: If combined text is empty or chunking produces no output.
        """
        combined_text = right_leaf.text + new_text
        if not combined_text:
            raise ValueError("Append produced no content to index")

        new_chunks = self.splitter.split_text(combined_text)
        if not new_chunks:
            raise ValueError("Chunking produced no output for appended text")

        from ragzoom.validate import validate, validate_chunk_sizes

        chunk_objects = []
        for idx, chunk in enumerate(new_chunks):
            chunk_obj = type("ChunkObj", (), {"text": chunk, "id": f"append_{idx}"})()
            chunk_objects.append(chunk_obj)

        validate(
            lambda: validate_chunk_sizes(
                chunk_objects, self.config.target_chunk_tokens
            ),
            "append chunk size validation",
        )

        return new_chunks

    def _track_mutable_nodes_for_telemetry(
        self,
        patch: TreePatch,
        tracking: PatchTracking,
        reporter: TelemetryCollector | None,
    ) -> None:
        """Record telemetry for mutable nodes in the patch."""
        if not reporter:
            return

        for mutable_id in tracking.mutable_node_ids:
            node = patch.lookup.get(mutable_id)
            if node is None:
                continue
            reporter.track_node_created(
                node_id=node.id,
                height=int(node.height),
                span=(int(node.span_start), int(node.span_end)),
            )
            if int(node.height) == 0:
                reporter.record_chunk_created(
                    node.id,
                    self.tokenizer.count_tokens(node.text),
                )

    def _capture_rollback_vectors(
        self,
        vector_node_ids: list[str],
        document_id: str,
    ) -> tuple[list[VectorPayload], set[str]]:
        """Capture existing vectors for rollback before writing new ones.

        Returns:
            Tuple of (rollback_vectors to restore, rollback_delete_ids to delete).
        """
        rollback_vectors: list[VectorPayload] = []
        rollback_delete_ids: set[str] = set()

        for node_id in vector_node_ids:
            try:
                existing_vector = self.vector_index.get_vectors([node_id])[0]
            except (KeyError, IndexError):
                rollback_delete_ids.add(node_id)
            except Exception as exc:  # pragma: no cover - defensive logging
                rollback_delete_ids.add(node_id)
                logger.warning(
                    "Failed to load existing vector before append: doc=%s node=%s error=%s",
                    document_id,
                    node_id,
                    exc,
                )
            else:
                rollback_vectors.append(
                    (
                        existing_vector.id,
                        [float(x) for x in existing_vector.vec.tolist()],
                        dict(existing_vector.meta),
                    )
                )

        return rollback_vectors, rollback_delete_ids

    def _persist_with_rollback(
        self,
        mutated_node_objs: list[DomainNode],
        neighbor_updates: list[tuple[str, str | None, str | None]],
        vectors_written: int,
        rollback_vectors: list[VectorPayload],
        rollback_delete_ids: set[str],
        document_id: str,
    ) -> None:
        """Persist nodes with transaction rollback for vectors on failure."""
        from itertools import groupby

        mutated_sorted = sorted(mutated_node_objs, key=lambda n: n.height, reverse=True)

        try:
            with self.document_store.transaction() as session:
                for _, nodes_group in groupby(mutated_sorted, key=lambda n: n.height):
                    payload = [self._domain_to_repo_dict(node) for node in nodes_group]
                    self.document_store.nodes.upsert_nodes_batch(
                        payload, session=session
                    )

                if neighbor_updates:
                    self.document_store.nodes.update_neighbors_batch(
                        neighbor_updates, session=session
                    )

        except Exception:
            if vectors_written:
                if rollback_vectors:
                    try:
                        self.vector_index.upsert(rollback_vectors)
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.error(
                            "Failed to restore vectors after append rollback: doc=%s error=%s",
                            document_id,
                            exc,
                        )
                if rollback_delete_ids:
                    try:
                        self.vector_index.delete(ids=list(rollback_delete_ids))
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.error(
                            "Failed to delete new vectors after append rollback: doc=%s ids=%s error=%s",
                            document_id,
                            sorted(rollback_delete_ids),
                            exc,
                        )
            raise

    def _finalize_append(
        self,
        tracking: PatchTracking,
        patch: TreePatch,
        neighbor_updates: list[tuple[str, str | None, str | None]],
        document_id: str,
    ) -> list[TreeNode]:
        """Clear caches, validate results, and log stats. Returns leaves_after."""
        if neighbor_updates:
            affected_for_depth = set(tracking.mutable_node_ids)
            affected_for_depth.update(node_id for node_id, _, _ in neighbor_updates)
        else:
            affected_for_depth = set(tracking.mutable_node_ids)
        self.document_store.tree.clear_depth_cache(list(affected_for_depth))

        leaves_after = self.document_store.nodes.get_leaves()
        leaves_after.sort(key=lambda n: int(n.span_start))
        reconstructed = "".join(leaf.text for leaf in leaves_after)

        logger.debug(
            "Append stats doc=%s mutated=%d new_leaves=%d resummarized=%d",
            document_id,
            len(tracking.mutable_node_ids),
            max(tracking.leaf_delta, 0),
            len(tracking.summary_node_ids),
        )

        from ragzoom.validate import validate

        validate(
            lambda: self._validate_append_results(
                tracking,
                patch,
                leaves_after,
                reconstructed,
            ),
            "incremental append",
        )

        return leaves_after

    async def _collect_telemetry_and_build_stats(
        self,
        tracking: PatchTracking,
        leaves_after: list[TreeNode],
        document_id: str,
        reporter: TelemetryCollector | None,
    ) -> AppendStats:
        """Finalize telemetry and build return stats."""
        telemetry_payload: TelemetryDataDict | None = None
        if reporter:
            reporter.record_append_metadata(
                span_start=tracking.tail_start,
                span_end=tracking.tail_start + len(tracking.tail_text),
                mutated_nodes=len(tracking.mutable_node_ids),
                summary_nodes=len(tracking.summary_node_ids),
                leaf_delta=tracking.leaf_delta,
            )
            token_limit = getattr(
                self.llm_service, "_embedding_batch_token_limit", 8000
            )
            max_items = getattr(
                self.llm_service, "_provider_max_embedding_batch_size", 1000
            )
            await compute_fidelity_for_telemetry(
                document_store=self.document_store,
                collector=reporter,
                vector_index=self.vector_index,
                embedder=self.llm_service,
                token_limit=token_limit,
                max_batch_items=max_items,
            )
            telemetry_payload = reporter.finalize()

        return AppendStats(
            document_id=document_id,
            mutated_nodes=len(tracking.mutable_node_ids),
            resummarized_nodes=len(tracking.summary_node_ids),
            new_leaves=max(tracking.leaf_delta, 0),
            total_leaves=len(leaves_after),
            telemetry=telemetry_payload,
        )

    async def append_text_async(
        self,
        new_text: str,
        show_progress: bool = True,
        reporter: TelemetryCollector | None = None,
    ) -> AppendStats:
        """Append new text to an existing document incrementally."""

        document_id = self._validate_append_preconditions(new_text)

        right_leaf = self.document_store.nodes.get_rightmost_leaf_for_document(
            document_id
        )
        if right_leaf is None:
            return await self._append_into_empty_document(
                new_text,
                show_progress=show_progress,
                reporter=reporter,
            )

        new_chunks = self._prepare_append_chunks(right_leaf, new_text)

        patch, tracking = self._patch_builder.build_patch(
            right_leaf, new_chunks, document_id
        )

        progress, async_progress = self._setup_progress_tracking(
            max(1, len(patch.embedding_node_ids)), show_progress
        )

        self._track_mutable_nodes_for_telemetry(patch, tracking, reporter)

        try:
            await run_tree_patch(
                patch=patch,
                llm_service=self.llm_service,
                target_tokens=self.config.target_chunk_tokens,
                max_summary_concurrency=30,
                max_embedding_concurrency=10,
                embedding_batch_size=self.config.embedding_batch_size,
                processing_strategy=ProcessingStrategy(self.config.processing_strategy),
                reporter=reporter,
                progress=async_progress,
            )
        finally:
            if progress:
                progress.close()

        mutated_node_objs = [
            patch.lookup[node_id] for node_id in tracking.mutable_node_ids
        ]

        vector_upserts, vector_node_ids = self._prepare_vector_upserts(
            mutated_node_objs
        )

        rollback_vectors, rollback_delete_ids = self._capture_rollback_vectors(
            vector_node_ids, document_id
        )

        vectors_written = len(vector_upserts)
        if vector_upserts:
            self.vector_index.upsert(vector_upserts)

        neighbor_updates = self._build_neighbor_updates(mutated_node_objs, tracking)

        self._persist_with_rollback(
            mutated_node_objs,
            neighbor_updates,
            vectors_written,
            rollback_vectors,
            rollback_delete_ids,
            document_id,
        )

        leaves_after = self._finalize_append(
            tracking, patch, neighbor_updates, document_id
        )

        return await self._collect_telemetry_and_build_stats(
            tracking, leaves_after, document_id, reporter
        )

    def append_text(
        self,
        new_text: str,
        show_progress: bool = True,
    ) -> AppendStats:
        """Sync wrapper for append_text_async."""

        return asyncio.run(
            self.append_text_async(new_text, show_progress=show_progress)
        )

    def _validate_append_results(
        self,
        tracking: PatchTracking,
        patch: TreePatch,
        leaves_after: list[TreeNode],
        reconstructed_text: str,
    ) -> str | None:
        """Execute expensive validations for incremental append when enabled."""

        if not tracking.tail_text:
            return None

        if tracking.tail_start < 0 or tracking.tail_start > len(reconstructed_text):
            return (
                f"Tail start {tracking.tail_start} is outside document bounds "
                f"(doc length {len(reconstructed_text)})"
            )

        tail_after = reconstructed_text[tracking.tail_start :]
        if tail_after != tracking.tail_text:
            return (
                "Tail reconstruction mismatch after append; expected length "
                f"{len(tracking.tail_text)} but observed {len(tail_after)}"
            )

        for node_id, original_height in tracking.original_heights.items():
            node = patch.lookup.get(node_id)
            if node is None:
                return f"Expected node {node_id} missing from patch lookup"
            if int(node.height) != int(original_height):
                return (
                    f"Height changed for reused node {node_id}: "
                    f"was {original_height}, now {node.height}"
                )

        for node_id in tracking.summary_node_ids:
            node = patch.lookup.get(node_id)
            if node is None:
                continue
            if not node.left_child_id and not node.right_child_id:
                continue
            left = patch.lookup.get(node.left_child_id) if node.left_child_id else None
            right = (
                patch.lookup.get(node.right_child_id) if node.right_child_id else None
            )
            if left is None:
                return f"Parent {node_id} missing left child {node.left_child_id}"
            expected_start = int(left.span_start)
            expected_end = int(right.span_end) if right else int(left.span_end)
            if (
                int(node.span_start) != expected_start
                or int(node.span_end) != expected_end
            ):
                return (
                    f"Parent span mismatch for {node_id}: [{node.span_start}, {node.span_end}) "
                    f"vs expected [{expected_start}, {expected_end})"
                )

        return None

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
