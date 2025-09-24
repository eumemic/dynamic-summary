"""Tree building and indexing functionality for RagZoom."""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TypeAlias, cast, overload

import numpy as np
from numpy.typing import NDArray

from ragzoom.config import IndexConfig, SecretStr
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


@dataclass
class PatchTracking:
    """Tracking information for incremental append patches."""

    mutable_node_ids: set[str]
    context_node_ids: set[str]
    original_neighbors: dict[str, tuple[str | None, str | None]]
    neighbor_updates: list[tuple[str, str | None, str | None]] = field(
        default_factory=list
    )
    leaf_delta: int = 0
    tail_start: int = 0
    tail_text: str = ""
    summary_node_ids: set[str] = field(default_factory=set)
    original_heights: dict[str, int] = field(default_factory=dict)


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

    def _node_to_domain(self, node: TreeNode) -> DomainNode:
        """Convert a stored TreeNode into a DomainNode for patch construction."""

        document_id = node.document_id or (self.document_store.document_id or "")
        is_pinned_raw = getattr(node, "is_pinned", False)
        if isinstance(is_pinned_raw, int):
            is_pinned = bool(is_pinned_raw)
        else:
            is_pinned = bool(is_pinned_raw)

        return DomainNode(
            id=node.id,
            document_id=document_id,
            parent_id=node.parent_id,
            left_child_id=node.left_child_id,
            right_child_id=node.right_child_id,
            span_start=int(node.span_start),
            span_end=int(node.span_end),
            text=node.text or "",
            token_count=int(getattr(node, "token_count", 0)),
            height=int(getattr(node, "height", 0)),
            is_pinned=is_pinned,
            depth=int(getattr(node, "depth", 0)),
            preceding_neighbor_id=getattr(node, "preceding_neighbor_id", None),
            following_neighbor_id=getattr(node, "following_neighbor_id", None),
            embedding=None,
        )

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        return str(uuid.uuid4())

    def _collect_spine(self, leaf: TreeNode) -> list[TreeNode]:
        """Collect nodes along the rightmost spine starting from the provided leaf."""

        spine: list[TreeNode] = [leaf]
        current = leaf
        while current.parent_id:
            parent = self.document_store.nodes.get(current.parent_id)
            if parent is None:
                raise ValueError(
                    "Encountered missing ancestor while tracing right spine"
                )
            spine.append(parent)
            current = parent
        return spine

    def _domain_to_repo_dict(self, node: DomainNode) -> dict[str, object]:
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
        }

    def _ensure_context_nodes(
        self,
        lookup: dict[str, DomainNode],
        candidates: list[str | None],
        tracking: PatchTracking,
    ) -> None:
        """Ensure referenced neighbor nodes are present in the patch lookup."""

        for node_id in candidates:
            if not node_id:
                continue
            if node_id in lookup:
                tracking.context_node_ids.add(node_id)
                continue
            node = self.document_store.nodes.get(node_id)
            if node is None:
                continue
            domain = self._node_to_domain(node)
            lookup[domain.id] = domain
            tracking.context_node_ids.add(domain.id)
            tracking.original_neighbors.setdefault(
                domain.id,
                (domain.preceding_neighbor_id, domain.following_neighbor_id),
            )

    def _build_append_patch(
        self,
        right_leaf: TreeNode,
        new_chunks: list[str],
        document_id: str,
    ) -> tuple[TreePatch, PatchTracking]:
        """Construct a TreePatch for incremental append."""

        if not new_chunks:
            raise ValueError("Append requires at least one chunk of text")

        lookup: dict[str, DomainNode] = {}
        tracking = PatchTracking(set(), set(), {})

        spine_nodes = self._collect_spine(right_leaf)
        if not spine_nodes:
            raise ValueError("Rightmost leaf is missing its ancestor chain")

        spine_domains: dict[str, DomainNode] = {}
        for node in spine_nodes:
            domain = self._node_to_domain(node)
            lookup[domain.id] = domain
            spine_domains[domain.id] = domain
            tracking.mutable_node_ids.add(domain.id)
            tracking.original_neighbors[domain.id] = (
                domain.preceding_neighbor_id,
                domain.following_neighbor_id,
            )
            tracking.original_heights[domain.id] = int(domain.height)

        leaf_domain = spine_domains[right_leaf.id]
        tracking.tail_start = int(leaf_domain.span_start)
        original_following = leaf_domain.following_neighbor_id
        self._ensure_context_nodes(
            lookup, [leaf_domain.preceding_neighbor_id], tracking
        )

        new_leaf_domains: list[DomainNode] = []
        span_cursor = leaf_domain.span_start
        for idx, chunk in enumerate(new_chunks):
            span_end = span_cursor + len(chunk)
            token_count = self.tokenizer.count_tokens(chunk)

            if idx == 0:
                leaf_domain.text = chunk
                leaf_domain.span_start = span_cursor
                leaf_domain.span_end = span_end
                leaf_domain.token_count = token_count
            else:
                new_leaf = DomainNode(
                    id=self._generate_node_id(),
                    document_id=document_id,
                    parent_id=None,
                    left_child_id=None,
                    right_child_id=None,
                    span_start=span_cursor,
                    span_end=span_end,
                    text=chunk,
                    token_count=token_count,
                    height=0,
                    preceding_neighbor_id=None,
                    following_neighbor_id=None,
                )
                lookup[new_leaf.id] = new_leaf
                new_leaf_domains.append(new_leaf)
                tracking.mutable_node_ids.add(new_leaf.id)
                tracking.original_neighbors[new_leaf.id] = (None, None)

            span_cursor = span_end

        tracking.tail_text = "".join(new_chunks)

        last_leaf_id = leaf_domain.id
        if new_leaf_domains:
            leaf_domain.following_neighbor_id = new_leaf_domains[0].id
            for idx, leaf in enumerate(new_leaf_domains):
                leaf.preceding_neighbor_id = (
                    new_leaf_domains[idx - 1].id if idx > 0 else leaf_domain.id
                )
                leaf.following_neighbor_id = (
                    new_leaf_domains[idx + 1].id
                    if idx + 1 < len(new_leaf_domains)
                    else original_following
                )
            last_leaf_id = new_leaf_domains[-1].id
        else:
            leaf_domain.following_neighbor_id = original_following

        tracking.leaf_delta = len(new_chunks) - 1

        if original_following:
            following_node = self.document_store.nodes.get(original_following)
            following_follow = (
                getattr(following_node, "following_neighbor_id", None)
                if following_node
                else None
            )
            # Record how the right-edge neighbor chain is rewritten so rollback can restore it
            tracking.neighbor_updates.append(
                (original_following, last_leaf_id, following_follow)
            )
            if following_node is not None:
                tracking.context_node_ids.add(original_following)
                tracking.original_neighbors.setdefault(
                    original_following,
                    (
                        getattr(following_node, "preceding_neighbor_id", None),
                        getattr(following_node, "following_neighbor_id", None),
                    ),
                )

        embedding_ids = [leaf_domain.id] + [leaf.id for leaf in new_leaf_domains]

        current_level: list[DomainNode] = []
        # Include left sibling when the path node is a right child
        if len(spine_nodes) > 1:
            parent_tree = spine_nodes[1]
            if parent_tree.right_child_id == right_leaf.id:
                left_sibling_id = parent_tree.left_child_id
                if left_sibling_id:
                    sibling_domain = lookup.get(left_sibling_id)
                    if sibling_domain is None:
                        sibling_node = self.document_store.nodes.get(left_sibling_id)
                        if sibling_node is not None:
                            sibling_domain = self._node_to_domain(sibling_node)
                            lookup[sibling_domain.id] = sibling_domain
                        if sibling_domain is not None:
                            tracking.context_node_ids.add(sibling_domain.id)
                            tracking.original_neighbors.setdefault(
                                sibling_domain.id,
                                (
                                    sibling_domain.preceding_neighbor_id,
                                    sibling_domain.following_neighbor_id,
                                ),
                            )
                    if sibling_domain is not None:
                        current_level.append(sibling_domain)
                        self._ensure_context_nodes(
                            lookup,
                            [sibling_domain.preceding_neighbor_id],
                            tracking,
                        )

        current_level.append(leaf_domain)
        current_level.extend(new_leaf_domains)

        summary_root_ids: list[str] = []

        # Traverse up the spine, reusing existing parents when present
        for level_index in range(len(spine_nodes) - 1):
            parent_tree = spine_nodes[level_index + 1]
            parent_domain = spine_domains[parent_tree.id]

            next_level, summary_ids = self._build_parent_level(
                current_level,
                parent_domain,
                document_id,
                lookup,
                tracking,
            )
            summary_root_ids.extend(summary_ids)
            current_level = next_level

            if level_index + 1 < len(spine_nodes) - 1:
                next_parent_tree = spine_nodes[level_index + 2]
                if next_parent_tree.right_child_id == parent_domain.id:
                    left_sibling_id = next_parent_tree.left_child_id
                    if left_sibling_id:
                        sibling_domain = lookup.get(left_sibling_id)
                        if sibling_domain is None:
                            sibling_node = self.document_store.nodes.get(
                                left_sibling_id
                            )
                            if sibling_node is not None:
                                sibling_domain = self._node_to_domain(sibling_node)
                                lookup[sibling_domain.id] = sibling_domain
                            if sibling_domain is not None:
                                tracking.context_node_ids.add(sibling_domain.id)
                                tracking.original_neighbors.setdefault(
                                    sibling_domain.id,
                                    (
                                        sibling_domain.preceding_neighbor_id,
                                        sibling_domain.following_neighbor_id,
                                    ),
                                )
                        if sibling_domain is not None and all(
                            sibling_domain.id != existing.id
                            for existing in current_level
                        ):
                            # Pull the left sibling into the patch to keep adjacency consistent
                            current_level.insert(0, sibling_domain)
                            self._ensure_context_nodes(
                                lookup,
                                [sibling_domain.preceding_neighbor_id],
                                tracking,
                            )

        # Build any additional parents if new nodes extended the tree height
        while len(current_level) > 1:
            current_level, summary_ids = self._build_parent_level(
                current_level,
                None,
                document_id,
                lookup,
                tracking,
            )
            summary_root_ids.extend(summary_ids)

        if current_level:
            current_level[0].parent_id = None
            self._assign_patch_depths(current_level[0].id, lookup)

        patch = TreePatch(
            lookup=lookup,
            embedding_node_ids=embedding_ids,
            summary_root_ids=summary_root_ids,
        )
        return patch, tracking

    def _assign_patch_depths(self, root_id: str, lookup: dict[str, DomainNode]) -> None:
        """Assign depth values within a patch so batching logic remains correct."""

        queue: deque[tuple[str, int]] = deque([(root_id, 0)])
        visited: set[str] = set()

        while queue:
            node_id, depth = queue.popleft()
            node = lookup.get(node_id)
            if node is None or node_id in visited:
                continue
            visited.add(node_id)
            node.depth = depth

            if node.left_child_id:
                queue.append((node.left_child_id, depth + 1))
            if node.right_child_id:
                queue.append((node.right_child_id, depth + 1))

    def _build_parent_level(
        self,
        nodes: list[DomainNode],
        existing_parent: DomainNode | None,
        document_id: str,
        lookup: dict[str, DomainNode],
        tracking: PatchTracking,
    ) -> tuple[list[DomainNode], list[str]]:
        """Pair nodes into parents, reusing spine ancestor when provided."""

        if not nodes:
            return [], []

        summary_ids: list[str] = []
        next_level: list[DomainNode] = []
        idx = 0
        reuse_available = existing_parent is not None

        while idx < len(nodes):
            left = nodes[idx]
            right = nodes[idx + 1] if idx + 1 < len(nodes) else None

            if reuse_available:
                assert existing_parent is not None
                parent = existing_parent
                reuse_available = False
            else:
                parent = DomainNode(
                    id=self._generate_node_id(),
                    document_id=document_id,
                    parent_id=None,
                    left_child_id=None,
                    right_child_id=None,
                    span_start=0,
                    span_end=0,
                    text="",
                    token_count=0,
                    height=0,
                )
                lookup[parent.id] = parent
                tracking.mutable_node_ids.add(parent.id)
                tracking.original_neighbors[parent.id] = (
                    parent.preceding_neighbor_id,
                    parent.following_neighbor_id,
                )

            parent.document_id = document_id
            parent.left_child_id = left.id
            parent.span_start = left.span_start
            left.parent_id = parent.id

            if right is not None:
                parent.right_child_id = right.id
                parent.span_end = right.span_end
                parent.height = max(int(left.height), int(right.height)) + 1
                right.parent_id = parent.id
                step = 2
            else:
                parent.right_child_id = None
                parent.span_end = left.span_end
                parent.height = int(left.height) + 1
                step = 1

            parent.text = ""
            parent.token_count = 0
            parent.embedding = None

            summary_ids.append(parent.id)
            tracking.summary_node_ids.add(parent.id)
            self._ensure_context_nodes(lookup, [left.preceding_neighbor_id], tracking)

            next_level.append(parent)

            idx += step

        # Update neighbor links for the newly formed level
        for i, parent in enumerate(next_level):
            if i > 0:
                parent.preceding_neighbor_id = next_level[i - 1].id
            if i + 1 < len(next_level):
                parent.following_neighbor_id = next_level[i + 1].id

        return next_level, summary_ids

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
            patch = build_full_document_patch(
                chunks=chunks,
                document_id=document_id,
                reporter=reporter,
            )
            tree_nodes = await run_tree_patch(
                patch=patch,
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

    async def append_text_async(
        self,
        new_text: str,
        show_progress: bool = True,
        reporter: TelemetryCollector | None = None,
    ) -> AppendStats:
        """Append new text to an existing document incrementally."""

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

        right_leaf = nodes_repo.get_rightmost_leaf_for_document(document_id)
        if right_leaf is None:
            result = await self._add_document_impl(
                new_text,
                show_progress=show_progress,
                reporter=reporter,
            )

            if reporter:
                document_id, initial_telemetry = cast(
                    tuple[str, TelemetryDataDict], result
                )
            else:
                document_id = cast(str, result)
                initial_telemetry = None

            leaves_after = self.document_store.nodes.get_leaves()
            total_leaves = len(leaves_after)
            total_nodes = self.document_store.nodes.count()
            mutated_nodes = int(total_nodes)
            resummarized_nodes = max(mutated_nodes - total_leaves, 0)

            return AppendStats(
                document_id=document_id,
                mutated_nodes=mutated_nodes,
                resummarized_nodes=resummarized_nodes,
                new_leaves=total_leaves,
                total_leaves=total_leaves,
                telemetry=initial_telemetry,
            )

        combined_text = (right_leaf.text or "") + new_text
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

        doc_version = self.document_store.get_version() or 1
        new_version = doc_version + 1

        patch, tracking = self._build_append_patch(right_leaf, new_chunks, document_id)

        progress, async_progress = self._setup_progress_tracking(
            max(1, len(patch.embedding_node_ids)), show_progress
        )

        if reporter:
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
                        self.tokenizer.count_tokens(node.text or ""),
                    )

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

        vector_upserts: list[VectorPayload] = []
        vector_node_ids: list[str] = []
        for node in mutated_node_objs:
            if node.embedding is None:
                continue
            meta = {
                "span_start": int(node.span_start),
                "span_end": int(node.span_end),
                "parent_id": node.parent_id or "",
                "document_id": node.document_id,
                "is_leaf": 1 if int(node.height) == 0 else 0,
            }
            vector_node_ids.append(node.id)
            vector_upserts.append((node.id, [float(x) for x in node.embedding], meta))

        rollback_vectors: list[VectorPayload] = []
        rollback_delete_ids: set[str] = set()

        if vector_node_ids:
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

        vectors_written = len(vector_upserts)
        if vector_upserts:
            self.vector_index.upsert(vector_upserts)

        from itertools import groupby

        mutated_sorted = sorted(mutated_node_objs, key=lambda n: n.height, reverse=True)

        neighbor_map: dict[str, tuple[str | None, str | None]] = {}
        for node_id, preceding, following in tracking.neighbor_updates:
            neighbor_map[node_id] = (preceding, following)

        for node in mutated_node_objs:
            original = tracking.original_neighbors.get(node.id)
            new_pair = (node.preceding_neighbor_id, node.following_neighbor_id)
            if original != new_pair:
                neighbor_map[node.id] = new_pair

        neighbor_updates = [
            (node_id, values[0], values[1]) for node_id, values in neighbor_map.items()
        ]

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

                self.document_store.set_metadata(version=new_version, session=session)
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

        if neighbor_updates:
            affected_for_depth = set(tracking.mutable_node_ids)
            affected_for_depth.update(node_id for node_id, _, _ in neighbor_updates)
        else:
            affected_for_depth = set(tracking.mutable_node_ids)
        self.document_store.tree.clear_depth_cache(list(affected_for_depth))

        leaves_after = self.document_store.nodes.get_leaves()
        leaves_after.sort(key=lambda n: int(n.span_start))
        reconstructed = "".join(leaf.text or "" for leaf in leaves_after)
        new_hash = DocumentStore.compute_content_hash(reconstructed)
        self.document_store.set_metadata(content_hash=new_hash)

        logger.debug(
            "Append stats doc=%s version=%d->%d mutated=%d new_leaves=%d resummarized=%d",
            document_id,
            doc_version,
            new_version,
            len(tracking.mutable_node_ids),
            max(tracking.leaf_delta, 0),
            len(tracking.summary_node_ids),
        )

        validate(
            lambda: self._validate_append_results(
                tracking,
                patch,
                leaves_after,
                reconstructed,
            ),
            "incremental append",
        )

        telemetry_payload: TelemetryDataDict | None = None
        if reporter:
            reporter.record_append_metadata(
                document_version=new_version,
                span_start=tracking.tail_start,
                span_end=tracking.tail_start + len(tracking.tail_text),
                mutated_nodes=len(tracking.mutable_node_ids),
                summary_nodes=len(tracking.summary_node_ids),
                leaf_delta=tracking.leaf_delta,
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
