"""Left-to-right indexing engine for document processing.

This module implements a simple job-based indexing system that processes work
left-to-right through documents, ensuring optimal context quality for
summarization and embedding generation.

Key concepts:
- Only document roots are eligible for work (leaves for embedding, sibling pairs
  for summarization)
- Work is processed left-to-right (by span_start order) to maximize context
  quality
- Jobs self-trigger: when a job completes, it triggers discovery of new work
- Parallelism is capped at a configurable maximum across all documents
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.tree_node import TreeNode

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from openai import OpenAI

    from ragzoom.config import IndexConfig
    from ragzoom.contracts.storage_backend import StorageBackend
    from ragzoom.contracts.vector_index import VectorIndex
    from ragzoom.retrieve import RetrievalResult, Retriever
    from ragzoom.services.llm_service import LLMService
    from ragzoom.telemetry_collection import TelemetryCollector

# Type alias for vector index factory
VectorIndexFactory = Callable[[str], "VectorIndex"]

# Callback type for document idle events (receives document_id)
OnDocumentIdleCallback = Callable[[str], "Awaitable[None]"]

logger = logging.getLogger(__name__)


class TilingValidationError(Exception):
    """Raised when a tiling fails validation during indexing.

    This error intentionally kills the indexing process to preserve
    the database state for debugging the retrieval bug.
    """

    pass


def _expected_total_from_leaf_count(n: int) -> int:
    """Calculate expected total jobs for N leaves.

    For N leaves building a binary tree:
    - N embedding jobs (one per leaf)
    - N - popcount(N) summary jobs (where popcount = count of 1-bits)

    The popcount accounts for forests: odd leaf counts at any level
    leave unpaired roots, reducing the summary count.

    Examples:
        - 8 leaves (0b1000): 8 + (8-1) = 15 jobs → 1 root
        - 7 leaves (0b111):  7 + (7-3) = 11 jobs → 3 roots
        - 5 leaves (0b101):  5 + (5-2) = 8 jobs  → 2 roots
    """
    if n <= 0:
        return 0
    popcount = bin(n).count("1")
    return 2 * n - popcount


# ---------------------------------------------------------------------------
# Job types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingJob:
    """Job to generate contextual embedding for a leaf node."""

    document_id: str
    leaf_id: str

    def __hash__(self) -> int:
        return hash(("embedding", self.document_id, self.leaf_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EmbeddingJob):
            return False
        return self.document_id == other.document_id and self.leaf_id == other.leaf_id


@dataclass(frozen=True)
class SummaryJob:
    """Job to summarize a pair of sibling roots into a parent node."""

    document_id: str
    left_id: str
    right_id: str

    def __hash__(self) -> int:
        return hash(("summary", self.document_id, self.left_id, self.right_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SummaryJob):
            return False
        return (
            self.document_id == other.document_id
            and self.left_id == other.left_id
            and self.right_id == other.right_id
        )


IndexingJob = EmbeddingJob | SummaryJob


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexingStatus:
    """Snapshot of indexing activity."""

    in_flight: int
    in_flight_by_document: dict[str, int]
    completed_by_document: dict[str, int]
    expected_total_by_document: dict[str, int]


# ---------------------------------------------------------------------------
# Per-document state
# ---------------------------------------------------------------------------


@dataclass
class DocumentContext:
    """Per-document state for indexing."""

    telemetry_collector: TelemetryCollector | None = None
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled: bool = False
    # Track permanently failed jobs to avoid infinite retry loops
    failed_job_ids: set[IndexingJob] | None = None
    # Expected total jobs for progress tracking (set at append time)
    expected_total_jobs: int = 0

    def mark_failed(self, job: IndexingJob) -> None:
        """Mark a job as permanently failed."""
        if self.failed_job_ids is None:
            self.failed_job_ids = set()
        self.failed_job_ids.add(job)
        self.failed_jobs += 1

    def is_failed(self, job: IndexingJob) -> bool:
        """Check if a job has already failed."""
        if self.failed_job_ids is None:
            return False
        return job in self.failed_job_ids


# ---------------------------------------------------------------------------
# Indexing engine
# ---------------------------------------------------------------------------


class IndexingEngine:
    """Drives left-to-right document indexing.

    The engine:
    1. Queries roots ordered by span_start (left-to-right)
    2. Finds eligible jobs: leaves needing embeddings, or sibling pairs
    3. Runs up to max_parallelism jobs concurrently across all documents
    4. Re-triggers work discovery when jobs complete
    """

    def __init__(
        self,
        *,
        store: StorageBackend,
        llm_service: LLMService,
        index_config: IndexConfig,
        openai_client: OpenAI,
        vector_index_factory: VectorIndexFactory | None = None,
        max_parallelism: int = 30,
        on_document_idle: OnDocumentIdleCallback | None = None,
    ) -> None:
        self._store = store
        self._llm_service = llm_service
        self._index_config = index_config
        self._openai_client = openai_client
        self._vector_index_factory = vector_index_factory
        self._max_parallelism = max(1, max_parallelism)
        self._on_document_idle = on_document_idle

        self._active_jobs: set[IndexingJob] = set()
        self._lock = asyncio.Lock()
        self._idle_event = asyncio.Event()
        self._idle_event.set()

        # Cache vector index by model to avoid creating new instances each call
        self._vector_index_cache: dict[str, VectorIndex] = {}

        # Per-document context (telemetry, completion tracking)
        self._document_contexts: dict[str, DocumentContext] = {}

        # Track documents with pending work
        self._active_documents: set[str] = set()

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    async def trigger_work(
        self,
        document_id: str,
        *,
        telemetry_collector: TelemetryCollector | None = None,
    ) -> None:
        """Start or resume indexing for a document.

        Call this after document splitting creates leaves, or on server startup
        to resume incomplete indexing.

        Args:
            document_id: Document to index
            telemetry_collector: Optional collector for timing/metrics
        """
        # Count leaves to calculate expected total work
        store = self._store.for_document(document_id)
        leaf_count = sum(1 for n in store.nodes.get_all() if n.height == 0)
        expected_total = _expected_total_from_leaf_count(leaf_count)

        async with self._lock:
            # Create or update document context
            if document_id not in self._document_contexts:
                self._document_contexts[document_id] = DocumentContext(
                    telemetry_collector=telemetry_collector,
                    expected_total_jobs=expected_total,
                )
            else:
                ctx = self._document_contexts[document_id]
                if telemetry_collector is not None:
                    ctx.telemetry_collector = telemetry_collector
                ctx.cancelled = False
                ctx.completed_jobs = 0
                ctx.expected_total_jobs = expected_total

            self._active_documents.add(document_id)
            self._idle_event.clear()

        await self._find_and_start_jobs(document_id)

    async def cancel_document(self, document_id: str) -> None:
        """Cancel pending work for a document.

        Marks the document as cancelled so no new jobs will start.
        Waits for any in-flight jobs to complete.
        """
        async with self._lock:
            ctx = self._document_contexts.get(document_id)
            if ctx is not None:
                ctx.cancelled = True
            self._active_documents.discard(document_id)

        # Wait for in-flight jobs to complete
        await self.wait_until_idle(document_id)

        # Clean up context
        async with self._lock:
            self._document_contexts.pop(document_id, None)

    async def wait_until_idle(self, document_id: str | None = None) -> None:
        """Wait until all work is complete.

        Args:
            document_id: If provided, wait only for this document.
                        If None, wait for all documents.
        """
        while True:
            async with self._lock:
                if document_id is None:
                    if not self._active_jobs and not self._active_documents:
                        return
                else:
                    doc_jobs = [
                        j for j in self._active_jobs if j.document_id == document_id
                    ]
                    if not doc_jobs and document_id not in self._active_documents:
                        return

                # Clear event before waiting so we can detect new signals
                self._idle_event.clear()

            # Wait for state change signal (job completion or no more work)
            await self._idle_event.wait()

    async def status(self) -> IndexingStatus:
        """Get current indexing activity snapshot."""
        async with self._lock:
            in_flight_by_doc: dict[str, int] = {}
            for job in self._active_jobs:
                in_flight_by_doc[job.document_id] = (
                    in_flight_by_doc.get(job.document_id, 0) + 1
                )

            completed_by_doc: dict[str, int] = {}
            expected_total_by_doc: dict[str, int] = {}
            for doc_id, ctx in self._document_contexts.items():
                if ctx.completed_jobs > 0:
                    completed_by_doc[doc_id] = ctx.completed_jobs
                if ctx.expected_total_jobs > 0:
                    expected_total_by_doc[doc_id] = ctx.expected_total_jobs

            return IndexingStatus(
                in_flight=len(self._active_jobs),
                in_flight_by_document=in_flight_by_doc,
                completed_by_document=completed_by_doc,
                expected_total_by_document=expected_total_by_doc,
            )

    async def shutdown(self) -> None:
        """Wait for all work to complete."""
        await self.wait_until_idle()

    # -----------------------------------------------------------------------
    # Job discovery and scheduling
    # -----------------------------------------------------------------------

    async def _find_and_start_jobs(self, document_id: str) -> None:
        """Find and start eligible jobs up to max parallelism."""
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > 100:
                logger.warning(
                    "_find_and_start_jobs loop count %d, breaking", loop_count
                )
                break
            # Check state under lock
            async with self._lock:
                ctx = self._document_contexts.get(document_id)
                if ctx is not None and ctx.cancelled:
                    self._active_documents.discard(document_id)
                    self._notify_state_change()
                    return

                if len(self._active_jobs) >= self._max_parallelism:
                    return

                # Copy active jobs to check against outside lock
                current_active = set(self._active_jobs)

            # Find job outside lock (does I/O)
            job = self._find_next_job(document_id, current_active, ctx)

            # Acquire lock to add job
            async with self._lock:
                if job is None:
                    # Check if there are any active jobs for this document
                    # Only fire idle callback when no work in progress
                    has_active_jobs = any(
                        j.document_id == document_id for j in self._active_jobs
                    )

                    if has_active_jobs:
                        # Jobs still running - they'll re-trigger when done
                        return

                    self._active_documents.discard(document_id)
                    self._notify_state_change()

                    # Fire callback outside lock - document is truly idle
                    if self._on_document_idle is not None:
                        asyncio.create_task(self._safe_on_document_idle(document_id))
                    return

                # Re-check job not already added by another task
                if job in self._active_jobs:
                    continue

                # Re-check parallelism limit
                if len(self._active_jobs) >= self._max_parallelism:
                    return

                self._active_jobs.add(job)
                asyncio.create_task(self._run_job(job))
                logger.debug(
                    "engine: started job %s (active=%d)",
                    job,
                    len(self._active_jobs),
                )

    def _notify_state_change(self) -> None:
        """Wake up waiters after state changes. Must hold lock."""
        # Always set the event to wake up waiters so they can re-check conditions
        self._idle_event.set()

    async def _safe_on_document_idle(self, document_id: str) -> None:
        """Call on_document_idle callback with error handling."""
        if self._on_document_idle is None:
            return
        try:
            await self._on_document_idle(document_id)
        except Exception:
            logger.exception("on_document_idle callback failed for %s", document_id)

    def _find_next_job(
        self,
        document_id: str,
        active_jobs: set[IndexingJob],
        ctx: DocumentContext | None,
    ) -> IndexingJob | None:
        """Scan roots left-to-right for next eligible job.

        Returns the first job that:
        1. Is not already active
        2. Has not previously failed
        3. Is either a leaf needing embedding, or an eligible sibling pair
        """
        store = self._store.for_document(document_id)
        roots = store.nodes.get_root_nodes(document_id)

        for i, root in enumerate(roots):
            # Check for leaf needing embedding
            if self._is_leaf(root) and not self._has_embedding(root, document_id):
                embedding_job = EmbeddingJob(document_id, root.id)
                if embedding_job not in active_jobs:
                    # Skip if this job already failed
                    if ctx is not None and ctx.is_failed(embedding_job):
                        continue
                    return embedding_job

            # Check for eligible sibling pair
            if i + 1 < len(roots):
                right = roots[i + 1]
                if self._is_eligible_pair(root, right, document_id):
                    summary_job = SummaryJob(document_id, root.id, right.id)
                    if summary_job not in active_jobs:
                        # Skip if this job already failed
                        if ctx is not None and ctx.is_failed(summary_job):
                            continue
                        return summary_job

        return None

    def _is_leaf(self, node: TreeNode) -> bool:
        """Check if a node is a leaf (height 0)."""
        return int(getattr(node, "height", 0)) == 0

    def _get_vector_index(self) -> VectorIndex | None:
        """Get the vector index for the configured embedding model (cached)."""
        if self._vector_index_factory is None:
            return None

        model = self._index_config.embedding_model
        if model not in self._vector_index_cache:
            self._vector_index_cache[model] = self._vector_index_factory(model)
        return self._vector_index_cache[model]

    def _has_embedding(self, node: TreeNode, document_id: str) -> bool:
        """Check if a leaf node already has an embedding."""
        vector_index = self._get_vector_index()
        if vector_index is None:
            return False

        # Check vector index for existing embedding by ID lookup
        results = vector_index.get_vectors([node.id])
        return len(results) > 0

    def _is_eligible_pair(
        self, left: TreeNode, right: TreeNode, document_id: str
    ) -> bool:
        """Check if two adjacent roots form an eligible sibling pair ready for summary.

        Two roots are eligible siblings if:
        1. They have the same height
        2. Left has an even level_index (is a left child position)
        3. Right has level_index = left.level_index + 1 (is adjacent)
        4. For leaves (height 0): both must have embeddings
        """
        left_height = int(getattr(left, "height", 0))
        right_height = int(getattr(right, "height", 0))
        if left_height != right_height:
            return False

        left_level = int(getattr(left, "level_index", 0))
        right_level = int(getattr(right, "level_index", 0))

        # Left must be at even index (left child position)
        if left_level % 2 != 0:
            return False

        # Right must be adjacent (left's sibling)
        if right_level != left_level + 1:
            return False

        # For leaves, both must have embeddings before summarization
        if left_height == 0:
            if not self._has_embedding(left, document_id):
                return False
            if not self._has_embedding(right, document_id):
                return False

        return True

    # -----------------------------------------------------------------------
    # Job execution
    # -----------------------------------------------------------------------

    async def _run_job(self, job: IndexingJob) -> None:
        """Execute a job with cleanup and re-trigger."""
        document_id = job.document_id
        job_failed = False
        try:
            if isinstance(job, EmbeddingJob):
                await self._embed_leaf(job)
            else:
                await self._summarize_pair(job)
        except Exception:
            logger.exception("Job failed: %s", job)
            job_failed = True
        finally:
            async with self._lock:
                self._active_jobs.discard(job)

                # Track success/failure
                ctx = self._document_contexts.get(document_id)
                if ctx is not None:
                    if job_failed:
                        ctx.mark_failed(job)
                    else:
                        ctx.completed_jobs += 1

                logger.debug(
                    "engine: finished job %s (active=%d, failed=%s)",
                    job,
                    len(self._active_jobs),
                    job_failed,
                )

                # Wake up any waiters to re-check their conditions
                self._idle_event.set()

            # Re-trigger to find more work
            await self._find_and_start_jobs(document_id)

    async def _embed_leaf(self, job: EmbeddingJob) -> None:
        """Generate contextual embedding for a leaf node.

        Steps:
        1. Retrieve preceding context
        2. Summarize context into prefix (LLM call)
        3. Generate embedding with context prefix
        4. Write to vector index
        """
        import time

        from ragzoom.utils.tokenization import tokenizer

        store = self._store.for_document(job.document_id)
        leaf = store.nodes.get(job.leaf_id)
        if leaf is None:
            logger.warning(
                "embed: leaf not found doc=%s leaf=%s",
                job.document_id,
                job.leaf_id,
            )
            # Still set preceding_context to empty string for consistency
            store.nodes._repo.update_preceding_context(job.leaf_id, "")
            return

        leaf_text = leaf.text or ""
        if not leaf_text:
            logger.warning(
                "embed: leaf has no text doc=%s leaf=%s",
                job.document_id,
                job.leaf_id,
            )
            # Still set preceding_context to empty string for consistency
            store.nodes._repo.update_preceding_context(job.leaf_id, "")
            return

        span_start = int(getattr(leaf, "span_start", 0))
        span_end = int(getattr(leaf, "span_end", 0))

        # Get telemetry collector for this document
        ctx = self._document_contexts.get(job.document_id)
        telemetry = ctx.telemetry_collector if ctx else None

        # Retrieve preceding context
        # Nodes at span_start=0 get empty tiling, others get retrieved context

        context_result: RetrievalResult | None = None
        if span_start > 0:
            retriever = self._create_retriever(job.document_id)
            if retriever is not None:
                try:
                    context_result = await retriever.retrieve_for_context(
                        query_text=leaf_text,
                        span_end_limit=span_start,
                        budget_tokens=self._index_config.preceding_summary_budget_tokens,
                        document_id=job.document_id,
                        recent_verbatim_token_budget=0,
                    )
                    # Validate tiling covers [0, span_start) completely
                    self._validate_tiling(
                        context_result,
                        span_start,
                        job.document_id,
                        job.leaf_id,
                        "embed_leaf",
                    )
                except TilingValidationError:
                    # Re-raise to kill indexing and preserve DB state for debugging
                    raise
                except Exception:
                    logger.exception(
                        "embed: failed to retrieve context doc=%s leaf=%s",
                        job.document_id,
                        job.leaf_id,
                    )
                    # context_result stays as None

        # Extract tiling IDs and assemble context text
        tiling_ids: list[str] = []
        context_prefix = ""
        if (
            context_result is not None
            and context_result.tiling
            and context_result.nodes
        ):
            tiling_ids = list(context_result.tiling)
            context_prefix = "\n\n".join(
                context_result.nodes[nid].text or ""
                for nid in tiling_ids
                if nid in context_result.nodes
            )

        # Store preceding_context as JSON array of node IDs on the leaf node
        import json

        store = self._store.for_document(job.document_id)
        preceding_context_json = json.dumps(tiling_ids)
        store.nodes._repo.update_preceding_context(job.leaf_id, preceding_context_json)

        # Limit text_to_embed to stay within embedding token limit (8000)
        # This prevents the ValueError that occurs when context + leaf exceeds the limit
        text_to_embed = self._build_embedding_text(leaf_text, context_prefix)

        # Record embedding start time for telemetry
        embed_start_time = time.time()
        embeddings = await self._llm_service.embed_texts([text_to_embed])
        if not embeddings:
            logger.error(
                "embed: no embedding returned doc=%s leaf=%s",
                job.document_id,
                job.leaf_id,
            )
            return

        # Record embedding telemetry
        if telemetry is not None:
            text_tokens = tokenizer.count_tokens(text_to_embed)
            telemetry.record_embedding_call_v2(
                node_embeddings=[(job.leaf_id, text_tokens)],
                batch_size=1,
                model=self._index_config.embedding_model,
                start_time=embed_start_time,
            )

        # Write to vector index
        vector_index = self._get_vector_index()
        if vector_index is not None:
            embedding_array = np.asarray(embeddings[0], dtype=np.float64)
            metadata: dict[str, object] = {
                "node_id": job.leaf_id,
                "document_id": job.document_id,
                "span_start": span_start,
                "span_end": span_end,
                "height": 0,
                "is_leaf": 1,
                "parent_id": getattr(leaf, "parent_id", None) or "",
            }
            vector_index.upsert([(job.leaf_id, embedding_array, metadata)])
            logger.debug(
                "embed: wrote vector doc=%s leaf=%s",
                job.document_id,
                job.leaf_id,
            )

    def _build_embedding_text(
        self, leaf_text: str, context_prefix: str, token_limit: int = 8000
    ) -> str:
        """Build text for embedding, ensuring it stays within token limit.

        The embedding model (text-embedding-3-small) has an 8000 token limit.
        When context_prefix + leaf_text would exceed this, we truncate the
        context_prefix to make room for the leaf_text.

        Args:
            leaf_text: The leaf node's text (required, not truncated)
            context_prefix: Retrieved preceding context (may be truncated)
            token_limit: Maximum tokens for the combined text (default 8000)

        Returns:
            Combined text within token limit
        """
        from ragzoom.utils.tokenization import tokenizer

        if not context_prefix:
            return leaf_text

        leaf_tokens = tokenizer.count_tokens(leaf_text)

        # If leaf alone exceeds limit, just return it (will fail at embed time)
        if leaf_tokens >= token_limit:
            logger.warning(
                "embed: leaf text (%d tokens) exceeds embedding limit (%d)",
                leaf_tokens,
                token_limit,
            )
            return leaf_text

        # Calculate available budget for context (minus 1 for newline separator)
        context_budget = token_limit - leaf_tokens - 1

        if context_budget <= 0:
            return leaf_text

        # Check if context fits within budget
        context_tokens = tokenizer.count_tokens(context_prefix)
        if context_tokens <= context_budget:
            return f"{context_prefix}\n{leaf_text}"

        # Truncate context to fit within budget
        # Truncate from the beginning since earlier context is less relevant
        truncated_context = tokenizer.truncate_to_token_limit(
            context_prefix, context_budget, from_end=False
        )

        if not truncated_context or not truncated_context.strip():
            return leaf_text

        logger.debug(
            "embed: truncated context from %d to %d tokens (leaf=%d)",
            context_tokens,
            tokenizer.count_tokens(truncated_context),
            leaf_tokens,
        )
        return f"{truncated_context}\n{leaf_text}"

    async def _summarize_pair(self, job: SummaryJob) -> None:
        """Summarize a sibling pair into a parent node.

        Steps:
        1. Get left and right nodes
        2. Generate summary (LLM call)
        3. Retrieve preceding context
        4. Create parent node in database
        """
        store = self._store.for_document(job.document_id)

        left = store.nodes.get(job.left_id)
        right = store.nodes.get(job.right_id)

        if left is None:
            logger.warning(
                "summarize: left not found doc=%s left=%s",
                job.document_id,
                job.left_id,
            )
            return

        if right is None:
            logger.warning(
                "summarize: right not found doc=%s right=%s",
                job.document_id,
                job.right_id,
            )
            return

        # Check if this pair was already summarized by another concurrent job
        left_parent = getattr(left, "parent_id", None)
        right_parent = getattr(right, "parent_id", None)
        if left_parent is not None or right_parent is not None:
            logger.debug(
                "summarize: pair already has parent doc=%s left=%s right=%s",
                job.document_id,
                job.left_id,
                job.right_id,
            )
            return

        # Get telemetry collector if available
        ctx = self._document_contexts.get(job.document_id)
        telemetry = ctx.telemetry_collector if ctx else None

        # Extract node properties
        left_text = left.text or ""
        right_text = right.text or ""
        left_tokens = int(getattr(left, "token_count", 0))
        right_tokens = int(getattr(right, "token_count", 0))
        left_height = int(getattr(left, "height", 0))
        right_height = int(getattr(right, "height", 0))
        left_level_index = int(getattr(left, "level_index", 0))

        span_start = int(getattr(left, "span_start", 0))
        span_end = int(getattr(right, "span_end", 0))
        parent_height = max(left_height, right_height) + 1
        parent_level_index = left_level_index // 2

        parent_id = str(uuid.uuid4())

        # Track node creation in telemetry BEFORE summary (summary attempts reference it)
        if telemetry is not None:
            telemetry.track_node_created(
                node_id=parent_id,
                height=parent_height,
                span=(span_start, span_end),
            )

        # Retrieve preceding context BEFORE summarization
        # Uses combined children text as query since we don't have a summary yet
        # Nodes at span_start=0 get empty tiling, others get retrieved context

        context_result: RetrievalResult | None = None
        if span_start > 0:
            retriever = self._create_retriever(job.document_id)
            if retriever is not None:
                query_text = f"{left_text}\n{right_text}"
                context_result = await retriever.retrieve_for_context(
                    query_text=query_text,
                    span_end_limit=span_start,
                    budget_tokens=self._index_config.preceding_summary_budget_tokens,
                    document_id=job.document_id,
                    recent_verbatim_token_budget=0,
                )
                # Validate tiling covers [0, span_start) completely
                self._validate_tiling(
                    context_result,
                    span_start,
                    job.document_id,
                    parent_id,
                    "summarize_pair",
                )

        # Extract tiling IDs and assemble context text
        import json

        tiling_ids: list[str] = []
        context_text = ""
        if (
            context_result is not None
            and context_result.tiling
            and context_result.nodes
        ):
            tiling_ids = list(context_result.tiling)
            context_text = "\n\n".join(
                context_result.nodes[nid].text or ""
                for nid in tiling_ids
                if nid in context_result.nodes
            )
        preceding_context_json = json.dumps(tiling_ids)

        # Generate summary with preceding context
        summary, _retry_count, summary_tokens = await self._llm_service._summarize_text(
            left_text,
            right_text,
            self._index_config.target_chunk_tokens,
            parent_id=parent_id,
            reporter=telemetry,
            prev_context=context_text,
            left_token_count=left_tokens,
            right_token_count=right_tokens,
        )

        # Determine neighbor IDs at parent level
        preceding_parent_id: str | None = None
        following_parent_id: str | None = None

        if parent_level_index > 0:
            prev_parent = store.nodes.get_by_height_and_level(
                height=parent_height,
                level_index=parent_level_index - 1,
            )
            if prev_parent is not None:
                preceding_parent_id = prev_parent.id

        next_parent = store.nodes.get_by_height_and_level(
            height=parent_height,
            level_index=parent_level_index + 1,
        )
        if next_parent is not None:
            following_parent_id = next_parent.id

        # Create parent node
        node_payload: NodeDataDict = {
            "node_id": parent_id,
            "text": summary,
            "span_start": span_start,
            "span_end": span_end,
            "parent_id": None,
            "left_child_id": left.id,
            "right_child_id": right.id,
            "document_id": job.document_id,
            "token_count": summary_tokens,
            "height": parent_height,
            "preceding_neighbor_id": preceding_parent_id,
            "following_neighbor_id": following_parent_id,
            "level_index": parent_level_index,
            "preceding_context": preceding_context_json,
        }

        # Commit to database - each operation auto-commits
        store.nodes.add_batch([node_payload])
        store.nodes.update_parent_references_batch(
            [(left.id, parent_id), (right.id, parent_id)]
        )

        # Update neighbor links
        neighbors_update: list[tuple[str, str | None, str | None]] = [
            (parent_id, preceding_parent_id, following_parent_id)
        ]
        if preceding_parent_id is not None:
            prev_parent = store.nodes.get(preceding_parent_id)
            if prev_parent is not None:
                neighbors_update.append(
                    (
                        preceding_parent_id,
                        getattr(prev_parent, "preceding_neighbor_id", None),
                        parent_id,
                    )
                )
        if following_parent_id is not None:
            next_parent = store.nodes.get(following_parent_id)
            if next_parent is not None:
                neighbors_update.append(
                    (
                        following_parent_id,
                        parent_id,
                        getattr(next_parent, "following_neighbor_id", None),
                    )
                )

        store.nodes.update_neighbors_batch(neighbors_update)

        logger.debug(
            "summarize: created parent doc=%s parent=%s left=%s right=%s h=%d",
            job.document_id,
            parent_id,
            left.id,
            right.id,
            parent_height,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _validate_tiling(
        self,
        context_result: RetrievalResult,
        expected_end: int,
        document_id: str,
        node_id: str,
        operation: str,
    ) -> None:
        """Validate that a tiling covers [0, expected_end) with no gaps.

        Raises TilingValidationError if the tiling is incomplete or has gaps,
        killing the indexing process to preserve database state for debugging.
        """
        if not context_result.tiling or not context_result.nodes:
            raise TilingValidationError(
                f"{operation}: Empty tiling for node {node_id} "
                f"(expected coverage [0, {expected_end}))"
            )

        # Get nodes in tiling order and sort by span_start
        tiling_nodes = [
            context_result.nodes[nid]
            for nid in context_result.tiling
            if nid in context_result.nodes
        ]
        if not tiling_nodes:
            raise TilingValidationError(
                f"{operation}: No valid nodes in tiling for node {node_id}"
            )

        sorted_nodes = sorted(tiling_nodes, key=lambda n: n.span_start)

        # Check first node starts at 0
        first = sorted_nodes[0]
        if first.span_start != 0:
            raise TilingValidationError(
                f"{operation}: Tiling does not start at 0 for node {node_id}: "
                f"first tiling node {first.id} starts at {first.span_start}"
            )

        # Check for gaps
        for i in range(len(sorted_nodes) - 1):
            curr = sorted_nodes[i]
            next_node = sorted_nodes[i + 1]
            if curr.span_end != next_node.span_start:
                raise TilingValidationError(
                    f"{operation}: Gap in tiling for node {node_id}: "
                    f"{curr.id} ends at {curr.span_end}, "
                    f"{next_node.id} starts at {next_node.span_start}"
                )

        # Check last node ends at expected_end
        last = sorted_nodes[-1]
        if last.span_end != expected_end:
            raise TilingValidationError(
                f"{operation}: Tiling does not end at {expected_end} for node {node_id}: "
                f"last tiling node {last.id} ends at {last.span_end}"
            )

    def _create_retriever(self, document_id: str) -> Retriever | None:
        """Create a retriever for the given document."""
        if self._vector_index_factory is None:
            return None

        from ragzoom.config import QueryConfig
        from ragzoom.retrieval.budget_planner import BudgetPlanner
        from ragzoom.retrieval.embedding_service import EmbeddingService
        from ragzoom.retrieve import Retriever

        document_store = self._store.for_document(document_id)
        vector_index = self._vector_index_factory(self._index_config.embedding_model)

        # Create per-document services
        embedding_service = EmbeddingService(
            self._openai_client,
            document_store,
            self._index_config.embedding_model,
            async_client=self._llm_service.client,
        )
        budget_planner = BudgetPlanner(
            document_store,
            self._index_config.target_chunk_tokens,
        )

        query_config = QueryConfig(
            budget_tokens=self._index_config.preceding_summary_budget_tokens,
        )

        return Retriever(
            query_config=query_config,
            document_store=document_store,
            embedding_service=embedding_service,
            budget_planner=budget_planner,
            vector_index=vector_index,
        )
