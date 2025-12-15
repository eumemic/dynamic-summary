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
import time
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

    from ragzoom.config import IndexConfig, PrecedingContextConfig
    from ragzoom.contracts.storage_backend import StorageBackend
    from ragzoom.contracts.vector_index import VectorIndex
    from ragzoom.document_store import DocumentStore
    from ragzoom.retrieve import Retriever
    from ragzoom.services.llm_service import LLMService
    from ragzoom.telemetry_collection import TelemetryCollector

# Type alias for vector index factory
VectorIndexFactory = Callable[[str], "VectorIndex"]

# Callback type for document idle events (receives document_id)
OnDocumentIdleCallback = Callable[[str], "Awaitable[None]"]

logger = logging.getLogger(__name__)


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


def _min_roots_for_leaf_count(n: int) -> int:
    """Return minimum number of roots for a complete forest of n leaves.

    This equals popcount(n) - the number of 1-bits in binary representation.
    A complete binary forest over n leaves has exactly popcount(n) trees.

    Examples:
        - 8 leaves (0b1000): popcount=1 → 1 root (one perfect tree of 8)
        - 7 leaves (0b111):  popcount=3 → 3 roots (trees of 4, 2, 1)
        - 5 leaves (0b101):  popcount=2 → 2 roots (trees of 4 and 1)
    """
    if n <= 0:
        return 0
    return bin(n).count("1")


def _optimal_max_height(n: int) -> int:
    """Return max height of an optimal forest for n leaves.

    This is the position of the highest set bit, i.e., floor(log2(n)).

    Examples:
        - 8 (0b1000): max_height = 3
        - 7 (0b111):  max_height = 2
        - 13 (0b1101): max_height = 3
    """
    if n <= 0:
        return 0
    return n.bit_length() - 1


def _forest_completeness(num_roots: int, max_height: int, leaf_count: int) -> float:
    """Compute forest completeness using tiling cost estimate.

    The metric is: (optimal_roots + optimal_max_h) / (actual_roots + actual_max_h)

    This estimates the ratio of worst-case tiling costs. A tiling that zooms
    to a single seed needs at most (num_roots + max_height) tiles:
    - num_roots for the minimal coverage
    - max_height additional tiles to zoom down in the tallest tree

    The ratio is always <= 1.0 because:
    - Fewer merges → more roots but lower max height
    - The sum is minimized when the forest is optimally compressed

    Examples:
        - 8 leaves, 1 root height 3: (1+3)/(1+3) = 1.0 (optimal)
        - 8 leaves, 8 roots height 0: (1+3)/(8+0) = 0.5 (no merges)
        - 8 leaves, 2 roots heights [2,2]: (1+3)/(2+2) = 1.0 (equivalent)
    """
    if leaf_count <= 1:
        return 1.0  # Trivially complete

    optimal_roots = _min_roots_for_leaf_count(leaf_count)
    optimal_max_h = _optimal_max_height(leaf_count)

    optimal_cost = optimal_roots + optimal_max_h
    actual_cost = num_roots + max_height

    if actual_cost == 0:
        return 1.0

    return optimal_cost / actual_cost


@dataclass
class PrecedingContextResult:
    """Result from preceding context retrieval."""

    tiling_ids: list[str]
    nodes: dict[str, TreeNode]
    tiling_tokens: int


def get_verbatim_context_nodes(
    store: DocumentStore,
    document_id: str,
    span_end_limit: int,
    verbatim_tokens: int,
) -> PrecedingContextResult:
    """Get rightmost nodes before span_end_limit totaling >= verbatim_tokens.

    This is a simplified retrieval path used when verbatim_nodes_only=true.
    It skips the full retrieval algorithm (embedding query, MMR, coverage building)
    and just walks backwards collecting leaves until the token budget is met.

    Args:
        store: Document store for node retrieval
        document_id: Document to retrieve from
        span_end_limit: Only include nodes with span_end <= this value
        verbatim_tokens: Target token budget for verbatim content

    Returns:
        PrecedingContextResult with node IDs in document order and their data.
    """
    if span_end_limit <= 0 or verbatim_tokens <= 0:
        return PrecedingContextResult(tiling_ids=[], nodes={}, tiling_tokens=0)

    # Use existing repository method that handles the windowed budget query
    leaves = store.nodes._repo.get_recent_leaves_within_budget_before(
        document_id, verbatim_tokens, span_end_limit
    )

    if not leaves:
        return PrecedingContextResult(tiling_ids=[], nodes={}, tiling_tokens=0)

    # Build result in document order (already sorted by span_start)
    tiling_ids = [leaf.id for leaf in leaves]
    nodes = {leaf.id: leaf for leaf in leaves}
    tiling_tokens = sum(leaf.token_count for leaf in leaves)

    return PrecedingContextResult(
        tiling_ids=tiling_ids,
        nodes=nodes,
        tiling_tokens=tiling_tokens,
    )


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
        """Find the next eligible job by checking both embedding and summary jobs.

        Calls separate scanners for embedding and summary jobs, then returns
        the leftmost job by span_start. This ensures leaves get embedded even
        after being summarized (since embedding jobs scan all leaves, not just roots).

        Each job type uses its own forest completeness gating:
        - Embedding jobs use leaf.min_forest_completeness (typically 1.0)
        - Summary jobs use inner.min_forest_completeness (typically 0.0 = no gating)
        """
        store = self._store.for_document(document_id)
        roots = store.nodes.get_root_nodes(document_id)

        # Calculate separate frontiers for each job type
        leaf_config = self._index_config.preceding_context.leaf
        inner_config = self._index_config.preceding_context.inner

        embedding_frontier = self._calculate_eligibility_frontier(
            roots, store, document_id, leaf_config.min_forest_completeness
        )
        summary_frontier = self._calculate_eligibility_frontier(
            roots, store, document_id, inner_config.min_forest_completeness
        )

        # Find candidates from each scanner with their respective frontiers
        embedding_job = self._find_next_embedding_job(
            store, document_id, active_jobs, ctx, embedding_frontier
        )
        summary_job = self._find_next_summary_job(
            roots, document_id, active_jobs, ctx, summary_frontier
        )

        # Return leftmost job by span_start
        if embedding_job is None:
            return summary_job
        if summary_job is None:
            return embedding_job

        # Both exist - compare span_start to pick leftmost
        embed_leaf = store.nodes.get(embedding_job.leaf_id)
        summary_left = store.nodes.get(summary_job.left_id)
        embed_span = int(getattr(embed_leaf, "span_start", 0)) if embed_leaf else 0
        summary_span = (
            int(getattr(summary_left, "span_start", 0)) if summary_left else 0
        )

        if embed_span <= summary_span:
            return embedding_job
        return summary_job

    def _find_next_embedding_job(
        self,
        store: DocumentStore,
        document_id: str,
        active_jobs: set[IndexingJob],
        ctx: DocumentContext | None,
        frontier: int | None,
    ) -> EmbeddingJob | None:
        """Scan all leaves for the first one needing an embedding.

        Scans all leaves (not just roots) because leaves may have been
        summarized before their embedding job started. Leaves are sorted
        by span_start for deterministic left-to-right processing.
        """
        leaves = store.nodes.get_leaves()
        # Sort by span_start for left-to-right order
        leaves.sort(key=lambda n: int(getattr(n, "span_start", 0)))

        for leaf in leaves:
            leaf_span_start = int(getattr(leaf, "span_start", 0))

            # Check frontier (embedding jobs respect the same frontier as summaries)
            if frontier is not None and leaf_span_start > frontier:
                return None

            # Check if needs embedding
            if not self._has_embedding(leaf, document_id):
                embedding_job = EmbeddingJob(document_id, leaf.id)
                if embedding_job not in active_jobs:
                    if ctx is not None and ctx.is_failed(embedding_job):
                        continue
                    return embedding_job

        return None

    def _find_next_summary_job(
        self,
        roots: list[TreeNode],
        document_id: str,
        active_jobs: set[IndexingJob],
        ctx: DocumentContext | None,
        frontier: int | None,
    ) -> SummaryJob | None:
        """Scan roots for the first eligible sibling pair to summarize."""
        for i, root in enumerate(roots):
            root_span_start = int(getattr(root, "span_start", 0))

            # Check frontier
            if frontier is not None and root_span_start > frontier:
                return None

            # Check for eligible sibling pair
            if i + 1 < len(roots):
                right = roots[i + 1]
                if self._is_eligible_pair(root, right, document_id):
                    summary_job = SummaryJob(document_id, root.id, right.id)
                    if summary_job not in active_jobs:
                        if ctx is not None and ctx.is_failed(summary_job):
                            continue
                        return summary_job

        return None

    def _calculate_eligibility_frontier(
        self,
        roots: list[TreeNode],
        store: DocumentStore,
        document_id: str,
        min_forest_completeness: float,
    ) -> int | None:
        """Calculate the last eligible position for jobs based on forest completeness.

        Returns the maximum span_start position where jobs can be scheduled, or
        None if there's no restriction (all jobs are eligible).

        The frontier is calculated as:
            first_ineligible_root.span_start + verbatim_tokens * avg_chars_per_token

        This allows jobs to extend past the strict completeness boundary by the
        verbatim token budget, since that content will be fetched as raw leaves
        rather than summaries.

        Forest completeness is measured by comparing the height histogram of
        actual roots against the optimal forest structure. This is scale-invariant
        because it compares structural shape rather than absolute counts.

        - 1.0 = heights match optimal forest exactly
        - 0.0 = maximum deviation from optimal

        Args:
            roots: Current root nodes in document order
            store: Document store for metadata lookups
            document_id: Document being indexed
            min_forest_completeness: Minimum completeness threshold (0.0 = no gating)
        """
        # No gating if min_forest_completeness is 0
        if min_forest_completeness <= 0.0:
            return None

        # Use leaf config for verbatim token budget
        leaf_config = self._index_config.preceding_context.leaf
        verbatim_tokens = leaf_config.verbatim_tokens

        # Track cumulative forest statistics as we scan left-to-right
        preceding_leaves = 0
        preceding_roots = 0
        preceding_max_height = 0

        for root in roots:
            root_height = int(getattr(root, "height", 0))
            leaves_in_subtree = 2**root_height

            # Check completeness in PRECEDING forest (before this root)
            if preceding_leaves > 1:
                completeness = _forest_completeness(
                    preceding_roots, preceding_max_height, preceding_leaves
                )

                if completeness < min_forest_completeness:
                    # Found first ineligible root - calculate frontier
                    root_span_start = int(getattr(root, "span_start", 0))

                    # Get avg chars per token for this document
                    avg_chars = store.nodes.get_avg_chars_per_token(document_id)
                    if avg_chars is None:
                        avg_chars = 4.0  # Reasonable default for English text

                    verbatim_chars = int(verbatim_tokens * avg_chars)
                    frontier = root_span_start + verbatim_chars

                    logger.debug(
                        "eligibility frontier at %d "
                        "(first_ineligible_span=%d + verbatim_chars=%d, "
                        "completeness=%.2f < min=%.2f, "
                        "preceding_leaves=%d, preceding_roots=%d)",
                        frontier,
                        root_span_start,
                        verbatim_chars,
                        completeness,
                        min_forest_completeness,
                        preceding_leaves,
                        preceding_roots,
                    )
                    return frontier

            # Update preceding stats for next iteration
            preceding_leaves += leaves_in_subtree
            preceding_roots += 1
            preceding_max_height = max(preceding_max_height, root_height)

        # No ineligibility found - all jobs are eligible
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
        """Check if a node already has an embedding."""
        return node.embedding is not None

    def _is_eligible_pair(
        self, left: TreeNode, right: TreeNode, document_id: str
    ) -> bool:
        """Check if two adjacent roots form an eligible sibling pair ready for summary.

        Two roots are eligible siblings if:
        1. They have the same height
        2. Left has an even level_index (is a left child position)
        3. Right has level_index = left.level_index + 1 (is adjacent)

        Note: No embedding requirements for summarization eligibility.
        - Leaf embeddings are for query-time retrieval, not summarization.
          Embedding jobs run in parallel with summary jobs.
        - Inner nodes use verbatim_nodes_only=True for preceding context
          (enforced by config), so they don't need embeddings either.
        - Inner nodes no longer store embeddings - score propagation happens
          at query time from descendant leaf embeddings.
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

        # Get leaf-specific preceding context config
        leaf_config = self._index_config.preceding_context.leaf

        # Retrieve preceding context (unified function handles span_start=0 case)
        retrieval_start_time = time.time()
        context_result = await self._get_preceding_context(
            store=store,
            document_id=job.document_id,
            span_start=span_start,
            config=leaf_config,
            query_text=leaf_text,
        )
        tiling_ids = context_result.tiling_ids
        context_nodes = context_result.nodes
        tiling_tokens = context_result.tiling_tokens

        # Record retrieval telemetry
        if span_start > 0 and telemetry is not None:
            telemetry.record_retrieval_call(
                node_id=job.leaf_id,
                tiling_node_count=len(tiling_ids),
                tiling_tokens=tiling_tokens,
                start_time=retrieval_start_time,
            )

        # Assemble context text from tiling nodes
        context_prefix = ""
        if tiling_ids and context_nodes:
            context_prefix = "\n\n".join(
                context_nodes[nid].text or ""
                for nid in tiling_ids
                if nid in context_nodes
            )

        # Store preceding_context as JSON array of node IDs on the leaf node
        import json

        store = self._store.for_document(job.document_id)
        preceding_context_json = json.dumps(tiling_ids)
        store.nodes._repo.update_preceding_context(job.leaf_id, preceding_context_json)

        # Summarize preceding context if present, then build embedding text
        context_summary = ""
        if context_prefix:
            # Generate a summary of the preceding context (target: target_chunk_tokens)
            context_summary, _retry_count, _summary_tokens = (
                await self._llm_service._summarize_text(
                    context_prefix,
                    self._index_config.target_chunk_tokens,
                    parent_id=job.leaf_id,
                    reporter=telemetry,
                    prev_context=None,
                    text_tokens=tiling_tokens,
                )
            )
            # Store the summary in the database
            store.nodes._repo.update_preceding_context_summary(
                job.leaf_id, context_summary
            )

        # Build embedding text: summary + leaf (no truncation needed)
        if context_summary:
            text_to_embed = f"{context_summary}\n{leaf_text}"
        else:
            text_to_embed = leaf_text

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

        # Store embedding on the node for algorithmic access
        embedding_array = np.asarray(embeddings[0], dtype=np.float64)
        store.nodes._repo.update_embedding(job.leaf_id, embedding_array)

        # Write to vector index for similarity search
        vector_index = self._get_vector_index()
        if vector_index is not None:
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

        # Get inner-specific preceding context config
        inner_config = self._index_config.preceding_context.inner

        # Retrieve preceding context (unified function handles span_start=0 case)
        # Inner nodes pass query_text=None (no semantic search, just minimal tiling)
        import json

        retrieval_start_time = time.time()
        context_result = await self._get_preceding_context(
            store=store,
            document_id=job.document_id,
            span_start=span_start,
            config=inner_config,
            query_text=None,
        )
        tiling_ids = context_result.tiling_ids
        context_nodes = context_result.nodes
        tiling_tokens = context_result.tiling_tokens

        # Record retrieval telemetry
        if span_start > 0 and telemetry is not None:
            telemetry.record_retrieval_call(
                node_id=parent_id,
                tiling_node_count=len(tiling_ids),
                tiling_tokens=tiling_tokens,
                start_time=retrieval_start_time,
            )

        # Assemble context text from tiling nodes
        context_text = ""
        if tiling_ids and context_nodes:
            context_text = "\n\n".join(
                context_nodes[nid].text or ""
                for nid in tiling_ids
                if nid in context_nodes
            )
        preceding_context_json = json.dumps(tiling_ids)

        # Generate summary with preceding context
        combined_text = f"{left_text} {right_text}".strip()
        combined_tokens = (
            left_tokens + right_tokens
            if left_tokens is not None and right_tokens is not None
            else None
        )
        summary, _retry_count, summary_tokens = await self._llm_service._summarize_text(
            combined_text,
            self._index_config.target_chunk_tokens,
            parent_id=parent_id,
            reporter=telemetry,
            prev_context=context_text,
            text_tokens=combined_tokens,
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

        # Note: Inner nodes no longer store embeddings (to enable parallel
        # summarization). Embeddings only exist on leaf nodes for query-time
        # retrieval. Score propagation at query time computes inner node scores
        # from descendant leaf embeddings.

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
            budget_tokens=self._index_config.preceding_context_budget,
        )

        return Retriever(
            query_config=query_config,
            document_store=document_store,
            embedding_service=embedding_service,
            budget_planner=budget_planner,
            vector_index=vector_index,
        )

    async def _get_preceding_context(
        self,
        store: DocumentStore,
        document_id: str,
        span_start: int,
        config: PrecedingContextConfig,
        query_text: str | None,
    ) -> PrecedingContextResult:
        """Retrieve preceding context for a node.

        Unified function for both leaf embedding and inner node summarization.

        Args:
            store: Document store for the document
            document_id: Document ID
            span_start: Start position of the node (context covers [0, span_start))
            config: Preceding context configuration (leaf or inner)
            query_text: Query text for semantic retrieval. Pass the node's text for
                leaves, or None for inner nodes (which use minimal tiling).

        Returns:
            PrecedingContextResult with tiling node IDs, node data, and token count.
        """
        if span_start <= 0:
            return PrecedingContextResult(tiling_ids=[], nodes={}, tiling_tokens=0)

        if config.verbatim_nodes_only:
            # Simplified path: get rightmost N leaves within token budget
            return get_verbatim_context_nodes(
                store,
                document_id,
                span_start,
                config.verbatim_tokens,
            )

        # Standard retrieval path with semantic search (or minimal tiling if no query)
        retriever = self._create_retriever(document_id)
        if retriever is None:
            return PrecedingContextResult(tiling_ids=[], nodes={}, tiling_tokens=0)

        try:
            context_result = await retriever.retrieve_for_context(
                query_text=query_text or "",
                span_end_limit=span_start,
                budget_tokens=self._index_config.preceding_context_budget,
                document_id=document_id,
                recent_verbatim_token_budget=config.verbatim_tokens,
                num_seeds=config.num_seeds,
            )
            if (
                context_result is not None
                and context_result.tiling
                and context_result.nodes
            ):
                tiling_ids = list(context_result.tiling)
                context_nodes = context_result.nodes
                tiling_tokens = sum(
                    context_nodes[nid].token_count
                    for nid in tiling_ids
                    if nid in context_nodes
                )
                return PrecedingContextResult(
                    tiling_ids=tiling_ids,
                    nodes=context_nodes,
                    tiling_tokens=tiling_tokens,
                )
        except Exception:
            logger.exception(
                "Failed to retrieve preceding context doc=%s span_start=%d",
                document_id,
                span_start,
            )

        return PrecedingContextResult(tiling_ids=[], nodes={}, tiling_tokens=0)
