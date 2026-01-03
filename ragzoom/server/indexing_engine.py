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
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.server.run_manager import TelemetryRunManager

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


def _apply_token_cap(
    tiling_ids: list[str],
    nodes: dict[str, TreeNode],
    token_cap: int,
) -> tuple[list[str], int]:
    """Select smallest suffix of tiling with total tokens >= token_cap.

    Walks backward from the end of the tiling, accumulating nodes until
    the cumulative token count meets or exceeds token_cap. Returns the
    selected node IDs (in document order) and the total token count.

    This rounds UP to whole nodes - if token_cap=200 and the last node
    has 150 tokens, we include the previous node even if it pushes us
    over the budget. The semantics are "at least token_cap tokens".

    The tiling is a complete, gapless sequence of adjacent nodes covering
    some portion of the document. This function selects the rightmost
    portion of that tiling.

    Args:
        tiling_ids: Node IDs in document order (left to right)
        nodes: Node data keyed by ID
        token_cap: Minimum token budget to meet

    Returns:
        Tuple of (selected_ids in document order, total_tokens)
    """
    if not tiling_ids or token_cap <= 0:
        return [], 0

    # Walk backward accumulating tokens until we meet the cap
    cumulative = 0
    start_idx = len(tiling_ids)

    for i in range(len(tiling_ids) - 1, -1, -1):
        node = nodes[tiling_ids[i]]
        cumulative += node.token_count
        start_idx = i
        if cumulative >= token_cap:
            break

    selected = tiling_ids[start_idx:]
    return selected, cumulative


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
    telemetry_by_run: dict[str, TelemetryCollector] = field(default_factory=dict)
    run_queue: deque[str] = field(default_factory=deque)
    run_assignments: dict[str, str] = field(default_factory=dict)
    inflight_by_run: dict[str, int] = field(default_factory=dict)
    job_run_ids: dict[IndexingJob, str] = field(default_factory=dict)
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled: bool = False
    # Track permanently failed jobs to avoid infinite retry loops
    failed_job_ids: set[IndexingJob] | None = None
    # Expected total jobs for progress tracking (set at append time)
    expected_total_jobs: int = 0
    # Leaf count when document last went idle (baseline for incremental progress)
    leaves_at_last_idle: int = 0

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
        telemetry_run_manager: TelemetryRunManager | None = None,
        max_parallelism: int = 30,
        on_document_idle: OnDocumentIdleCallback | None = None,
    ) -> None:
        self._store = store
        self._llm_service = llm_service
        self._index_config = index_config
        self._openai_client = openai_client
        self._vector_index_factory = vector_index_factory
        self._telemetry_run_manager = telemetry_run_manager
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

        # Scheduling coalescing - prevents N completing jobs from triggering N
        # separate scheduling calls. Instead, they mark documents dirty and a
        # single scheduler task processes all of them.
        self._dirty_documents: set[str] = set()
        self._scheduler_task: asyncio.Task[None] | None = None

        # Store references to running job tasks to prevent garbage collection.
        # Without this, tasks can be GC'd mid-execution, cancelling the coroutine.
        self._job_tasks: set[asyncio.Task[None]] = set()

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
            telemetry_collector: Optional legacy collector for timing/metrics
        """
        # Count leaves to calculate expected work
        store = self._store.for_document(document_id)
        leaf_count = store.nodes.leaf_count()

        async with self._lock:
            # Create or update document context
            if document_id not in self._document_contexts:
                # First trigger: all leaves are new work
                self._document_contexts[document_id] = DocumentContext(
                    telemetry_collector=telemetry_collector,
                    expected_total_jobs=_expected_total_from_leaf_count(leaf_count),
                )
            else:
                ctx = self._document_contexts[document_id]
                if telemetry_collector is not None and not ctx.run_queue:
                    ctx.telemetry_collector = telemetry_collector
                ctx.cancelled = False
                # Calculate incremental expected jobs since last idle
                total_expected = _expected_total_from_leaf_count(leaf_count)
                baseline_expected = _expected_total_from_leaf_count(
                    ctx.leaves_at_last_idle
                )
                ctx.expected_total_jobs = total_expected - baseline_expected
                # Reset completed count when starting new work after idle
                if document_id not in self._active_documents:
                    ctx.completed_jobs = 0

            self._active_documents.add(document_id)
            self._idle_event.clear()

        await self._find_and_start_jobs(document_id)

    async def register_run(
        self,
        document_id: str,
        *,
        run_id: str,
        telemetry_collector: TelemetryCollector | None,
        new_leaf_ids: Sequence[str],
    ) -> None:
        """Attach a telemetry run to new leaves so jobs use the right collector."""
        if telemetry_collector is None:
            return

        async with self._lock:
            ctx = self._document_contexts.get(document_id)
            if ctx is None:
                ctx = DocumentContext()
                self._document_contexts[document_id] = ctx

            if run_id not in ctx.run_queue:
                ctx.run_queue.append(run_id)
            ctx.telemetry_by_run[run_id] = telemetry_collector

            for leaf_id in new_leaf_ids:
                if leaf_id:
                    ctx.run_assignments[leaf_id] = run_id

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
        """Get current indexing activity snapshot.

        Progress is computed from actual database state:
        - completed = leaves with embeddings + inner nodes (all existing inner nodes are complete)
        - expected = 2*N - popcount(N) where N = leaf count (total nodes in a complete forest)
        """
        async with self._lock:
            in_flight_by_doc: dict[str, int] = {}
            for job in self._active_jobs:
                in_flight_by_doc[job.document_id] = (
                    in_flight_by_doc.get(job.document_id, 0) + 1
                )

            # Get active document IDs (either have active jobs or are being tracked)
            active_doc_ids = set(in_flight_by_doc.keys()) | set(
                self._document_contexts.keys()
            )

        # Query actual DB state outside the lock to avoid blocking
        completed_by_doc: dict[str, int] = {}
        expected_total_by_doc: dict[str, int] = {}

        for doc_id in active_doc_ids:
            store = self._store.for_document(doc_id)

            # Completed = leaves with embeddings + inner nodes
            # Inner nodes are created atomically with their summary, so all existing
            # inner nodes (height > 0) are complete
            leaves_with_embeddings = store.nodes.count_leaves_with_embeddings()
            total_nodes = store.nodes.count()
            leaf_count = store.nodes.leaf_count()
            inner_nodes = total_nodes - leaf_count

            completed = leaves_with_embeddings + inner_nodes

            # Expected total = 2*N - popcount(N) where N = leaf count
            # This is the number of nodes in a complete binary forest
            expected = _expected_total_from_leaf_count(leaf_count)

            if completed > 0:
                completed_by_doc[doc_id] = completed
            if expected > 0:
                expected_total_by_doc[doc_id] = expected

        return IndexingStatus(
            in_flight=len(in_flight_by_doc),
            in_flight_by_document=in_flight_by_doc,
            completed_by_document=completed_by_doc,
            expected_total_by_document=expected_total_by_doc,
        )

    async def shutdown(self) -> None:
        """Wait for all work to complete and clean up tasks."""
        await self.wait_until_idle()
        # Cancel any remaining job tasks to allow clean shutdown
        for task in list(self._job_tasks):
            if not task.done():
                task.cancel()
        self._job_tasks.clear()

    # -----------------------------------------------------------------------
    # Job discovery and scheduling
    # -----------------------------------------------------------------------

    def _request_scheduling(self, document_id: str) -> None:
        """Request scheduling for a document, coalescing multiple requests.

        When multiple jobs complete simultaneously, each calls this method.
        Instead of triggering N separate scheduling passes, we mark the document
        dirty and ensure a single scheduler task processes all dirty documents.
        """
        self._dirty_documents.add(document_id)

        # Start scheduler if not already running
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(self._run_scheduler())

    async def _run_scheduler(self) -> None:
        """Process all dirty documents, finding and starting jobs for each.

        Yields once at start to let more completions accumulate, then processes
        all dirty documents in a loop until none remain.
        """
        # Yield to let more job completions accumulate their scheduling requests
        await asyncio.sleep(0)

        while self._dirty_documents:
            # Pop one document at a time (set iteration isn't safe during modification)
            document_id = self._dirty_documents.pop()
            await self._find_and_start_jobs(document_id)

    async def _find_and_start_jobs(self, document_id: str) -> None:
        """Find and start eligible jobs up to max parallelism.

        Uses batch discovery to find multiple jobs at once, then fires them
        all simultaneously. This reduces job discovery overhead and ensures
        concurrent API requests hit the server together.
        """
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > 100:
                logger.warning(
                    "_find_and_start_jobs loop count %d, breaking", loop_count
                )
                break

            # Check state and compute available slots under lock
            async with self._lock:
                ctx = self._document_contexts.get(document_id)
                if ctx is not None and ctx.cancelled:
                    self._active_documents.discard(document_id)
                    self._notify_state_change()
                    return

                available_slots = self._max_parallelism - len(self._active_jobs)
                if available_slots <= 0:
                    return

                # Copy active jobs to check against outside lock
                current_active = set(self._active_jobs)

            # Find multiple jobs at once outside lock (does I/O)
            jobs = self._find_next_n_jobs(
                document_id, current_active, ctx, available_slots
            )

            finalize_runs = False
            # Acquire lock to add jobs and fire them all at once
            async with self._lock:
                if not jobs:
                    # No jobs found - check if document is truly idle
                    has_active_jobs = any(
                        j.document_id == document_id for j in self._active_jobs
                    )

                    if has_active_jobs:
                        # Jobs still running - they'll re-trigger when done
                        return

                    self._active_documents.discard(document_id)
                    self._notify_state_change()

                    # Update leaves count but keep progress counters so displays
                    # can show the final "completed=X/X inflight=0" state.
                    # Counters reset on the next trigger_work() call.
                    ctx = self._document_contexts.get(document_id)
                    if ctx is not None:
                        store = self._store.for_document(document_id)
                        ctx.leaves_at_last_idle = store.nodes.leaf_count()

                    # Fire callback outside lock - document is truly idle
                    if self._on_document_idle is not None:
                        asyncio.create_task(self._safe_on_document_idle(document_id))
                    finalize_runs = True
                else:
                    # Filter out any jobs already added by another task
                    new_jobs = [j for j in jobs if j not in self._active_jobs]
                    if not new_jobs:
                        continue

                    # Re-check parallelism limit and trim if needed
                    available_now = self._max_parallelism - len(self._active_jobs)
                    if available_now <= 0:
                        return
                    new_jobs = new_jobs[:available_now]

                    # Add all jobs to active set
                    for job in new_jobs:
                        self._active_jobs.add(job)

                        ctx = self._document_contexts.get(document_id)
                        if ctx is not None:
                            run_id = self._assign_run_id_locked(
                                ctx, self._job_node_id(job)
                            )
                            if run_id is not None:
                                ctx.inflight_by_run[run_id] = (
                                    ctx.inflight_by_run.get(run_id, 0) + 1
                                )
                                ctx.job_run_ids[job] = run_id

                    # Fire all jobs simultaneously.
                    # Store task references to prevent garbage collection - without
                    # this, tasks can be GC'd mid-execution, cancelling coroutines.
                    for job in new_jobs:
                        task = asyncio.create_task(self._run_job(job))
                        self._job_tasks.add(task)
                        task.add_done_callback(self._job_tasks.discard)

                    logger.debug(
                        "engine: started %d jobs (active=%d)",
                        len(new_jobs),
                        len(self._active_jobs),
                    )

            if finalize_runs:
                await self._maybe_complete_runs(document_id)
                return

    def _notify_state_change(self) -> None:
        """Wake up waiters after state changes. Must hold lock."""
        # Always set the event to wake up waiters so they can re-check conditions
        self._idle_event.set()

    def _job_node_id(self, job: IndexingJob) -> str:
        if isinstance(job, EmbeddingJob):
            return job.leaf_id
        return job.left_id

    def _assign_run_id_locked(self, ctx: DocumentContext, node_id: str) -> str | None:
        """Resolve and attach a run ID for a node. Caller must hold _lock."""
        run_id = ctx.run_assignments.get(node_id)
        if run_id is None and ctx.run_queue:
            run_id = ctx.run_queue[-1]
            ctx.run_assignments[node_id] = run_id
        return run_id

    def _telemetry_for_node_locked(
        self, ctx: DocumentContext, node_id: str
    ) -> TelemetryCollector | None:
        run_id = self._assign_run_id_locked(ctx, node_id)
        if run_id is None:
            return ctx.telemetry_collector
        collector = ctx.telemetry_by_run.get(run_id)
        if collector is None:
            raise RuntimeError(
                f"Telemetry run {run_id} missing collector for node {node_id}"
            )
        return collector

    async def _run_context_for_node(
        self, document_id: str, node_id: str
    ) -> tuple[str | None, TelemetryCollector | None]:
        async with self._lock:
            ctx = self._document_contexts.get(document_id)
            if ctx is None:
                return None, None
            run_id = self._assign_run_id_locked(ctx, node_id)
            telemetry = self._telemetry_for_node_locked(ctx, node_id)
            return run_id, telemetry

    def _detach_run_locked(self, ctx: DocumentContext, run_id: str) -> None:
        try:
            ctx.run_queue.remove(run_id)
        except ValueError:
            pass
        ctx.telemetry_by_run.pop(run_id, None)
        ctx.inflight_by_run.pop(run_id, None)

        if ctx.job_run_ids:
            for job, assigned in list(ctx.job_run_ids.items()):
                if assigned == run_id:
                    ctx.job_run_ids.pop(job, None)

        if ctx.run_assignments:
            for node_id, assigned in list(ctx.run_assignments.items()):
                if assigned == run_id:
                    ctx.run_assignments.pop(node_id, None)

    async def _maybe_complete_runs(self, document_id: str) -> None:
        if self._telemetry_run_manager is None:
            return

        while True:
            async with self._lock:
                ctx = self._document_contexts.get(document_id)
                if ctx is None or not ctx.run_queue:
                    return
                run_id = ctx.run_queue[0]
                inflight = ctx.inflight_by_run.get(run_id, 0)
                active_jobs = set(self._active_jobs)

            if inflight != 0:
                return

            if self._has_eligible_job_for_run(document_id, ctx, run_id, active_jobs):
                return

            await self._telemetry_run_manager.complete_run(run_id, error=None)

            async with self._lock:
                ctx = self._document_contexts.get(document_id)
                if ctx is None:
                    return
                self._detach_run_locked(ctx, run_id)

    def _has_eligible_job_for_run(
        self,
        document_id: str,
        ctx: DocumentContext,
        run_id: str,
        active_jobs: set[IndexingJob],
    ) -> bool:
        store = self._store.for_document(document_id)

        leaf_config = self._index_config.preceding_context.leaf
        inner_config = self._index_config.preceding_context.inner

        embedding_frontier = self._calculate_eligibility_frontier(
            store,
            document_id,
            leaf_config.min_forest_completeness,
            leaf_config.verbatim_tokens,
        )
        summary_frontier = self._calculate_eligibility_frontier(
            store,
            document_id,
            inner_config.min_forest_completeness,
            inner_config.verbatim_tokens,
        )

        embedding_jobs = self._find_next_n_embedding_jobs(
            store,
            document_id,
            active_jobs,
            ctx,
            embedding_frontier,
            max_jobs=1,
            run_id=run_id,
        )
        if embedding_jobs:
            return True

        summary_jobs = self._find_next_n_summary_jobs(
            store,
            document_id,
            active_jobs,
            ctx,
            summary_frontier,
            inner_config.max_forest_height_differential,
            max_jobs=1,
            run_id=run_id,
        )
        return bool(summary_jobs)

    def _find_next_job(
        self,
        document_id: str,
        active_jobs: set[IndexingJob],
        ctx: DocumentContext | None,
    ) -> IndexingJob | None:
        """Find the next eligible job (convenience wrapper for tests).

        This is a thin wrapper around _find_next_n_jobs that returns the first
        job or None. Used by tests; the main scheduling loop uses _find_next_n_jobs
        directly for batch discovery.
        """
        jobs = self._find_next_n_jobs(document_id, active_jobs, ctx, max_jobs=1)
        return jobs[0] if jobs else None

    async def _safe_on_document_idle(self, document_id: str) -> None:
        """Call on_document_idle callback with error handling."""
        if self._on_document_idle is None:
            return
        try:
            await self._on_document_idle(document_id)
        except Exception:
            logger.exception("on_document_idle callback failed for %s", document_id)

    def _find_next_n_jobs(
        self,
        document_id: str,
        active_jobs: set[IndexingJob],
        ctx: DocumentContext | None,
        max_jobs: int,
    ) -> list[IndexingJob]:
        """Find up to max_jobs eligible jobs, sorted by span_start.

        Calls batch scanners for embedding and summary jobs, merges them
        sorted by span_start, and returns up to max_jobs.

        Each job type uses its own forest completeness gating:
        - Embedding jobs use leaf.min_forest_completeness (typically 1.0)
        - Summary jobs use inner.min_forest_completeness (typically 0.0 = no gating)

        Uses streaming iterators to avoid loading all nodes into memory.
        """
        store = self._store.for_document(document_id)

        # Calculate separate frontiers for each job type
        leaf_config = self._index_config.preceding_context.leaf
        inner_config = self._index_config.preceding_context.inner

        embedding_frontier = self._calculate_eligibility_frontier(
            store,
            document_id,
            leaf_config.min_forest_completeness,
            leaf_config.verbatim_tokens,
        )
        summary_frontier = self._calculate_eligibility_frontier(
            store,
            document_id,
            inner_config.min_forest_completeness,
            inner_config.verbatim_tokens,
        )

        # Find candidates from each scanner (returns (span_start, job) tuples)
        embedding_jobs = self._find_next_n_embedding_jobs(
            store,
            document_id,
            active_jobs,
            ctx,
            embedding_frontier,
            max_jobs,
        )
        summary_jobs = self._find_next_n_summary_jobs(
            store,
            document_id,
            active_jobs,
            ctx,
            summary_frontier,
            inner_config.max_forest_height_differential,
            max_jobs,
        )

        # Log job discovery results for diagnostics
        logger.info(
            "JOB_DISCOVERY: doc=%s embed_jobs=%d summary_jobs=%d "
            "embed_frontier=%s summary_frontier=%s",
            document_id[:8],
            len(embedding_jobs),
            len(summary_jobs),
            embedding_frontier,
            summary_frontier,
        )

        # Merge both lists sorted by span_start, take up to max_jobs
        all_jobs: list[tuple[int, IndexingJob]] = []
        all_jobs.extend(embedding_jobs)
        all_jobs.extend(summary_jobs)
        all_jobs.sort(key=lambda x: x[0])

        return [job for _, job in all_jobs[:max_jobs]]

    def _find_next_n_embedding_jobs(
        self,
        store: DocumentStore,
        document_id: str,
        active_jobs: set[IndexingJob],
        ctx: DocumentContext | None,
        frontier: int | None,
        max_jobs: int,
        run_id: str | None = None,
    ) -> list[tuple[int, EmbeddingJob]]:
        """Scan leaves for up to max_jobs eligible embedding jobs.

        Returns list of (span_start, job) tuples for merging with summary jobs.
        Scans all leaves (not just roots) because leaves may have been
        summarized before their embedding job started.

        Uses iter_leaves() for memory-efficient streaming - leaves are yielded
        ordered by span_start, allowing early exit without loading all leaves.

        Note: max_forest_height_differential is not supported for embedding jobs
        in this streaming implementation. The default config has it set to null
        for leaf nodes anyway.
        """
        results: list[tuple[int, EmbeddingJob]] = []

        for leaf in store.nodes.iter_leaves():
            if len(results) >= max_jobs:
                break

            leaf_span_start = int(getattr(leaf, "span_start", 0))

            # Check frontier
            if frontier is not None and leaf_span_start > frontier:
                break

            # Check if needs embedding
            if not self._has_embedding(leaf, document_id):
                if run_id is not None:
                    assigned_run = ctx.run_assignments.get(leaf.id) if ctx else None
                    if assigned_run != run_id:
                        continue
                embedding_job = EmbeddingJob(document_id, leaf.id)
                if embedding_job not in active_jobs:
                    if ctx is not None and ctx.is_failed(embedding_job):
                        continue
                    results.append((leaf_span_start, embedding_job))

        return results

    def _find_next_n_summary_jobs(
        self,
        store: DocumentStore,
        document_id: str,
        active_jobs: set[IndexingJob],
        ctx: DocumentContext | None,
        frontier: int | None,
        max_height_diff: int | None,
        max_jobs: int,
        run_id: str | None = None,
    ) -> list[tuple[int, SummaryJob]]:
        """Scan roots for up to max_jobs eligible sibling pairs to summarize.

        Returns list of (span_start, job) tuples for merging with embedding jobs.

        Uses iter_root_nodes() for memory-efficient streaming with a sliding window
        pattern - tracks previous root and min_preceding_height incrementally.
        """
        results: list[tuple[int, SummaryJob]] = []
        # Track which roots are already used in a job (can't use same root twice)
        used_roots: set[str] = set()

        # Sliding window state for iterator-based approach
        prev_root: TreeNode | None = None
        min_preceding_height: int | None = None

        # Diagnostic counters
        roots_scanned = 0
        pairs_checked = 0
        pairs_ineligible = 0
        pairs_height_blocked = 0
        pairs_active = 0
        pairs_failed = 0

        for root in store.nodes.iter_root_nodes():
            roots_scanned += 1
            if len(results) >= max_jobs:
                break

            root_span_start = int(getattr(root, "span_start", 0))
            root_height = int(getattr(root, "height", 0))

            # Check frontier
            if frontier is not None and root_span_start > frontier:
                break

            # Try to form pair with previous root (if prev_root exists and eligible)
            if prev_root is not None and prev_root.id not in used_roots:
                if root.id not in used_roots:
                    pairs_checked += 1
                    if self._is_eligible_pair(prev_root, root, document_id):
                        # Check height differential constraint
                        if (
                            max_height_diff is not None
                            and min_preceding_height is not None
                        ):
                            left_height = int(getattr(prev_root, "height", 0))
                            parent_height = left_height + 1
                            if parent_height - min_preceding_height > max_height_diff:
                                pairs_height_blocked += 1
                                logger.info(
                                    "HEIGHT_BLOCKED: parent_h=%d min_preceding=%d "
                                    "diff=%d max_allowed=%d",
                                    parent_height,
                                    min_preceding_height,
                                    parent_height - min_preceding_height,
                                    max_height_diff,
                                )
                                break

                        if run_id is not None:
                            assigned_run = (
                                ctx.run_assignments.get(prev_root.id) if ctx else None
                            )
                            if assigned_run != run_id:
                                # Skip this pair but continue scanning
                                prev_root = root
                                if (
                                    min_preceding_height is None
                                    or root_height < min_preceding_height
                                ):
                                    min_preceding_height = root_height
                                continue

                        prev_span_start = int(getattr(prev_root, "span_start", 0))
                        summary_job = SummaryJob(document_id, prev_root.id, root.id)
                        if summary_job not in active_jobs:
                            if ctx is not None and ctx.is_failed(summary_job):
                                pairs_failed += 1
                                # Skip this pair but continue scanning
                                prev_root = root
                                if (
                                    min_preceding_height is None
                                    or root_height < min_preceding_height
                                ):
                                    min_preceding_height = root_height
                                continue
                            results.append((prev_span_start, summary_job))
                            used_roots.add(prev_root.id)
                            used_roots.add(root.id)
                        else:
                            pairs_active += 1
                    else:
                        pairs_ineligible += 1

            # Update sliding window state
            prev_root = root
            # Update min_preceding_height for next iteration
            if min_preceding_height is None or root_height < min_preceding_height:
                min_preceding_height = root_height

        # Log diagnostic info if no jobs found but roots exist
        if len(results) == 0 and roots_scanned > 1:
            # Collect root info for diagnostics (height, level, span_start)
            root_info: list[tuple[int, int, int]] = []
            for r in store.nodes.iter_root_nodes():
                root_info.append(
                    (
                        int(getattr(r, "height", 0)),
                        int(getattr(r, "level_index", 0)),
                        int(getattr(r, "span_start", 0)),
                    )
                )
                if len(root_info) >= 10:  # Limit to first 10
                    break
            logger.info(
                "SUMMARY_SCAN: doc=%s roots=%d checked=%d "
                "inelig=%d h_block=%d active=%d failed=%d "
                "root_info(h,lvl,span)=%s",
                document_id[:8],
                roots_scanned,
                pairs_checked,
                pairs_ineligible,
                pairs_height_blocked,
                pairs_active,
                pairs_failed,
                root_info,
            )

        return results

    def _calculate_eligibility_frontier(
        self,
        store: DocumentStore,
        document_id: str,
        min_forest_completeness: float,
        verbatim_tokens: int,
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

        Uses iter_root_nodes() for memory-efficient streaming - exits early when
        first ineligible root is found.

        Args:
            store: Document store for metadata lookups
            document_id: Document being indexed
            min_forest_completeness: Minimum completeness threshold (0.0 = no gating)
            verbatim_tokens: Token budget for verbatim context (used for frontier offset)
        """
        # No gating if min_forest_completeness is 0
        if min_forest_completeness <= 0.0:
            return None

        # Track cumulative forest statistics as we scan left-to-right
        preceding_leaves = 0
        preceding_roots = 0
        preceding_max_height = 0

        for root in store.nodes.iter_root_nodes():
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
        - Inner nodes use token_cap for preceding context to select the
          rightmost portion of the tiling, avoiding embedding lookups.
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
        job_type = "embed" if isinstance(job, EmbeddingJob) else "summarize"
        logger.info("JOB_START: %s job=%s", job_type, job)
        try:
            if isinstance(job, EmbeddingJob):
                await self._embed_leaf(job)
            else:
                await self._summarize_pair(job)
            logger.info("JOB_END: %s job=%s success=True", job_type, job)
        except asyncio.CancelledError:
            logger.info("JOB_END: %s job=%s cancelled=True", job_type, job)
            raise
        except Exception:
            logger.info("JOB_END: %s job=%s success=False", job_type, job)
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
                    run_id = ctx.job_run_ids.pop(job, None)
                    if run_id is not None:
                        inflight = ctx.inflight_by_run.get(run_id, 0)
                        if inflight > 1:
                            ctx.inflight_by_run[run_id] = inflight - 1
                        else:
                            ctx.inflight_by_run.pop(run_id, None)

                logger.debug(
                    "engine: finished job %s (active=%d, failed=%s)",
                    job,
                    len(self._active_jobs),
                    job_failed,
                )

                # Wake up any waiters to re-check their conditions
                self._idle_event.set()

                # Request scheduling (coalesced with other job completions)
                self._request_scheduling(document_id)
            await self._maybe_complete_runs(document_id)

    async def _embed_leaf(self, job: EmbeddingJob) -> None:
        """Generate contextual embedding for a leaf node.

        Steps:
        1. Retrieve preceding context (with query embedding for cost tracking)
        2. Summarize context into prefix (LLM call)
        3. Generate embedding with context prefix
        4. Write to vector index
        """
        from ragzoom.contracts.embedding_model import EmbeddingUsageInfo

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

        _run_id, telemetry = await self._run_context_for_node(
            job.document_id, job.leaf_id
        )

        # Get leaf-specific preceding context config
        leaf_config = self._index_config.preceding_context.leaf

        # Pre-compute query embedding with usage tracking (only if we need semantic retrieval)
        retrieval_embedding_usage: EmbeddingUsageInfo = {"total_tokens": 0, "model": ""}
        query_embedding: list[float] | None = None
        retrieval_start_time = time.time()

        # Only compute query embedding if span_start > 0 AND num_seeds != 0
        # (num_seeds=0 means skip semantic search, so no embedding needed)
        needs_semantic_retrieval = span_start > 0 and (leaf_config.num_seeds or 0) != 0
        if needs_semantic_retrieval:
            # Get retrieval embedding with usage info
            retriever = self._create_retriever(job.document_id)
            if retriever is not None:
                query_embedding, retrieval_embedding_usage = (
                    await retriever.embedding_service.get_query_embedding_async_with_usage(
                        leaf_text, job.document_id
                    )
                )

        # Retrieve preceding context (pass pre-computed embedding to skip API call)
        logger.info(
            "EMBED_CONTEXT_START: leaf=%s span_start=%s", job.leaf_id, span_start
        )
        context_result = await self._get_preceding_context(
            store=store,
            document_id=job.document_id,
            span_start=span_start,
            config=leaf_config,
            query_text=leaf_text,
            query_embedding=query_embedding,
        )
        logger.info(
            "EMBED_CONTEXT_END: leaf=%s tiling_count=%s",
            job.leaf_id,
            len(context_result.tiling_ids),
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

        # Contextualize preceding context if present, then build embedding text
        context_summary = ""
        contextualization_result = None
        if context_prefix:
            # Generate a contextualizing summary of preceding context
            # (extracts only information relevant to understanding the leaf)
            logger.info("EMBED_LLM_START: leaf=%s contextualize", job.leaf_id)
            contextualization_result = await self._llm_service._contextualize_text(
                preceding_context=context_prefix,
                target_text=leaf_text,
                target_tokens=self._index_config.target_chunk_tokens,
                parent_id=job.leaf_id,
                reporter=telemetry,
            )
            logger.info("EMBED_LLM_END: leaf=%s contextualize", job.leaf_id)
            context_summary = contextualization_result.summary
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
        logger.info("EMBED_API_START: leaf=%s", job.leaf_id)
        embed_result = await self._llm_service.embed_texts_with_usage([text_to_embed])
        logger.info("EMBED_API_END: leaf=%s", job.leaf_id)
        embeddings = embed_result["embeddings"]
        leaf_embedding_usage = embed_result["usage"]
        if not embeddings:
            logger.error(
                "embed: no embedding returned doc=%s leaf=%s",
                job.document_id,
                job.leaf_id,
            )
            return

        # Combine retrieval and leaf embedding tokens for telemetry and cost
        retrieval_tokens = retrieval_embedding_usage.get("total_tokens", 0)
        leaf_tokens = leaf_embedding_usage.get("total_tokens", 0)
        total_embedding_tokens = retrieval_tokens + leaf_tokens

        # Record embedding telemetry (combined for both API calls)
        if telemetry is not None:
            telemetry.record_embedding_call_v2(
                node_embeddings=[(job.leaf_id, total_embedding_tokens)],
                batch_size=2 if retrieval_tokens > 0 else 1,
                model=self._index_config.embedding_model,
                start_time=embed_start_time,
            )

        # Compute and store leaf cost (embedding + contextualization if present)
        from ragzoom.config import get_embedding_cost, get_llm_costs
        from ragzoom.cost import (
            calculate_completion_cost,
            calculate_embedding_cost,
            calculate_prompt_cost_with_cache,
        )
        from ragzoom.model_info import ModelInfo

        embedding_cost_per_1k = get_embedding_cost(self._index_config.embedding_model)
        leaf_cost = calculate_embedding_cost(
            total_embedding_tokens, embedding_cost_per_1k
        )

        # Add contextualization LLM cost if we made that call
        if contextualization_result is not None:
            input_cost_per_1k, output_cost_per_1k = get_llm_costs(
                self._index_config.summary_model
            )
            model_info = ModelInfo()
            cache_discount = model_info.get_cache_discount(
                self._index_config.summary_model
            )
            usage = contextualization_result.usage
            leaf_cost += calculate_prompt_cost_with_cache(
                usage.prompt_tokens,
                usage.cached_tokens,
                input_cost_per_1k,
                cache_discount,
            ) + calculate_completion_cost(usage.completion_tokens, output_cost_per_1k)

        store.nodes._repo.update_cost(job.leaf_id, leaf_cost)

        # Store embedding on the node for algorithmic access
        embedding_array = np.asarray(embeddings[0], dtype=np.float64)
        store.nodes._repo.update_embedding(job.leaf_id, embedding_array)

        # Write to vector index for similarity search
        vector_index = self._get_vector_index()
        if vector_index is None:
            raise RuntimeError(
                f"Cannot persist embedding for leaf {job.leaf_id}: "
                "no vector_index_factory configured. "
                "Set RAGZOOM_VECTOR_BACKEND to 'pgvector', 'chroma', or 'python'."
            )
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

        run_id, telemetry = await self._run_context_for_node(
            job.document_id, job.left_id
        )

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
        logger.info("CONTEXT_START: span_start=%s doc=%s", span_start, job.document_id)
        context_result = await self._get_preceding_context(
            store=store,
            document_id=job.document_id,
            span_start=span_start,
            config=inner_config,
            query_text=None,
        )
        logger.info(
            "CONTEXT_END: span_start=%s tiling_count=%s",
            span_start,
            len(context_result.tiling_ids),
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
            context_parts = []
            for node_id in tiling_ids:
                node = context_nodes.get(node_id)
                if node and node.text:
                    context_parts.append(node.text)
            context_text = " ".join(context_parts)

        preceding_context_json = json.dumps(tiling_ids)

        # Generate summary with preceding context
        combined_text = f"{left_text} {right_text}".strip()
        combined_tokens = (
            left_tokens + right_tokens
            if left_tokens is not None and right_tokens is not None
            else None
        )
        logger.info(
            "SUMMARIZE_START: parent_id=%s combined_tokens=%s",
            parent_id,
            combined_tokens,
        )
        summary_result = await self._llm_service._summarize_text(
            combined_text,
            self._index_config.target_chunk_tokens,
            parent_id=parent_id,
            reporter=telemetry,
            prev_context=context_text,
            text_tokens=combined_tokens,
        )
        logger.info("SUMMARIZE_END: parent_id=%s", parent_id)
        summary = summary_result.summary
        summary_tokens = summary_result.summary_tokens

        # Compute summary cost using actual token counts from API response
        from ragzoom.config import get_llm_costs
        from ragzoom.cost import (
            calculate_completion_cost,
            calculate_prompt_cost_with_cache,
        )
        from ragzoom.model_info import ModelInfo

        input_cost_per_1k, output_cost_per_1k = get_llm_costs(
            self._index_config.summary_model
        )
        model_info = ModelInfo()
        cache_discount = model_info.get_cache_discount(self._index_config.summary_model)

        # Use actual token counts from accumulated usage across all attempts
        usage = summary_result.usage
        summary_cost = calculate_prompt_cost_with_cache(
            usage.prompt_tokens,
            usage.cached_tokens,
            input_cost_per_1k,
            cache_discount,
        ) + calculate_completion_cost(usage.completion_tokens, output_cost_per_1k)

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
            "cost": summary_cost,
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

        if run_id is not None:
            async with self._lock:
                ctx = self._document_contexts.get(job.document_id)
                if ctx is not None:
                    ctx.run_assignments[parent_id] = run_id

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
        from ragzoom.config import QueryConfig
        from ragzoom.retrieval.budget_planner import BudgetPlanner
        from ragzoom.retrieval.embedding_service import EmbeddingService
        from ragzoom.retrieve import Retriever

        document_store = self._store.for_document(document_id)
        vector_index = self._get_vector_index()
        if vector_index is None:
            return None

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
        query_embedding: list[float] | None = None,
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
            query_embedding: Pre-computed query embedding. If provided, skips the
                embedding API call in retrieval.

        Returns:
            PrecedingContextResult with tiling node IDs, node data, and token count.
        """
        if span_start <= 0:
            return PrecedingContextResult(tiling_ids=[], nodes={}, tiling_tokens=0)

        # Skip retrieval entirely if token_cap is 0
        if config.token_cap == 0:
            return PrecedingContextResult(tiling_ids=[], nodes={}, tiling_tokens=0)

        # Always use standard retrieval path
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
                query_embedding=query_embedding,
            )
            if (
                context_result is not None
                and context_result.tiling
                and context_result.nodes
            ):
                tiling_ids = list(context_result.tiling)
                context_nodes = context_result.nodes

                # Apply token_cap if set: select rightmost nodes totaling >= token_cap
                if config.token_cap is not None:
                    tiling_ids, tiling_tokens = _apply_token_cap(
                        tiling_ids, context_nodes, config.token_cap
                    )
                else:
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
