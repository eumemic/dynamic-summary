"""Worker coordination utilities for server-managed summarization."""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from ragzoom.config import IndexConfig, OperationalConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.server.run_manager import IndexRunContext, TelemetryRunManager
from ragzoom.telemetry_embeddings import compute_fidelity_for_telemetry
from ragzoom.vector_factory import create_vector_index

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from ragzoom.telemetry_collection import TelemetryCollector
else:  # pragma: no cover - fallback for runtime when telemetry isn’t available
    TelemetryCollector = object

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadyParentCandidate:
    """Payload describing a parent node that should be built."""

    document_id: str
    left_child_id: str
    height: int
    level_index: int
    span_start: int
    run_id: str | None = None


NodeFieldValue = str | int | float | bool | list[float] | NDArray[np.float64] | None


@dataclass(frozen=True)
class WorkerStatus:
    """Snapshot of queue depth and in-flight activity."""

    queue_depth: int
    in_flight: int
    pending_by_document: dict[str, int]
    inflight_by_document: dict[str, int]
    completed_by_document: dict[str, int]


@dataclass(slots=True)
class DependencySnapshot:
    """Resolved neighbors required to build a parent."""

    preceding: TreeNode | None
    left: TreeNode | None
    right: TreeNode | None


@dataclass(slots=True)
class DocumentState:
    """Mutable per-document scheduling state."""

    queued: set[str]
    inflight: set[str]
    tombstones: set[str]
    completed: int
    run_queue: deque[str]
    pending_by_run: dict[str, int]
    inflight_by_run: dict[str, int]
    run_assignments: dict[str, str]


# jscpd:ignore-start - Signature must match LLMService._summarize_text exactly for type safety
class SummaryBackend(Protocol):
    async def _summarize_text(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: TelemetryCollector | None = None,
        prev_context: str | None = None,
        left_token_count: int | None = None,
        right_token_count: int | None = None,
    ) -> tuple[str, int, int]: ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


# jscpd:ignore-end


def compute_ready_parent_candidates(store: DocumentStore) -> list[ReadyParentCandidate]:
    document_id = store.document_id or ""
    left_ids = store.nodes.get_ready_left_children()
    nodes_by_id = {node.id: node for node in store.nodes.get_nodes(left_ids)}

    candidates: list[ReadyParentCandidate] = []
    for left_id in left_ids:
        node = nodes_by_id.get(left_id)
        if node is None:
            logger.warning(
                "Ready-left candidate %s missing node record in store", left_id
            )
            continue
        candidates.append(
            ReadyParentCandidate(
                document_id=document_id,
                left_child_id=left_id,
                height=int(getattr(node, "height", 0)),
                level_index=int(getattr(node, "level_index", 0)),
                span_start=int(getattr(node, "span_start", 0)),
                run_id=None,
            )
        )
    return candidates


class WorkerCoordinator:
    """Async queue that schedules and executes parent summarisation jobs."""

    def __init__(
        self,
        *,
        store: StorageBackend,
        index_config: IndexConfig,
        operational_config: OperationalConfig,
        llm_service: SummaryBackend,
        run_manager: TelemetryRunManager | None = None,
        vector_index_factory: Callable[[str], VectorIndex] | None = None,
        worker_count: int = 30,
    ) -> None:
        self._store = store
        self._index_config = index_config
        self._operational_config = operational_config
        self._llm_service = llm_service
        self._worker_count = max(worker_count, 1)
        self._run_manager = run_manager

        # Active run contexts keyed by document_id

        if vector_index_factory is None:
            vector_index_factory = lambda _doc_id: create_vector_index(  # noqa: E731
                operational_config.vector_backend,
                operational_config.database_url,
                index_config.embedding_model,
            )
        self._vector_index_factory = vector_index_factory

        self._queue: asyncio.PriorityQueue[
            tuple[tuple[int, int, int, int], ReadyParentCandidate]
        ] = asyncio.PriorityQueue()
        self._sequence = count()
        self._documents: dict[str, DocumentState] = {}
        self._pending_counts: defaultdict[str, int] = defaultdict(int)
        self._inflight_counts: defaultdict[str, int] = defaultdict(int)
        self._doc_locks: dict[str, asyncio.Lock] = {}
        self._cancelled_documents: set[str] = set()

        self._coord_lock = asyncio.Lock()
        self._shutdown = asyncio.Event()
        self._workers: list[asyncio.Task[None]] = []
        self._next_worker_id = 0
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._shutdown.is_set():
            self._shutdown = asyncio.Event()
        if self._workers:
            self._ensure_workers()
            return
        for _ in range(self._worker_count):
            self._spawn_worker()

    async def shutdown(self) -> None:
        self._shutdown.set()
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._idle_event.set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def enqueue_document(
        self,
        document_id: str,
        *,
        deleted_node_ids: Iterable[str] | None = None,
        new_root_ids: Iterable[str] | None = None,
        run_context: IndexRunContext | None = None,
    ) -> None:
        if document_id in self._cancelled_documents:
            return
        deleted_list = list(deleted_node_ids) if deleted_node_ids is not None else None
        new_root_list = list(new_root_ids) if new_root_ids is not None else None
        logger.debug(
            "workers[%s]: enqueue_document start (deleted=%d, new_roots=%d)",
            document_id,
            0 if deleted_list is None else len(deleted_list),
            0 if new_root_list is None else len(new_root_list),
        )
        state = self._get_or_create_document_state(document_id)
        async with self._coord_lock:
            if (
                self._pending_counts.get(document_id, 0) == 0
                and self._inflight_counts.get(document_id, 0) == 0
            ):
                state.completed = 0
            if run_context is not None:
                if run_context.run_id not in state.run_queue:
                    state.run_queue.append(run_context.run_id)
                state.pending_by_run.setdefault(run_context.run_id, 0)
                state.inflight_by_run.setdefault(run_context.run_id, 0)
        await self._resync_document(
            document_id,
            state,
            deleted_node_ids=deleted_list,
            new_root_ids=new_root_list,
            run_id=run_context.run_id if run_context is not None else None,
        )
        await self._maybe_finish_run(document_id)
        logger.debug(
            "workers[%s]: enqueue_document done (pending=%d, inflight=%d)",
            document_id,
            self._pending_counts.get(document_id, 0),
            self._inflight_counts.get(document_id, 0),
        )

    async def wait_until_idle(self, document_id: str | None = None) -> None:
        while True:
            self._ensure_workers()
            if document_id is None:
                if (
                    not self._pending_counts
                    and self._queue.empty()
                    and sum(self._inflight_counts.values()) == 0
                ):
                    return
            else:
                if (
                    self._pending_counts.get(document_id, 0) == 0
                    and self._inflight_counts.get(document_id, 0) == 0
                ):
                    return

            await self._idle_event.wait()

    def queue_depth(self, document_id: str | None = None) -> int:
        if document_id is None:
            return sum(self._pending_counts.values())
        return self._pending_counts.get(document_id, 0)

    def _spawn_worker(self) -> None:
        if self._shutdown.is_set():
            return
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        task = asyncio.create_task(
            self._worker_loop(worker_id), name=f"worker-{worker_id}"
        )
        self._workers.append(task)

    def _ensure_workers(self) -> None:
        if self._shutdown.is_set():
            return
        alive: list[asyncio.Task[None]] = []
        for task in self._workers:
            if task.done():
                try:
                    exc = task.exception()
                except asyncio.CancelledError:
                    exc = None
                if task.cancelled():
                    logger.warning("Worker task cancelled unexpectedly")
                elif exc is not None:
                    logger.exception("Worker task failed unexpectedly", exc_info=exc)
                continue
            alive.append(task)
        missing = self._worker_count - len(alive)
        self._workers = alive
        for _ in range(missing):
            self._spawn_worker()

    async def attach_run(self, context: IndexRunContext) -> None:
        async with self._coord_lock:
            state = self._get_or_create_document_state(context.document_id)
            state.run_queue.append(context.run_id)
            state.pending_by_run.setdefault(context.run_id, 0)
            state.inflight_by_run.setdefault(context.run_id, 0)

    async def detach_run(self, document_id: str, run_id: str) -> None:
        async with self._coord_lock:
            state = self._documents.get(document_id)
            if state is None:
                return
            try:
                state.run_queue.remove(run_id)
            except ValueError:
                pass
            state.pending_by_run.pop(run_id, None)
            state.inflight_by_run.pop(run_id, None)
            if state.run_assignments:
                removable = [
                    node_id
                    for node_id, assigned in state.run_assignments.items()
                    if assigned == run_id
                ]
                for node_id in removable:
                    state.run_assignments.pop(node_id, None)

    async def status(self) -> WorkerStatus:
        async with self._coord_lock:
            doc_ids = set(self._documents)
            doc_ids.update(self._pending_counts)
            doc_ids.update(self._inflight_counts)
            completed_by_document: dict[str, int] = {}
            for doc_id in doc_ids:
                state = self._documents.get(doc_id)
                completed_by_document[doc_id] = state.completed if state else 0
            return WorkerStatus(
                queue_depth=sum(self._pending_counts.values()),
                in_flight=sum(self._inflight_counts.values()),
                pending_by_document=dict(self._pending_counts),
                inflight_by_document=dict(self._inflight_counts),
                completed_by_document=completed_by_document,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_or_create_document_state(self, document_id: str) -> DocumentState:
        state = self._documents.get(document_id)
        if state is None:
            state = DocumentState(
                queued=set(),
                inflight=set(),
                tombstones=set(),
                completed=0,
                run_queue=deque(),
                pending_by_run={},
                inflight_by_run={},
                run_assignments={},
            )
            self._documents[document_id] = state
        return state

    def _candidate_key(self, candidate: ReadyParentCandidate) -> str:
        return f"{candidate.document_id}:{candidate.left_child_id}"

    def _is_cancelled(self, document_id: str) -> bool:
        return document_id in self._cancelled_documents

    def _doc_lock(self, document_id: str) -> asyncio.Lock:
        lock = self._doc_locks.get(document_id)
        if lock is None:
            lock = asyncio.Lock()
            self._doc_locks[document_id] = lock
        return lock

    async def _resync_document(
        self,
        document_id: str,
        state: DocumentState,
        *,
        deleted_node_ids: Iterable[str] | None,
        new_root_ids: Iterable[str] | None,
        run_id: str | None = None,
    ) -> None:
        if self._is_cancelled(document_id):
            return
        async with self._doc_lock(document_id):
            if deleted_node_ids:
                await self._handle_deleted_nodes(document_id, state, deleted_node_ids)

            store = self._store.for_document(document_id)
            doc_span_end = self._document_span_end(document_id, store)
            explicit_roots: set[str] | None = None
            if new_root_ids is None:
                roots = store.nodes.get_root_nodes()
                root_ids = [node.id for node in roots]
            else:
                filtered = [root_id for root_id in new_root_ids if root_id]
                root_ids = filtered
                explicit_roots = set(filtered)

        if explicit_roots:
            reinstated = explicit_roots & state.tombstones
            if reinstated:
                state.tombstones.difference_update(reinstated)
                logger.debug(
                    "workers[%s]: cleared tombstones for reinstated nodes %s",
                    document_id,
                    sorted(reinstated),
                )

        for root_id in root_ids:
            if run_id is not None:
                state.run_assignments[root_id] = run_id
            await self._process_new_root(
                document_id,
                root_id,
                state,
                store,
                doc_span_end,
                run_id=state.run_assignments.get(root_id, run_id),
            )

        self._prune_resolved_tombstones(state, store)

    async def _handle_deleted_nodes(
        self,
        document_id: str,
        state: DocumentState,
        deleted_node_ids: Iterable[str],
    ) -> None:
        deleted_ids = {node_id for node_id in deleted_node_ids if node_id}
        if not deleted_ids:
            return

        state.tombstones.update(deleted_ids)
        for node_id in deleted_ids:
            state.run_assignments.pop(node_id, None)

        async with self._coord_lock:
            queue_list = self._queue._queue  # type: ignore[attr-defined]
            retained: list[tuple[tuple[int, int, int, int], ReadyParentCandidate]] = []
            removed = 0

            for priority, candidate in queue_list:
                if (
                    candidate.document_id == document_id
                    and candidate.left_child_id in deleted_ids
                ):
                    removed += 1
                    state.queued.discard(candidate.left_child_id)
                    if candidate.run_id is not None:
                        pending = state.pending_by_run.get(candidate.run_id, 0)
                        if pending > 0:
                            state.pending_by_run[candidate.run_id] = pending - 1
                    logger.debug(
                        "workers[%s]: dropping queued candidate %s due to deletion",
                        document_id,
                        candidate.left_child_id,
                    )
                    continue
                retained.append((priority, candidate))

            if removed == 0:
                logger.debug(
                    "workers[%s]: deleted nodes %s did not match queued candidates",
                    document_id,
                    sorted(deleted_ids),
                )
                return

            self._queue._queue = retained  # type: ignore[attr-defined]
            heapq.heapify(self._queue._queue)  # type: ignore[attr-defined]

            unfinished = getattr(self._queue, "_unfinished_tasks", 0)
            if unfinished > 0:
                self._queue._unfinished_tasks = max(0, unfinished - removed)  # type: ignore[attr-defined]

            current_pending = self._pending_counts.get(document_id, 0)
            if current_pending > 0:
                remaining = current_pending - removed
                if remaining > 0:
                    self._pending_counts[document_id] = remaining
                else:
                    self._pending_counts.pop(document_id, None)

            if (
                self._queue.empty()
                and not self._pending_counts
                and sum(self._inflight_counts.values()) == 0
            ):
                self._idle_event.set()
        logger.debug(
            "workers[%s]: removed %d queued candidates due to deletion",
            document_id,
            removed,
        )

    def _prune_resolved_tombstones(
        self, state: DocumentState, store: DocumentStore
    ) -> None:
        if not state.tombstones:
            return

        removable: set[str] = set()
        for node_id in state.tombstones:
            if store.nodes.get(node_id) is None:
                removable.add(node_id)

        if not removable:
            return

        state.tombstones.difference_update(removable)
        logger.debug(
            "workers[%s]: pruned resolved tombstones %s",
            store.document_id,
            sorted(removable),
        )

    async def _process_new_root(
        self,
        document_id: str,
        root_id: str,
        state: DocumentState,
        store: DocumentStore,
        doc_span_end: int | None,
        *,
        run_id: str | None = None,
    ) -> None:
        if not root_id or root_id in state.tombstones:
            return

        root = store.nodes.get(root_id)
        if root is None:
            return

        assigned_run = run_id or state.run_assignments.get(root_id)
        if assigned_run is None and state.run_queue:
            assigned_run = state.run_queue[-1]
        if assigned_run is not None:
            state.run_assignments[root_id] = assigned_run

        level_index = int(getattr(root, "level_index", 0))
        effective_span_end = doc_span_end
        if effective_span_end is None:
            effective_span_end = self._document_span_end(document_id, store)
        if effective_span_end is None:
            return

        if level_index % 2 == 0:
            if assigned_run is not None:
                state.run_assignments.setdefault(root.id, assigned_run)
            await self._possibly_enqueue_left_child(
                document_id,
                root,
                state,
                store,
                effective_span_end,
            )
            return

        preceding_id = getattr(root, "preceding_neighbor_id", None)
        if preceding_id and preceding_id not in state.tombstones:
            left_sibling = store.nodes.get(preceding_id)
            if left_sibling is not None:
                if assigned_run is not None:
                    state.run_assignments.setdefault(left_sibling.id, assigned_run)
                await self._possibly_enqueue_left_child(
                    document_id,
                    left_sibling,
                    state,
                    store,
                    effective_span_end,
                )

        following_id = getattr(root, "following_neighbor_id", None)
        if following_id and following_id not in state.tombstones:
            following_neighbor = store.nodes.get(following_id)
            if following_neighbor is not None:
                if assigned_run is not None:
                    state.run_assignments.setdefault(
                        following_neighbor.id, assigned_run
                    )
                await self._possibly_enqueue_left_child(
                    document_id,
                    following_neighbor,
                    state,
                    store,
                    effective_span_end,
                )

    async def _possibly_enqueue_left_child(
        self,
        document_id: str,
        left_node: TreeNode,
        state: DocumentState,
        store: DocumentStore,
        doc_span_end: int | None,
    ) -> None:
        ready, _ = self._resolve_dependencies(
            document_id,
            left_node,
            state,
            store,
            doc_span_end,
        )
        if not ready:
            return

        run_id = state.run_assignments.get(left_node.id)
        if run_id is None and state.run_queue:
            run_id = state.run_queue[-1]
            state.run_assignments[left_node.id] = run_id

        candidate = ReadyParentCandidate(
            document_id=document_id,
            left_child_id=left_node.id,
            height=int(getattr(left_node, "height", 0)),
            level_index=int(getattr(left_node, "level_index", 0)),
            span_start=int(getattr(left_node, "span_start", 0)),
            run_id=run_id,
        )
        await self._enqueue_candidate(candidate, state)

    def _document_span_end(self, document_id: str, store: DocumentStore) -> int | None:
        rightmost = store.nodes.get_rightmost_leaf_for_document(document_id)
        if rightmost is None:
            return None
        return int(getattr(rightmost, "span_end", 0))

    async def _enqueue_candidate(
        self, candidate: ReadyParentCandidate, state: DocumentState
    ) -> None:
        if self._is_cancelled(candidate.document_id):
            return
        key = candidate.left_child_id
        if key in state.tombstones:
            return

        async with self._coord_lock:
            if key in state.queued or key in state.inflight:
                return

            priority = (
                max(candidate.height, 0),
                max(candidate.level_index, 0),
                max(candidate.span_start, 0),
                next(self._sequence),
            )
            await self._queue.put((priority, candidate))
            state.queued.add(key)
            self._pending_counts[candidate.document_id] += 1
            if candidate.run_id is not None:
                state.pending_by_run[candidate.run_id] = (
                    state.pending_by_run.get(candidate.run_id, 0) + 1
                )
            pending = self._pending_counts[candidate.document_id]
            self._idle_event.clear()
            logger.debug(
                "workers[%s]: enqueued candidate left=%s height=%d level=%d pending=%d",
                candidate.document_id,
                candidate.left_child_id,
                candidate.height,
                candidate.level_index,
                pending,
            )
        self._ensure_workers()

    def _dependency_signature(
        self, snapshot: DependencySnapshot
    ) -> tuple[str | None, str | None, str | None]:
        return (
            snapshot.preceding.id if snapshot.preceding is not None else None,
            snapshot.left.id if snapshot.left is not None else None,
            snapshot.right.id if snapshot.right is not None else None,
        )

    def _resolve_dependencies(
        self,
        document_id: str,
        left: TreeNode,
        state: DocumentState,
        store: DocumentStore,
        doc_span_end: int | None,
    ) -> tuple[bool, DependencySnapshot]:
        if left.id in state.tombstones:
            return False, DependencySnapshot(None, None, None)

        parent_id = getattr(left, "parent_id", None)
        if parent_id is not None:
            parent = store.nodes.get(parent_id)
            if parent is not None:
                return False, DependencySnapshot(None, None, None)
            store.nodes.update_parent_references_batch([(left.id, None)])
            try:
                setattr(left, "parent_id", None)
            except Exception:
                pass
            parent_id = None

        level_index = int(getattr(left, "level_index", 0))
        if level_index % 2 != 0:
            raise RuntimeError(
                f"Node {left.id} at level_index {level_index} is not a left child"
            )

        if doc_span_end is None:
            doc_span_end = self._document_span_end(document_id, store)
        if doc_span_end is None:
            raise RuntimeError(
                f"Document {document_id} has no span end while processing {left.id}"
            )

        span_start = int(getattr(left, "span_start", 0))
        span_end = int(getattr(left, "span_end", 0))

        if span_start == 0 and span_end == doc_span_end:
            return False, DependencySnapshot(None, left, None)

        preceding: TreeNode | None = None
        preceding_id = getattr(left, "preceding_neighbor_id", None)
        if span_start > 0:
            if preceding_id is None or preceding_id in state.tombstones:
                return False, DependencySnapshot(None, left, None)
            preceding = store.nodes.get(preceding_id)
            if preceding is None:
                return False, DependencySnapshot(None, left, None)
        elif preceding_id is not None:
            if preceding_id in state.tombstones:
                return False, DependencySnapshot(None, left, None)
            preceding = store.nodes.get(preceding_id)
            if preceding is None:
                return False, DependencySnapshot(None, left, None)

        following: TreeNode | None = None
        following_id = getattr(left, "following_neighbor_id", None)
        if span_end < doc_span_end:
            if following_id is None or following_id in state.tombstones:
                return False, DependencySnapshot(preceding, left, None)
            following = store.nodes.get(following_id)
            if following is None:
                return False, DependencySnapshot(preceding, left, None)
        elif following_id is not None:
            if following_id in state.tombstones:
                return False, DependencySnapshot(preceding, left, None)
            following = store.nodes.get(following_id)
            if following is None:
                return False, DependencySnapshot(preceding, left, None)

        return True, DependencySnapshot(preceding, left, following)

    def _dependencies_match(
        self, before: DependencySnapshot, after: DependencySnapshot
    ) -> bool:
        return self._dependency_signature(before) == self._dependency_signature(after)

    async def _retry_candidate(
        self, candidate: ReadyParentCandidate, state: DocumentState
    ) -> None:
        if self._is_cancelled(candidate.document_id):
            return

        store = self._store.for_document(candidate.document_id)
        left = store.nodes.get(candidate.left_child_id)
        if left is None:
            return

        doc_span_end = self._document_span_end(candidate.document_id, store)
        ready, _ = self._resolve_dependencies(
            candidate.document_id,
            left,
            state,
            store,
            doc_span_end,
        )
        if not ready:
            logger.debug(
                "workers[%s]: dependencies still unstable for left=%s; rescanning",
                candidate.document_id,
                candidate.left_child_id,
            )
            await self._scan_document(candidate.document_id)
            return

        retry_candidate = ReadyParentCandidate(
            document_id=candidate.document_id,
            left_child_id=left.id,
            height=int(getattr(left, "height", 0)),
            level_index=int(getattr(left, "level_index", 0)),
            span_start=int(getattr(left, "span_start", 0)),
            run_id=candidate.run_id,
        )
        logger.debug(
            "workers[%s]: requeuing candidate after dependency change left=%s",
            candidate.document_id,
            left.id,
        )
        await self._enqueue_candidate(retry_candidate, state)

    def _check_dependencies_still_valid(
        self,
        document_id: str,
        left_child_id: str,
        state: DocumentState,
    ) -> tuple[bool, DependencySnapshot]:
        if left_child_id in state.tombstones:
            return False, DependencySnapshot(None, None, None)

        store = self._store.for_document(document_id)
        left = store.nodes.get(left_child_id)
        if left is None:
            return False, DependencySnapshot(None, None, None)

        doc_span_end = self._document_span_end(document_id, store)
        ready, snapshot = self._resolve_dependencies(
            document_id,
            left,
            state,
            store,
            doc_span_end,
        )
        return ready, snapshot

    async def _scan_document(self, document_id: str) -> None:
        state = self._get_or_create_document_state(document_id)
        await self._resync_document(
            document_id,
            state,
            deleted_node_ids=None,
            new_root_ids=None,
            run_id=None,
        )

    async def _worker_loop(self, worker_id: int) -> None:
        while not self._shutdown.is_set():
            try:
                priority, candidate = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                return

            state = self._get_or_create_document_state(candidate.document_id)
            key = candidate.left_child_id
            async with self._coord_lock:
                state.queued.discard(key)
                state.inflight.add(key)
                self._inflight_counts[candidate.document_id] += 1
                if candidate.run_id is not None:
                    pending = state.pending_by_run.get(candidate.run_id, 0)
                    if pending > 0:
                        state.pending_by_run[candidate.run_id] = pending - 1
                    state.inflight_by_run[candidate.run_id] = (
                        state.inflight_by_run.get(candidate.run_id, 0) + 1
                    )
            logger.debug(
                "worker-%d: start candidate doc=%s left=%s level=%d height=%d",
                worker_id,
                candidate.document_id,
                candidate.left_child_id,
                candidate.level_index,
                candidate.height,
            )

            error_exc: Exception | None = None
            new_roots: list[str] = []
            retry_candidate = False
            try:
                new_roots, retry_candidate = await self._process_candidate(
                    candidate, state
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                error_exc = exc
                logger.exception(
                    "Worker failed while processing %s", candidate, exc_info=True
                )
            finally:
                async with self._coord_lock:
                    doc_id = candidate.document_id
                    state.inflight.discard(key)
                    if error_exc is None:
                        state.completed += 1
                    current_inflight = self._inflight_counts.get(doc_id, 0)
                    if current_inflight > 1:
                        self._inflight_counts[doc_id] = current_inflight - 1
                    else:
                        self._inflight_counts.pop(doc_id, None)
                    if candidate.run_id is not None:
                        inflight = state.inflight_by_run.get(candidate.run_id, 0)
                        if inflight > 1:
                            state.inflight_by_run[candidate.run_id] = inflight - 1
                        else:
                            state.inflight_by_run.pop(candidate.run_id, None)

                    current_pending = self._pending_counts.get(doc_id, 0)
                    if current_pending > 1:
                        self._pending_counts[doc_id] = current_pending - 1
                    else:
                        self._pending_counts.pop(doc_id, None)
                    if (
                        not self._pending_counts
                        and self._queue.empty()
                        and sum(self._inflight_counts.values()) == 0
                    ):
                        self._idle_event.set()

                self._queue.task_done()
            logger.debug(
                "worker-%d: finish candidate doc=%s left=%s pending=%d inflight=%d error=%s",
                worker_id,
                candidate.document_id,
                candidate.left_child_id,
                self._pending_counts.get(candidate.document_id, 0),
                self._inflight_counts.get(candidate.document_id, 0),
                error_exc.__class__.__name__ if error_exc else None,
            )

            if retry_candidate and error_exc is None:
                await self._retry_candidate(candidate, state)

            if error_exc is None:
                try:
                    if new_roots:
                        store = self._store.for_document(candidate.document_id)
                        doc_span_end = self._document_span_end(
                            candidate.document_id, store
                        )
                        for root_id in new_roots:
                            if candidate.run_id is not None:
                                state.run_assignments[root_id] = candidate.run_id
                            await self._process_new_root(
                                candidate.document_id,
                                root_id,
                                state,
                                store,
                                doc_span_end,
                                run_id=candidate.run_id,
                            )
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception(
                        "Failed to queue follow-up work for %s",
                        candidate.document_id,
                        exc_info=True,
                    )
                await self._maybe_finish_run(candidate.document_id)
                logger.debug(
                    "worker-%d: candidate doc=%s left=%s produced new_parents=%d",
                    worker_id,
                    candidate.document_id,
                    candidate.left_child_id,
                    len(new_roots),
                )
            else:
                await self._handle_worker_failure(
                    candidate.document_id, candidate.run_id, error_exc
                )
                logger.debug(
                    "worker-%d: candidate doc=%s left=%s failed with %s",
                    worker_id,
                    candidate.document_id,
                    candidate.left_child_id,
                    type(error_exc).__name__,
                )

        while not self._queue.empty():
            _, candidate = await self._queue.get()
            async with self._coord_lock:
                state = self._get_or_create_document_state(candidate.document_id)
                state.queued.discard(candidate.left_child_id)
                doc_id = candidate.document_id
                current_pending = self._pending_counts.get(doc_id, 0)
                if current_pending > 1:
                    self._pending_counts[doc_id] = current_pending - 1
                else:
                    self._pending_counts.pop(doc_id, None)
            self._queue.task_done()

    async def _handle_worker_failure(
        self, document_id: str, run_id: str | None, error: Exception
    ) -> None:
        if self._run_manager is None or run_id is None:
            return
        message = f"Worker failure for {document_id}: {error}"
        await self._run_manager.complete_run(run_id, error=message)
        await self.detach_run(document_id, run_id)
        await self._maybe_finish_run(document_id)

    async def _maybe_finish_run(self, document_id: str) -> None:
        if self._run_manager is None:
            return
        async with self._coord_lock:
            state = self._documents.get(document_id)
            if state is None or not state.run_queue:
                return
            ready: list[str] = []
            for run_id in list(state.run_queue):
                pending = state.pending_by_run.get(run_id, 0)
                inflight = state.inflight_by_run.get(run_id, 0)
                if pending == 0 and inflight == 0:
                    ready.append(run_id)
                else:
                    break
        for run_id in ready:
            context = await self._run_manager.get_run(run_id)
            collector = context.telemetry_collector if context else None
            if collector:
                doc_store = self._store.for_document(document_id)
                vector_index = self._vector_index_factory(document_id)
                token_limit = getattr(
                    self._llm_service, "_embedding_batch_token_limit", 8000
                )
                max_items = getattr(
                    self._llm_service, "_provider_max_embedding_batch_size", 1000
                )
                await compute_fidelity_for_telemetry(
                    document_store=doc_store,
                    collector=collector,
                    vector_index=vector_index,
                    embedder=self._llm_service,
                    token_limit=token_limit,
                    max_batch_items=max_items,
                )
            await self._run_manager.complete_run(run_id, error=None)
            await self.detach_run(document_id, run_id)

    async def _process_candidate(
        self, candidate: ReadyParentCandidate, state: DocumentState
    ) -> tuple[list[str], bool]:
        if self._is_cancelled(candidate.document_id):
            logger.debug(
                "worker: candidate doc=%s left=%s aborted (document cancelled)",
                candidate.document_id,
                candidate.left_child_id,
            )
            return [], False
        store = self._store.for_document(candidate.document_id)

        def _skip(stage: str, reason: str) -> tuple[list[str], bool]:
            logger.debug(
                "worker: skip candidate doc=%s left=%s stage=%s reason=%s",
                candidate.document_id,
                candidate.left_child_id,
                stage,
                reason,
            )
            return [], True

        context: IndexRunContext | None = None
        if self._run_manager is not None and candidate.run_id is not None:
            context = await self._run_manager.get_run(candidate.run_id)
        collector = context.telemetry_collector if context else None

        logger.debug(
            "worker: candidate doc=%s left=%s dependency check",
            candidate.document_id,
            candidate.left_child_id,
        )

        ready, snapshot = self._check_dependencies_still_valid(
            candidate.document_id, candidate.left_child_id, state
        )
        if not ready or snapshot.left is None:
            return _skip("dependency-check", "dependencies not ready or left missing")

        left = snapshot.left
        right = snapshot.right
        preceding = snapshot.preceding

        left_level_index = int(getattr(left, "level_index", 0))
        parent_level_index = left_level_index // 2

        span_start = int(getattr(left, "span_start", 0))
        span_end = (
            int(getattr(right, "span_end", span_start))
            if right is not None
            else int(getattr(left, "span_end", span_start))
        )
        height = int(getattr(left, "height", 0)) + 1
        if right is not None:
            height = max(height, int(getattr(right, "height", 0)) + 1)

        left_text = left.text or ""
        right_text = right.text or "" if right is not None else ""
        left_tokens = int(getattr(left, "token_count", 0))
        right_tokens = int(getattr(right, "token_count", 0)) if right else 0

        prev_context = None
        if preceding is not None and preceding.text:
            prev_context = preceding.text

        parent_id = str(uuid.uuid4())

        if collector is not None:
            collector.track_node_created(
                node_id=parent_id,
                height=height,
                span=(span_start, span_end),
            )

        summary, _retry_count, summary_tokens = await self._llm_service._summarize_text(
            left_text,
            right_text,
            self._index_config.target_chunk_tokens,
            parent_id=parent_id,
            reporter=collector,
            prev_context=prev_context,
            left_token_count=left_tokens,
            right_token_count=right_tokens,
        )
        logger.debug(
            "worker: summarized doc=%s left=%s right=%s parent=%s span=(%d,%d)",
            candidate.document_id,
            left.id,
            right.id if right else None,
            parent_id,
            span_start,
            span_end,
        )

        if self._is_cancelled(candidate.document_id):
            logger.debug(
                "worker: candidate doc=%s left=%s aborted after summary (cancelled)",
                candidate.document_id,
                candidate.left_child_id,
            )
            return [], False

        ready, post_summary = self._check_dependencies_still_valid(
            candidate.document_id, candidate.left_child_id, state
        )
        if not ready or not self._dependencies_match(snapshot, post_summary):
            return _skip("post-summary", "dependencies changed after summary")

        if post_summary.left is None:
            return _skip("post-summary", "left missing after summary")

        start_time = time.time()
        embeddings = await self._llm_service.embed_texts([summary])
        if len(embeddings) != 1:
            raise ValueError("Embedding provider returned unexpected batch size")
        embedding_vector = np.asarray(embeddings[0], dtype=np.float64)
        logger.debug(
            "worker: embedded parent doc=%s parent=%s",
            candidate.document_id,
            parent_id,
        )

        if collector is not None:
            collector.record_embedding_call_v2(
                [(parent_id, summary_tokens)],
                batch_size=1,
                model=self._index_config.embedding_model,
                start_time=start_time,
            )

        if self._is_cancelled(candidate.document_id):
            logger.debug(
                "worker: candidate doc=%s left=%s aborted after embedding (cancelled)",
                candidate.document_id,
                candidate.left_child_id,
            )
            return [], False

        ready, post_embedding = self._check_dependencies_still_valid(
            candidate.document_id, candidate.left_child_id, state
        )
        if not ready or not self._dependencies_match(post_summary, post_embedding):
            return _skip("post-embedding", "dependencies changed after embedding")

        left_after_embedding = post_embedding.left
        right_after_embedding = post_embedding.right
        if left_after_embedding is None:
            return _skip("post-embedding", "left missing after embedding")

        following_neighbor_id = (
            getattr(right_after_embedding, "following_neighbor_id", None)
            if right_after_embedding is not None
            else None
        )

        vector_index = self._vector_index_factory(candidate.document_id)

        affected_ids = {parent_id, left_after_embedding.id}
        parent_refs: list[tuple[str, str | None]] = [
            (left_after_embedding.id, parent_id)
        ]
        if right_after_embedding is not None:
            affected_ids.add(right_after_embedding.id)
            parent_refs.append((right_after_embedding.id, parent_id))

        ready, final_snapshot = self._check_dependencies_still_valid(
            candidate.document_id, left_after_embedding.id, state
        )
        if (
            not ready
            or not self._dependencies_match(post_embedding, final_snapshot)
            or final_snapshot.left is None
        ):
            return _skip("final-check", "dependencies changed before commit")

        left_final = final_snapshot.left
        right_final = final_snapshot.right
        if left_final is None:
            return _skip("final-check", "left missing before commit")

        span_start_final = int(getattr(left_final, "span_start", span_start))
        span_end_final = int(
            getattr(right_final, "span_end", span_end)
            if right_final is not None
            else getattr(left_final, "span_end", span_end)
        )

        preceding_node_id_final = getattr(left_final, "preceding_neighbor_id", None)

        vector_written = False
        try:
            with store.transaction() as session:
                if self._is_cancelled(candidate.document_id):
                    logger.debug(
                        "worker: candidate doc=%s left=%s aborted during commit (cancelled)",
                        candidate.document_id,
                        candidate.left_child_id,
                    )
                    return [], False
                preceding_parent_id: str | None = None
                preceding_parent_node = None
                if preceding_node_id_final:
                    refreshed_preceding = store.nodes.get(preceding_node_id_final)
                    if refreshed_preceding is None:
                        return _skip(
                            "commit",
                            f"preceding neighbor {preceding_node_id_final} missing",
                        )
                    parent_candidate = getattr(refreshed_preceding, "parent_id", None)
                    if parent_candidate:
                        preceding_parent_id = str(parent_candidate)
                        preceding_parent_node = store.nodes.get(preceding_parent_id)
                        if preceding_parent_node is None:
                            store.nodes.update_parent_references_batch(
                                [(refreshed_preceding.id, None)]
                            )
                            preceding_parent_id = None
                if preceding_parent_id is None and parent_level_index > 0:
                    fallback_prev = store.nodes.get_by_height_and_level(
                        height=height,
                        level_index=parent_level_index - 1,
                    )
                    if fallback_prev is not None:
                        preceding_parent_id = fallback_prev.id
                        preceding_parent_node = fallback_prev
                        affected_ids.add(preceding_parent_id)

                following_parent_id: str | None = None
                following_parent_node = None
                if following_neighbor_id:
                    refreshed_following = store.nodes.get(following_neighbor_id)
                    if refreshed_following is None:
                        return _skip(
                            "commit",
                            f"following neighbor {following_neighbor_id} missing",
                        )
                    parent_candidate = getattr(refreshed_following, "parent_id", None)
                    if parent_candidate:
                        following_parent_id = str(parent_candidate)
                        following_parent_node = store.nodes.get(following_parent_id)
                        if following_parent_node is None:
                            store.nodes.update_parent_references_batch(
                                [(refreshed_following.id, None)]
                            )
                            following_parent_id = None
                if following_parent_id is None:
                    fallback_next = store.nodes.get_by_height_and_level(
                        height=height,
                        level_index=parent_level_index + 1,
                    )
                    if fallback_next is not None:
                        following_parent_id = fallback_next.id
                        following_parent_node = fallback_next
                        affected_ids.add(following_parent_id)

                node_payload: dict[str, NodeFieldValue] = {
                    "node_id": parent_id,
                    "text": summary,
                    "span_start": span_start_final,
                    "span_end": span_end_final,
                    "parent_id": None,
                    "left_child_id": left_final.id,
                    "right_child_id": right_final.id if right_final else None,
                    "document_id": candidate.document_id,
                    "token_count": summary_tokens,
                    "height": height,
                    "preceding_neighbor_id": preceding_parent_id,
                    "following_neighbor_id": following_parent_id,
                    "level_index": parent_level_index,
                }

                neighbors_update: list[tuple[str, str | None, str | None]] = [
                    (parent_id, preceding_parent_id, following_parent_id)
                ]

                if preceding_parent_id and preceding_parent_node is not None:
                    neighbors_update.append(
                        (
                            preceding_parent_id,
                            getattr(
                                preceding_parent_node,
                                "preceding_neighbor_id",
                                None,
                            ),
                            parent_id,
                        )
                    )
                    affected_ids.add(preceding_parent_id)

                if following_parent_id and following_parent_node is not None:
                    neighbors_update.append(
                        (
                            following_parent_id,
                            parent_id,
                            getattr(
                                following_parent_node,
                                "following_neighbor_id",
                                None,
                            ),
                        )
                    )
                    affected_ids.add(following_parent_id)

                store.nodes.add_batch([node_payload], session=session)
                store.nodes.update_parent_references_batch(parent_refs, session=session)
                if neighbors_update:
                    store.nodes.update_neighbors_batch(
                        neighbors_update, session=session
                    )

                vector_index.upsert(
                    [
                        (
                            parent_id,
                            embedding_vector,
                            {
                                "document_id": candidate.document_id,
                                "span_start": span_start_final,
                                "span_end": span_end_final,
                                "is_leaf": 0,
                                "height": height,
                                "level_index": parent_level_index,
                                "coord_version": 1,
                            },
                        )
                    ]
                )
                vector_written = True

            store.tree.clear_depth_cache(list(affected_ids))
        except Exception:
            if vector_written:
                try:  # pragma: no cover - best-effort cleanup
                    vector_index.delete(ids=[parent_id])
                except Exception:
                    logger.exception(
                        "Failed to delete vector during rollback", exc_info=True
                    )
            raise

        if context is not None:
            context.register_summary_node(parent_id)
            if self._run_manager is not None:
                await self._run_manager.record_node_committed(
                    context,
                    node_id=parent_id,
                    height=height,
                    span_start=span_start_final,
                    span_end=span_end_final,
                )

        logger.debug(
            "worker: committed parent doc=%s parent=%s left=%s right=%s height=%d level=%d",
            candidate.document_id,
            parent_id,
            left_final.id,
            right_final.id if right_final else None,
            height,
            parent_level_index,
        )

        return [parent_id], False

    async def cancel_document(self, document_id: str) -> None:
        run_ids: list[str] = []
        async with self._coord_lock:
            self._cancelled_documents.add(document_id)
            state = self._documents.get(document_id)
            if state is not None:
                state.queued.clear()
                state.inflight.clear()
                state.tombstones.clear()
                state.completed = 0
                run_ids = list(state.run_queue)
                state.run_queue.clear()
                state.pending_by_run.clear()
                state.inflight_by_run.clear()
                state.run_assignments.clear()

            queue_list = getattr(self._queue, "_queue")
            retained: list[tuple[tuple[int, int, int, int], ReadyParentCandidate]] = []
            removed = 0
            for priority, candidate in queue_list:
                if candidate.document_id == document_id:
                    removed += 1
                    continue
                retained.append((priority, candidate))
            if removed:
                setattr(self._queue, "_queue", retained)
                heapq.heapify(getattr(self._queue, "_queue"))
                unfinished = getattr(self._queue, "_unfinished_tasks", 0)
                if unfinished:
                    setattr(
                        self._queue,
                        "_unfinished_tasks",
                        max(0, unfinished - removed),
                    )

            self._pending_counts.pop(document_id, None)
            self._idle_event.clear()

        for run_id in run_ids:
            if self._run_manager is not None:
                await self._run_manager.complete_run(run_id, error="Document cleared")
            await self.detach_run(document_id, run_id)
        await self.wait_until_idle(document_id)

        async with self._coord_lock:
            self._pending_counts.pop(document_id, None)
            self._inflight_counts.pop(document_id, None)
            self._documents.pop(document_id, None)
            self._cancelled_documents.discard(document_id)
            self._idle_event.set()


__all__ = [
    "ReadyParentCandidate",
    "SummaryBackend",
    "WorkerStatus",
    "compute_ready_parent_candidates",
    "WorkerCoordinator",
]
