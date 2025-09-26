"""Worker coordination utilities for server-managed summarization."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from ragzoom.config import IndexConfig, OperationalConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.server.run_manager import IndexRunContext, TelemetryRunManager
from ragzoom.vector_factory import create_vector_index

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from ragzoom.telemetry_collection import TelemetryCollector
else:  # pragma: no cover - fallback for runtime when telemetry isn’t available
    TelemetryCollector = object

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReadyParentCandidate:
    """Payload describing a parent node that should be built."""

    document_id: str
    left_child_id: str


NodeFieldValue = str | int | float | bool | list[float] | NDArray[np.float64] | None


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    """Snapshot of queue depth and in-flight activity."""

    queue_depth: int
    in_flight: int
    pending_by_document: dict[str, int]
    inflight_by_document: dict[str, int]


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
    return [
        ReadyParentCandidate(document_id=document_id, left_child_id=left_id)
        for left_id in left_ids
    ]


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
        worker_count: int = 2,
    ) -> None:
        self._store = store
        self._index_config = index_config
        self._operational_config = operational_config
        self._llm_service = llm_service
        self._worker_count = max(worker_count, 1)
        self._run_manager = run_manager

        # Active run contexts keyed by document_id
        self._runs: dict[str, IndexRunContext] = {}

        if vector_index_factory is None:
            vector_index_factory = lambda _doc_id: create_vector_index(  # noqa: E731
                operational_config.vector_backend,
                operational_config.database_url,
                index_config.embedding_model,
            )
        self._vector_index_factory = vector_index_factory

        self._queue: asyncio.PriorityQueue[
            tuple[tuple[int, int], ReadyParentCandidate]
        ] = asyncio.PriorityQueue()
        self._sequence = count()
        self._queued: set[str] = set()
        self._inflight: set[str] = set()
        self._pending_counts: defaultdict[str, int] = defaultdict(int)
        self._inflight_counts: defaultdict[str, int] = defaultdict(int)
        self._doc_locks: dict[str, asyncio.Lock] = {}

        self._coord_lock = asyncio.Lock()
        self._shutdown = asyncio.Event()
        self._workers: list[asyncio.Task[None]] = []
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._workers:
            return
        for worker_id in range(self._worker_count):
            task = asyncio.create_task(
                self._worker_loop(worker_id), name=f"worker-{worker_id}"
            )
            self._workers.append(task)

    async def shutdown(self) -> None:
        self._shutdown.set()
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def enqueue_document(self, document_id: str) -> None:
        await self._scan_document(document_id)
        await self._maybe_finish_run(document_id)

    async def wait_until_idle(self, document_id: str | None = None) -> None:
        while True:
            if document_id is None:
                if (
                    not self._pending_counts
                    and self._queue.empty()
                    and not self._inflight
                ):
                    return
            else:
                if self._pending_counts.get(document_id, 0) == 0:
                    return

            await self._idle_event.wait()

    def queue_depth(self, document_id: str | None = None) -> int:
        if document_id is None:
            return sum(self._pending_counts.values())
        return self._pending_counts.get(document_id, 0)

    async def attach_run(self, context: IndexRunContext) -> None:
        async with self._coord_lock:
            self._runs[context.document_id] = context

    async def detach_run(self, document_id: str) -> None:
        async with self._coord_lock:
            if document_id in self._runs:
                self._runs.pop(document_id)

    async def status(self) -> WorkerStatus:
        async with self._coord_lock:
            return WorkerStatus(
                queue_depth=sum(self._pending_counts.values()),
                in_flight=len(self._inflight),
                pending_by_document=dict(self._pending_counts),
                inflight_by_document=dict(self._inflight_counts),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _candidate_key(self, candidate: ReadyParentCandidate) -> str:
        return f"{candidate.document_id}:{candidate.left_child_id}"

    def _doc_lock(self, document_id: str) -> asyncio.Lock:
        lock = self._doc_locks.get(document_id)
        if lock is None:
            lock = asyncio.Lock()
            self._doc_locks[document_id] = lock
        return lock

    async def _scan_document(self, document_id: str) -> None:
        async with self._doc_lock(document_id):
            store = self._store.for_document(document_id)
            candidates = compute_ready_parent_candidates(store)

            async with self._coord_lock:
                added = False
                for candidate in candidates:
                    key = self._candidate_key(candidate)
                    if key in self._queued or key in self._inflight:
                        continue

                    priority = (0, next(self._sequence))
                    await self._queue.put((priority, candidate))
                    self._queued.add(key)
                    self._pending_counts[candidate.document_id] += 1
                    added = True

                if added:
                    self._idle_event.clear()

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

            key = self._candidate_key(candidate)
            async with self._coord_lock:
                self._queued.discard(key)
                self._inflight.add(key)
                self._inflight_counts[candidate.document_id] += 1

            error_exc: Exception | None = None
            try:
                await self._process_candidate(candidate)
            except Exception as exc:  # pragma: no cover - defensive logging
                error_exc = exc
                logger.exception(
                    "Worker failed while processing %s", candidate, exc_info=True
                )
            finally:
                async with self._coord_lock:
                    self._inflight.discard(key)
                    doc_id = candidate.document_id
                    current_inflight = self._inflight_counts.get(doc_id, 0)
                    if current_inflight > 1:
                        self._inflight_counts[doc_id] = current_inflight - 1
                    else:
                        if doc_id in self._inflight_counts:
                            self._inflight_counts.pop(doc_id)
                    self._pending_counts[doc_id] -= 1
                    if self._pending_counts[doc_id] <= 0:
                        if doc_id in self._pending_counts:
                            self._pending_counts.pop(doc_id)
                    if (
                        not self._pending_counts
                        and self._queue.empty()
                        and not self._inflight
                    ):
                        self._idle_event.set()

                self._queue.task_done()

            if error_exc is None:
                try:
                    await self._scan_document(candidate.document_id)
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception(
                        "Failed to rescan document %s",
                        candidate.document_id,
                        exc_info=True,
                    )
                await self._maybe_finish_run(candidate.document_id)
            else:
                await self._handle_worker_failure(candidate.document_id, error_exc)

        while not self._queue.empty():
            _, candidate = await self._queue.get()
            key = self._candidate_key(candidate)
            async with self._coord_lock:
                self._queued.discard(key)
            self._queue.task_done()

    async def _handle_worker_failure(self, document_id: str, error: Exception) -> None:
        if self._run_manager is None:
            return
        message = f"Worker failure for {document_id}: {error}"
        await self._finalize_run(document_id, error=message)

    async def _maybe_finish_run(self, document_id: str) -> None:
        if self._run_manager is None:
            return
        async with self._coord_lock:
            context = self._runs.get(document_id)
            if context is None:
                return
            if self._pending_counts.get(document_id, 0) > 0:
                return
            if self._inflight_counts.get(document_id, 0) > 0:
                return
        await self._finalize_run(document_id, error=None)

    async def _finalize_run(self, document_id: str, *, error: str | None) -> None:
        if self._run_manager is None:
            return
        async with self._coord_lock:
            context = self._runs.get(document_id)
        if context is None:
            return
        await self._run_manager.complete_run(context.run_id, error=error)
        await self.detach_run(document_id)

    async def _process_candidate(self, candidate: ReadyParentCandidate) -> None:
        store = self._store.for_document(candidate.document_id)

        async with self._coord_lock:
            context = self._runs.get(candidate.document_id)
        collector = context.telemetry_collector if context else None

        left = store.nodes.get(candidate.left_child_id)
        if left is None or left.parent_id is not None:
            return

        right_id = getattr(left, "following_neighbor_id", None)
        if right_id is None:
            return
        right = store.nodes.get(right_id)
        if right is None or right.parent_id is not None:
            return
        if int(getattr(right, "height", -1)) != int(getattr(left, "height", -1)):
            return
        if getattr(right, "preceding_neighbor_id", None) != left.id:
            return

        left_level_index = int(getattr(left, "level_index", 0))
        if left_level_index % 2 != 0:
            return
        parent_level_index = left_level_index // 2

        span_start = int(getattr(left, "span_start", 0))
        span_end = int(getattr(right, "span_end", span_start))
        height = (
            max(int(getattr(left, "height", 0)), int(getattr(right, "height", 0))) + 1
        )

        left_text = left.text or ""
        right_text = right.text or ""
        left_tokens = int(getattr(left, "token_count", 0))
        right_tokens = int(getattr(right, "token_count", 0))

        prev_context = None
        preceding_node = None
        if left.preceding_neighbor_id:
            preceding_node = store.nodes.get(left.preceding_neighbor_id)
            if preceding_node is None:
                return
            if preceding_node.text:
                prev_context = preceding_node.text

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

        start_time = time.time()
        embeddings = await self._llm_service.embed_texts([summary])
        if len(embeddings) != 1:
            raise ValueError("Embedding provider returned unexpected batch size")
        embedding_vector = np.asarray(embeddings[0], dtype=np.float64)

        if collector is not None:
            collector.record_embedding_call_v2(
                [(parent_id, summary_tokens)],
                batch_size=1,
                model=self._index_config.embedding_model,
                start_time=start_time,
            )

        preceding_parent_id = None
        if preceding_node and preceding_node.parent_id:
            preceding_parent_id = preceding_node.parent_id

        following_parent_id = None
        following_neighbor_id = getattr(right, "following_neighbor_id", None)
        if following_neighbor_id:
            following_neighbor = store.nodes.get(following_neighbor_id)
            if following_neighbor and following_neighbor.parent_id:
                following_parent_id = following_neighbor.parent_id

        vector_index = self._vector_index_factory(candidate.document_id)

        neighbors_update: list[tuple[str, str | None, str | None]] = [
            (parent_id, preceding_parent_id, following_parent_id)
        ]

        affected_ids = {parent_id, left.id, right.id}
        parent_refs: list[tuple[str, str | None]] = [
            (left.id, parent_id),
            (right.id, parent_id),
        ]

        if preceding_parent_id:
            preceding_parent = store.nodes.get(preceding_parent_id)
            if preceding_parent is not None:
                neighbors_update.append(
                    (
                        preceding_parent_id,
                        getattr(preceding_parent, "preceding_neighbor_id", None),
                        parent_id,
                    )
                )
                affected_ids.add(preceding_parent_id)

        if following_parent_id:
            following_parent = store.nodes.get(following_parent_id)
            if following_parent is not None:
                neighbors_update.append(
                    (
                        following_parent_id,
                        parent_id,
                        getattr(following_parent, "following_neighbor_id", None),
                    )
                )
                affected_ids.add(following_parent_id)

        node_payload: dict[str, NodeFieldValue] = {
            "node_id": parent_id,
            "text": summary,
            "span_start": span_start,
            "span_end": span_end,
            "parent_id": None,
            "left_child_id": left.id,
            "right_child_id": right.id if right else None,
            "document_id": candidate.document_id,
            "token_count": summary_tokens,
            "height": height,
            "preceding_neighbor_id": preceding_parent_id,
            "following_neighbor_id": following_parent_id,
            "level_index": parent_level_index,
        }

        vector_written = False
        try:
            with store.transaction() as session:
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
                                "span_start": span_start,
                                "span_end": span_end,
                                "is_leaf": 0,
                                "height": height,
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


__all__ = [
    "ReadyParentCandidate",
    "SummaryBackend",
    "WorkerStatus",
    "compute_ready_parent_candidates",
    "WorkerCoordinator",
]
