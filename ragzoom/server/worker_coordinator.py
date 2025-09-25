"""Worker coordination utilities for server-managed summarization."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
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
    right_child_id: str | None
    height: int
    span_start: int
    span_end: int


NodeFieldValue = str | int | float | bool | list[float] | NDArray[np.float64] | None


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    """Snapshot of queue depth and in-flight activity."""

    queue_depth: int
    in_flight: int
    pending_by_document: dict[str, int]
    inflight_by_document: dict[str, int]


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


def _span_bounds(nodes: Iterable[TreeNode]) -> int:
    max_end = 0
    for node in nodes:
        max_end = max(max_end, int(getattr(node, "span_end", 0)))
    return max_end


def compute_ready_parent_candidates(store: DocumentStore) -> list[ReadyParentCandidate]:
    document_id = store.document_id or ""
    nodes = store.nodes.get_all()
    if not nodes:
        return []

    by_id: dict[str, TreeNode] = {node.id: node for node in nodes}
    doc_span_end = _span_bounds(nodes)

    candidates: list[ReadyParentCandidate] = []
    claimed: set[str] = set()

    for node in sorted(nodes, key=lambda n: int(getattr(n, "span_start", 0))):
        node_id = node.id
        if node_id in claimed or node.parent_id is not None:
            continue

        span_start = int(getattr(node, "span_start", 0))
        span_end = int(getattr(node, "span_end", 0))
        height = int(getattr(node, "height", 0))

        right_id = getattr(node, "following_neighbor_id", None)
        if right_id:
            right_node = by_id.get(right_id)
            if (
                right_node
                and right_node.parent_id is None
                and int(getattr(right_node, "height", -1)) == height
                and getattr(right_node, "preceding_neighbor_id", None) == node_id
            ):
                candidates.append(
                    ReadyParentCandidate(
                        document_id=document_id,
                        left_child_id=node_id,
                        right_child_id=right_id,
                        height=height,
                        span_start=span_start,
                        span_end=int(getattr(right_node, "span_end", span_end)),
                    )
                )
                claimed.add(node_id)
                claimed.add(right_id)
                continue

        if (
            right_id is None
            and span_start == 0
            and span_end == doc_span_end
            and getattr(node, "left_child_id", None) is None
            and getattr(node, "right_child_id", None) is None
        ):
            candidates.append(
                ReadyParentCandidate(
                    document_id=document_id,
                    left_child_id=node_id,
                    right_child_id=None,
                    height=height,
                    span_start=span_start,
                    span_end=span_end,
                )
            )
            claimed.add(node_id)

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
        vector_index_factory: Callable[[str], VectorIndex] | None = None,
        worker_count: int = 2,
    ) -> None:
        self._store = store
        self._index_config = index_config
        self._operational_config = operational_config
        self._llm_service = llm_service
        self._worker_count = max(worker_count, 1)

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
        right = candidate.right_child_id or "-"
        return f"{candidate.document_id}:{candidate.left_child_id}:{right}"

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

                    priority = (candidate.height, next(self._sequence))
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

            try:
                await self._process_candidate(candidate)
            except Exception:  # pragma: no cover - defensive logging
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
                        self._inflight_counts.pop(doc_id, None)
                    self._pending_counts[doc_id] -= 1
                    if self._pending_counts[doc_id] <= 0:
                        self._pending_counts.pop(doc_id, None)
                    if (
                        not self._pending_counts
                        and self._queue.empty()
                        and not self._inflight
                    ):
                        self._idle_event.set()

                self._queue.task_done()

            try:
                await self._scan_document(candidate.document_id)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Failed to rescan document %s", candidate.document_id, exc_info=True
                )

        while not self._queue.empty():
            _, candidate = await self._queue.get()
            key = self._candidate_key(candidate)
            async with self._coord_lock:
                self._queued.discard(key)
            self._queue.task_done()

    async def _process_candidate(self, candidate: ReadyParentCandidate) -> None:
        store = self._store.for_document(candidate.document_id)

        left = store.nodes.get(candidate.left_child_id)
        if left is None or left.parent_id is not None:
            return

        right = (
            store.nodes.get(candidate.right_child_id)
            if candidate.right_child_id is not None
            else None
        )
        if right is not None and right.parent_id is not None:
            return

        span_start = int(left.span_start)
        span_end = int(right.span_end) if right else int(left.span_end)
        height = (
            max(int(left.height), int(right.height) if right else int(left.height)) + 1
        )

        if right is not None and int(right.height) != int(left.height):
            return

        left_text = left.text or ""
        right_text = right.text or "" if right else ""
        left_tokens = int(getattr(left, "token_count", 0))
        right_tokens = int(getattr(right, "token_count", 0)) if right else 0

        prev_context = None
        if left.preceding_neighbor_id:
            preceding = store.nodes.get(left.preceding_neighbor_id)
            if preceding and preceding.text:
                prev_context = preceding.text

        parent_id = str(uuid.uuid4())

        summary, _retry_count, summary_tokens = await self._llm_service._summarize_text(
            left_text,
            right_text,
            self._index_config.target_chunk_tokens,
            parent_id=parent_id,
            reporter=None,
            prev_context=prev_context,
            left_token_count=left_tokens,
            right_token_count=right_tokens,
        )

        embeddings = await self._llm_service.embed_texts([summary])
        if len(embeddings) != 1:
            raise ValueError("Embedding provider returned unexpected batch size")
        embedding_vector = np.asarray(embeddings[0], dtype=np.float64)

        preceding_parent_id = None
        if left.preceding_neighbor_id:
            preceding_neighbor = store.nodes.get(left.preceding_neighbor_id)
            if preceding_neighbor and preceding_neighbor.parent_id:
                preceding_parent_id = preceding_neighbor.parent_id

        following_parent_id = None
        sibling = right or left
        following_neighbor_id = getattr(sibling, "following_neighbor_id", None)
        if following_neighbor_id:
            following_neighbor = store.nodes.get(following_neighbor_id)
            if following_neighbor and following_neighbor.parent_id:
                following_parent_id = following_neighbor.parent_id

        vector_index = self._vector_index_factory(candidate.document_id)

        neighbors_update: list[tuple[str, str | None, str | None]] = [
            (parent_id, preceding_parent_id, following_parent_id)
        ]

        affected_ids = {parent_id, left.id}
        parent_refs: list[tuple[str, str | None]] = [(left.id, parent_id)]
        if right is not None:
            parent_refs.append((right.id, parent_id))
            affected_ids.add(right.id)

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


__all__ = [
    "ReadyParentCandidate",
    "SummaryBackend",
    "WorkerStatus",
    "compute_ready_parent_candidates",
    "WorkerCoordinator",
]
