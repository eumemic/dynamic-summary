"""Core runtime for RagZoom document indexing operations."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import dataclass
from typing import Literal, cast

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.server.append_executor import AppendExecutor, AppendOutcome
from ragzoom.server.indexing_engine import IndexingEngine, IndexingStatus
from ragzoom.server.run_manager import IndexRunContext, TelemetryRunManager
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


VectorIndexFactory = Callable[[str], VectorIndex]
ProgressCallback = Callable[["ProgressEvent"], None]


@dataclass
class ClearedDocumentResult:
    """Outcome from clearing a document."""

    document_id: str
    deleted_nodes: int
    document_existed: bool


@dataclass
class TruncateResult:
    """Outcome from truncating a document at a span position."""

    document_id: str
    deleted_node_ids: list[str]
    span_start: int


@dataclass
class TruncateFromTimeResult:
    """Outcome from truncating a temporal document at a cutoff time.

    Used for time-based truncation where all nodes with time_end > cutoff_time
    are deleted. This is the temporal analog of TruncateResult (span-based).
    """

    document_id: str
    deleted_node_ids: list[str]
    cutoff_time: float


@dataclass
class ProgressEvent:
    """Snapshot of worker progress for a document."""

    stage: Literal["update", "complete"]
    pending: int
    inflight: int
    completed: int
    total: int
    error: str | None = None


class ProgressHandle:
    """Handle used to remove a previously registered progress listener."""

    def __init__(
        self,
        *,
        document_id: str,
        callback: ProgressCallback,
        runtime: IndexerRuntime,
    ) -> None:
        self._document_id = document_id
        self._callback = callback
        self._runtime = runtime
        self._closed = False

    def unsubscribe(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runtime._remove_progress_listener(self._document_id, self._callback)


class IndexerRuntime:
    """Factory for per-document indexing sessions."""

    def __init__(
        self,
        *,
        store: StorageBackend,
        index_config: IndexConfig,
        append_executor: AppendExecutor,
        indexing_engine: IndexingEngine,
        telemetry_manager: TelemetryRunManager | None,
        vector_index_factory: VectorIndexFactory,
    ) -> None:
        self._store = store
        self._index_config = index_config
        self._append_executor = append_executor
        self._indexing_engine = indexing_engine
        self._telemetry_manager = telemetry_manager
        self._vector_index_factory = vector_index_factory

        self._listeners: dict[str, set[ProgressCallback]] = defaultdict(set)
        self._listener_lock: asyncio.Lock | None = None
        self._listener_loop: asyncio.AbstractEventLoop | None = None
        self._last_progress: dict[str, tuple[str, int, int, int, int]] = {}
        self._progress_task: asyncio.Task[None] | None = None
        self._poll_interval = 0.5

    def get_session(
        self,
        document_id: str,
        *,
        file_path: str | None = None,
    ) -> DocumentIndexSession:
        return DocumentIndexSession(
            runtime=self,
            document_id=document_id,
            file_path=file_path,
        )

    async def _emit_status(self, document_id: str) -> None:
        listeners = self._listeners.get(document_id)
        if not listeners:
            return
        status = await self._indexing_engine.status()
        event = self._build_progress_event(status, document_id)
        if event is not None:
            self._dispatch_progress(document_id, event)

    def _build_progress_event(
        self, status: IndexingStatus, document_id: str, *, error: str | None = None
    ) -> ProgressEvent | None:
        # New status model: in_flight (no pending queue)
        inflight = status.in_flight_by_document.get(document_id, 0)
        completed = status.completed_by_document.get(document_id, 0)
        pending = 0  # No explicit queue in new model
        total = status.expected_total_by_document.get(document_id, inflight + completed)

        stage: Literal["update", "complete"]
        if inflight == 0:
            stage = "complete"
        else:
            stage = "update"

        progress_key = (stage, pending, inflight, completed, total)
        last_key = self._last_progress.get(document_id)
        if last_key == progress_key and error is None:
            return None

        if stage == "complete" and total == 0 and last_key is not None:
            # Preserve total from previous snapshot when queue drains completely.
            total = last_key[4]

        self._last_progress[document_id] = progress_key
        return ProgressEvent(
            stage=stage,
            pending=pending,
            inflight=inflight,
            completed=completed,
            total=total,
            error=error,
        )

    def _dispatch_progress(self, document_id: str, event: ProgressEvent) -> None:
        listeners = self._listeners.get(document_id)
        if not listeners:
            return
        for callback in list(listeners):
            try:
                callback(event)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Progress listener for %s raised an exception", document_id
                )

    def _get_listener_lock(self) -> asyncio.Lock:
        if self._listener_lock is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError as exc:  # pragma: no cover - API misuse
                raise RuntimeError(
                    "Progress operations require an active event loop"
                ) from exc
            self._listener_lock = asyncio.Lock()
            self._listener_loop = loop
        return self._listener_lock

    async def _progress_poll_loop(self) -> None:
        try:
            while True:
                lock = self._get_listener_lock()
                async with lock:
                    active_ids = [
                        doc_id
                        for doc_id, callbacks in self._listeners.items()
                        if callbacks
                    ]
                if not active_ids:
                    break

                try:
                    status = await self._indexing_engine.status()
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to poll worker status for progress loop")
                else:
                    for document_id in active_ids:
                        event = self._build_progress_event(status, document_id)
                        if event is not None:
                            self._dispatch_progress(document_id, event)

                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        finally:
            self._progress_task = None

    def _schedule_progress_loop(self) -> None:
        if self._progress_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no loop running
            logger.warning(
                "register_progress_listener called outside running loop; progress updates disabled"
            )
            return
        self._listener_loop = loop
        self._progress_task = loop.create_task(self._progress_poll_loop())

    async def _add_progress_listener(
        self, document_id: str, callback: ProgressCallback
    ) -> None:
        lock = self._get_listener_lock()
        async with lock:
            self._listeners[document_id].add(callback)
        self._schedule_progress_loop()
        await self._emit_status(document_id)

    def _remove_progress_listener(
        self, document_id: str, callback: ProgressCallback
    ) -> None:
        async def _remove() -> None:
            lock = self._get_listener_lock()
            async with lock:
                callbacks = self._listeners.get(document_id)
                if callbacks and callback in callbacks:
                    callbacks.remove(callback)
                if callbacks and not callbacks:
                    self._listeners.pop(document_id, None)

        loop = self._listener_loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover - no loop
                raise RuntimeError(
                    "unsubscribe requires the runtime event loop to be running"
                ) from None
        loop.call_soon_threadsafe(asyncio.create_task, _remove())

    def register_progress_listener(
        self, document_id: str, callback: ProgressCallback
    ) -> ProgressHandle:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "register_progress_listener requires an active event loop"
            ) from None
        loop.create_task(self._add_progress_listener(document_id, callback))
        return ProgressHandle(document_id=document_id, callback=callback, runtime=self)


class DocumentIndexSession:
    """Per-document orchestrator for append and clear operations."""

    def __init__(
        self,
        *,
        runtime: IndexerRuntime,
        document_id: str,
        file_path: str | None,
    ) -> None:
        self._runtime = runtime
        self._document_id = document_id
        self._file_path = file_path

    @property
    def document_id(self) -> str:
        return self._document_id

    async def append_text(
        self,
        text: str,
        *,
        replace_existing: bool,
        collect_telemetry: bool = False,
        timestamp: str | tuple[str, str] | None = None,
        summarization_guidance: str | None = None,
    ) -> IndexingResult:
        # Allow empty text only when target_chunk_tokens=None (client-managed mode)
        if not text and self._runtime._index_config.target_chunk_tokens is not None:
            raise ValueError("text must be non-empty")

        if replace_existing:
            await self.clear()

        store = self._runtime._store
        telemetry_manager = self._runtime._telemetry_manager
        lock_cm = self._lock_document(store)

        run_context: IndexRunContext | None = None
        outcome: AppendOutcome | None = None
        result: IndexingResult | None = None
        previous_leaf_count = 0

        try:
            with lock_cm:
                doc_record = store.get_document_by_id(self._document_id)
                embedding_model = (
                    getattr(doc_record, "embedding_model", None)
                    if doc_record is not None
                    else None
                ) or self._runtime._index_config.embedding_model
                summary_model = (
                    getattr(doc_record, "summary_model", None)
                    if doc_record is not None
                    else None
                ) or self._runtime._index_config.summary_model
                stored_path = getattr(doc_record, "file_path", None)
                resolved_path = (
                    self._file_path if self._file_path is not None else stored_path
                )

                if doc_record is None:
                    store.add_document(
                        document_id=self._document_id,
                        file_path=resolved_path,
                        embedding_model=embedding_model,
                        summary_model=summary_model,
                        summarization_guidance=summarization_guidance,
                    )
                    doc_record = store.get_document_by_id(self._document_id)

                document_store = store.for_document(self._document_id)
                previous_leaf_count = document_store.nodes.leaf_count()

                if collect_telemetry and telemetry_manager is not None:
                    existing_tokens = document_store.nodes.sum_leaf_tokens()
                    new_tokens = tokenizer.count_tokens(text)
                    source_tokens = existing_tokens + new_tokens
                    run_context = await telemetry_manager.start_run(
                        self._document_id,
                        collect=True,
                        source_tokens=source_tokens,
                        document_path=(
                            self._file_path
                            if self._file_path is not None
                            else getattr(doc_record, "file_path", None)
                        ),
                        replace_existing=replace_existing,
                    )

                outcome = await self._runtime._append_executor.append(
                    store=document_store,
                    document_id=self._document_id,
                    new_text=text,
                    timestamp=timestamp,
                    reporter=run_context.telemetry_collector if run_context else None,
                    run_context=run_context,
                    telemetry_manager=telemetry_manager,
                )

                if (
                    telemetry_manager is not None
                    and run_context is not None
                    and outcome.deleted_node_ids
                ):
                    await telemetry_manager.record_nodes_deleted(
                        run_context,
                        node_ids=outcome.deleted_node_ids,
                    )

                root = document_store.tree.get_root()
                tree_depth = int(getattr(root, "height", 0) or 0) if root else 0
                mutated_nodes = len(outcome.new_leaf_ids) + len(
                    outcome.deleted_node_ids
                )
                new_leaves = len(outcome.new_leaf_ids)

                result = IndexingResult(
                    document_id=outcome.document_id,
                    chunks_created=outcome.total_leaves,
                    tree_depth=tree_depth,
                    span_start=outcome.appended_span_start,
                    span_end=outcome.appended_span_end,
                    mutated_nodes=mutated_nodes,
                    resummarized_nodes=0,
                    new_leaves=new_leaves,
                    telemetry=None,
                    telemetry_run_id=run_context.run_id if run_context else None,
                )

                if run_context is not None:
                    run_context.register_append_outcome(
                        span_start=outcome.appended_span_start,
                        span_end=outcome.appended_span_end,
                        mutated_nodes=mutated_nodes,
                        new_leaves=new_leaves,
                        previous_leaf_count=previous_leaf_count,
                        total_leaves=outcome.total_leaves,
                    )

            if run_context is not None:
                await self._runtime._indexing_engine.register_run(
                    self._document_id,
                    run_id=run_context.run_id,
                    telemetry_collector=run_context.telemetry_collector,
                    new_leaf_ids=outcome.new_leaf_ids,
                )

            # Update chars_per_token after append (client-managed chunking mode)
            self._runtime._indexing_engine.update_chars_per_token_after_append(
                self._document_id
            )

            # Trigger indexing work - engine discovers leaves and sibling pairs
            await self._runtime._indexing_engine.trigger_work(self._document_id)

            await self._runtime._emit_status(self._document_id)
            assert result is not None
            return result
        except Exception as exc:
            logger.exception(
                "Append failed for document %s", self._document_id, exc_info=True
            )
            if run_context is not None and telemetry_manager is not None:
                with suppress(Exception):
                    await telemetry_manager.complete_run(
                        run_context.run_id, error=str(exc)
                    )
            raise

    # jscpd:ignore-start - Parallel structure to append_text intentional (batch vs single)
    async def batch_append_text(
        self,
        units: list[str],
        *,
        collect_telemetry: bool = False,
        timestamps: list[str | tuple[str, str]] | None = None,
        summarization_guidance: str | None = None,
    ) -> IndexingResult:
        """Append multiple text units with forced split boundaries between them.

        Each unit creates a forced split boundary, meaning text is never merged
        across unit boundaries. This is semantically equivalent to calling
        append_text() for each unit sequentially, but executed in a single
        transaction for efficiency.

        Args:
            units: List of text units, each creating a forced boundary
            collect_telemetry: Whether to collect telemetry data
            timestamps: Optional list of timestamps parallel to units. Each entry
                can be an ISO 8601 string (used for both start and end) or a tuple
                of (start, end) strings.
            summarization_guidance: Optional guidance for summary generation.
                Stored on document at creation time and used by the summarizer.

        Returns:
            IndexingResult with combined stats for all appended units
        """
        import time

        t0 = time.perf_counter()
        store = self._runtime._store
        telemetry_manager = self._runtime._telemetry_manager
        lock_cm = self._lock_document(store)

        run_context: IndexRunContext | None = None
        outcome: AppendOutcome | None = None
        result: IndexingResult | None = None
        previous_leaf_count = 0

        try:
            with lock_cm:
                t_lock = time.perf_counter()
                doc_record = store.get_document_by_id(self._document_id)
                embedding_model = (
                    getattr(doc_record, "embedding_model", None)
                    if doc_record is not None
                    else None
                ) or self._runtime._index_config.embedding_model
                summary_model = (
                    getattr(doc_record, "summary_model", None)
                    if doc_record is not None
                    else None
                ) or self._runtime._index_config.summary_model
                stored_path = getattr(doc_record, "file_path", None)
                resolved_path = (
                    self._file_path if self._file_path is not None else stored_path
                )

                if doc_record is None:
                    store.add_document(
                        document_id=self._document_id,
                        file_path=resolved_path,
                        embedding_model=embedding_model,
                        summary_model=summary_model,
                        summarization_guidance=summarization_guidance,
                    )
                    doc_record = store.get_document_by_id(self._document_id)

                document_store = store.for_document(self._document_id)
                previous_leaf_count = document_store.nodes.leaf_count()
                t_setup = time.perf_counter()

                if collect_telemetry and telemetry_manager is not None:
                    existing_tokens = document_store.nodes.sum_leaf_tokens()
                    new_tokens = sum(tokenizer.count_tokens(u) for u in units if u)
                    source_tokens = existing_tokens + new_tokens
                    run_context = await telemetry_manager.start_run(
                        self._document_id,
                        collect=True,
                        source_tokens=source_tokens,
                        document_path=(
                            self._file_path
                            if self._file_path is not None
                            else getattr(doc_record, "file_path", None)
                        ),
                        replace_existing=False,
                    )

                outcome = await self._runtime._append_executor.append_batch(
                    store=document_store,
                    document_id=self._document_id,
                    units=units,
                    timestamps=timestamps,
                    reporter=run_context.telemetry_collector if run_context else None,
                    run_context=run_context,
                    telemetry_manager=telemetry_manager,
                )
                t_append = time.perf_counter()
                logger.info(
                    "[TIMING] batch_append_text: lock=%.3fs setup=%.3fs append=%.3fs",
                    t_lock - t0,
                    t_setup - t_lock,
                    t_append - t_setup,
                )

                if (
                    telemetry_manager is not None
                    and run_context is not None
                    and outcome.deleted_node_ids
                ):
                    await telemetry_manager.record_nodes_deleted(
                        run_context,
                        node_ids=outcome.deleted_node_ids,
                    )

                root = document_store.tree.get_root()
                tree_depth = int(getattr(root, "height", 0) or 0) if root else 0
                mutated_nodes = len(outcome.new_leaf_ids) + len(
                    outcome.deleted_node_ids
                )
                new_leaves = len(outcome.new_leaf_ids)

                result = IndexingResult(
                    document_id=outcome.document_id,
                    chunks_created=outcome.total_leaves,
                    tree_depth=tree_depth,
                    span_start=outcome.appended_span_start,
                    span_end=outcome.appended_span_end,
                    mutated_nodes=mutated_nodes,
                    resummarized_nodes=0,
                    new_leaves=new_leaves,
                    telemetry=None,
                    telemetry_run_id=run_context.run_id if run_context else None,
                )

                if run_context is not None:
                    run_context.register_append_outcome(
                        span_start=outcome.appended_span_start,
                        span_end=outcome.appended_span_end,
                        mutated_nodes=mutated_nodes,
                        new_leaves=new_leaves,
                        previous_leaf_count=previous_leaf_count,
                        total_leaves=outcome.total_leaves,
                    )

            if run_context is not None:
                await self._runtime._indexing_engine.register_run(
                    self._document_id,
                    run_id=run_context.run_id,
                    telemetry_collector=run_context.telemetry_collector,
                    new_leaf_ids=outcome.new_leaf_ids,
                )

            # Update chars_per_token after append (client-managed chunking mode)
            self._runtime._indexing_engine.update_chars_per_token_after_append(
                self._document_id
            )

            # Trigger indexing work - engine discovers leaves and sibling pairs
            t_before_trigger = time.perf_counter()
            await self._runtime._indexing_engine.trigger_work(self._document_id)
            t_after_trigger = time.perf_counter()

            await self._runtime._emit_status(self._document_id)
            t_end = time.perf_counter()
            logger.info(
                "[TIMING] batch_append_text: trigger_work=%.3fs emit_status=%.3fs "
                "total=%.3fs",
                t_after_trigger - t_before_trigger,
                t_end - t_after_trigger,
                t_end - t0,
            )
            assert result is not None
            return result
        except Exception as exc:
            logger.exception(
                "Batch append failed for document %s", self._document_id, exc_info=True
            )
            if run_context is not None and telemetry_manager is not None:
                with suppress(Exception):
                    await telemetry_manager.complete_run(
                        run_context.run_id, error=str(exc)
                    )
            raise

    # jscpd:ignore-end

    async def clear(self) -> ClearedDocumentResult:
        await self._runtime._indexing_engine.cancel_document(self._document_id)

        store = self._runtime._store
        lock_cm = self._lock_document(store)

        with lock_cm:
            doc_record = store.get_document_by_id(self._document_id)
            if doc_record is None:
                result = ClearedDocumentResult(
                    document_id=self._document_id,
                    deleted_nodes=0,
                    document_existed=False,
                )
                await self._runtime._emit_status(self._document_id)
                return result

            embedding_model = (
                getattr(doc_record, "embedding_model", None)
                or self._runtime._index_config.embedding_model
            )

            try:
                vector_index = self._runtime._vector_index_factory(embedding_model)
                vector_index.delete(filter={"document_id": self._document_id})
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Failed to clear vectors for document %s", self._document_id
                )

            deleted_nodes = store.clear_document(self._document_id)

        await self._runtime._emit_status(self._document_id)
        telemetry_manager = self._runtime._telemetry_manager
        if telemetry_manager is not None:
            await telemetry_manager.clear_document(self._document_id)
        return ClearedDocumentResult(
            document_id=self._document_id,
            deleted_nodes=deleted_nodes,
            document_existed=True,
        )

    async def truncate_from_span(self, span_start: int) -> TruncateResult:
        """Delete all nodes with span_start >= given position.

        Used for rolling back a document after a conversation revert.
        Cancels any in-flight indexing, deletes nodes from storage,
        and removes corresponding vectors from the vector index.

        Args:
            span_start: Delete all nodes where node.span_start >= this value

        Returns:
            TruncateResult with document_id, deleted node IDs, and span_start
        """
        await self._runtime._indexing_engine.cancel_document(self._document_id)

        store = self._runtime._store
        lock_cm = self._lock_document(store)

        with lock_cm:
            doc_record = store.get_document_by_id(self._document_id)
            if doc_record is None:
                return TruncateResult(
                    document_id=self._document_id,
                    deleted_node_ids=[],
                    span_start=span_start,
                )

            embedding_model = (
                getattr(doc_record, "embedding_model", None)
                or self._runtime._index_config.embedding_model
            )

            # Delete nodes from storage
            deleted_node_ids = store.delete_nodes_from_span(
                self._document_id, span_start
            )

            # Delete vectors for those nodes
            if deleted_node_ids:
                try:
                    vector_index = self._runtime._vector_index_factory(embedding_model)
                    vector_index.delete(filter={"node_id": {"$in": deleted_node_ids}})
                except Exception:
                    logger.exception(
                        "Failed to delete vectors for truncated nodes in document %s",
                        self._document_id,
                    )

        await self._runtime._emit_status(self._document_id)
        return TruncateResult(
            document_id=self._document_id,
            deleted_node_ids=deleted_node_ids,
            span_start=span_start,
        )

    async def truncate_from_time(self, cutoff_time: float) -> TruncateFromTimeResult:
        """Delete all nodes with time_end > cutoff_time.

        Used for rolling back a temporal document to a specific point in time.
        Cancels any in-flight indexing, deletes nodes from storage,
        and removes corresponding vectors from the vector index.

        This is the temporal analog of truncate_from_span.

        Args:
            cutoff_time: Delete all nodes where node.time_end > this value
                        (Unix timestamp in seconds)

        Returns:
            TruncateFromTimeResult with document_id, deleted node IDs, and cutoff_time
        """
        await self._runtime._indexing_engine.cancel_document(self._document_id)

        store = self._runtime._store
        lock_cm = self._lock_document(store)
        deleted_node_ids: list[str] = []

        with lock_cm:
            doc_record = store.get_document_by_id(self._document_id)
            if doc_record is None:
                return TruncateFromTimeResult(
                    document_id=self._document_id,
                    deleted_node_ids=[],
                    cutoff_time=cutoff_time,
                )

            embedding_model = (
                getattr(doc_record, "embedding_model", None)
                or self._runtime._index_config.embedding_model
            )

            # Delete nodes from storage
            deleted_node_ids = store.delete_nodes_from_time(
                self._document_id, cutoff_time
            )

            # Delete vectors for those nodes
            if deleted_node_ids:
                try:
                    vector_index = self._runtime._vector_index_factory(embedding_model)
                    vector_index.delete(filter={"node_id": {"$in": deleted_node_ids}})
                except Exception:
                    logger.exception(
                        "Failed to delete vectors for time-truncated nodes in document %s",
                        self._document_id,
                    )

        await self._runtime._emit_status(self._document_id)
        return TruncateFromTimeResult(
            document_id=self._document_id,
            deleted_node_ids=deleted_node_ids,
            cutoff_time=cutoff_time,
        )

    def register_progress_listener(self, callback: ProgressCallback) -> ProgressHandle:
        return self._runtime.register_progress_listener(self._document_id, callback)

    def _lock_document(self, store: StorageBackend) -> AbstractContextManager[object]:
        lock_fn = getattr(store, "lock_document", None)
        candidate: AbstractContextManager[object] | None = None
        if callable(lock_fn):
            maybe_lock = lock_fn(self._document_id)
            if (
                maybe_lock is not None
                and hasattr(maybe_lock, "__enter__")
                and hasattr(maybe_lock, "__exit__")
            ):
                candidate = cast(AbstractContextManager[object], maybe_lock)
        if candidate is not None:
            return candidate
        return cast(AbstractContextManager[object], nullcontext())
