"""Telemetry run lifecycle management for server-side indexing."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from ragzoom.config import IndexConfig
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.telemetry_log import DocumentTelemetryLog
from ragzoom.telemetry_types import TelemetryDataDict

RunStatus = Literal["in_progress", "completed", "failed"]

logger = logging.getLogger(__name__)


@dataclass
class IndexRunContext:
    """Mutable record describing a telemetry-enabled append run."""

    run_id: str
    document_id: str
    start_time: float
    telemetry_collector: TelemetryCollector | None
    append_id: str
    collect_telemetry: bool
    replace_existing: bool
    document_path: str | None
    source_tokens: int | None
    status: RunStatus = "in_progress"
    end_time: float | None = None
    result: TelemetryDataDict | None = None
    error: str | None = None
    awaiters: list[asyncio.Future[IndexRunContext]] = field(default_factory=list)
    append_span_start: int | None = None
    append_span_end: int | None = None
    mutated_nodes: int | None = None
    new_leaves: int | None = None
    leaf_delta: int | None = None
    previous_leaf_count: int | None = None
    total_leaves: int | None = None
    summary_node_ids: set[str] = field(default_factory=set)

    def register_append_outcome(
        self,
        *,
        span_start: int,
        span_end: int,
        mutated_nodes: int,
        new_leaves: int,
        previous_leaf_count: int,
        total_leaves: int,
    ) -> None:
        self.append_span_start = span_start
        self.append_span_end = span_end
        self.mutated_nodes = mutated_nodes
        self.new_leaves = new_leaves
        self.previous_leaf_count = previous_leaf_count
        self.total_leaves = total_leaves
        self.leaf_delta = total_leaves - previous_leaf_count

    def register_summary_node(self, node_id: str) -> None:
        self.summary_node_ids.add(node_id)

    def _finalize_telemetry(self) -> TelemetryDataDict | None:
        collector = self.telemetry_collector
        if collector is None:
            return None

        if (
            self.append_span_start is not None
            and self.append_span_end is not None
            and self.mutated_nodes is not None
            and self.leaf_delta is not None
        ):
            collector.record_append_metadata(
                span_start=self.append_span_start,
                span_end=self.append_span_end,
                mutated_nodes=self.mutated_nodes,
                summary_nodes=len(self.summary_node_ids),
                leaf_delta=self.leaf_delta,
            )

        return collector.finalize()

    def finalize(self, *, error: str | None) -> None:
        if self.status != "in_progress":
            return

        telemetry = self._finalize_telemetry()
        self.result = telemetry
        self.error = error
        self.status = "failed" if error else "completed"
        self.end_time = time.time()
        for future in self.awaiters:
            if not future.done():
                future.set_result(self)
        self.awaiters.clear()

    def add_awaiter(self) -> asyncio.Future[IndexRunContext]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[IndexRunContext] = loop.create_future()
        if self.status != "in_progress":
            future.set_result(self)
        else:
            self.awaiters.append(future)
        return future


class TelemetryRunManager:
    """Tracks active and recently completed telemetry-enabled runs."""

    def __init__(
        self,
        index_config: IndexConfig,
        *,
        ttl_seconds: float = 60 * 60 * 24,
        telemetry_log: DocumentTelemetryLog | None = None,
    ) -> None:
        self._index_config = index_config
        self._ttl_seconds = ttl_seconds
        self._runs: dict[str, IndexRunContext] = {}
        self._active_by_document: dict[str, IndexRunContext] = {}
        self._lock = asyncio.Lock()
        self._telemetry_log = telemetry_log

    async def start_run(
        self,
        document_id: str,
        *,
        collect: bool,
        source_tokens: int,
        document_path: str | None,
        replace_existing: bool,
    ) -> IndexRunContext:
        async with self._lock:
            self._prune_expired_locked()
            run_id = str(uuid.uuid4())
            collector = (
                TelemetryCollector(
                    document_id,
                    source_tokens,
                    self._index_config,
                    document_path=document_path,
                )
                if collect
                else None
            )
            context = IndexRunContext(
                run_id=run_id,
                document_id=document_id,
                start_time=time.time(),
                telemetry_collector=collector,
                append_id=run_id,
                collect_telemetry=collect,
                replace_existing=replace_existing,
                document_path=document_path,
                source_tokens=source_tokens,
            )
            self._runs[run_id] = context
            if collect:
                self._active_by_document[document_id] = context
        await self._log_run_started(context)
        return context

    async def get_run(self, run_id: str) -> IndexRunContext | None:
        async with self._lock:
            return self._runs.get(run_id)

    async def latest_for_document(self, document_id: str) -> IndexRunContext | None:
        async with self._lock:
            context = self._active_by_document.get(document_id)
            if context is not None:
                return context
            contexts = [
                ctx for ctx in self._runs.values() if ctx.document_id == document_id
            ]
            contexts.sort(key=lambda ctx: ctx.start_time, reverse=True)
            return contexts[0] if contexts else None

    async def complete_run(
        self,
        run_id: str,
        *,
        error: str | None = None,
    ) -> IndexRunContext | None:
        log_context: IndexRunContext | None = None
        async with self._lock:
            context = self._runs.get(run_id)
            if context is None:
                return None
            if context.status != "in_progress":
                return context
            context.finalize(error=error)
            if context.document_id in self._active_by_document:
                self._active_by_document.pop(context.document_id)
            self._prune_expired_locked()
            log_context = context
        if log_context is not None:
            await self._log_run_completed(log_context)
        return log_context

    async def wait_for_completion(self, run: IndexRunContext) -> IndexRunContext:
        future = run.add_awaiter()
        result = await future
        return result

    async def prune_expired(self) -> None:
        async with self._lock:
            self._prune_expired_locked()

    def _prune_expired_locked(self) -> None:
        now = time.time()
        expired: list[str] = []
        for run_id, context in list(self._runs.items()):
            if context.status == "in_progress":
                continue
            if context.end_time is None:
                continue
            if now - context.end_time > self._ttl_seconds:
                expired.append(run_id)
        if not expired:
            return
        for run_id in expired:
            if run_id not in self._runs:
                continue
            context = self._runs.pop(run_id)
            if context is None:
                continue
            current = self._active_by_document.get(context.document_id)
            if current is context:
                self._active_by_document.pop(context.document_id)

    async def record_node_committed(
        self,
        context: IndexRunContext,
        *,
        node_id: str,
        height: int,
        span_start: int,
        span_end: int,
    ) -> None:
        if not self._should_log(context):
            return

        telemetry_log = self._telemetry_log
        if telemetry_log is None:
            return

        event: dict[str, object] = {
            "event": "node_committed",
            "run_id": context.run_id,
            "append_id": context.append_id,
            "node_id": node_id,
            "height": height,
            "span_start": span_start,
            "span_end": span_end,
        }

        await telemetry_log.append_event(context.document_id, event)

    async def clear_document(self, document_id: str) -> None:
        if self._telemetry_log is None:
            return
        await self._telemetry_log.clear(document_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _should_log(self, context: IndexRunContext) -> bool:
        return context.collect_telemetry and self._telemetry_log is not None

    async def _log_run_started(self, context: IndexRunContext) -> None:
        if not self._should_log(context):
            return

        telemetry_log = self._telemetry_log
        if telemetry_log is None:
            return

        collector = context.telemetry_collector
        if collector is None:
            return

        metadata = collector.metadata_snapshot()
        try:
            await telemetry_log.ensure_metadata(
                context.document_id,
                metadata,
                reset=context.replace_existing,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to write telemetry metadata for %s", context.document_id
            )

        source_tokens = int(context.source_tokens or 0)
        start_event: dict[str, object] = {
            "event": "append_started",
            "run_id": context.run_id,
            "append_id": context.append_id,
            "source_tokens": source_tokens,
            "replace_existing": context.replace_existing,
        }
        if context.document_path:
            start_event["document_path"] = context.document_path

        await telemetry_log.append_event(context.document_id, start_event)

    async def _log_run_completed(self, context: IndexRunContext) -> None:
        if not self._should_log(context):
            return

        telemetry_log = self._telemetry_log
        if telemetry_log is None:
            return

        duration = None
        if context.end_time is not None:
            duration = context.end_time - context.start_time

        base_event: dict[str, object] = {
            "run_id": context.run_id,
            "append_id": context.append_id,
        }
        if duration is not None:
            base_event["duration"] = duration

        if context.status == "failed":
            event: dict[str, object] = {
                **base_event,
                "event": "append_failed",
            }
            if context.error:
                event["error"] = context.error
            await telemetry_log.append_event(context.document_id, event)
            return

        telemetry = context.result
        nodes_payload: list[dict[str, object]] = []
        append_metadata: dict[str, object] | None = None
        if telemetry is not None:
            raw_nodes = telemetry.get("nodes", [])
            if isinstance(raw_nodes, list):
                for node_payload in raw_nodes:
                    if isinstance(node_payload, dict):
                        nodes_payload.append(dict(node_payload))
            meta = telemetry.get("append_metadata")
            if isinstance(meta, dict):
                append_metadata = dict(meta)

        mutated_nodes = (
            context.mutated_nodes if context.mutated_nodes is not None else 0
        )
        new_leaves = context.new_leaves if context.new_leaves is not None else 0
        leaf_delta = context.leaf_delta if context.leaf_delta is not None else 0
        previous_leaf_count = (
            context.previous_leaf_count
            if context.previous_leaf_count is not None
            else 0
        )
        total_leaves = context.total_leaves if context.total_leaves is not None else 0

        outcome: dict[str, object] = {
            "mutated_nodes": mutated_nodes,
            "new_leaves": new_leaves,
            "leaf_delta": leaf_delta,
            "previous_leaf_count": previous_leaf_count,
            "total_leaves": total_leaves,
            "summary_nodes": len(context.summary_node_ids),
            "nodes": nodes_payload,
        }
        if context.append_span_start is not None:
            outcome["span_start"] = context.append_span_start
        if context.append_span_end is not None:
            outcome["span_end"] = context.append_span_end
        if append_metadata is not None:
            outcome["append_metadata"] = append_metadata

        completed_event: dict[str, object] = {
            **base_event,
            "event": "append_completed",
            "outcome": outcome,
        }

        try:
            await telemetry_log.append_event(context.document_id, completed_event)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to append telemetry event for %s", context.document_id
            )


__all__ = ["IndexRunContext", "TelemetryRunManager"]
