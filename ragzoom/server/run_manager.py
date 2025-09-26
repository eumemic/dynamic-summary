"""Telemetry run lifecycle management for server-side indexing."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from ragzoom.config import IndexConfig
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.telemetry_types import TelemetryDataDict

RunStatus = Literal["in_progress", "completed", "failed"]


@dataclass(slots=True)
class IndexRunContext:
    """Mutable record describing a telemetry-enabled append run."""

    run_id: str
    document_id: str
    start_time: float
    telemetry_collector: TelemetryCollector | None
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
    ) -> None:
        self._index_config = index_config
        self._ttl_seconds = ttl_seconds
        self._runs: dict[str, IndexRunContext] = {}
        self._active_by_document: dict[str, IndexRunContext] = {}
        self._lock = asyncio.Lock()

    async def start_run(
        self,
        document_id: str,
        *,
        collect: bool,
        source_tokens: int,
        document_path: str | None,
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
            )
            self._runs[run_id] = context
            if collect:
                self._active_by_document[document_id] = context
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
            return context

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


__all__ = ["IndexRunContext", "TelemetryRunManager"]
