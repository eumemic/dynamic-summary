"""High-level Python client for the RagZoom gRPC services."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar

from ragzoom.client.grpc_client import (
    ExecuteQueryOutput,
    GrpcRagzoomClient,
    TruncateResult,
)
from ragzoom.constants import DEFAULT_GRPC_ADDRESS
from ragzoom.services.indexing_service import IndexingResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ragzoom.indexing import ClearedDocumentResult
    from ragzoom.indexing.runtime import TruncateResult as RuntimeTruncateResult


def _resolve_address(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_value = os.environ.get("RAGZOOM_SERVER_ADDRESS")
    if env_value:
        return env_value
    return DEFAULT_GRPC_ADDRESS


T = TypeVar("T")


@dataclass
class QueryResponse:
    """Convenience container for query outputs."""

    summary: str
    token_count: int
    nodes_retrieved: int
    tiling_size: int
    raw: ExecuteQueryOutput


class _SessionProtocol(Protocol):
    async def append_text(
        self,
        text: str,
        *,
        replace_existing: bool,
        collect_telemetry: bool,
        timestamp: str | tuple[str, str] | None = None,
    ) -> IndexingResult: ...

    async def batch_append_text(
        self,
        units: list[str],
        *,
        collect_telemetry: bool,
        timestamps: list[str | tuple[str, str]] | None = None,
    ) -> IndexingResult: ...

    async def clear(self) -> ClearedDocumentResult: ...

    async def truncate_from_span(self, span_start: int) -> RuntimeTruncateResult: ...


class _RuntimeProtocol(Protocol):
    def get_session(
        self, document_id: str, *, file_path: str | None = None
    ) -> _SessionProtocol: ...


class RagZoom:
    """Synchronous wrapper around the RagZoom gRPC services."""

    def __init__(
        self,
        *,
        server_address: str | None = None,
        timeout: float | None = None,
        runtime: _RuntimeProtocol | None = None,
    ) -> None:
        self._runtime = runtime
        self._address: str | None
        if runtime is None or server_address is not None:
            self._address = _resolve_address(server_address)
        else:
            self._address = None
        self._timeout = timeout

    def _client(self) -> GrpcRagzoomClient:
        if self._address is None:
            raise RuntimeError(
                "RagZoom was configured without a server address; network operations require a gRPC endpoint"
            )
        return GrpcRagzoomClient(self._address, timeout=self._timeout)

    def _run_runtime(self, awaitable: Coroutine[object, object, T]) -> T:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        if loop.is_running():
            raise RuntimeError(
                "Runtime-backed RagZoom operations must use AsyncRagZoom when an event loop is already running"
            )
        return loop.run_until_complete(awaitable)

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------
    def index(
        self,
        document_id: str,
        text: str,
        *,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Rebuild a document by clearing existing nodes then appending text."""

        return self._append(
            document_id=document_id,
            text=text,
            collect_telemetry=collect_telemetry,
            replace_existing=True,
        )

    def append(
        self,
        document_id: str,
        text: str,
        *,
        collect_telemetry: bool = False,
        timestamp: str | tuple[str, str] | None = None,
    ) -> IndexingResult:
        """Append text to an existing document without clearing it.

        Args:
            document_id: The document to append to
            text: Text content to append
            collect_telemetry: Whether to collect telemetry data
            timestamp: Optional ISO 8601 timestamp. Can be a single string
                (used for both start and end) or a tuple of (start, end) strings.
        """

        return self._append(
            document_id=document_id,
            text=text,
            collect_telemetry=collect_telemetry,
            replace_existing=False,
            timestamp=timestamp,
        )

    def batch_append(
        self,
        document_id: str,
        units: list[str],
        *,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Append multiple text units with forced split boundaries between them.

        Each unit creates a forced boundary, meaning text is never merged across
        unit boundaries. Semantically equivalent to calling append() for each unit
        sequentially, but executed in a single transaction for efficiency.
        """
        if not document_id:
            raise ValueError("document_id is required")
        if not units:
            raise ValueError("units must be non-empty")

        if self._runtime is not None:
            session = self._runtime.get_session(document_id)
            return self._run_runtime(
                session.batch_append_text(
                    units,
                    collect_telemetry=collect_telemetry,
                )
            )

        with self._client() as client:
            return client.batch_append_text(
                document_id=document_id,
                units=units,
                collect_telemetry=collect_telemetry,
            )

    def _append(
        self,
        *,
        document_id: str,
        text: str,
        collect_telemetry: bool,
        replace_existing: bool,
        timestamp: str | tuple[str, str] | None = None,
    ) -> IndexingResult:
        if not document_id:
            raise ValueError("document_id is required")
        if not text:
            raise ValueError("text must be non-empty")

        if self._runtime is not None:
            session = self._runtime.get_session(document_id)
            return self._run_runtime(
                session.append_text(
                    text,
                    replace_existing=replace_existing,
                    collect_telemetry=collect_telemetry,
                    timestamp=timestamp,
                )
            )

        with self._client() as client:
            return client.append_text(
                document_id=document_id,
                content=text.encode("utf-8"),
                collect_telemetry=collect_telemetry,
                replace_existing=replace_existing,
                timestamp=timestamp,
            )

    def clear(self, document_id: str) -> None:
        """Delete all nodes for a document and cancel in-flight work."""

        if not document_id:
            raise ValueError("document_id is required")
        if self._runtime is not None:
            session = self._runtime.get_session(document_id)
            self._run_runtime(session.clear())
            return

        with self._client() as client:
            client.clear_document(document_id)

    def truncate(self, document_id: str, span_start: int) -> TruncateResult:
        """Delete all nodes at or after the given span position."""

        if not document_id:
            raise ValueError("document_id is required")
        if span_start < 0:
            raise ValueError("span_start must be non-negative")

        if self._runtime is not None:
            session = self._runtime.get_session(document_id)
            runtime_result = self._run_runtime(session.truncate_from_span(span_start))
            return TruncateResult(
                document_id=runtime_result.document_id,
                deleted_node_ids=runtime_result.deleted_node_ids,
                span_start=runtime_result.span_start,
            )

        with self._client() as client:
            return client.truncate_document(
                document_id=document_id,
                span_start=span_start,
            )

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def query(
        self,
        document_id: str,
        query_text: str,
        *,
        budget_tokens: int | None = None,
        num_seeds: int | None = None,
        embedding_model: str | None = None,
        debug: bool = False,
        viz_width: int = 120,
        use_token_coords: bool = False,
    ) -> QueryResponse:
        if not document_id:
            raise ValueError("document_id is required")
        if not query_text:
            raise ValueError("query_text must be non-empty")

        with self._client() as client:
            output = client.execute_query(
                query=query_text,
                document_id=document_id,
                budget_tokens=budget_tokens,
                num_seeds=num_seeds,
                embedding_model=embedding_model,
                debug=debug,
                viz_width=viz_width,
                use_token_coords=use_token_coords,
            )

        result = output.query_result
        return QueryResponse(
            summary=result.summary,
            token_count=result.token_count,
            nodes_retrieved=result.nodes_retrieved,
            tiling_size=result.tiling_size,
            raw=output,
        )


class AsyncRagZoom:
    """Async wrapper around the RagZoom gRPC services.

    When a runtime is provided, async methods call the session directly
    on the current event loop. This is critical for background tasks
    (like indexing jobs) to not be cancelled - they must run on the
    same loop as the caller.

    When no runtime is provided (gRPC client mode), operations are
    delegated to a thread pool since the gRPC client may be sync.
    """

    # jscpd:ignore-start - Async wrapper intentionally mirrors sync class init
    def __init__(
        self,
        *,
        server_address: str | None = None,
        timeout: float | None = None,
        runtime: _RuntimeProtocol | None = None,
    ) -> None:
        self._runtime = runtime
        self._address: str | None
        if runtime is None or server_address is not None:
            self._address = _resolve_address(server_address)
        else:
            self._address = None
        self._timeout = timeout
        # jscpd:ignore-end
        # Lazy sync wrapper - only created when needed for query()
        self._sync: RagZoom | None = None
        self._server_address_for_sync = server_address

    def _get_sync(self) -> RagZoom:
        """Get or create the sync wrapper for gRPC operations."""
        if self._sync is None:
            self._sync = RagZoom(
                server_address=self._server_address_for_sync,
                timeout=self._timeout,
                runtime=None,  # Don't pass runtime - we handle it directly
            )
        return self._sync

    def _client(self) -> GrpcRagzoomClient:
        if self._address is None:
            raise RuntimeError(
                "AsyncRagZoom was configured without a server address; "
                "network operations require a gRPC endpoint"
            )
        return GrpcRagzoomClient(self._address, timeout=self._timeout)

    async def _call_sync(
        self,
        func: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Run a sync function in a thread pool."""
        return await asyncio.to_thread(func, *args, **kwargs)

    # jscpd:ignore-start - Async wrappers intentionally mirror sync API
    async def index(
        self,
        document_id: str,
        text: str,
        *,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        return await self._append(
            document_id=document_id,
            text=text,
            collect_telemetry=collect_telemetry,
            replace_existing=True,
        )

    async def append(
        self,
        document_id: str,
        text: str,
        *,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        return await self._append(
            document_id=document_id,
            text=text,
            collect_telemetry=collect_telemetry,
            replace_existing=False,
        )

    async def _append(
        self,
        *,
        document_id: str,
        text: str,
        collect_telemetry: bool,
        replace_existing: bool,
    ) -> IndexingResult:
        if not document_id:
            raise ValueError("document_id is required")
        if not text:
            raise ValueError("text must be non-empty")

        if self._runtime is not None:
            # Call session directly on current loop - critical for background tasks
            session = self._runtime.get_session(document_id)
            return await session.append_text(
                text,
                replace_existing=replace_existing,
                collect_telemetry=collect_telemetry,
            )

        # gRPC client is sync - run in thread to avoid blocking event loop
        def _do_append() -> IndexingResult:
            with self._client() as client:
                return client.append_text(
                    document_id=document_id,
                    content=text.encode("utf-8"),
                    collect_telemetry=collect_telemetry,
                    replace_existing=replace_existing,
                )

        return await asyncio.to_thread(_do_append)

    async def batch_append(
        self,
        document_id: str,
        units: list[str],
        *,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        if not document_id:
            raise ValueError("document_id is required")
        if not units:
            raise ValueError("units must be non-empty")

        if self._runtime is not None:
            # Call session directly on current loop - critical for background tasks
            session = self._runtime.get_session(document_id)
            return await session.batch_append_text(
                units,
                collect_telemetry=collect_telemetry,
            )

        # gRPC client is sync - run in thread to avoid blocking event loop
        def _do_batch_append() -> IndexingResult:
            with self._client() as client:
                return client.batch_append_text(
                    document_id=document_id,
                    units=units,
                    collect_telemetry=collect_telemetry,
                )

        return await asyncio.to_thread(_do_batch_append)

    async def clear(self, document_id: str) -> None:
        if not document_id:
            raise ValueError("document_id is required")

        if self._runtime is not None:
            session = self._runtime.get_session(document_id)
            await session.clear()
            return

        # gRPC client is sync - run in thread to avoid blocking event loop
        def _do_clear() -> None:
            with self._client() as client:
                client.clear_document(document_id)

        await asyncio.to_thread(_do_clear)

    async def truncate(self, document_id: str, span_start: int) -> TruncateResult:
        if not document_id:
            raise ValueError("document_id is required")
        if span_start < 0:
            raise ValueError("span_start must be non-negative")

        if self._runtime is not None:
            session = self._runtime.get_session(document_id)
            runtime_result = await session.truncate_from_span(span_start)
            return TruncateResult(
                document_id=runtime_result.document_id,
                deleted_node_ids=runtime_result.deleted_node_ids,
                span_start=runtime_result.span_start,
            )

        # gRPC client is sync - run in thread to avoid blocking event loop
        def _do_truncate() -> TruncateResult:
            with self._client() as client:
                return client.truncate_document(
                    document_id=document_id,
                    span_start=span_start,
                )

        return await asyncio.to_thread(_do_truncate)

    async def query(
        self,
        document_id: str,
        query_text: str,
        *,
        budget_tokens: int | None = None,
        num_seeds: int | None = None,
        embedding_model: str | None = None,
        debug: bool = False,
        viz_width: int = 120,
        use_token_coords: bool = False,
    ) -> QueryResponse:
        # Query always goes through gRPC client (no runtime path)
        return await self._call_sync(
            self._get_sync().query,
            document_id,
            query_text,
            budget_tokens=budget_tokens,
            num_seeds=num_seeds,
            embedding_model=embedding_model,
            debug=debug,
            viz_width=viz_width,
            use_token_coords=use_token_coords,
        )

    # jscpd:ignore-end


P = ParamSpec("P")
R = TypeVar("R")
