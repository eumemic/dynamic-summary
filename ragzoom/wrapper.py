"""High-level Python client for the RagZoom gRPC services."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar

from ragzoom.client.grpc_client import ExecuteQueryOutput, GrpcRagzoomClient
from ragzoom.constants import DEFAULT_GRPC_ADDRESS
from ragzoom.services.indexing_service import IndexingResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ragzoom.indexing import ClearedDocumentResult


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
    ) -> IndexingResult: ...

    async def clear(self) -> ClearedDocumentResult: ...


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
    ) -> IndexingResult:
        """Append text to an existing document without clearing it."""

        return self._append(
            document_id=document_id,
            text=text,
            collect_telemetry=collect_telemetry,
            replace_existing=False,
        )

    def _append(
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
            session = self._runtime.get_session(document_id)
            return self._run_runtime(
                session.append_text(
                    text,
                    replace_existing=replace_existing,
                    collect_telemetry=collect_telemetry,
                )
            )

        with self._client() as client:
            return client.append_text(
                document_id=document_id,
                content=text.encode("utf-8"),
                collect_telemetry=collect_telemetry,
                replace_existing=replace_existing,
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
        tiling_strategy: str | None = None,
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
                tiling_strategy=tiling_strategy,
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
    """Async wrapper around the RagZoom gRPC services."""

    def __init__(
        self,
        *,
        server_address: str | None = None,
        timeout: float | None = None,
        runtime: _RuntimeProtocol | None = None,
    ) -> None:
        self._sync = RagZoom(
            server_address=server_address,
            timeout=timeout,
            runtime=runtime,
        )

    async def _call(
        self,
        func: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return await asyncio.to_thread(func, *args, **kwargs)

    # jscpd:ignore-start - Async wrappers intentionally mirror sync API
    async def index(
        self,
        document_id: str,
        text: str,
        *,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        return await self._call(
            self._sync.index,
            document_id,
            text,
            collect_telemetry=collect_telemetry,
        )

    async def append(
        self,
        document_id: str,
        text: str,
        *,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        return await self._call(
            self._sync.append,
            document_id,
            text,
            collect_telemetry=collect_telemetry,
        )

    async def clear(self, document_id: str) -> None:
        await self._call(self._sync.clear, document_id)

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
        return await self._call(
            self._sync.query,
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
