"""High-level Python client for the RagZoom gRPC services."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

from ragzoom.client.grpc_client import ExecuteQueryOutput, GrpcRagzoomClient
from ragzoom.constants import DEFAULT_GRPC_ADDRESS
from ragzoom.services.indexing_service import IndexingResult


def _resolve_address(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_value = os.environ.get("RAGZOOM_SERVER_ADDRESS")
    if env_value:
        return env_value
    return DEFAULT_GRPC_ADDRESS


@dataclass(slots=True)
class QueryResponse:
    """Convenience container for query outputs."""

    summary: str
    token_count: int
    nodes_retrieved: int
    tiling_size: int
    raw: ExecuteQueryOutput


class RagZoom:
    """Synchronous wrapper around the RagZoom gRPC services."""

    def __init__(
        self,
        *,
        server_address: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._address = _resolve_address(server_address)
        self._timeout = timeout

    def _client(self) -> GrpcRagzoomClient:
        return GrpcRagzoomClient(self._address, timeout=self._timeout)

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
    """Async wrapper around the RagZoom gRPC services."""

    def __init__(
        self,
        *,
        server_address: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._sync = RagZoom(server_address=server_address, timeout=timeout)

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
