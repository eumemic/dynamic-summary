"""Indexing service for RagZoom document processing."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

from ragzoom.config import IndexConfig, OperationalConfig
from ragzoom.constants import DEFAULT_GRPC_ADDRESS
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.telemetry_types import TelemetryDataDict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ragzoom.client.grpc_client import GrpcRagzoomClient

T = TypeVar("T")


class _IndexSession(Protocol):
    async def append_text(
        self,
        text: str,
        *,
        replace_existing: bool,
        collect_telemetry: bool,
    ) -> IndexingResult: ...


class _Runtime(Protocol):
    def get_session(
        self, document_id: str, *, file_path: str | None = None
    ) -> _IndexSession: ...


logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result from document indexing operation."""

    document_id: str
    chunks_created: int
    tree_depth: int
    span_start: int = 0
    """Character offset where appended content starts."""
    span_end: int = 0
    """Character offset where appended content ends."""
    mutated_nodes: int | None = None
    resummarized_nodes: int | None = None
    new_leaves: int | None = None
    telemetry: TelemetryDataDict | None = None
    telemetry_run_id: str | None = None


class IndexingService:
    """Service for document indexing operations."""

    def __init__(
        self,
        store: StorageBackend,
        index_config: IndexConfig,
        operational_config: OperationalConfig,
        *,
        grpc_address: str | None = None,
        client_factory: Callable[[str], GrpcRagzoomClient] | None = None,
        index_runtime: _Runtime | None = None,
    ):
        self.store = store
        self.index_config = index_config
        self.operational_config = operational_config
        if index_runtime is not None and (
            grpc_address is not None or client_factory is not None
        ):
            raise ValueError(
                "IndexingService cannot mix runtime and gRPC client configuration"
            )

        self._index_runtime: _Runtime | None = index_runtime
        self._client_factory: Callable[[str], GrpcRagzoomClient] | None
        self._grpc_address: str | None

        if index_runtime is None:
            resolved_address = self._resolve_address(grpc_address)
            self._grpc_address = resolved_address
            self._client_factory = client_factory or self._default_client_factory
        else:
            self._grpc_address = None
            self._client_factory = None

    def _resolve_address(self, explicit: str | None) -> str:
        if explicit:
            return explicit
        env_address = os.environ.get("RAGZOOM_SERVER_ADDRESS")
        if env_address:
            return env_address
        return DEFAULT_GRPC_ADDRESS

    def _default_client_factory(self, address: str) -> GrpcRagzoomClient:
        from ragzoom.client.grpc_client import GrpcRagzoomClient as _GrpcClient

        return _GrpcClient(address)

    def _append_via_grpc(
        self,
        *,
        document_id: str,
        text: str,
        collect_telemetry: bool,
        replace_existing: bool,
    ) -> IndexingResult:
        if self._client_factory is None or self._grpc_address is None:
            raise RuntimeError("IndexingService is not configured for gRPC access")
        if not text:
            raise ValueError("text must be non-empty")

        with self._client_factory(self._grpc_address) as client:
            result = client.append_text(
                document_id=document_id,
                content=text.encode("utf-8"),
                collect_telemetry=collect_telemetry,
                replace_existing=replace_existing,
            )
        return result

    def _append_via_runtime(
        self,
        *,
        document_id: str,
        text: str,
        collect_telemetry: bool,
        replace_existing: bool,
        file_path: str | None = None,
    ) -> IndexingResult:
        runtime = self._index_runtime
        if runtime is None:
            raise RuntimeError("IndexingService is not configured with a local runtime")
        if not text:
            raise ValueError("text must be non-empty")
        session = runtime.get_session(document_id, file_path=file_path)
        return self._await_runtime(
            session.append_text(
                text,
                replace_existing=replace_existing,
                collect_telemetry=collect_telemetry,
            )
        )

    def _await_runtime(self, awaitable: Coroutine[object, object, T]) -> T:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        if loop.is_running():
            raise RuntimeError(
                "Runtime-backed IndexingService methods must use the async API when an event loop is running"
            )
        return loop.run_until_complete(awaitable)

    def _index_via_grpc(
        self,
        *,
        document_id: str,
        text: str,
        collect_telemetry: bool,
    ) -> IndexingResult:
        if self._client_factory is None or self._grpc_address is None:
            raise RuntimeError("IndexingService is not configured for gRPC access")
        with self._client_factory(self._grpc_address) as client:
            result = client.index_document(
                document_id=document_id,
                content=text.encode("utf-8"),
                collect_telemetry=collect_telemetry,
            )
        return result

    def index_document(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Index (rebuild) a document by delegating to the gRPC server."""

        if not text:
            raise ValueError("text must be non-empty")

        resolved_id = document_id or ""
        if not resolved_id:
            if file_path:
                resolved_id = Path(file_path).name
        if not resolved_id:
            raise ValueError("document_id or file_path must be provided")

        if self._index_runtime is not None:
            return self._append_via_runtime(
                document_id=resolved_id,
                text=text,
                collect_telemetry=collect_telemetry,
                replace_existing=True,
                file_path=file_path,
            )

        return self._index_via_grpc(
            document_id=resolved_id,
            text=text,
            collect_telemetry=collect_telemetry,
        )

    # jscpd:ignore-start - Legitimate sync wrapper pattern
    def index_from_file(
        self,
        file_path: str,
        document_id: str | None = None,
        show_progress: bool = True,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Index a document from file.

        Args:
            file_path: Path to file to index
            document_id: Optional document ID (defaults to filename)
            show_progress: Whether to show progress bar
            collect_telemetry: Whether to collect telemetry data

        Returns:
            IndexingResult with document stats and optional telemetry

        Raises:
            OSError: If file cannot be read
        """
        # Read file
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")

        # Use filename as document ID if not provided
        if not document_id:
            document_id = path.name

        return self.index_document(
            text,
            document_id=document_id,
            file_path=str(path.absolute()),
            show_progress=show_progress,
            collect_telemetry=collect_telemetry,
        )

    # jscpd:ignore-end

    async def index_document_async(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = False,  # Default False for async
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Index a document asynchronously.

        Args:
            text: Document text to index
            document_id: Optional document ID
            file_path: Optional file path for metadata
            show_progress: Whether to show progress bar
            collect_telemetry: Whether to collect telemetry data

        Returns:
            IndexingResult with document stats and optional telemetry
        """
        if self._index_runtime is not None:
            resolved_id = document_id or ""
            if not resolved_id and file_path:
                resolved_id = Path(file_path).name
            if not resolved_id:
                raise ValueError("document_id or file_path must be provided")
            session = self._index_runtime.get_session(resolved_id, file_path=file_path)
            return await session.append_text(
                text,
                replace_existing=True,
                collect_telemetry=collect_telemetry,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.index_document(
                text,
                document_id=document_id,
                file_path=file_path,
                show_progress=show_progress,
                collect_telemetry=collect_telemetry,
            ),
        )

    def append_to_document(
        self,
        document_id: str,
        new_text: str,
        show_progress: bool = False,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Append new text to a document via gRPC."""

        if not document_id:
            raise ValueError("document_id is required for append")

        # Compatibility shim – progress display is handled client-side for gRPC workflows.
        _ = show_progress

        if self._index_runtime is not None:
            return self._append_via_runtime(
                document_id=document_id,
                text=new_text,
                collect_telemetry=collect_telemetry,
                replace_existing=False,
            )

        return self._append_via_grpc(
            document_id=document_id,
            text=new_text,
            collect_telemetry=collect_telemetry,
            replace_existing=False,
        )

    async def append_to_document_async(
        self,
        document_id: str,
        new_text: str,
        show_progress: bool = False,
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Async wrapper for append_to_document."""

        if self._index_runtime is not None:
            session = self._index_runtime.get_session(document_id)
            return await session.append_text(
                new_text,
                replace_existing=False,
                collect_telemetry=collect_telemetry,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.append_to_document(
                document_id=document_id,
                new_text=new_text,
                show_progress=show_progress,
                collect_telemetry=collect_telemetry,
            ),
        )
