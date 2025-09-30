"""Indexing service for RagZoom document processing."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ragzoom.config import IndexConfig, OperationalConfig
from ragzoom.constants import DEFAULT_GRPC_ADDRESS
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.telemetry_types import TelemetryDataDict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ragzoom.client.grpc_client import GrpcRagzoomClient

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result from document indexing operation."""

    document_id: str
    chunks_created: int
    tree_depth: int
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
    ):
        self.store = store
        self.index_config = index_config
        self.operational_config = operational_config
        self._grpc_address = self._resolve_address(grpc_address)
        self._client_factory = client_factory or self._default_client_factory

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

        with self._client_factory(self._grpc_address) as client:
            result = client.index_document(
                document_id=resolved_id,
                content=text.encode("utf-8"),
                collect_telemetry=collect_telemetry,
            )
        return result

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
