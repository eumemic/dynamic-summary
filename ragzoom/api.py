"""FastAPI routes for RagZoom REST interface."""

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from typing_extensions import TypedDict

from ragzoom.api_middleware import create_error_handling_middleware
from ragzoom.client.grpc_client import GrpcRagzoomClient
from ragzoom.config import IndexConfig, IndexConfigDict, OperationalConfig, QueryConfig
from ragzoom.constants import DEFAULT_GRPC_ADDRESS
from ragzoom.query_log import QueryLog
from ragzoom.server.state import _resolve_query_log_path, _resolve_telemetry_dir
from ragzoom.services.document_service import DocumentInfo, DocumentService
from ragzoom.services.query_service import QueryService
from ragzoom.store import create_store_with_docker


def _resolve_grpc_address(configured: str | None = None) -> str:
    if configured:
        return configured
    env_value = os.environ.get("RAGZOOM_SERVER_ADDRESS")
    if env_value:
        return env_value
    return DEFAULT_GRPC_ADDRESS


_TELEMETRY_HEARTBEAT_SECONDS = 10.0
_STREAM_EVENT_TYPES = {
    "append_started",
    "append_completed",
    "append_failed",
    "node_committed",
    "nodes_deleted",
}


def _sanitize_document_id(document_id: str) -> str:
    """Map document identifiers to filesystem-safe paths (mirrors telemetry log)."""
    sanitized = re.sub(r"[^0-9A-Za-z._-]", "_", document_id)
    return sanitized or "document"


def _events_path_for_document(base_dir: Path, document_id: str) -> Path:
    return base_dir / _sanitize_document_id(document_id) / "telemetry.events.jsonl"


def _read_new_telemetry_lines(path: Path, offset: int) -> tuple[list[str], int]:
    if not path.exists():
        return [], offset

    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return [], offset

    if size < offset:
        offset = 0

    if size == offset:
        return [], offset

    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        lines = handle.readlines()
        offset = handle.tell()
    return lines, offset


async def _document_event_stream(
    document_id: str, telemetry_dir: Path
) -> AsyncGenerator[str, None]:
    """Async generator emitting SSE payloads for document telemetry events."""
    events_path = _events_path_for_document(telemetry_dir, document_id)
    offset = 0
    last_emit = time.monotonic()

    try:
        while True:
            if not events_path.exists():
                now = time.monotonic()
                if now - last_emit >= _TELEMETRY_HEARTBEAT_SECONDS:
                    yield ": keep-alive\n\n"
                    last_emit = now
                await asyncio.sleep(1.0)
                continue

            lines, offset = await asyncio.to_thread(
                _read_new_telemetry_lines,
                events_path,
                offset,
            )

            emitted = False
            for line in lines:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_type = payload.get("event")
                if event_type not in _STREAM_EVENT_TYPES:
                    continue
                data = json.dumps(payload, separators=(",", ":"))
                yield f"data: {data}\n\n"
                last_emit = time.monotonic()
                emitted = True

            if not emitted:
                now = time.monotonic()
                if now - last_emit >= _TELEMETRY_HEARTBEAT_SECONDS:
                    yield ": keep-alive\n\n"
                    last_emit = now
                await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        return


async def _documents_event_stream(
    services: "ServiceContainer",
) -> AsyncGenerator[str, None]:
    """Async generator emitting SSE payloads when the document catalog changes."""
    document_service = services.document_service
    last_emit = time.monotonic()
    known_documents: dict[str, DocumentInfo] = {}

    try:
        while True:
            documents = await asyncio.to_thread(document_service.list_documents)
            document_map = {doc.document_id: doc for doc in documents}

            added_ids = sorted(set(document_map) - set(known_documents))
            removed_ids = sorted(set(known_documents) - set(document_map))

            metadata_changed = False
            if not added_ids and not removed_ids:
                for doc_id, doc in document_map.items():
                    previous = known_documents.get(doc_id)
                    if previous is None:
                        continue
                    if (
                        previous.node_count != doc.node_count
                        or previous.file_path != doc.file_path
                        or previous.indexed_at != doc.indexed_at
                    ):
                        metadata_changed = True
                        break

            if added_ids or removed_ids or metadata_changed or not known_documents:
                payload = {
                    "event": "documents_changed",
                    "documents": [
                        DocumentInfoResponse.from_domain(doc).model_dump()
                        for doc in documents
                    ],
                    "added_ids": added_ids,
                    "removed_ids": removed_ids,
                }
                data = json.dumps(payload, separators=(",", ":"))
                yield f"data: {data}\n\n"
                known_documents = document_map
                last_emit = time.monotonic()
            else:
                now = time.monotonic()
                if now - last_emit >= _TELEMETRY_HEARTBEAT_SECONDS:
                    yield ": keep-alive\n\n"
                    last_emit = now

            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        return


logger = logging.getLogger(__name__)


# Service container for dependency injection
class ServiceContainer:
    """Container for RagZoom services.

    Document Isolation Architecture:
    - Services are initialized with the multi-document Store
    - Each service internally creates DocumentStore instances as needed
    - This ensures document isolation is enforced at the service layer
    - QueryService handles document scoping for retrieval operations
    - This pattern prevents cross-document contamination through the type system
    """

    def __init__(self) -> None:
        # Create configurations
        self.index_config = IndexConfig.load()
        self.query_config = QueryConfig()
        self.operational_config = OperationalConfig()

        # Initialize multi-document store
        # Services will create document-scoped stores internally as needed
        self.store = create_store_with_docker(
            self.operational_config, embedding_model=self.index_config.embedding_model
        )

        self.grpc_address = _resolve_grpc_address()
        self.query_log = QueryLog(_resolve_query_log_path(self.operational_config))

        # Initialize services with the multi-document store
        # Each service handles document isolation internally:
        # - DocumentService: manages document metadata across all documents
        # - QueryService: creates DocumentStore for each query operation
        # Indexing operations are proxied to the gRPC server via GrpcRagzoomClient
        self.document_service = DocumentService(self.store)
        self.query_service = QueryService(
            self.store, self.query_config, self.operational_config, self.query_log
        )

    def close(self) -> None:
        """Close store connections and cleanup resources."""
        if hasattr(self, "store"):
            self.store.close()


# Dependency injection - creates new service container per request
def get_service_container() -> ServiceContainer:
    """Create a new service container for thread safety."""
    return ServiceContainer()


# Create FastAPI app
app = FastAPI(
    title="RagZoom API",
    description="Incremental, hierarchical RAG memory system",
    version="0.1.0",
)


def _allowed_cors_origins() -> list[str]:
    configured = os.environ.get("RAGZOOM_CORS_ORIGINS")
    if configured:
        origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
        if origins:
            return origins
    # Development default to unblock local inspector
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:55300",
        "http://127.0.0.1:55300",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# Add error handling middleware
app.add_middleware(create_error_handling_middleware(include_traceback=False))


# Request/Response models
# Pydantic BaseModel inherits from type containing Any, required for serialization
class IndexDocumentRequest(BaseModel):  # type: ignore[explicit-any]
    """Request to append text to a document."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Document text to index")
    document_id: str | None = Field(None, description="Optional document ID")
    file_path: str | None = Field(
        None, description="Optional file path (used for default document ID)"
    )


# Pydantic BaseModel inherits from type containing Any, required for serialization
class IndexDocumentResponse(BaseModel):  # type: ignore[explicit-any]
    """Response from document append."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunks_created: int
    tree_depth: int


# Pydantic BaseModel inherits from type containing Any, required for serialization
class QueryRequest(BaseModel):  # type: ignore[explicit-any]
    """Request to query the system."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Query text")
    document_id: str = Field(description="Document ID to query within")
    num_seeds: int | None = Field(None, description="Override max nodes to retrieve")
    token_budget: int | None = Field(None, description="Override token budget")


# Pydantic BaseModel inherits from type containing Any, required for serialization
class QueryResponse(BaseModel):  # type: ignore[explicit-any]
    """Response from query."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    token_count: int
    nodes_retrieved: int
    tiling_size: int
    query_id: str


class QueryListItem(BaseModel):  # type: ignore[explicit-any]
    """Summary information for a logged query."""

    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    query_text: str
    budget_tokens: int | None
    num_seeds: int | None
    created_at: str


class DocumentQueriesResponse(BaseModel):  # type: ignore[explicit-any]
    """List of queries for a document."""

    model_config = ConfigDict(extra="forbid")

    queries: list[QueryListItem]


class QueryNodeEntry(BaseModel):  # type: ignore[explicit-any]
    """Logged node entry for a query tiling."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    score: float
    is_seed: bool
    position: int


class QueryDetailResponse(BaseModel):  # type: ignore[explicit-any]
    """Detailed logged query payload."""

    model_config = ConfigDict(extra="forbid")

    query: QueryListItem
    nodes: list[QueryNodeEntry]


# Pydantic BaseModel inherits from type containing Any, required for serialization
class PinNodeRequest(BaseModel):  # type: ignore[explicit-any]
    """Request to pin a node."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(description="Node ID to pin")


# Pydantic BaseModel inherits from type containing Any, required for serialization
class UpdateConfigRequest(BaseModel):  # type: ignore[explicit-any]
    """Request to update configuration."""

    model_config = ConfigDict(extra="forbid")

    budget_tokens: int | None = None
    leaf_tokens: int | None = None
    mmr_lambda: float | None = None
    # Deprecated fields removed - ttl_turns and freshness_decay no longer exist


# IndexConfigDict imported from config.py to avoid duplication


class QueryConfigDict(TypedDict):
    """Type definition for query configuration dictionary."""

    budget_tokens: int
    mmr_lambda: float
    mmr_k_multiplier: float
    embedding_model: str


class OperationalConfigDict(TypedDict):
    """Type definition for operational configuration dictionary (filtered)."""

    database_url: str
    cache_size: int
    log_level: str
    validate_pipeline: bool
    # Note: openai_api_key is deliberately excluded from API responses


class SystemConfigDict(TypedDict):
    """Type definition for complete system configuration dictionary."""

    index: IndexConfigDict
    query: QueryConfigDict
    operational: OperationalConfigDict


# Pydantic BaseModel inherits from type containing Any, required for serialization
class SystemStatusResponse(BaseModel):  # type: ignore[explicit-any]
    """System status information."""

    model_config = ConfigDict(extra="forbid")

    total_nodes: int
    leaf_nodes: int
    tree_depth: int
    pinned_nodes: int
    config: SystemConfigDict


# Pydantic BaseModel inherits from type containing Any, required for serialization
class DocumentInfoResponse(BaseModel):  # type: ignore[explicit-any]
    """Information about an indexed document for API response."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    file_path: str | None
    indexed_at: str
    node_count: int

    @classmethod
    def from_domain(cls, doc_info: DocumentInfo) -> "DocumentInfoResponse":
        """Create from domain DocumentInfo object."""
        return cls(
            document_id=doc_info.document_id,
            file_path=doc_info.file_path,
            indexed_at=doc_info.indexed_at.isoformat(),
            node_count=doc_info.node_count,
        )


# Pydantic BaseModel inherits from type containing Any, required for serialization
class DocumentsResponse(BaseModel):  # type: ignore[explicit-any]
    """Response listing all indexed documents."""

    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentInfoResponse]


# jscpd:ignore-start - NodeResponse schema mirrors the service snapshot dataclass
class NodeResponse(BaseModel):  # type: ignore[explicit-any]
    """Serialized node."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    document_id: str | None
    parent_id: str | None
    left_child_id: str | None
    right_child_id: str | None
    span_start: int
    span_end: int
    text: str
    token_count: int
    height: int
    level_index: int
    preceding_neighbor_id: str | None
    following_neighbor_id: str | None
    is_pinned: bool
    created_at: datetime | None


# jscpd:ignore-end


class NodesPageResponse(BaseModel):  # type: ignore[explicit-any]
    """Response for span-based node queries."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[NodeResponse]
    total_matching: int


class NodeBatchRequest(BaseModel):  # type: ignore[explicit-any]
    """Request body for batch node retrieval."""

    model_config = ConfigDict(extra="forbid")

    node_ids: list[str]


class NodeBatchResponse(BaseModel):  # type: ignore[explicit-any]
    """Response body for batch node retrieval."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[NodeResponse]


class ClearDocumentRequest(BaseModel):  # type: ignore[explicit-any]
    """Request to clear a document and delete its nodes."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(description="Document ID to clear")


class ClearDocumentResponse(BaseModel):  # type: ignore[explicit-any]
    """Response confirming document clearing."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    deleted_nodes: int
    document_existed: bool


# Routes
@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "RagZoom API", "version": "0.1.0"}


@app.post("/index", response_model=IndexDocumentResponse)
async def index_document(
    request: IndexDocumentRequest,
    services: ServiceContainer = Depends(get_service_container),
) -> IndexDocumentResponse:
    """Append text to a document via the gRPC server."""

    text = request.text or ""
    if not text:
        raise HTTPException(status_code=400, detail="`text` must be non-empty")

    resolved_document_id = request.document_id or ""
    if not resolved_document_id:
        if request.file_path:
            resolved_document_id = Path(request.file_path).name
    if not resolved_document_id:
        raise HTTPException(
            status_code=400,
            detail="`document_id` is required when `file_path` is not provided",
        )

    address = services.grpc_address
    try:
        with GrpcRagzoomClient(address) as client:
            result = client.append_text(
                document_id=resolved_document_id,
                content=text.encode("utf-8"),
                collect_telemetry=False,
                replace_existing=False,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return IndexDocumentResponse(
        document_id=result.document_id,
        chunks_created=result.chunks_created,
        tree_depth=result.tree_depth,
    )


@app.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    services: ServiceContainer = Depends(get_service_container),
) -> DocumentsResponse:
    """List all indexed documents."""
    doc_infos = services.document_service.list_documents()
    documents = [DocumentInfoResponse.from_domain(doc) for doc in doc_infos]
    return DocumentsResponse(documents=documents)
    # Error handling is now done by middleware


@app.get("/documents/events")
async def stream_documents_events(
    services: "ServiceContainer" = Depends(get_service_container),
) -> StreamingResponse:
    """Server-Sent Events stream for document catalog updates."""

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in _documents_event_stream(services):
                yield chunk
        finally:
            services.close()

    headers = {"Cache-Control": "no-cache"}
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


@app.get(
    "/documents/{document_id}/nodes",
    response_model=NodesPageResponse,
)
async def get_document_nodes(
    document_id: str,
    span_start: int = Query(..., ge=0),
    span_end: int = Query(..., gt=0),
    limit: int = Query(200, gt=0, le=2000),
    min_height: int | None = Query(None, ge=0),
    services: ServiceContainer = Depends(get_service_container),
) -> NodesPageResponse:
    """Return nodes within the requested span ordered from top to bottom."""
    try:
        result = services.document_service.get_nodes_in_span(
            document_id,
            span_start,
            span_end,
            limit=limit,
            min_height=min_height,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    nodes = [NodeResponse(**asdict(snapshot)) for snapshot in result.nodes]
    return NodesPageResponse(nodes=nodes, total_matching=result.total_matching)


@app.post(
    "/documents/{document_id}/nodes/batch",
    response_model=NodeBatchResponse,
)
async def get_nodes_batch(
    document_id: str,
    request: NodeBatchRequest,
    services: ServiceContainer = Depends(get_service_container),
) -> NodeBatchResponse:
    """Return details for a set of node IDs."""
    try:
        nodes = services.document_service.get_nodes_by_ids(
            document_id,
            request.node_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    serialized = [NodeResponse(**asdict(snapshot)) for snapshot in nodes]
    return NodeBatchResponse(nodes=serialized)


@app.get(
    "/documents/{document_id}/queries",
    response_model=DocumentQueriesResponse,
)
async def list_document_queries(
    document_id: str,
    limit: int = Query(50, gt=0, le=500),
    services: ServiceContainer = Depends(get_service_container),
) -> DocumentQueriesResponse:
    """List recent queries for a document (most recent first)."""
    summaries = services.query_log.list_queries(document_id, limit)
    items = [
        QueryListItem(
            id=summary.id,
            document_id=summary.document_id,
            query_text=summary.query_text,
            budget_tokens=summary.budget_tokens,
            num_seeds=summary.num_seeds,
            created_at=summary.created_at,
        )
        for summary in summaries
    ]
    return DocumentQueriesResponse(queries=items)


@app.get(
    "/queries/{query_id}",
    response_model=QueryDetailResponse,
)
async def get_query_detail(
    query_id: str,
    services: ServiceContainer = Depends(get_service_container),
) -> QueryDetailResponse:
    """Return a logged query with tiling nodes."""
    detail = services.query_log.get_query(query_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Query not found")

    query_item = QueryListItem(
        id=detail.id,
        document_id=detail.document_id,
        query_text=detail.query_text,
        budget_tokens=detail.budget_tokens,
        num_seeds=detail.num_seeds,
        created_at=detail.created_at,
    )
    nodes = [
        QueryNodeEntry(
            node_id=row.node_id,
            score=row.score,
            is_seed=row.is_seed,
            position=row.position,
        )
        for row in detail.nodes
    ]
    return QueryDetailResponse(query=query_item, nodes=nodes)


async def _query_event_stream(
    services: ServiceContainer,
    document_id: str,
    limit: int,
) -> AsyncGenerator[str, None]:
    """Async generator emitting SSE payloads for query list updates."""

    known_ids: list[str] = []
    last_emit = time.monotonic()
    try:
        while True:
            summaries = services.query_log.list_queries(document_id, limit)
            current_ids = [summary.id for summary in summaries]
            if current_ids != known_ids:
                items = [
                    QueryListItem(
                        id=summary.id,
                        document_id=summary.document_id,
                        query_text=summary.query_text,
                        budget_tokens=summary.budget_tokens,
                        num_seeds=summary.num_seeds,
                        created_at=summary.created_at,
                    ).model_dump()
                    for summary in summaries
                ]
                payload = {
                    "event": "queries_changed",
                    "queries": items,
                }
                data = json.dumps(payload, separators=(",", ":"))
                yield f"data: {data}\n\n"
                known_ids = current_ids
                last_emit = time.monotonic()
            else:
                now = time.monotonic()
                if now - last_emit >= _TELEMETRY_HEARTBEAT_SECONDS:
                    yield ": keep-alive\n\n"
                    last_emit = now
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        return
    finally:
        services.close()


@app.get("/documents/{document_id}/queries/events")
async def stream_document_query_events(
    document_id: str,
    limit: int = Query(50, gt=0, le=500),
    services: ServiceContainer = Depends(get_service_container),
) -> StreamingResponse:
    """Server-Sent Events stream for query history updates."""

    headers = {"Cache-Control": "no-cache"}

    async def event_generator() -> AsyncGenerator[str, None]:
        async for chunk in _query_event_stream(services, document_id, limit):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


@app.get("/documents/{document_id}/events")
async def stream_document_events(document_id: str) -> StreamingResponse:
    """Server-Sent Events stream for document telemetry."""
    telemetry_dir = _resolve_telemetry_dir(OperationalConfig(), None)
    if not telemetry_dir.exists():
        raise HTTPException(
            status_code=503,
            detail="Telemetry logging is not enabled on the server.",
        )

    stream = _document_event_stream(document_id, telemetry_dir)
    headers = {"Cache-Control": "no-cache"}
    return StreamingResponse(stream, media_type="text/event-stream", headers=headers)


@app.post("/clear", response_model=ClearDocumentResponse)
async def clear_document(
    request: ClearDocumentRequest,
    services: ServiceContainer = Depends(get_service_container),
) -> ClearDocumentResponse:
    """Clear a document via the gRPC worker service."""

    document_id = request.document_id.strip()
    if not document_id:
        raise HTTPException(status_code=400, detail="`document_id` must be provided")

    address = services.grpc_address
    try:
        with GrpcRagzoomClient(address) as client:
            result = client.clear_document(document_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ClearDocumentResponse(
        document_id=result.document_id,
        deleted_nodes=result.deleted_nodes,
        document_existed=result.document_existed,
    )


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest, services: ServiceContainer = Depends(get_service_container)
) -> QueryResponse:
    """Query the system."""
    result = await services.query_service.execute_query_async(
        request.query,
        request.document_id,
        num_seeds=request.num_seeds,
        token_budget=request.token_budget,
    )

    return QueryResponse(
        summary=result.summary,
        token_count=result.token_count,
        nodes_retrieved=result.nodes_retrieved,
        tiling_size=result.tiling_size,
        query_id=result.query_id,
    )
    # Error handling is now done by middleware


@app.post("/pin")
async def pin_node(
    request: PinNodeRequest, services: ServiceContainer = Depends(get_service_container)
) -> dict[str, str]:
    """Pin a node."""
    services.document_service.pin_node(request.node_id)
    return {"message": "Node pinned successfully", "node_id": request.node_id}
    # Specific exceptions are now handled by middleware
    # Error handling is now done by middleware


@app.patch("/config")
async def update_config(
    request: UpdateConfigRequest,
    services: ServiceContainer = Depends(get_service_container),
) -> dict[str, str]:
    """Update configuration dynamically."""
    # Update query service configuration
    services.query_service.update_config(
        budget_tokens=request.budget_tokens,
        mmr_lambda=request.mmr_lambda,
    )

    # Update index config if needed
    if request.leaf_tokens is not None:
        raise HTTPException(
            status_code=400,
            detail="Updating `leaf_tokens` is not supported via the REST API; configure the gRPC server instead.",
        )

    return {"message": "Configuration updated successfully"}
    # Error handling is now done by middleware


@app.get("/status", response_model=SystemStatusResponse)
async def get_status(
    services: ServiceContainer = Depends(get_service_container),
) -> SystemStatusResponse:
    """Get system status."""
    status = services.document_service.get_system_status()

    return SystemStatusResponse(
        total_nodes=status.total_nodes,
        leaf_nodes=status.leaf_nodes,
        tree_depth=status.tree_depth,
        pinned_nodes=status.pinned_nodes,
        config={
            "index": cast(IndexConfigDict, asdict(services.index_config)),
            "query": cast(QueryConfigDict, asdict(services.query_config)),
            "operational": cast(
                OperationalConfigDict,
                {
                    k: v
                    for k, v in asdict(services.operational_config).items()
                    if k != "openai_api_key"  # Don't expose API key
                },
            ),
        },
    )
    # Error handling is now done by middleware


# Health check
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    services: ServiceContainer = ServiceContainer()
    logging.basicConfig(
        level=getattr(logging, services.operational_config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    uvicorn.run(
        app, host="127.0.0.1", port=8000
    )  # nosec B104 - bind to localhost only for security
