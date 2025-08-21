"""FastAPI routes for RagZoom REST interface."""

import logging
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragzoom.api_middleware import ErrorHandlingMiddleware
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.exceptions import InvalidOperationError, NodeNotFoundError
from ragzoom.services.document_service import DocumentInfo, DocumentService
from ragzoom.services.indexing_service import IndexingService
from ragzoom.services.query_service import QueryService
from ragzoom.store import Store

logger = logging.getLogger(__name__)


# Service container for dependency injection
class ServiceContainer:
    """Container for RagZoom services."""

    def __init__(self) -> None:
        # Create configurations
        self.index_config = IndexConfig.load()
        self.query_config = QueryConfig()
        self.operational_config = OperationalConfig()

        # Initialize store  
        self.store = Store(
            self.operational_config, embedding_model=self.index_config.embedding_model
        )

        # Initialize services
        self.document_service = DocumentService(self.store)
        self.indexing_service = IndexingService(
            self.store, self.index_config, self.operational_config
        )
        self.query_service = QueryService(
            self.store, self.query_config, self.operational_config
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

# Add error handling middleware
app.add_middleware(ErrorHandlingMiddleware, include_traceback=False)


# Request/Response models
class IndexDocumentRequest(BaseModel):
    """Request to index a new document."""

    text: str = Field(..., description="Document text to index")
    document_id: str | None = Field(None, description="Optional document ID")
    file_path: str | None = Field(
        None, description="Optional file path (used for default document ID)"
    )


class IndexDocumentResponse(BaseModel):
    """Response from document indexing."""

    document_id: str
    chunks_created: int
    tree_depth: int


class QueryRequest(BaseModel):
    """Request to query the system."""

    query: str = Field(..., description="Query text")
    document_id: str = Field(..., description="Document ID to query within")
    num_seeds: int | None = Field(None, description="Override max nodes to retrieve")
    token_budget: int | None = Field(None, description="Override token budget")


class QueryResponse(BaseModel):
    """Response from query."""

    summary: str
    token_count: int
    nodes_retrieved: int
    tiling_size: int


class PinNodeRequest(BaseModel):
    """Request to pin a node."""

    node_id: str = Field(..., description="Node ID to pin")


class UpdateConfigRequest(BaseModel):
    """Request to update configuration."""

    budget_tokens: int | None = None
    leaf_tokens: int | None = None
    mmr_lambda: float | None = None
    # Deprecated fields removed - ttl_turns and freshness_decay no longer exist


class SystemStatusResponse(BaseModel):
    """System status information."""

    total_nodes: int
    leaf_nodes: int
    tree_depth: int
    pinned_nodes: int
    config: dict[str, Any]


class DocumentInfoResponse(BaseModel):
    """Information about an indexed document for API response."""

    document_id: str
    file_path: str | None
    indexed_at: str
    chunk_count: int
    node_count: int

    @classmethod
    def from_domain(cls, doc_info: DocumentInfo) -> "DocumentInfoResponse":
        """Create from domain DocumentInfo object."""
        return cls(
            document_id=doc_info.document_id,
            file_path=doc_info.file_path,
            indexed_at=doc_info.indexed_at.isoformat(),
            chunk_count=doc_info.chunk_count,
            node_count=doc_info.node_count,
        )


class DocumentsResponse(BaseModel):
    """Response listing all indexed documents."""

    documents: list[DocumentInfoResponse]


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
    """Index a new document."""
    # Use service layer - error handling is done by middleware
    result = await services.indexing_service.index_document_async(
        request.text,
        document_id=request.document_id,
        file_path=request.file_path,
        show_progress=False,
    )

    return IndexDocumentResponse(
        document_id=result.document_id,
        chunks_created=result.chunks_created,
        tree_depth=result.tree_depth,
    )
    # Error handling is now done by middleware


@app.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    services: ServiceContainer = Depends(get_service_container),
) -> DocumentsResponse:
    """List all indexed documents."""
    doc_infos = services.document_service.list_documents()
    documents = [DocumentInfoResponse.from_domain(doc) for doc in doc_infos]
    return DocumentsResponse(documents=documents)
    # Error handling is now done by middleware


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
        services.index_config = services.index_config.replace(
            target_chunk_tokens=request.leaf_tokens
        )
        # Recreate indexing service with new config
        services.indexing_service = IndexingService(
            services.store, services.index_config, services.operational_config
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
            "index": asdict(services.index_config),
            "query": asdict(services.query_config),
            "operational": {
                k: v
                for k, v in asdict(services.operational_config).items()
                if k != "openai_api_key"  # Don't expose API key
            },
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
    services = ServiceContainer()
    logging.basicConfig(
        level=getattr(logging, services.operational_config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    uvicorn.run(
        app, host="127.0.0.1", port=8000
    )  # nosec B104 - bind to localhost only for security
