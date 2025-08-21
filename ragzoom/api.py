"""FastAPI routes for RagZoom REST interface."""

import logging
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from ragzoom.api_middleware import ErrorHandlingMiddleware
from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import StoreManager

logger = logging.getLogger(__name__)


# Thread-safe service creation - new instance per request
class RagZoomService:
    """Service container for RagZoom components."""

    def __init__(self) -> None:
        # Create separate configs
        self.index_config = IndexConfig.load()  # Load defaults
        self.query_config = QueryConfig()
        self.operational_config: OperationalConfig = (
            OperationalConfig()
        )  # Will read OPENAI_API_KEY from env

        # Initialize components with specific configs
        self.store = StoreManager(
            self.operational_config, embedding_model=self.index_config.embedding_model
        )
        self.tree_builder = TreeBuilder(
            self.index_config,
            self.store,
            api_key=self.operational_config.openai_api_key,
            max_concurrent=30,  # Default concurrency limit for API calls
        )
        self.retriever = Retriever(
            self.query_config,
            self.store,
            api_key=self.operational_config.openai_api_key,
        )
        self.assembler = Assembler(self.store)

    def close(self) -> None:
        """Close store connections and cleanup resources."""
        if hasattr(self, "store"):
            self.store.close()


# Dependency injection - creates new service per request
def get_ragzoom_service() -> RagZoomService:
    """Create a new RagZoom service instance for thread safety."""
    return RagZoomService()


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


class DocumentInfo(BaseModel):
    """Information about an indexed document."""

    document_id: str
    file_path: str | None
    indexed_at: str
    chunk_count: int
    node_count: int


class DocumentsResponse(BaseModel):
    """Response listing all indexed documents."""

    documents: list[DocumentInfo]


# Routes
@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "RagZoom API", "version": "0.1.0"}


@app.post("/index", response_model=IndexDocumentResponse)
async def index_document(
    request: IndexDocumentRequest,
    service: RagZoomService = Depends(get_ragzoom_service),
) -> IndexDocumentResponse:
    """Index a new document."""
    # Add document to tree - use async version directly since we're in an async endpoint
    document_id = await service.tree_builder.add_document_async(
        request.text,
        request.document_id,
        file_path=request.file_path,
        show_progress=False,
    )

    # Get stats for this specific document using document store
    doc_store = service.store.for_document(document_id)
    doc_leaves = doc_store.nodes.get_leaves()
    root = doc_store.tree.get_root()

    tree_height = root.height if root else 0

    return IndexDocumentResponse(
        document_id=document_id,
        chunks_created=len(doc_leaves),
        tree_depth=tree_height,
    )
    # Error handling is now done by middleware


@app.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    service: RagZoomService = Depends(get_ragzoom_service),
) -> DocumentsResponse:
    """List all indexed documents."""
    documents = []

    with service.store.SessionLocal() as session:
        from ragzoom.store import Document, TreeNode

        docs = session.query(Document).all()

        for doc in docs:
            # Get node count for this document
            node_count = session.query(TreeNode).filter_by(document_id=doc.id).count()

            documents.append(
                DocumentInfo(
                    document_id=doc.id,
                    file_path=doc.file_path,
                    indexed_at=doc.indexed_at.isoformat(),
                    chunk_count=doc.chunk_count,
                    node_count=node_count,
                )
            )

    return DocumentsResponse(documents=documents)
    # Error handling is now done by middleware


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest, service: RagZoomService = Depends(get_ragzoom_service)
) -> QueryResponse:
    """Query the system."""
    # Use async version since we're in an async endpoint
    retrieval_result = await service.retriever.retrieve_async(
        request.query,
        request.num_seeds,
        request.token_budget,
        document_id=request.document_id,
    )

    # Assemble summary
    summary = service.assembler.assemble(retrieval_result)
    token_count = service.assembler.get_token_count(summary)

    return QueryResponse(
        summary=summary,
        token_count=token_count,
        nodes_retrieved=len(retrieval_result.node_ids),
        tiling_size=(len(retrieval_result.tiling) if retrieval_result.tiling else 0),
    )
    # Error handling is now done by middleware


@app.post("/pin")
async def pin_node(
    request: PinNodeRequest, service: RagZoomService = Depends(get_ragzoom_service)
) -> dict[str, str]:
    """Pin a node."""
    service.store.pin_node(request.node_id)
    return {"message": "Node pinned successfully", "node_id": request.node_id}
    # Specific exceptions are now handled by middleware
    # Error handling is now done by middleware


@app.patch("/config")
async def update_config(
    request: UpdateConfigRequest, service: RagZoomService = Depends(get_ragzoom_service)
) -> dict[str, str]:
    """Update configuration dynamically."""
    # Update query config fields
    query_updates: dict[str, int | float] = {}
    if request.budget_tokens is not None:
        query_updates["budget_tokens"] = request.budget_tokens
    if request.mmr_lambda is not None:
        query_updates["mmr_lambda"] = request.mmr_lambda

    if query_updates:
        service.query_config = service.query_config.replace(**query_updates)
        # Recreate retriever with new config
        service.retriever = Retriever(
            service.query_config,
            service.store,
            api_key=service.operational_config.openai_api_key,
        )

    # Update index config fields
    index_updates = {}
    if request.leaf_tokens is not None:
        index_updates["target_chunk_tokens"] = request.leaf_tokens

    if index_updates:
        service.index_config = service.index_config.replace(**index_updates)
        # Recreate components that use index config
        service.tree_builder = TreeBuilder(
            service.index_config,
            service.store,
            api_key=service.operational_config.openai_api_key,
            max_concurrent=30,  # Default concurrency limit for API calls
        )
        service.retriever = Retriever(
            service.query_config,
            service.store,
            api_key=service.operational_config.openai_api_key,
        )

    # Note: slope_cap and smoothing_pass_enabled were removed as they're not in any config

    return {"message": "Configuration updated successfully"}
    # Error handling is now done by middleware


@app.get("/status", response_model=SystemStatusResponse)
async def get_status(
    service: RagZoomService = Depends(get_ragzoom_service),
) -> SystemStatusResponse:
    """Get system status."""
    # Gather stats
    with service.store.SessionLocal() as session:
        from ragzoom.store import TreeNode

        all_nodes = session.query(TreeNode).count()
        # TODO: Implement system-wide stats for multi-document architecture
        # For now, return basic stats
        pinned = service.store.get_pinned_nodes()

        return SystemStatusResponse(
            total_nodes=all_nodes,
            leaf_nodes=0,  # TODO: Aggregate across all documents
            tree_depth=0,  # TODO: Maximum depth across all documents
            pinned_nodes=len(pinned),
            config={
                "index": asdict(service.index_config),
                "query": asdict(service.query_config),
                "operational": {
                    k: v
                    for k, v in asdict(service.operational_config).items()
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
    service = RagZoomService()
    logging.basicConfig(
        level=getattr(logging, service.operational_config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    uvicorn.run(
        app, host="127.0.0.1", port=8000
    )  # nosec B104 - bind to localhost only for security
