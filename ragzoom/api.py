"""FastAPI routes for RagZoom REST interface."""

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager
from threading import Lock

from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import Store

logger = logging.getLogger(__name__)

# Thread-safe service creation - new instance per request
class RagZoomService:
    """Service container for RagZoom components."""
    
    def __init__(self):
        self.config = RagZoomConfig()
        self.store = Store(self.config)
        # Each service gets its own OpenAI client to avoid thread issues
        self.tree_builder = TreeBuilder(self.config, self.store)
        self.retriever = Retriever(self.config, self.store)
        self.assembler = Assembler(self.config, self.store)

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


# Request/Response models
class IndexDocumentRequest(BaseModel):
    """Request to index a new document."""
    text: str = Field(..., description="Document text to index")
    document_id: Optional[str] = Field(None, description="Optional document ID")


class IndexDocumentResponse(BaseModel):
    """Response from document indexing."""
    document_id: str
    chunks_created: int
    tree_depth: int


class QueryRequest(BaseModel):
    """Request to query the system."""
    query: str = Field(..., description="Query text")
    n_max: Optional[int] = Field(None, description="Override max nodes to retrieve")
    token_budget: Optional[int] = Field(None, description="Override token budget")
    use_eviction: bool = Field(False, description="Use sliding queue eviction")


class QueryResponse(BaseModel):
    """Response from query."""
    summary: str
    token_count: int
    nodes_retrieved: int
    frontier_size: int


class PinNodeRequest(BaseModel):
    """Request to pin a node."""
    node_id: str = Field(..., description="Node ID to pin")


class UpdateConfigRequest(BaseModel):
    """Request to update configuration."""
    budget_tokens: Optional[int] = None
    leaf_tokens: Optional[int] = None
    mmr_lambda: Optional[float] = None
    slope_cap: Optional[bool] = None
    smoothing_pass_enabled: Optional[bool] = None
    ttl_turns: Optional[int] = None
    freshness_decay: Optional[float] = None


class SystemStatusResponse(BaseModel):
    """System status information."""
    total_nodes: int
    leaf_nodes: int
    tree_depth: int
    pinned_nodes: int
    config: dict


# Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RagZoom API", "version": "0.1.0"}


@app.post("/index", response_model=IndexDocumentResponse)
async def index_document(
    request: IndexDocumentRequest,
    service: RagZoomService = Depends(get_ragzoom_service)
):
    """Index a new document."""
    try:
        # Add document to tree
        document_id = service.tree_builder.add_document(request.text, request.document_id)
        
        # Get stats
        leaf_nodes = service.store.get_leaf_nodes()
        root = service.store.get_root_node()
        tree_depth = root.depth if root else 0
        
        return IndexDocumentResponse(
            document_id=document_id,
            chunks_created=len(leaf_nodes),
            tree_depth=tree_depth,
        )
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    service: RagZoomService = Depends(get_ragzoom_service)
):
    """Query the system."""
    try:
        # Retrieve with or without eviction
        if request.use_eviction:
            retrieval_result = service.retriever.retrieve_with_eviction(
                request.query, request.token_budget
            )
        else:
            retrieval_result = service.retriever.retrieve(request.query, request.n_max)
        
        # Assemble summary
        summary, token_count = service.assembler.assemble_with_budget(
            retrieval_result, request.token_budget
        )
        
        return QueryResponse(
            summary=summary,
            token_count=token_count,
            nodes_retrieved=len(retrieval_result.node_ids),
            frontier_size=len(retrieval_result.frontier_nodes),
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pin")
async def pin_node(
    request: PinNodeRequest,
    service: RagZoomService = Depends(get_ragzoom_service)
):
    """Pin a node."""
    try:
        success = service.store.pin_node(request.node_id)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot pin node {request.node_id} (doesn't exist or too deep)",
            )
        return {"message": "Node pinned successfully", "node_id": request.node_id}
    except Exception as e:
        logger.error(f"Error pinning node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/config")
async def update_config(
    request: UpdateConfigRequest,
    service: RagZoomService = Depends(get_ragzoom_service)
):
    """Update configuration dynamically."""
    try:
        # Update only provided fields
        if request.budget_tokens is not None:
            service.config.budget_tokens = request.budget_tokens
        if request.leaf_tokens is not None:
            service.config.leaf_tokens = request.leaf_tokens
        if request.mmr_lambda is not None:
            service.config.mmr_lambda = request.mmr_lambda
        if request.slope_cap is not None:
            service.config.slope_cap = request.slope_cap
        if request.smoothing_pass_enabled is not None:
            service.config.smoothing_pass_enabled = request.smoothing_pass_enabled
        if request.ttl_turns is not None:
            service.config.ttl_turns = request.ttl_turns
        if request.freshness_decay is not None:
            service.config.freshness_decay = request.freshness_decay
        
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=SystemStatusResponse)
async def get_status(
    service: RagZoomService = Depends(get_ragzoom_service)
):
    """Get system status."""
    try:
        # Gather stats
        all_nodes = service.store.collection.count()
        leaf_nodes = service.store.get_leaf_nodes()
        root = service.store.get_root_node()
        pinned = service.store.get_pinned_nodes()
        
        return SystemStatusResponse(
            total_nodes=all_nodes,
            leaf_nodes=len(leaf_nodes),
            tree_depth=root.depth if root else 0,
            pinned_nodes=len(pinned),
            config=service.config.model_dump(),
        )
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recompute")
async def recompute_summaries(
    service: RagZoomService = Depends(get_ragzoom_service)
):
    """Recompute summaries for dirty nodes."""
    try:
        count = service.tree_builder.recompute_dirty_summaries()
        return {
            "message": "Summaries recomputed",
            "nodes_updated": count,
        }
    except Exception as e:
        logger.error(f"Error recomputing summaries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    service = RagZoomService()
    logging.basicConfig(
        level=getattr(logging, service.config.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)