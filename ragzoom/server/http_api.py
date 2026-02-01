"""Simple HTTP API for RagZoom recall.

Runs alongside the gRPC server to support clients that can't use gRPC
(e.g., sandboxed containers with only curl available).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import uvicorn

if TYPE_CHECKING:
    from ragzoom.server.state import ServerState

logger = logging.getLogger(__name__)

# Global state reference (set by create_http_app)
_state: ServerState | None = None


class RecallRequest(BaseModel):
    """Request body for recall endpoint."""
    q: str | None = None
    query: str | None = None
    document_id: str
    budget: int = 2000
    start: str | None = None
    end: str | None = None
    time_start: str | None = None
    time_end: str | None = None


class RecallNode(BaseModel):
    """A single node from the recall tiling."""
    text: str
    time_start: str | float | None
    time_end: str | float | None
    height: int


class RecallResponse(BaseModel):
    """Response from recall endpoint."""
    nodes: list[RecallNode]
    query: str
    document_id: str
    budget: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str


app = FastAPI(title="RagZoom HTTP API", description="REST API for RagZoom recall queries")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/recall", response_model=RecallResponse)
async def recall_get(
    q: str | None = Query(None, description="Query text"),
    query: str | None = Query(None, description="Query text (alias for q)"),
    document_id: str = Query(..., description="Document ID to query"),
    budget: int = Query(2000, description="Token budget for results"),
    start: str | None = Query(None, description="Start time (ISO format)"),
    end: str | None = Query(None, description="End time (ISO format)"),
    time_start: str | None = Query(None, description="Start time (alias)"),
    time_end: str | None = Query(None, description="End time (alias)"),
):
    """Execute a recall query via GET request."""
    query_text = q or query
    if not query_text:
        raise HTTPException(status_code=400, detail="Missing 'q' or 'query' parameter")
    
    return await _execute_recall(
        query=query_text,
        document_id=document_id,
        budget=budget,
        time_start=start or time_start,
        time_end=end or time_end,
    )


@app.post("/recall", response_model=RecallResponse)
async def recall_post(request: RecallRequest):
    """Execute a recall query via POST request."""
    query_text = request.q or request.query
    if not query_text:
        raise HTTPException(status_code=400, detail="Missing 'q' or 'query' in request body")
    
    return await _execute_recall(
        query=query_text,
        document_id=request.document_id,
        budget=request.budget,
        time_start=request.start or request.time_start,
        time_end=request.end or request.time_end,
    )


async def _execute_recall(
    query: str,
    document_id: str,
    budget: int,
    time_start: str | None,
    time_end: str | None,
) -> RecallResponse:
    """Execute a recall query using the server state."""
    global _state
    if _state is None:
        raise HTTPException(status_code=500, detail="Server state not initialized")
    
    try:
        from openai import OpenAI
        from ragzoom.retrieval.budget_planner import BudgetPlanner
        from ragzoom.retrieval.embedding_service import EmbeddingService
        from ragzoom.retrieve import Retriever
        from ragzoom.vector_factory import create_vector_index
        
        # Build retriever the same way as the gRPC servicer
        resolved_embedding = _state.query_config.embedding_model
        document_store = _state.store.for_document(document_id)
        client = OpenAI(
            api_key=_state.operational_config.openai_api_key.get_secret_value(),
            timeout=_state.operational_config.openai_timeout,
        )
        embedding_service = EmbeddingService(client, document_store, resolved_embedding)
        
        chunk_tokens = (
            _state.index_config.target_chunk_tokens
            if _state.index_config.target_chunk_tokens is not None
            else _state.index_config.target_embedding_tokens
        )
        budget_planner = BudgetPlanner(document_store, chunk_tokens)
        vector_index = create_vector_index(
            _state.operational_config.vector_backend,
            _state.operational_config.database_url,
            resolved_embedding,
        )
        
        retriever = Retriever(
            _state.query_config,
            document_store,
            embedding_service,
            budget_planner,
            vector_index,
        )
        
        result = await retriever.retrieve_async(
            query=query,
            budget_tokens=budget,
            time_start=time_start,
            time_end=time_end,
        )
        
        # Format response
        nodes = []
        if result.tiling and result.nodes:
            for node_id in result.tiling:
                node = result.nodes.get(node_id)
                if node and node.text:
                    nodes.append(RecallNode(
                        text=node.text,
                        time_start=getattr(node, 'time_start', None),
                        time_end=getattr(node, 'time_end', None),
                        height=getattr(node, 'height', 0),
                    ))
        
        return RecallResponse(
            nodes=nodes,
            query=query,
            document_id=document_id,
            budget=budget,
        )
        
    except Exception as e:
        logger.exception("Recall query failed")
        raise HTTPException(status_code=500, detail=str(e))


class UvicornServer:
    """Wrapper to run uvicorn in the background."""
    
    def __init__(self, host: str, port: int):
        self.config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(self.config)
        self._task: asyncio.Task | None = None
    
    async def start(self):
        """Start the server in the background."""
        self._task = asyncio.create_task(self.server.serve())
    
    async def stop(self):
        """Stop the server."""
        if self.server:
            self.server.should_exit = True
            if self._task:
                await self._task


async def start_http_server(
    state: "ServerState",
    host: str = "127.0.0.1",
    port: int = 50053,
) -> UvicornServer:
    """Start the HTTP server.
    
    Returns the server so caller can clean up on shutdown.
    """
    global _state
    _state = state
    
    server = UvicornServer(host, port)
    await server.start()
    
    logger.info("Started RagZoom HTTP API on http://%s:%d", host, port)
    return server
