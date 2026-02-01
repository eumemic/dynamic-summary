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
    time_start: str | None
    time_end: str | None
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
        from ragzoom.retrieval.budget_planner import BudgetPlanner
        from ragzoom.retrieval.embedding_service import EmbeddingService
        from ragzoom.retrieve import Retriever
        
        embedding_service = EmbeddingService(_state.query_config)
        budget_planner = BudgetPlanner(_state.query_config)
        retriever = Retriever(
            store=_state.store,
            embedding_service=embedding_service,
            budget_planner=budget_planner,
            query_config=_state.query_config,
        )
        
        result = retriever.query(
            document_id=document_id,
            query=query,
            budget_tokens=budget,
            time_start=time_start,
            time_end=time_end,
        )
        
        # Format response
        nodes = []
        for node_id in result.tiling_ids:
            node = result.nodes.get(node_id)
            if node and node.text:
                nodes.append(RecallNode(
                    text=node.text,
                    time_start=node.time_start,
                    time_end=node.time_end,
                    height=node.height,
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
