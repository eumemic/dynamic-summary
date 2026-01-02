"""MCP server exposing the 'remember' tool for querying pre-compaction history."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from memory_service.ingestion.claude.transcript_sync import (
    SessionPidMapping,
    _get_state_dir,
)
from ragzoom.client.grpc_client import GrpcRagzoomClient


class RememberNode(BaseModel):  # type: ignore[explicit-any]
    """A node from the tiling result."""

    text: str
    span_start: int
    span_end: int
    height: int
    token_count: int


class RememberResult(BaseModel):  # type: ignore[explicit-any]
    """Structured result from the remember tool."""

    nodes: list[RememberNode]


mcp = FastMCP(name="RagZoom Memory")


def _get_session_id() -> str:
    """Find the session ID by matching our parent PID to transcript state files.

    The sync hook writes Claude Code's PID to the state file. We find our session
    by scanning state files for one whose last_pid matches our parent process.

    Returns:
        document_id (session_id)
    """
    claude_code_pid = os.getppid()

    state_dir = _get_state_dir()
    if not state_dir.exists():
        raise ValueError(
            f"No transcript state directory found at {state_dir}. "
            "Set RAGZOOM_STATE_DIR environment variable if using a custom location. "
            "Has the transcript been synced yet?"
        )

    for state_file in state_dir.glob("*.jsonl"):
        mapping = SessionPidMapping.load(state_file)
        if mapping is not None and mapping.last_pid == claude_code_pid:
            return mapping.document_id

    raise ValueError(
        f"No session found for PID {claude_code_pid}. "
        "The Stop hook should have synced the transcript. "
        "Check that hooks are configured correctly."
    )


def _get_compaction_span_end(session_id: str, user_id: str) -> int | None:
    """Get the compaction boundary span_end from the server.

    The server tracks the compaction boundary during ingestion.
    This is much more efficient than scanning the transcript locally.

    Returns:
        span_end just before post-compaction content, or None if no compaction.
    """
    server_address = os.environ.get("RAGZOOM_SERVER_ADDRESS", "localhost:50051")
    with GrpcRagzoomClient(server_address) as client:
        result = client.get_compaction_boundary(session_id=session_id, user_id=user_id)
        if result.has_boundary:
            return result.span_end
        return None


@mcp.tool()
def remember(
    query: str,
    token_budget: int = 2000,
    span_start: int = 0,
    span_end: int | None = None,
) -> RememberResult:
    """Search pre-compaction conversation history.

    Use keyword/phrase queries that match content semantically.
    For follow-up queries, use span_start/span_end to zoom into specific regions.

    Args:
        query: Keywords/phrases to search for (semantic search)
        token_budget: Max tokens for returned context (default 2000)
        span_start: Start of span range for zooming (default 0)
        span_end: End of span range (defaults to compaction boundary)

    Returns:
        RememberResult with array of nodes, each containing text, span info, and height.
        Height 0 = verbatim original text, higher heights = progressively compressed.

    ## How It Works

    The tool returns a structured array of nodes that fit within your token budget.
    Each node has a height: height=0 are verbatim leaves (original text),
    higher heights are progressively more compressed summaries.

    With a small budget, you get high-level summaries. With a larger budget
    or constrained span, you get more verbatim content.

    ## Iterative Zoom Workflow

    This tool is designed for iterative exploration, not single-shot search.

    **Step 1 - Survey:** Start with a broad query to get an overview:

        remember(query="authentication bug", token_budget=2000)

        # Returns structured nodes like:
        # {"nodes": [
        #   {"text": "...", "span_start": 0, "span_end": 45000, "height": 5, ...},
        #   {"text": "...", "span_start": 45000, "span_end": 72000, "height": 4, ...},
        #   {"text": "mentions auth bug...", "span_start": 72000, "span_end": 89000, "height": 3, ...},
        # ]}

    **Step 2 - Zoom:** Drill into the relevant span for more detail:

        remember(query="authentication bug", token_budget=2000,
                 span_start=72000, span_end=89000)

        # Same budget, smaller region = more verbatim (height=0) content
        # {"nodes": [
        #   {"text": "...", "span_start": 72000, "span_end": 75000, "height": 1, ...},
        #   {"text": "verbatim details...", "span_start": 75000, "span_end": 78500, "height": 0, ...},
        # ]}

    **Step 3 - Repeat:** Continue zooming for full detail if needed.

    ## Tips

    - **Same results for different queries?** Budget is too small to expand
      past root summaries. Either increase token_budget or constrain the span.

    - **Query drives seed selection:** Keywords mark relevant leaves as
      important; seeds expand first, pulling in matching detail.

    - **Maximum detail:** Use token_budget=5000+ with tight span constraints.

    - **Recent content:** Spans near the compaction boundary are often
      already height=0 (verbatim) since they haven't been summarized yet.
    """
    doc_id = _get_session_id()

    # Get user_id from environment (set by sync hook)
    user_id = os.environ.get("RAGZOOM_USER_ID", "")
    if not user_id:
        raise ValueError(
            "RAGZOOM_USER_ID environment variable not set. "
            "The sync hook should set this when starting the MCP server."
        )

    # Compute compaction boundary on-demand if span_end not specified
    if span_end is None:
        span_end = _get_compaction_span_end(doc_id, user_id)

    # Query RagZoom via gRPC
    server_address = os.environ.get("RAGZOOM_SERVER_ADDRESS", "localhost:50051")
    with GrpcRagzoomClient(server_address) as client:
        output = client.execute_query(
            query=query,
            document_id=doc_id,
            budget_tokens=token_budget,
            num_seeds=None,
            embedding_model=None,
            debug=False,
            viz_width=80,
            use_token_coords=False,
            span_start=span_start,
            span_end=span_end,
        )

    # Extract nodes from tiling
    retrieval = output.retrieval
    nodes: list[RememberNode] = []

    for node_id in retrieval.tiling_ids:
        node = retrieval.nodes.get(node_id)
        if node and node.text:
            nodes.append(
                RememberNode(
                    text=node.text,
                    span_start=node.span_start,
                    span_end=node.span_end,
                    height=node.height,
                    token_count=node.token_count,
                )
            )

    return RememberResult(nodes=nodes)


if __name__ == "__main__":
    mcp.run(transport="stdio")
