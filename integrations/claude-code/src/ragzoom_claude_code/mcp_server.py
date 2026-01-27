"""MCP server exposing the 'remember' tool for querying conversation history."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from ragzoom.client.grpc_client import GrpcRagzoomClient
from ragzoom_claude_code.transcript_sync import get_session_document_id

mcp = FastMCP(name="RagZoom Memory")


def _get_session_id() -> str:
    """Get the document ID for the current session.

    Identity resolution priority:
    1. RAGZOOM_DOCUMENT_ID env var (configured identity - Jarvis/Legion model)
    2. PID temp file lookup (discovered identity - Claude Code model)

    Returns:
        Document ID string
    """
    # Configured identity (Jarvis/Legion model)
    if doc_id := os.environ.get("RAGZOOM_DOCUMENT_ID"):
        return doc_id

    # Discovered identity (Claude Code model) - PID temp file lookup
    claude_code_pid = os.getppid()
    doc_id = get_session_document_id(claude_code_pid)

    if doc_id is not None:
        return doc_id

    raise ValueError(
        f"No session found for PID {claude_code_pid}. "
        "Either set RAGZOOM_DOCUMENT_ID or ensure SessionStart hook ran."
    )


def _format_response(
    summary: str,
    nodes: list[tuple[str | None, str | None, int]],
) -> str:
    """Format query response with node metadata for follow-up queries.

    Args:
        summary: The summary text from the query
        nodes: List of (time_start, time_end, height) tuples

    Returns:
        Formatted string with summary and time ranges
    """
    result = summary

    if nodes:
        result += "\n\n---\nTime ranges (for zooming):\n"
        for time_start, time_end, height in nodes:
            start = time_start or "?"
            end = time_end or "?"
            result += f"  [{start} to {end}] height={height}\n"

    return result


@mcp.tool()
def remember(
    query: str,
    token_budget: int = 2000,
    time_start: str | None = None,
    time_end: str | None = None,
) -> str:
    """Search conversation history.

    Use keyword/phrase queries that match content semantically.
    For follow-up queries, use time_start/time_end to zoom into specific periods.

    Args:
        query: Keywords/phrases to search for (semantic search)
        token_budget: Max tokens for returned context (default 2000)
        time_start: ISO timestamp to start from (e.g., "2024-01-15T10:00:00")
        time_end: ISO timestamp to end at (e.g., "2024-01-15T18:00:00")

    Returns:
        Summary text with time ranges for follow-up zoom queries

    ## How It Works

    The tool returns a "tiling" of nodes that fit within your token budget.
    Each node has a height: height=0 are verbatim leaves (original text),
    higher heights are progressively more compressed summaries.

    With a small budget, you get high-level summaries. With a larger budget
    or constrained time range, you get more verbatim content.

    ## Iterative Zoom Workflow

    This tool is designed for iterative exploration, not single-shot search.

    **Step 1 - Survey:** Start with a broad query to get an overview:

        remember(query="authentication bug", token_budget=2000)

        # Returns summaries + time ranges like:
        # [2024-01-10T09:00:00 to 2024-01-10T12:00:00] height=3
        # [2024-01-10T14:00:00 to 2024-01-10T16:30:00] height=2  <-- mentions auth bug
        # [2024-01-10T16:30:00 to 2024-01-10T18:00:00] height=1

    **Step 2 - Zoom:** Drill into the relevant time range for more detail:

        remember(query="authentication bug", token_budget=2000,
                 time_start="2024-01-10T14:00:00", time_end="2024-01-10T16:30:00")

        # Same budget, smaller time range = more verbatim content

    **Step 3 - Repeat:** Continue zooming for full detail if needed.

    ## Tips

    - **Same results for different queries?** Budget is too small to expand
      past root summaries. Either increase token_budget or constrain the time range.

    - **Query drives seed selection:** Keywords mark relevant leaves as
      important; seeds expand first, pulling in matching detail.

    - **Maximum detail:** Use token_budget=5000+ with tight time constraints.

    - **Recent content:** Recent time ranges often return height=0 (verbatim)
      since they haven't been summarized yet.
    """
    doc_id = _get_session_id()

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
            time_start=time_start,
            time_end=time_end,
        )

    # Extract summary from tiling
    retrieval = output.retrieval
    summary_parts = []
    node_info: list[tuple[str | None, str | None, int]] = []

    for node_id in retrieval.tiling_ids:
        node = retrieval.nodes.get(node_id)
        if node and node.text:
            summary_parts.append(node.text)
            node_info.append((node.time_start, node.time_end, node.height))

    summary = "\n\n".join(summary_parts)
    return _format_response(summary, node_info)


if __name__ == "__main__":
    mcp.run(transport="stdio")
