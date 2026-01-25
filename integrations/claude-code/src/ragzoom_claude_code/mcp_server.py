"""MCP server exposing the 'remember' tool for querying conversation history."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from ragzoom.client.grpc_client import GrpcRagzoomClient
from ragzoom_claude_code.transcript_sync import SessionState, _get_state_dir

mcp = FastMCP(name="RagZoom Memory")


def _get_session_id() -> tuple[str, SessionState]:
    """Find the session ID by matching our parent PID to transcript state files.

    The sync hook writes Claude Code's PID to the state file. We find our session
    by scanning state files for one whose last_pid matches our parent process.

    Returns:
        Tuple of (document_id, session_state)
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
        state = SessionState.load(state_file)
        if state is not None and state.header.last_pid == claude_code_pid:
            return state.header.document_id, state

    raise ValueError(
        f"No session found for PID {claude_code_pid}. "
        "The Stop hook should have synced the transcript. "
        "Check that hooks are configured correctly."
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
    doc_id, _ = _get_session_id()

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
