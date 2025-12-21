"""MCP server exposing the 'remember' tool for querying pre-compaction history."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ragzoom.claude_transcript import _load_state
from ragzoom.client.grpc_client import GrpcRagzoomClient

mcp = FastMCP(name="RagZoom Memory")


def _get_session_id() -> str:
    """Find the session ID by matching our parent PID to transcript state files.

    The sync hook writes Claude Code's PID to the state file. We find our session
    by scanning state files for one whose last_pid matches our parent process.
    """
    claude_code_pid = os.getppid()

    state_dir = Path("data/transcript-state")
    if not state_dir.exists():
        raise ValueError(
            f"No transcript state directory found at {state_dir}. "
            "Has the transcript been synced yet?"
        )

    for state_file in state_dir.glob("*.json"):
        doc_id = state_file.stem
        state = _load_state(doc_id)
        if state.last_pid == claude_code_pid:
            return doc_id

    raise ValueError(
        f"No session found for PID {claude_code_pid}. "
        "The UserPromptSubmit hook should have synced the transcript. "
        "Check that hooks are configured correctly."
    )


def _format_response(
    summary: str,
    nodes: list[tuple[int, int, int]],
) -> str:
    """Format query response with node metadata for follow-up queries.

    Args:
        summary: The summary text from the query
        nodes: List of (span_start, span_end, height) tuples

    Returns:
        Formatted string with summary and node spans
    """
    result = summary

    if nodes:
        result += "\n\n---\nNode spans (for zooming):\n"
        for span_start, span_end, height in nodes:
            result += f"  [{span_start}-{span_end}] height={height}\n"

    return result


@mcp.tool()
def remember(
    query: str,
    token_budget: int = 2000,
    span_start: int = 0,
    span_end: int | None = None,
) -> str:
    """Search pre-compaction conversation history.

    Use keyword/phrase queries that match content semantically.
    For follow-up queries, use span_start/span_end to zoom into specific regions.

    Args:
        query: Keywords/phrases to search for (semantic search)
        token_budget: Max tokens for returned context (default 2000)
        span_start: Start of span range for zooming (default 0)
        span_end: End of span range (defaults to compaction boundary)

    Returns:
        Summary text with node spans for follow-up zoom queries

    ## How It Works

    The tool returns a "tiling" of nodes that fit within your token budget.
    Each node has a height: height=0 are verbatim leaves (original text),
    higher heights are progressively more compressed summaries.

    With a small budget, you get high-level summaries. With a larger budget
    or constrained span, you get more verbatim content.

    ## Iterative Zoom Workflow

    This tool is designed for iterative exploration, not single-shot search.

    **Step 1 - Survey:** Start with a broad query to get an overview:

        remember(query="authentication bug", token_budget=2000)

        # Returns summaries + node spans like:
        # [0-45000] height=5
        # [45000-72000] height=4
        # [72000-89000] height=3  <-- mentions auth bug here
        # [89000-95000] height=1

    **Step 2 - Zoom:** Drill into the relevant span for more detail:

        remember(query="authentication bug", token_budget=2000,
                 span_start=72000, span_end=89000)

        # Same budget, smaller region = more verbatim content
        # Returns nodes like:
        # [72000-75000] height=1
        # [75000-78500] height=0  <-- verbatim leaf!
        # [78500-82000] height=0
        # [82000-89000] height=2

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

    # Default span_end to compaction boundary (pre-compaction only)
    if span_end is None:
        state = _load_state(doc_id)
        if state.compaction_span_end is not None:
            span_end = state.compaction_span_end

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

    # Extract summary from tiling
    retrieval = output.retrieval
    summary_parts = []
    node_info = []

    for node_id in retrieval.tiling_ids:
        node = retrieval.nodes.get(node_id)
        if node and node.text:
            summary_parts.append(node.text)
            node_info.append((node.span_start, node.span_end, node.height))

    summary = "\n\n".join(summary_parts)
    return _format_response(summary, node_info)


if __name__ == "__main__":
    mcp.run(transport="stdio")
