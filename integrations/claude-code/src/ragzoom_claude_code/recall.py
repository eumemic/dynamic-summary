"""Shared recall logic for CLI and MCP server."""

from __future__ import annotations

from dataclasses import dataclass

from ragzoom.client.grpc_client import GrpcRagzoomClient


@dataclass
class RecallNode:
    """A single node from the recall tiling."""

    text: str
    time_start: str | None
    time_end: str | None
    height: int


@dataclass
class RecallResult:
    """Result of a recall query."""

    nodes: list[RecallNode]


def execute_recall(
    query: str,
    document_id: str,
    token_budget: int = 2000,
    time_start: str | None = None,
    time_end: str | None = None,
    server_address: str = "localhost:50051",
) -> RecallResult:
    """Execute a recall query against RagZoom.

    Args:
        query: Keywords/phrases to search for (semantic search)
        document_id: The document to query
        token_budget: Max tokens for returned context
        time_start: ISO timestamp to start from (e.g., "2024-01-15T10:00:00")
        time_end: ISO timestamp to end at (e.g., "2024-01-15T18:00:00")
        server_address: RagZoom gRPC server address

    Returns:
        RecallResult with nodes from the tiling
    """
    with GrpcRagzoomClient(server_address) as client:
        output = client.execute_query(
            query=query,
            document_id=document_id,
            budget_tokens=token_budget,
            num_seeds=None,
            embedding_model=None,
            debug=False,
            viz_width=80,
            use_token_coords=False,
            time_start=time_start,
            time_end=time_end,
        )

    retrieval = output.retrieval

    # Handle empty results
    if not retrieval.tiling_ids:
        return RecallResult(nodes=[])

    nodes: list[RecallNode] = []
    for node_id in retrieval.tiling_ids:
        node = retrieval.nodes.get(node_id)
        if node and node.text:
            nodes.append(
                RecallNode(
                    text=node.text,
                    time_start=node.time_start,
                    time_end=node.time_end,
                    height=node.height,
                )
            )

    return RecallResult(nodes=nodes)


def format_for_mcp(result: RecallResult) -> str:
    """Format recall result for MCP tool output.

    Uses the same format as CLI for consistency.
    """
    return format_for_cli(result)


def format_for_cli(result: RecallResult) -> str:
    """Format recall result for human-readable CLI output.

    Args:
        result: The recall result to format

    Returns:
        Formatted output with XML-style spans and copy-pasteable timestamps
    """
    if not result.nodes:
        return "No conversation data found in the requested time range."

    lines: list[str] = []

    # Header explaining the format
    max_height = max(node.height for node in result.nodes)
    first_start = result.nodes[0].time_start or "?"
    last_end = result.nodes[-1].time_end or "?"

    lines.append("<Explanation>")
    lines.append(
        f"This is a variable-resolution summary of the events from {first_start} to {last_end}."
    )
    lines.append(
        "Each span's height indicates summarization level: "
        "height=0 is verbatim transcript, higher values are increasingly compressed."
    )
    if max_height > 0:
        lines.append(
            "To zoom in, invoke recall() with time_start/time_end to constrain the time range."
        )
    lines.append("</Explanation>")
    lines.append("")

    for node in result.nodes:
        start = node.time_start or "?"
        end = node.time_end or "?"

        # XML-style opening tag with copy-pasteable timestamps
        lines.append(
            f'<Span time_start="{start}" time_end="{end}" height={node.height}>'
        )

        # Text content
        lines.append(node.text)

        # Closing tag
        lines.append("</Span>")
        lines.append("")

    return "\n".join(lines).rstrip()
