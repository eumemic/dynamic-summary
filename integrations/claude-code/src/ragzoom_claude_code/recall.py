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

    Args:
        result: The recall result to format

    Returns:
        Formatted string with summary and time ranges for zooming
    """
    if not result.nodes:
        return "No conversation data found in the requested time range."

    summary = "\n\n".join(node.text for node in result.nodes)

    # Add time ranges footer for zoom workflow
    summary += "\n\n---\nTime ranges (for zooming):\n"
    for node in result.nodes:
        start = node.time_start or "?"
        end = node.time_end or "?"
        summary += f"  [{start} to {end}] height={node.height}\n"

    return summary


def format_for_cli(result: RecallResult) -> str:
    """Format recall result for human-readable CLI output.

    Args:
        result: The recall result to format

    Returns:
        Numbered, formatted output with timestamps
    """
    if not result.nodes:
        return "No conversation data found in the requested time range."

    lines: list[str] = []
    for i, node in enumerate(result.nodes, 1):
        # Format time range
        start = _format_time(node.time_start)
        end = _format_time(node.time_end)
        time_range = f"{start}-{end}" if start != end else start

        # Header with number, time range, and height
        lines.append(f"[{i}] {time_range} (height={node.height})")

        # Indented text content
        for text_line in node.text.split("\n"):
            lines.append(f"    {text_line}")

        lines.append("")  # Blank line between nodes

    return "\n".join(lines).rstrip()


def _format_time(iso_time: str | None) -> str:
    """Format ISO timestamp for display (HH:MM:SS or full date if needed)."""
    if iso_time is None:
        return "?"

    # Extract just the time portion if it's a full ISO timestamp
    # e.g., "2024-01-15T10:30:45.123Z" -> "10:30:45"
    if "T" in iso_time:
        time_part = iso_time.split("T")[1]
        # Remove timezone and milliseconds
        time_part = time_part.split(".")[0].split("Z")[0].split("+")[0]
        return time_part

    return iso_time
