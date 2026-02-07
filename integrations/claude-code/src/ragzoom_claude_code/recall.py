"""Shared recall logic for CLI and MCP server."""

from __future__ import annotations

from ragzoom.client.grpc_client import ExecuteQueryOutput, GrpcRagzoomClient
from ragzoom.output_formatters import format_tiling_spans


def execute_recall(
    query: str,
    document_id: str,
    token_budget: int = 2000,
    time_start: str | None = None,
    time_end: str | None = None,
    server_address: str = "localhost:50051",
) -> ExecuteQueryOutput:
    """Execute a recall query against RagZoom.

    Args:
        query: Keywords/phrases to search for (semantic search)
        document_id: The document to query
        token_budget: Max tokens for returned context
        time_start: ISO timestamp to start from (e.g., "2024-01-15T10:00:00")
        time_end: ISO timestamp to end at (e.g., "2024-01-15T18:00:00")
        server_address: RagZoom gRPC server address

    Returns:
        Full query output including retrieval metadata and tiling nodes.
        Use format_tiling_spans() to produce human-readable text output.
    """
    with GrpcRagzoomClient(server_address) as client:
        return client.execute_query(
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


__all__ = ["ExecuteQueryOutput", "execute_recall", "format_tiling_spans"]
