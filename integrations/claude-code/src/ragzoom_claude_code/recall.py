"""Shared search logic for CLI and MCP server."""

from __future__ import annotations

from ragzoom.client.grpc_client import GrpcRagzoomClient, SearchResultView


def execute_search(
    question: str,
    document_id: str,
    time_start: str | None = None,
    time_end: str | None = None,
    server_address: str = "localhost:50051",
    search_guidance: str | None = None,
) -> SearchResultView:
    """Execute an agentic search against RagZoom.

    The server-side search agent iteratively zooms into the document
    to find the best answer. Question in, answer out.

    Args:
        question: Natural language question to answer.
        document_id: Document to search within.
        time_start: ISO 8601 lower bound for search (optional).
        time_end: ISO 8601 upper bound for search (optional).
        server_address: RagZoom gRPC server address.
        search_guidance: Additional guidance appended to the search agent
            system prompt (e.g. persona instructions).

    Returns:
        SearchResultView with the answer.
    """
    with GrpcRagzoomClient(server_address) as client:
        return client.search(
            question=question,
            document_id=document_id,
            time_start=time_start,
            time_end=time_end,
            search_guidance=search_guidance,
        )


__all__ = [
    "SearchResultView",
    "execute_search",
]
