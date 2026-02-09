"""Shared search logic for CLI and MCP server."""

from __future__ import annotations

from ragzoom.client.grpc_client import GrpcRagzoomClient, SearchResultView


def execute_search(
    question: str,
    document_id: str,
    server_address: str = "localhost:50051",
) -> SearchResultView:
    """Execute an agentic search against RagZoom.

    The server-side search agent iteratively zooms into the document
    to find the best answer. Question in, answer out.

    Args:
        question: Natural language question to answer.
        document_id: Document to search within.
        server_address: RagZoom gRPC server address.

    Returns:
        SearchResultView with the answer.
    """
    with GrpcRagzoomClient(server_address) as client:
        return client.search(question=question, document_id=document_id)


__all__ = [
    "SearchResultView",
    "execute_search",
]
