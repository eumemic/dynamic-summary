"""MCP server exposing the 'recall' tool for querying conversation history."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from ragzoom_claude_code.recall import execute_search
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


@mcp.tool()
def recall(query: str) -> str:
    """Search conversation history.

    Use keyword/phrase queries that match content semantically.
    The server automatically searches through the conversation at multiple
    levels of detail to find the best answer.

    Args:
        query: Keywords/phrases to search for (semantic search)

    Returns:
        A concise answer synthesized from conversation history
    """
    doc_id = _get_session_id()
    server_address = os.environ.get("RAGZOOM_SERVER_ADDRESS", "localhost:50051")

    result = execute_search(
        question=query,
        document_id=doc_id,
        server_address=server_address,
    )

    return result.answer


if __name__ == "__main__":
    mcp.run(transport="stdio")
