"""MCP server exposing the 'recall' tool for querying conversation history."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from ragzoom_claude_code.recall import execute_search
from ragzoom_claude_code.transcript_sync import get_session_document_id

mcp = FastMCP(name="RagZoom Memory")

_MEMORY_PERSONA_GUIDANCE = """\
You are answering questions about the user's own conversation history — \
this is their memory, not a third-party document.

Framing rules:
- Use second person: "You were working on..." not "The conversation discussed..."
- Refer to the user's actions directly: "You decided to..." not "It was decided..."
- When quoting the user, attribute naturally: "You said..." or "You asked..."
- When quoting the assistant, say "Claude suggested..." or "The assistant recommended..."
- Treat the content as the user's lived experience, not an abstract record."""


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
def recall(
    query: str,
    time_start: str | None = None,
    time_end: str | None = None,
) -> str:
    """Search conversation history.

    Use keyword/phrase queries that match content semantically.
    The server automatically searches through the conversation at multiple
    levels of detail to find the best answer.

    For follow-up queries, use time_start/time_end to zoom into specific periods.

    Args:
        query: The question to answer from conversation history
        time_start: ISO timestamp to start from (e.g., "2024-01-15T10:00:00")
        time_end: ISO timestamp to end at (e.g., "2024-01-15T18:00:00")

    Returns:
        Summary text with time ranges for follow-up zoom queries
    """
    doc_id = _get_session_id()
    server_address = os.environ.get("RAGZOOM_SERVER_ADDRESS", "localhost:50051")

    result = execute_search(
        question=query,
        document_id=doc_id,
        time_start=time_start,
        time_end=time_end,
        server_address=server_address,
        search_guidance=_MEMORY_PERSONA_GUIDANCE,
    )

    return result.answer


if __name__ == "__main__":
    mcp.run(transport="stdio")
