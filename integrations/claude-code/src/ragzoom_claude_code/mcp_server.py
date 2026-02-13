"""MCP server exposing the 'recall' tool for querying conversation history."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

from ragzoom_claude_code.recall import execute_search
from ragzoom_claude_code.transcript_sync import get_session_document_id

mcp = FastMCP(name="RagZoom Memory")

_MEMORY_PERSONA_GUIDANCE = """\
The conversation history you are searching is between users and an \
assistant agent. You are being queried by that same assistant agent — \
it is recalling its own past. In the transcript, "assistant" messages \
are the caller's own words and actions.

Framing rules:
- Use second person for the assistant/caller: "You built...", "You suggested..."
- Refer to humans as "the user" (or by name when known): "The user asked you to..."
- Example: input "❯ review the memory service" → "The user asked you to review the memory service" """


def _ensure_timezone(ts: str | None) -> str | None:
    """Parse an ISO timestamp and ensure it has timezone info (default UTC)."""
    if ts is None:
        return None

    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


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
    session_id: str | None = None,
) -> str:
    """Search conversation history.

    Use this tool like a conversation with your memory. Ask natural questions
    about your past — what you worked on, what you decided, what happened —
    and follow up with more questions to dig deeper.

    Args:
        query: The question to answer from conversation history
        time_start: ISO timestamp to start from (e.g., "2024-01-15T10:00:00Z")
        time_end: ISO timestamp to end at (e.g., "2024-01-15T18:00:00Z")
        session_id: Resume a previous search session for follow-up questions

    Returns:
        Summary text with time ranges for follow-up zoom queries
    """
    doc_id = _get_session_id()
    server_address = os.environ.get("RAGZOOM_SERVER_ADDRESS", "localhost:50051")

    result = execute_search(
        question=query,
        document_id=doc_id,
        time_start=_ensure_timezone(time_start),
        time_end=_ensure_timezone(time_end),
        server_address=server_address,
        search_guidance=_MEMORY_PERSONA_GUIDANCE,
        session_id=session_id,
    )

    if result.session_id:
        return f"{result.answer}\n\nSession: {result.session_id}"
    return result.answer


if __name__ == "__main__":
    mcp.run(transport="stdio")
