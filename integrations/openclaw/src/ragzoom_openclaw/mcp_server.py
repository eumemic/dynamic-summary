"""MCP server exposing the 'recall' tool for querying OpenClaw conversation history."""

from __future__ import annotations

import os
import sys

# Add claude-code integration to path for recall module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../claude-code/src"))

from mcp.server.fastmcp import FastMCP

from ragzoom_claude_code.recall import execute_recall, format_for_mcp

mcp = FastMCP(name="RagZoom OpenClaw Memory")


def _get_config() -> tuple[str, str]:
    """Get document ID and server address from environment.
    
    Returns:
        (document_id, server_address)
    """
    doc_id = os.environ.get("RAGZOOM_DOCUMENT_ID")
    if not doc_id:
        raise ValueError(
            "RAGZOOM_DOCUMENT_ID environment variable required. "
            "Set it to your document ID (e.g., 'jarvis-main')."
        )
    
    server_address = os.environ.get("RAGZOOM_SERVER_ADDRESS", "localhost:50052")
    return doc_id, server_address


@mcp.tool()
def recall(
    query: str,
    token_budget: int = 2000,
    time_start: str | None = None,
    time_end: str | None = None,
) -> str:
    """Search your conversation history using semantic search.

    Use this to recall past discussions, decisions, context from earlier sessions.
    Results include time ranges you can use to zoom into specific periods.

    Args:
        query: Keywords/phrases to search for (semantic search)
        token_budget: Max tokens for returned context (default 2000)
        time_start: ISO timestamp to start from (e.g., "2026-01-15T10:00:00Z")
        time_end: ISO timestamp to end at (e.g., "2026-01-15T18:00:00Z")

    Returns:
        Summary text with time ranges. Each span shows:
        - time_start/time_end: when this content happened
        - height: 0 = verbatim, higher = more summarized
        
    ## Workflow
    
    1. Start broad: recall(query="what we discussed about X")
    2. See time ranges in results
    3. Zoom in: recall(query="X", time_start="...", time_end="...")
    4. Repeat until you have the detail you need
    
    ## Tips
    
    - Recent content may already be in your context window
    - Use time constraints + higher token_budget for more detail
    - Thinking blocks (💭) are included in memory
    """
    doc_id, server_address = _get_config()

    result = execute_recall(
        query=query,
        document_id=doc_id,
        token_budget=token_budget,
        time_start=time_start,
        time_end=time_end,
        server_address=server_address,
    )

    return format_for_mcp(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
