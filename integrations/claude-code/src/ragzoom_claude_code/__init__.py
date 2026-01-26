"""Claude Code memory integration - transcript syncing and MCP tool.

This package provides Claude Code with access to pre-compaction conversation
history through RagZoom's hierarchical summarization. It's a separate layer
on top of the core RagZoom platform.

Components:
- jsonl_reader: Streaming JSONL parser for Claude Code transcripts
- transcript_sync: Revert-aware sync with UUID→span tracking
- mcp_server: MCP 'remember' tool for querying historical context
- cli: Command-line interface for sync operations
"""

from ragzoom_claude_code.transcript_sync import (
    SessionState,
    SessionStateHeader,
    SyncResult,
    execute_sync,
    get_state_path,
    set_session_pid,
)

__all__ = [
    "SessionState",
    "SessionStateHeader",
    "SyncResult",
    "execute_sync",
    "get_state_path",
    "set_session_pid",
]
