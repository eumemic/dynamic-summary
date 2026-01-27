# Claude Code Integration

This package provides Claude Code with access to pre-compaction conversation history through RagZoom's hierarchical summarization.

## Components

| Module | Purpose |
|--------|---------|
| `jsonl_reader.py` | Streaming JSONL parser with forward/reverse iteration |
| `transcript_sync.py` | Stateless revert-aware sync using RagZoom document status |
| `mcp_server.py` | MCP `remember` tool for querying historical context |
| `cli.py` | Command-line interface for sync operations |

## Key Concepts

### Turn-Level Granularity

Messages are grouped into "turns" - a user prompt through the assistant's complete response cycle. Each turn becomes one leaf node with temporal metadata (`time_start`, `time_end`). This enables time-based queries like "what did we discuss yesterday?"

### Revert Detection

Claude Code supports branching - users can revert to earlier points and take different paths. The sync algorithm:

1. Builds a `parent_map` (uuid → parentUuid) from the JSONL transcript
2. Finds the common ancestor between the last indexed uuid and current head
3. If the ancestor is before the last indexed point, truncates the document and re-syncs

Turn-level tracking means if a revert falls *within* a turn (not at a boundary), we truncate the entire turn rather than keeping partial content.

### MCP Server

The MCP server exposes a `remember` tool that Claude Code can use to query pre-compaction history:

```python
remember(query="authentication bug", token_budget=2000)
```

The server finds its session by reading a PID temp file written by the SessionStart hook.

## CLI Usage

```bash
# Sync a transcript
ragzoom-claude-code sync ~/.claude/projects/.../session.jsonl

# Start MCP server (usually via scripts/start-mcp-server)
ragzoom-claude-code mcp-server
```

## Session Discovery

The MCP server discovers its session document ID via PID temp files:

1. The SessionStart hook writes the document ID to `/tmp/ragzoom-session-{pid}`
2. The MCP server reads this file using its parent PID to find the session

The sync algorithm itself is stateless, deriving all state from the transcript and RagZoom document status API.
