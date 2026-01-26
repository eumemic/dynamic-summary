# Claude Code Integration

This package provides Claude Code with access to pre-compaction conversation history through RagZoom's hierarchical summarization.

## Components

| Module | Purpose |
|--------|---------|
| `jsonl_reader.py` | Streaming JSONL parser with forward/reverse iteration |
| `transcript_sync.py` | Revert-aware sync with UUID→span tracking |
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

The server finds its session by matching the parent PID to state files written by the sync hook.

## CLI Usage

```bash
# Sync a transcript
ragzoom-claude-code sync ~/.claude/projects/.../session.jsonl

# Set session PID (called by SessionStart hook)
ragzoom-claude-code set-pid <document_id> <pid>

# Start MCP server (usually via scripts/start-mcp-server)
ragzoom-claude-code mcp-server
```

## State Files

Session state is stored in `$RAGZOOM_STATE_DIR/<document_id>.jsonl` (default: `data/transcript-state/`):

```json
{"document_id": "abc123", "last_pid": 12345}
```

The file contains a single header line with the document ID and optional PID for session discovery. The stateless sync algorithm derives all other state from the transcript and RagZoom document status API.
