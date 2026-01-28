# Claude Code Integration

This package provides Claude Code with access to pre-compaction conversation history through RagZoom's hierarchical summarization.

## Components

| Module | Purpose |
|--------|---------|
| `jsonl_reader.py` | Streaming JSONL parser with forward/reverse iteration |
| `transcript_sync.py` | Stateless revert-aware sync using RagZoom document status |
| `mcp_server.py` | MCP `recall` tool for querying historical context |
| `cli.py` | Command-line interface for sync operations |

## Key Concepts

### Step-Level Granularity

A "step" is any user or assistant message (excluding meta records, compaction summaries, and queue operations). Each step becomes its own leaf node with `time_start = time_end = record.timestamp`. This enables precise temporal queries - the `recall` tool can zoom to specific messages rather than being limited to coarser conversation chunks.

### Revert Detection

Claude Code supports branching - users can revert to earlier points and take different paths. The sync algorithm:

1. Builds a `parent_map` (uuid → parentUuid) from the JSONL transcript
2. Finds the common ancestor between the last indexed uuid and current head
3. If the ancestor is before the last indexed point, truncates the document and re-syncs

With step-level tracking, every record is a valid truncation point - no special boundary detection needed.

### MCP Server

The MCP server exposes a `recall` tool that Claude Code can use to query pre-compaction history:

```python
recall(query="authentication bug", token_budget=2000)
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

## Plugin Development

The Claude Code plugin source lives in `integrations/claude-code/plugin/`.

**After modifying plugin files, deploy to make changes available globally:**

```bash
./integrations/claude-code/scripts/deploy-plugin.sh
```

This copies the plugin to `~/.claude/plugins/ragzoom-memory/` with a backup of the previous version.

**Plugin structure:**
- `.claude-plugin/plugin.json` - Plugin manifest
- `.mcp.json` - MCP server configuration
- `hooks/hooks.json` - SessionStart, UserPromptSubmit, Stop hooks
- `scripts/` - Hook scripts (session-start.sh, sync-transcript.sh, iTerm2 status)
- `skills/memory-tool-usage/` - Memory retrieval guidance

Changes require restarting Claude Code to take effect.
