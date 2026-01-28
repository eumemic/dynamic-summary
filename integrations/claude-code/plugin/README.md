# RagZoom Memory Plugin

Persistent conversation memory for Claude Code via RagZoom hierarchical summarization.

## Prerequisites

1. **Install ragzoom packages:**
   ```bash
   pip install -e /path/to/dynamic-summary
   pip install -e /path/to/dynamic-summary/integrations/claude-code
   ```

2. **Start RagZoom server:**
   ```bash
   ragzoom server start --daemon
   ```

## What This Plugin Does

- **SessionStart hook**: Registers session for MCP server lookup
- **Stop hook**: Syncs transcript to RagZoom on session end
- **UserPromptSubmit hook**: Syncs transcript on each user message (keeps memory up-to-date)
- **MCP server**: Exposes `recall` tool for querying conversation history
- **Skill**: Guidance on effective memory retrieval patterns

## Usage

After installation, Claude Code automatically:
1. Syncs your conversation to RagZoom
2. Can use `recall(query, ...)` to recall pre-compaction context

## Configuration

Set `RAGZOOM_DISABLE_SYNC=1` to disable transcript sync (useful during builds).

## Troubleshooting

**MCP server not finding session:**
- Ensure SessionStart hook ran (check for `/tmp/ragzoom-session-*` files)
- Verify RagZoom server is running: `ragzoom server status`

**Transcript not syncing:**
- Check Stop hook is registered: `claude --debug` shows hook execution
- Verify `ragzoom-claude-code` CLI is available: `which ragzoom-claude-code`
