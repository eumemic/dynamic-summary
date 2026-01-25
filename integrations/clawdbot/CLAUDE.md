# Clawdbot Integration

This package syncs Clawdbot session transcripts to RagZoom for historical context retrieval.

## Components

| Module | Purpose |
|--------|---------|
| `transcript_sync.py` | Linear sync with JSONL normalization |
| `cli.py` | Command-line interface for sync operations |

## Key Differences from Claude Code

| Aspect | Claude Code | Clawdbot |
|--------|-------------|----------|
| Branching | Supports revert/branching | Linear only |
| Sync model | Revert-aware with truncation | Append-only |
| Message format | Native format | Normalized to Claude Code format |
| MCP server | Yes (`remember` tool) | No |

## JSONL Format Normalization

Clawdbot uses a different message format than Claude Code. The `normalize_clawdbot_entry()` function converts:

```json
// Clawdbot format
{"type": "message", "id": "...", "parentId": "...", "message": {"role": "user|assistant|toolResult", ...}}

// Normalized to Claude Code format
{"type": "user|assistant", "uuid": "...", "parentUuid": "...", "message": {...}}
```

Special handling:
- `toolCall` blocks → `tool_use` blocks
- `toolResult` messages → user messages with `toolUseResult`
- `thinking` blocks → filtered out

## CLI Usage

```bash
# Sync a Clawdbot transcript
ragzoom-clawdbot sync session.jsonl

# With custom state directory
ragzoom-clawdbot sync session.jsonl --state-dir ./my-state

# With custom document ID
ragzoom-clawdbot sync session.jsonl --document-id my-session
```

## State Files

State is stored in `<state-dir>/<filename>.jsonl` (default: `data/clawdbot-state/`):

```json
{"document_id": "clawdbot-session", "last_message_id": "uuid", "span_end": 5000, "turns_synced": 10}
{"last_id": "uuid1", "first_id": "uuid1", "span_end": 1000}
{"last_id": "uuid2", "first_id": "uuid2", "span_end": 2500}
```

Since Clawdbot is linear (no branching), we simply track `turns_synced` and skip already-processed turns on subsequent syncs.
