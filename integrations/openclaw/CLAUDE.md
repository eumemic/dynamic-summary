# OpenClaw Integration

This package syncs OpenClaw session transcripts to RagZoom for historical context retrieval.

## Components

| Module | Purpose |
|--------|---------|
| `transcript_sync.py` | Per-step sync with thinking preservation |
| `cli.py` | Command-line interface for sync operations |

## Key Differences from Claude Code

| Aspect | Claude Code | OpenClaw |
|--------|-------------|----------|
| Branching | Supports revert/branching | Linear only |
| Sync model | Revert-aware with truncation | Append-only |
| Message format | Native format | id/parentId with message.role |
| Granularity | Per-step | Per-step |
| Thinking blocks | Filtered out | **Preserved with 💭** |
| MCP server | Yes (`remember` tool) | Not yet |

## JSONL Format

OpenClaw uses a different structure than Claude Code:

```json
// OpenClaw format
{"type": "message", "id": "...", "parentId": "...", "timestamp": "...", "message": {"role": "user|assistant", "content": [...]}}

// Compaction markers
{"type": "compaction", "id": "...", ...}
```

## Thinking Block Preservation

Unlike other integrations, OpenClaw **preserves thinking blocks** in the indexed transcript:

```
Assistant: 💭 Let me think about this approach...

The solution involves three steps:
1. ...
```

This is intentional — thinking is part of the agent's identity and memory.

## Tool Call Summarization

Verbose tool calls are summarized for cleaner transcripts:

- File ops: `[Wrote: filename.py]`, `[Edited: config.json]`
- Exec: `[Ran: git status...]`
- Web: `[Searched: query]`, `[Fetched: url]`
- Internal ops (memory_search, process, etc.): Skipped entirely

## CLI Usage

```bash
# Sync an OpenClaw transcript
ragzoom-openclaw sync session.jsonl

# With custom document ID
ragzoom-openclaw sync session.jsonl --document-id jarvis-main

# With custom server address
ragzoom-openclaw sync session.jsonl --server-address localhost:50052
```

## State Files

State is stored in `<state-dir>/<filename>.jsonl` (default: `data/openclaw-state/`):

```json
{"document_id": "jarvis-main", "last_message_id": "uuid", "span_end": 5000, "steps_synced": 100}
{"uuid": "uuid1", "span_end": 1000}
{"uuid": "uuid2", "span_end": 2500}
```

Per-step tracking enables fine-grained temporal queries.
