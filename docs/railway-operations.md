# Railway Operations Guide

This guide covers operational tasks for the hosted RagZoom service on Railway.

## Environment Setup

The Railway CLI must be linked to the project:

```bash
cd /path/to/dynamic-summary
railway link  # Select magnificent-harmony project
```

## Key Services

| Service | Purpose |
|---------|---------|
| `dynamic-summary` | gRPC server for memory ingestion |
| `pgvector-rW-f` | PostgreSQL with pgvector extension |
| `pgvector` | (Legacy, not actively used) |

## Database Access

### Get Database URLs

```bash
# Internal URL (only works from within Railway network)
railway variables --kv --service dynamic-summary | grep RAGZOOM_DATABASE_URL

# Public URL (works from anywhere)
railway variables --kv --service pgvector-rW-f | grep DATABASE_PUBLIC_URL
```

### Run CLI Commands Against Production Database

```bash
# Set the public database URL
export RAGZOOM_DATABASE_URL="$(railway variables --kv --service pgvector-rW-f | grep DATABASE_PUBLIC_URL | cut -d= -f2-)"

# Now run any ragzoom CLI command
ragzoom validate <document-id>
ragzoom status
```

## Finding Session/Document IDs

Sessions are identified by UUIDs. For Claude Code sessions:

```bash
# Find the most recently modified session for a worktree
ls -lt ~/.claude/projects/-Users-tom-code-dynamic-summary-worktrees-worktree-1/*.jsonl | head -1

# Output example:
# -rw-------@ 1 tom  staff  91883447 Jan  3 17:14 .../7cdd0798-4f29-4ce6-bfc9-6dc3b7bb2153.jsonl
#                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                                     This UUID is the session/document ID
```

The session ID is the filename (without `.jsonl` extension) of the transcript file.

## Admin CLI

The `memory_service.admin` module provides direct database access for admin operations:

```bash
# Set database URL first
export RAGZOOM_DATABASE_URL="$(railway variables --kv --service pgvector-rW-f | grep DATABASE_PUBLIC_URL | cut -d= -f2-)"

# Show service status and session inventory
python -m memory_service.admin status

# Reset a session for full re-index
python -m memory_service.admin reset <session-id>

# Transcribe stored JSONL to text
python -m memory_service.admin transcribe <session-id> [-o output.txt]

# Validate indexed leaves match transcription
python -m memory_service.admin validate <session-id> [--from-compaction]

# Debug commands
python -m memory_service.admin chain <session-id>        # Show ancestor chain
python -m memory_service.admin segments <session-id>     # Show segment boundaries
python -m memory_service.admin inspect-uuid <session-id> <uuid>
python -m memory_service.admin inspect-leaves <session-id> <offset>
```

Session IDs can be specified as prefixes for convenience.

### Status Command Output

The `status` command is a comprehensive dashboard showing database, sessions, indexing
progress (embeddings AND summaries), tree structure, job queue, and validation:

```
Memory Service Status
============================================================

📊 Database: postgresql://***@nozomi.proxy.rlwy.net:30284/railway
   ✅ Connected

📋 Sessions: 2

   7cdd0798-4f2...
      user: tom
      offset: 92,400,393 bytes      # File bytes processed
      span_end: 1130850             # Character span indexed
      last_synced: 442a504d         # Last processed UUID
      stored: 46,448,304 bytes      # JSONL content stored

📄 Documents: 1
🌳 Tree nodes: 2,318

────────────────────────────────────────────────────────────
📄 Document: 7cdd0798-4f29-4ce6-bfc9-6dc3b7bb2153

   📈 Indexing Progress:
      Leaves: 1,806
      Embeddings: 546/1,806 (30.2%) ⏳ 1,260 pending
      Summaries: 513/1,805 (28.4%) ⏳ ~1,292 pending
      Tree: height=8 | 🌲 Forest (1293 roots)
      Queue: 643 mergeable pairs

   🔍 Validation:
      ✅ PASSED | Nodes: 2,319 | Leaves: 1,806
         ℹ️  Internal roots (normal during indexing): 29
```

Key metrics explained:
- **Embeddings**: Leaves with vector embeddings (needed for semantic search)
- **Summaries**: Internal nodes created by merging sibling pairs
- **Forest**: Multiple perfect binary trees. Complete when no two roots share the same height (all mergeable pairs merged). Only becomes a single tree if leaf count is a power of 2.
- **Queue**: Sibling pairs at same height eligible for summarization
- **Validation**: Checks for duplicate coordinates, broken parent refs, etc.

## Common Operations

### Validate a Document

Check tree invariants for corruption:

```bash
export RAGZOOM_DATABASE_URL="$(railway variables --kv --service pgvector-rW-f | grep DATABASE_PUBLIC_URL | cut -d= -f2-)"
ragzoom validate <session-uuid>

# Example output for corrupted document:
# ❌ Document validation failed
#    Nodes: 3203, Leaves: 1783, Parentless: 365
```

### Check Deployment Logs

```bash
# Build logs
railway logs --service dynamic-summary

# Deploy logs (live)
railway logs --service dynamic-summary --deploy
```

### View Service Variables

```bash
railway variables --service dynamic-summary
railway variables --service pgvector-rW-f
```

## Troubleshooting

### "Service not found" Error

Ensure you're in a directory linked to the Railway project:

```bash
railway status  # Should show project and environment
railway link    # Re-link if needed
```

### Database Connection Failures

If you see `nodename nor servname provided`:
- You're using the internal URL from outside Railway
- Use `DATABASE_PUBLIC_URL` from the `pgvector-rW-f` service instead

### Orphaned Roots / Tree Corruption

Symptoms:
- `ragzoom validate` shows "Parentless" nodes > 0
- Indexing stalls with jobs stuck
- `HEIGHT_BLOCKED` in logs

Root cause: Non-atomic summary job failure (fixed in commit `527c0aa`).

To diagnose:
```bash
ragzoom validate <session-uuid>
# Look for:
# - Nodes sharing same (height, level_index)
# - Children pointing to wrong parent
# - Parentless count > 0
```

To fix (reset and re-index):
```bash
# Reset the sync cursor - next sync will detect revert and rebuild from scratch
export RAGZOOM_DATABASE_URL="$(railway variables --kv --service pgvector-rW-f | grep DATABASE_PUBLIC_URL | cut -d= -f2-)"
python -m memory_service.admin reset <session-uuid>

# Output:
# Resetting session: <session-uuid>
#    Current offset: 92,213,209
#    Current span_end: 1130832
#    Current last_synced: <uuid>
#
# ✅ Cursor reset. Next sync will trigger full re-index.
```

The reset command clears `last_synced_uuid` and `original_file_offset` while preserving
`span_end`. The next sync detects that `span_end > 0` but there's no sync cursor,
triggering a full revert: the index is truncated and rebuilt from the beginning.

## MCP Server Configuration

The memory MCP server connects to Railway via gRPC:

```json
// .mcp.json
{
  "mcpServers": {
    "ragzoom-memory": {
      "command": "python",
      "args": ["-m", "memory_service.ingestion.claude.mcp_server"],
      "env": {
        "RAGZOOM_SERVER_ADDRESS": "switchback.proxy.rlwy.net:11553",
        "RAGZOOM_USER_ID": "tom"
      }
    }
  }
}
```

The server address is the Railway TCP proxy for the gRPC service.
