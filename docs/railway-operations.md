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
