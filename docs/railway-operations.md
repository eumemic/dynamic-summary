# Railway Operations Guide

This guide covers operational tasks for the hosted RagZoom service on Railway.

## Environment Setup

### Non-Interactive Linking (Recommended)

The Railway CLI link often gets lost. Use these project details for non-interactive linking:

- **Project ID**: `9d168ba6-ac78-4739-a53c-7ca04e211678`
- **Project Name**: `magnificent-harmony`
- **Production Environment ID**: `9fabe46f-fb02-49bc-afa6-0a9b0f87b51a`

```bash
# Link non-interactively using project ID (works even without TTY)
railway link -p 9d168ba6-ac78-4739-a53c-7ca04e211678 -e production

# Verify link status
railway status
```

This is the **preferred method** - it works in any directory and doesn't require
interactive prompts.

## Key Services

| Service | Purpose |
|---------|---------|
| `dynamic-summary` | gRPC server for memory ingestion |
| `pgvector` | PostgreSQL with pgvector extension (the active database) |
| `pgvector-rW-f` | Legacy/reference service (not used for data) |

**Important**: The database with actual data is `pgvector`, NOT `pgvector-rW-f`.

## Database Access

### Get Database URLs

```bash
# First ensure you're linked (see above)
railway link -p 9d168ba6-ac78-4739-a53c-7ca04e211678 -e production

# Get all database variables (use 'pgvector' service, NOT 'pgvector-rW-f')
railway variables --service pgvector --json

# Get just the public URL for external access
railway variables --service pgvector --kv | grep DATABASE_PUBLIC_URL
```

### Run CLI Commands Against Production Database

```bash
# Step 1: Link to project (if not already linked)
railway link -p 9d168ba6-ac78-4739-a53c-7ca04e211678 -e production

# Step 2: Get the public database URL
railway variables --service pgvector --kv | grep DATABASE_PUBLIC_URL
# Output: DATABASE_PUBLIC_URL=postgresql://postgres:PASSWORD@tramway.proxy.rlwy.net:48318/railway

# Step 3: Set it explicitly (copy-paste the URL)
export RAGZOOM_DATABASE_URL="postgresql://postgres:PASSWORD@tramway.proxy.rlwy.net:48318/railway"

# Step 4: Run commands
ragzoom validate <document-id>
python -m memory_service.admin status
```

**Note**: Don't use `export RAGZOOM_DATABASE_URL="$(railway ...)"` - command
substitution can mangle special characters in the password. Copy-paste the URL directly.

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
# Set database URL first (see "Run CLI Commands Against Production Database" above)
export RAGZOOM_DATABASE_URL="postgresql://postgres:PASSWORD@host:port/railway"

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
      offset: 93,090,415 bytes      # File bytes processed
      span_end: 1148356             # Character span indexed
      last_synced: f109d8cd         # Last processed UUID
      stored: 46,643,997 bytes      # JSONL content stored

📄 Documents: 1
🌳 Tree nodes: 3,669

────────────────────────────────────────────────────────────
📄 Document: 7cdd0798-4f29-4ce6-bfc9-6dc3b7bb2153

   📈 Indexing Progress:
      Leaves: 1,838
      Embeddings: 1,838/1,838 (100.0%) ✅
      Summaries: 1,831/1,831 (100.0%) ✅
      Tree: height=10 | ✅ Complete forest (7 trees)

   🔍 Validation:
      ✅ Tree: PASSED | Nodes: 3,669 | Leaves: 1,838
      ✅ Transcript: 1,838 leaves verified
         ℹ️  Internal roots (normal during indexing): 7
```

Key metrics explained:
- **Embeddings**: Leaves with vector embeddings (needed for semantic search)
- **Summaries**: Internal nodes created by merging sibling pairs
- **Forest**: Multiple perfect binary trees. Complete when no two roots share the same height (all mergeable pairs merged). Only becomes a single tree if leaf count is a power of 2.
- **Validation**: Two-layer validation:
  - **Tree**: Checks for duplicate coordinates, broken parent refs, orphaned nodes
  - **Transcript**: Verifies each leaf's text appears in order in the JSONL transcript

## Common Operations

### Validate a Document

Check tree invariants for corruption:

```bash
# Set database URL first (see "Run CLI Commands Against Production Database" above)
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
# (Set database URL first - see "Run CLI Commands Against Production Database" above)
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
