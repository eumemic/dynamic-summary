# Memory Service Operations - Detailed Reference

## Environment Model

### Production

Production is the canonical memory store for all agents. Every agent's memory syncs here regardless of what they're working on.

- **Environment name**: `production`
- **Database service**: `pgvector`
- **Database public proxy**: `tramway.proxy.rlwy.net:48318`

Use production for:
- All status checks and debugging
- Resetting sessions that need re-indexing
- Validating tree structure and content

### PR Test Sandbox

PR environments are isolated sandboxes for testing memory service changes. They have their own database so testing can't corrupt production.

- **Environment name pattern**: `dynamic-summary-pr-{PR_NUMBER}`
- **Database service**: `pgvector-rW-f`
- **Database public proxy**: `nozomi.proxy.rlwy.net:30284`

Use PR environment only for:
- Testing sync/indexing code changes
- Running manual syncs of test transcripts
- Verifying tree-building before merge

## Railway Project Details

- **Project ID**: `9d168ba6-ac78-4739-a53c-7ca04e211678`
- **Project Name**: `magnificent-harmony`

## Admin CLI Reference

### Status Command

Shows database connection, sessions, indexing progress, tree structure, and validation:

```bash
python -m memory_service.admin status
```

Output includes:
- **Sessions**: All tracked sessions with sync progress
- **Embeddings**: Percentage of leaves with vector embeddings
- **Summaries**: Internal nodes created by merging sibling pairs
- **Forest**: Tree structure - complete when no two roots share the same height
- **Validation**: Tree structure checks + transcript content verification

### Reset Command

Clears sync cursor to trigger full re-index on next sync:

```bash
python -m memory_service.admin reset {session-uuid}
```

The reset preserves `span_end` but clears `last_synced_uuid`. Next sync detects this and rebuilds from scratch.

### Validate Command

Check tree invariants:

```bash
python -m memory_service.admin validate {session-id}
```

Or via ragzoom CLI:

```bash
ragzoom validate {session-uuid}
```

### Transcribe Command

Extract text from stored JSONL:

```bash
python -m memory_service.admin transcribe {session-id} [-o output.txt]
```

### Debug Commands

```bash
python -m memory_service.admin chain {session-id}        # Show ancestor chain
python -m memory_service.admin segments {session-id}     # Show segment boundaries
python -m memory_service.admin inspect-uuid {session-id} {uuid}
python -m memory_service.admin inspect-leaves {session-id} {offset}
```

## gRPC Endpoint

The MCP server connects via gRPC. The address is in `.mcp.json`:

```json
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

## Troubleshooting

### "Service not found" Error

Re-link to the project:

```bash
railway status  # Check current link
railway link -p 9d168ba6-ac78-4739-a53c-7ca04e211678 -e production
```

### Database Connection Failures

If you see `nodename nor servname provided`:
- You're using an internal URL from outside Railway
- Use `DATABASE_PUBLIC_URL` from the correct database service

### Tree Corruption / Validation Failures

Symptoms:
- `ragzoom validate` shows errors
- Indexing stalls
- Validation shows "Leaf X not in transcript"

Fix: Reset and re-index:
```bash
python -m memory_service.admin reset {session-uuid}
```

The JSONL transcript is the source of truth. Re-indexing rebuilds the tree from scratch.

### Deployment Issues

Railway auto-deploys on push. If deployment seems stuck:

```bash
# Check deployment status
railway deployment list --service dynamic-summary

# Check logs
railway logs --service dynamic-summary
```

**Never use** `railway deployment redeploy` - it redeploys old code.
