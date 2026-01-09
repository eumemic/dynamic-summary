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

## Network Endpoints

Railway exposes services via two different mechanisms. Understanding the difference is critical.

### TCP Proxy (for gRPC)

gRPC uses TCP proxy, **not** HTTPS. Discover the address from Railway variables:

```bash
# Link to environment first
railway link -p 9d168ba6-ac78-4739-a53c-7ca04e211678 -e production  # or -e dynamic-summary-pr-{N}

# Get gRPC address components
railway variables --service dynamic-summary --kv | grep RAILWAY_TCP_PROXY
# RAILWAY_TCP_PROXY_DOMAIN=<hostname>
# RAILWAY_TCP_PROXY_PORT=<port>
```

The gRPC address is `${RAILWAY_TCP_PROXY_DOMAIN}:${RAILWAY_TCP_PROXY_PORT}`.

### HTTPS Domain (for REST)

HTTPS endpoints use `RAILWAY_PUBLIC_DOMAIN:443`. The memory service doesn't expose REST endpoints, so this is unused for our purposes.

**Common mistake**: Using `RAILWAY_PUBLIC_DOMAIN:443` for gRPC results in 502 errors because there's no HTTP endpoint on the gRPC service.

### MCP Server Configuration

The MCP server address in `.mcp.json` should match the production gRPC endpoint. Discover it with:

```bash
railway link -p 9d168ba6-ac78-4739-a53c-7ca04e211678 -e production
railway variables --service dynamic-summary --kv | grep RAILWAY_TCP_PROXY
```

Then update `.mcp.json` with the discovered `DOMAIN:PORT`.

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

### gRPC 502 Errors

Symptoms:
- `test-sync` fails with "Received http2 header with status: 502"
- gRPC calls return UNAVAILABLE

Cause: Using HTTPS domain instead of TCP proxy for gRPC.

Fix: Ensure gRPC uses `RAILWAY_TCP_PROXY_DOMAIN:RAILWAY_TCP_PROXY_PORT`, not `RAILWAY_PUBLIC_DOMAIN:443`.

Verify correct address:
```bash
railway variables --service dynamic-summary --kv | grep TCP_PROXY
```

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
