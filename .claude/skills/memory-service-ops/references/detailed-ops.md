# Railway Operations - Detailed Reference

## Project Details

- **Project ID**: `9d168ba6-ac78-4739-a53c-7ca04e211678`
- **Project Name**: `magnificent-harmony`

## Environment Details

### Production Environment

- **Environment name**: `production`
- **Database service**: `pgvector`
- **Database internal URL**: `postgresql://postgres:{password}@pgvector.railway.internal:5432/railway`
- **Database public proxy**: `tramway.proxy.rlwy.net:48318`

### PR Environments

- **Environment name pattern**: `dynamic-summary-pr-{PR_NUMBER}`
- **Database service**: `pgvector-rW-f`
- **Database internal URL**: `postgresql://postgres:{password}@pgvector-rw-f.railway.internal:5432/railway`
- **Database public proxy**: `nozomi.proxy.rlwy.net:30284`

## Key Services

| Service | Purpose | Notes |
|---------|---------|-------|
| `dynamic-summary` | gRPC server for memory ingestion | Main application service |
| `pgvector` | PostgreSQL with pgvector (production) | Production database |
| `pgvector-rW-f` | PostgreSQL with pgvector (PR envs) | PR environment database |

## gRPC Endpoint

The gRPC service address is configured in `.mcp.json` under `RAGZOOM_SERVER_ADDRESS`. This address is Railway's TCP proxy for the gRPC service.

To verify which endpoint has data:

```python
import grpc
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc

address = 'switchback.proxy.rlwy.net:11553'  # from .mcp.json
channel = grpc.insecure_channel(address)
stub = pb2_grpc.WorkerServiceStub(channel)
req = pb2.GetDocumentRequest(document_id='YOUR_SESSION_ID')
resp = stub.GetDocument(req, timeout=10, metadata=[('user_id', 'tom')])
print(f'Leaves: {resp.status.leaf_count}, Depth: {resp.status.tree_depth}')
```

## Admin CLI Commands

### Status Command

Shows database, sessions, indexing progress, tree structure, job queue, and validation:

```bash
python -m memory_service.admin status
```

Key metrics:
- **Embeddings**: Leaves with vector embeddings (needed for semantic search)
- **Summaries**: Internal nodes created by merging sibling pairs
- **Forest**: Multiple perfect binary trees. Complete when no two roots share the same height.
- **Validation**: Two-layer (tree structure + transcript content)

### Reset Command

Clears sync cursor to trigger full re-index on next sync:

```bash
python -m memory_service.admin reset {session-uuid}
```

### Validate Command

Check tree invariants:

```bash
ragzoom validate {session-uuid}
```

### Debug Commands

```bash
python -m memory_service.admin chain {session-id}        # Show ancestor chain
python -m memory_service.admin segments {session-id}     # Show segment boundaries
python -m memory_service.admin inspect-uuid {session-id} {uuid}
python -m memory_service.admin inspect-leaves {session-id} {offset}
```

## Finding Session/Document IDs

Sessions are identified by UUIDs. For Claude Code sessions:

```bash
ls -lt ~/.claude/projects/-Users-tom-code-dynamic-summary-worktrees-worktree-1/*.jsonl | head -1
```

The session ID is the filename (without `.jsonl` extension).

## Deployment Workflow

Railway deploys automatically when pushing to the PR branch:

```bash
# Push to the PR branch - Railway auto-deploys
git push origin {branch-name}

# Check deployment status
railway deployment list --service dynamic-summary
```

**Never use**:
- `railway deployment redeploy` - redeploys old code
- `railway deploy` - doesn't pick up new code from git

## Troubleshooting

### "Service not found" Error

Re-link to the project:

```bash
railway status  # Check current link
railway link -p 9d168ba6-ac78-4739-a53c-7ca04e211678 -e {environment}
```

### Database Connection Failures

If you see `nodename nor servname provided`:
- Using internal URL from outside Railway
- Use `DATABASE_PUBLIC_URL` instead

### Orphaned Roots / Tree Corruption

Symptoms:
- `ragzoom validate` shows "Parentless" nodes > 0
- Indexing stalls with jobs stuck

Fix with reset and re-index:
```bash
python -m memory_service.admin reset {session-uuid}
```

## MCP Server Configuration

The memory MCP server connects to Railway via gRPC:

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
