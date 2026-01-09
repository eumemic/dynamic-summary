---
name: Memory Service Operations
description: This skill should be used when the user asks to "check memory service status", "debug memory service", "check PR environment", "check production environment", "get database URL", "run admin commands", or mentions memory service, hosted service, or Railway operations.
---

# Memory Service Operations

Guidance for operating the hosted RagZoom memory service.

## Environment Model

**Production** is where all agents' memories live. **PR environment** is a test sandbox for memory service development only.

### Which Environment to Check?

1. **On `master`?** → Use production
2. **On any other branch?** → Check PR environment first (`--test`), then production if not found

### Environment Details

**Production** (default):
- All agents' real memories live here
- Use for: status checks, debugging memory issues, normal operations

**PR environment** (`--test`):
- Isolated test sandbox with separate database
- Use for: testing sync/indexing changes, manual test syncs, validating tree-building

Agents' own memories always sync to production, even when developing memory service changes. The JSONL transcript is the source of truth - production can always be re-indexed if needed.

## Wrapper Script (Recommended)

The `scripts/memory-admin` wrapper handles all Railway ceremony automatically:

```bash
# Production (default)
scripts/memory-admin status
scripts/memory-admin reset <session-id>
scripts/memory-admin validate <session-id>

# PR test environment
scripts/memory-admin status --test
scripts/memory-admin reset <session-id> --test
```

## Manual Workflow (If Needed)

### Production

```bash
railway link -p 9d168ba6-ac78-4739-a53c-7ca04e211678 -e production
railway variables --service pgvector --kv | grep DATABASE_PUBLIC_URL
RAGZOOM_DATABASE_URL="postgresql://..." python -m memory_service.admin status
```

### PR Test Environment

```bash
railway link -p 9d168ba6-ac78-4739-a53c-7ca04e211678 -e dynamic-summary-pr-{NUMBER}
railway variables --service pgvector-rW-f --kv | grep DATABASE_PUBLIC_URL
RAGZOOM_DATABASE_URL="postgresql://..." python -m memory_service.admin status
```

## Common Admin Commands

```bash
status                    # Service overview and session inventory
status <session-id>       # Status for a specific session (prefix OK)
reset <session-id>        # Reset session for full re-index
validate <session-id>     # Validate indexed content matches transcript
transcribe <session-id>   # Extract text from stored JSONL
```

## Deployment

The service auto-deploys when pushing to PR branches:
```bash
git push origin {branch}
```

**Never use** `railway deployment redeploy` - it redeploys old code.

## Additional Resources

- **`references/detailed-ops.md`** - Detailed procedures and troubleshooting
