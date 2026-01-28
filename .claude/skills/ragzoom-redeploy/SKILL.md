---
description: This skill should be used when the user asks to "redeploy ragzoom", "reinstall ragzoom", "reset ragzoom database", "reindex sessions", "update production ragzoom", or mentions needing to deploy code changes to the local ragzoom daemon.
---

# Redeploy RagZoom Production

Workflow for redeploying the local production ragzoom daemon after code changes.

## Prerequisites

- Code changes merged to master
- Working directory: `/Users/tom/code/dynamic-summary`

## Redeploy Steps

### 1. Stop Server

```bash
ragzoom server stop
```

### 2. Reinstall Packages (Non-Editable)

Both core and integration packages must be reinstalled:

```bash
pip uninstall ragzoom ragzoom-claude-code -y
pip install /Users/tom/code/dynamic-summary
pip install /Users/tom/code/dynamic-summary/integrations/claude-code
```

**Important:** Production MUST be non-editable (`pip install .`, not `pip install -e .`). This ensures code changes don't affect the running daemon until explicitly reinstalled.

### 3. Start Daemon

The daemon needs `OPENAI_API_KEY` from `.env`:

```bash
cd /Users/tom/code/dynamic-summary
source .env && ragzoom server start --daemon
```

### 4. Verify

```bash
ragzoom server status   # Should show "Running"
ragzoom doctor          # Should show all green
ragzoom documents       # List indexed documents
```

## Optional: Reset Database

If schema changes require reindexing all sessions:

```bash
ragzoom clear --confirm
```

This clears all documents. Sessions will reindex on next sync.

## Quick One-Liner

For routine redeploys:

```bash
ragzoom server stop; pip uninstall ragzoom ragzoom-claude-code -y && pip install /Users/tom/code/dynamic-summary && pip install /Users/tom/code/dynamic-summary/integrations/claude-code && source .env && ragzoom server start --daemon && sleep 3 && ragzoom server status
```

## Troubleshooting

If daemon fails to start, check:

1. **API key missing:** Ensure `source .env` was run
2. **Stale lease:** See `daemon-troubleshooting` skill
3. **Port in use:** Check `lsof -i :50051`
