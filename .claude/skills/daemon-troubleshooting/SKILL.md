---
description: This skill should be used when the user reports "daemon won't start", "daemon timeout", "server not responding", "timed out waiting for healthy state", "stale daemon", "failed to acquire indexer lease", or mentions daemon troubleshooting issues.
---

# Daemon Troubleshooting

Diagnose and fix ragzoom daemon state issues when commands fail with timeout or connection errors.

## Symptoms

- "Daemon failed to start: timed out after 30.0s waiting for healthy state"
- "gRPC UNAVAILABLE: Connection refused"
- `ragzoom server status` says "Not running" but processes exist
- Commands hang or timeout unexpectedly
- "Failed to acquire indexer lease after 90s"

## Diagnostic Steps

### 1. Check for Running Processes

```bash
pgrep -f "ragzoom.*server" && ps aux | grep ragzoom | grep -v grep
```

Note the PIDs of any running processes.

### 2. Check State Files

```bash
ls -la ~/.local/state/ragzoom/
cat ~/.local/state/ragzoom/daemon.pid 2>/dev/null
cat ~/.local/state/ragzoom/daemon.port 2>/dev/null
```

**Compare:** Does the PID in `daemon.pid` match any running process? If not, state is stale.

### 3. Check Daemon Logs

```bash
tail -30 ~/.local/state/ragzoom/daemon.log
```

Look for:
- "Failed to acquire indexer lease" - lease contention from stale processes
- "Starting RagZoom gRPC server" - confirms server started successfully
- "Acquiring global indexer lease" without success - server still starting

## Resolution

### Clear Stale Indexer Lease

If daemon won't start with "Failed to acquire indexer lease after 90s":

```bash
# Clear leases without full nuclear reset
sqlite3 data/sqlite.db "DELETE FROM indexer_leases;"

# Then start normally
ragzoom server start --daemon --config ~/.local/state/ragzoom/daemon.config.json
```

This is faster than a nuclear reset when only the lease is stale (no process/state file issues).

### Nuclear Reset (Most Reliable)

When state is corrupted, do a full reset:

```bash
# Kill all ragzoom processes
pkill -9 -f "ragzoom"
sleep 2

# Clear all state files
rm -rf ~/.local/state/ragzoom/*

# Start fresh
ragzoom server start --daemon --config <your-config.json>

# Wait for lease acquisition (~5-10s)
sleep 5

# Verify
ragzoom server status
ragzoom documents
```

### Why Timeouts Happen

1. **Stale processes** - Old daemons from previous sessions still running, holding the indexer lease
2. **State file mismatch** - PID/port files don't match actual running processes
3. **Lease acquisition** - After restart, server needs 5-10s to acquire the indexer lease before gRPC listens
4. **Health check timing** - Auto-start checks health before server is fully ready
5. **Stale code** - All ragzoom operations (including `ragzoom-claude-code sync`) go through gRPC to the daemon. With editable installs, the daemon doesn't pick up source changes until restarted.

### Prevention

- Always use `ragzoom server stop` before ending a session
- If using `--config`, the config is persisted to `daemon.config.json` for auto-start
- Don't run multiple terminal sessions that each try to start daemons

## Dev/Prod Code Separation

Production (`ragzoom`) must be a **non-editable install** so code changes don't affect the running daemon until explicitly reinstalled.

### Verify Production Install

```bash
pip show ragzoom | grep Editable
```

- **No output** = correct (non-editable)
- **Shows "Editable project location"** = wrong (will break on code changes)

### Fix Editable Install

```bash
pip uninstall ragzoom -y && pip install /Users/tom/code/dynamic-summary
```

### After Merging Daemon-Affecting Changes

When merging changes to files the daemon loads (`ragzoom/retrieve.py`, `ragzoom/server/`, etc.):

1. Verify production is non-editable (check above)
2. If editable, reinstall as non-editable
3. Restart production server to pick up new code: `ragzoom server stop && ragzoom server start`

Development always uses `python -m ragzoom.cli` which runs from source.
