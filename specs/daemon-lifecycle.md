---
status: READY
---

# Daemon Lifecycle Management

## Overview

RagZoom's gRPC server should run as a managed daemon with automatic lifecycle management. Users should never need to manually start the server - it starts automatically when needed and recovers from failures transparently.

## Goals

1. **Zero manual server management** - Server auto-starts on first client command
2. **Transparent crash recovery** - Stale/crashed servers are automatically replaced
3. **Clean shutdown** - Graceful stop with proper cleanup
4. **Observable state** - Easy to check if daemon is running and healthy

## Non-Goals

- Systemd/launchd service integration (future work)
- Multi-user daemon sharing
- Remote daemon management
- Idle timeout / auto-stop

## Architecture

### State Files

All daemon state lives in XDG-compliant directories:

```
~/.local/state/ragzoom/
├── daemon.pid          # PID of running daemon
├── daemon.port         # Port daemon is listening on
└── daemon.log          # Daemon stdout/stderr (rotated)
```

Environment variable `RAGZOOM_STATE_DIR` overrides the default location.

### Daemon Process

The daemon is a background process forked from the CLI:

```
ragzoom server start --daemon    # Fork to background, write PID file
ragzoom server start             # Foreground (current behavior, unchanged)
```

When `--daemon` is used:
1. Fork to background
2. Redirect stdout/stderr to `daemon.log`
3. Write PID to `daemon.pid`
4. Write port to `daemon.port`
5. Detach from terminal (setsid)

### Auto-Start

Any client command that needs the server triggers auto-start:

```python
def ensure_server_running() -> str:
    """Ensure daemon is running, return server address."""
    if is_server_healthy():
        return get_server_address()

    # Server not running or unhealthy - start it
    cleanup_stale_state()
    start_daemon()
    wait_for_healthy(timeout=30)
    return get_server_address()
```

Commands that trigger auto-start:
- `ragzoom index`
- `ragzoom query`
- `ragzoom clear`
- `ragzoom status`
- Any command with `--server-address` that uses default

Commands that do NOT auto-start:
- `ragzoom server start` (explicit start)
- `ragzoom server stop` (explicit stop)
- `ragzoom server status` (just reports)
- `ragzoom doctor` (diagnostic only)
- `ragzoom config` (local only)

### Health Check

Health check combines PID verification with gRPC probe:

```python
def is_server_healthy() -> bool:
    """Check if daemon is running and responsive."""
    pid = read_pid_file()
    if pid is None:
        return False

    if not is_process_running(pid):
        return False  # Stale PID file

    # Verify gRPC is responsive
    try:
        grpc_health_check(get_server_address(), timeout=2)
        return True
    except:
        return False
```

### Crash Recovery

When a command detects an unhealthy server:

1. Kill the stale process (if PID exists and process is running)
2. Remove stale state files
3. Start fresh daemon
4. Wait for healthy state
5. Proceed with original command

This is transparent to the user - they just see their command succeed.

### CLI Commands

#### `ragzoom server start`

```
ragzoom server start [--daemon] [--port PORT] [--host HOST]

Start the gRPC server.

Options:
  --daemon    Run as background daemon (default for auto-start)
  --port      Port to listen on (default: 50051)
  --host      Host to bind (default: 127.0.0.1)

Without --daemon, runs in foreground (current behavior).
```

#### `ragzoom server stop`

```
ragzoom server stop [--force]

Stop the running daemon.

Options:
  --force     Kill immediately (SIGKILL) instead of graceful (SIGTERM)

Sends SIGTERM, waits up to 10s for graceful shutdown.
If --force or timeout, sends SIGKILL.
Cleans up state files.
```

#### `ragzoom server status`

```
ragzoom server status

Show daemon status.

Output:
  Running: PID 12345, port 50051, uptime 2h 15m
  -- or --
  Not running
```

#### `ragzoom server logs`

```
ragzoom server logs [-f] [-n LINES]

Show daemon logs.

Options:
  -f          Follow log output (like tail -f)
  -n LINES    Number of lines to show (default: 50)
```

## Implementation Notes

### Process Daemonization

Use standard Unix daemonization:
1. Fork once
2. Call `setsid()` to become session leader
3. Fork again (prevents acquiring controlling terminal)
4. Close stdin, redirect stdout/stderr to log
5. Write PID file

Python's `daemon` library or manual implementation both work.

### Signal Handling

Daemon should handle:
- `SIGTERM` - Graceful shutdown (finish in-flight requests)
- `SIGINT` - Same as SIGTERM
- `SIGHUP` - Reload configuration (future)

### Exit Cleanup

**Critical:** State files (PID, port) must be cleaned up on ALL exit paths:

1. **Signal handlers** - SIGTERM/SIGINT already handled
2. **Normal exit** - Use `atexit.register()` to ensure cleanup when `run_server()` returns normally (e.g., when `server.wait_for_termination()` completes)
3. **Lease failure** - When lease acquisition fails with `sys.exit(1)`, cleanup must run first

```python
# In daemon mode, register atexit cleanup BEFORE run_server()
if daemon:
    daemonize()
    write_port_file(port)
    install_shutdown_handlers()
    atexit.register(cleanup_stale_state)  # Catches normal exits

run_server(options)
```

Without atexit cleanup, stale PID/port files persist after normal server termination, breaking subsequent auto-start attempts.

### Config Persistence

Auto-started daemons need configuration to work correctly. Persist config in well-known location:

```
~/.local/state/ragzoom/
├── daemon.pid
├── daemon.port
├── daemon.log
└── daemon.config.json    # Persisted config for auto-started daemons
```

When manually starting with `--config`:
1. Copy relevant config to `daemon.config.json`
2. Auto-start uses this config if present

When no config file present, auto-start uses defaults.

**Key config fields to persist:**
- `target_chunk_tokens` - Critical for temporal documents (must be `null`)
- `summary_system_prompt` / `summarization_guidance` - Document-specific prompts
- Database connection settings

Environment variable `RAGZOOM_DAEMON_CONFIG` overrides the default location.

### Log Rotation

Keep last 5 log files, 10MB each:
- `daemon.log` (current)
- `daemon.log.1` through `daemon.log.4`

### Port Selection

Default port: 50051

If port is in use:
- Auto-start: Try ports 50051-50060, use first available
- Explicit start: Fail with clear error

Store actual port in `daemon.port` file.

## Testing

### Unit Tests

- PID file read/write/cleanup
- Health check logic (mock gRPC)
- Auto-start trigger detection
- atexit cleanup registered in daemon mode
- Config persistence read/write

### Integration Tests

- Start daemon, verify PID file created
- Health check passes for running daemon
- Kill daemon, verify auto-restart on next command
- Graceful stop cleans up state files
- Concurrent commands don't race on auto-start
- **Normal server exit cleans up state files** (atexit path)
- **Lease failure cleans up state files before exit**
- Config persistence survives daemon restart

### Manual Testing

```bash
# Fresh start
ragzoom query "test" -d doc.txt  # Should auto-start daemon

# Check status
ragzoom server status  # Should show running

# Kill and recover
kill $(cat ~/.local/state/ragzoom/daemon.pid)
ragzoom query "test" -d doc.txt  # Should auto-restart

# Clean stop
ragzoom server stop
ragzoom server status  # Should show not running
```

## Rollout

1. Add daemon infrastructure (PID files, health check, start/stop)
2. Add `ragzoom server status/stop/logs` commands
3. Add auto-start to client commands (behind feature flag initially)
4. Remove feature flag, make auto-start default
