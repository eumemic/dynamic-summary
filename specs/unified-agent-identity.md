---
status: COMPLETE
---

# Unified Agent Identity

## Overview

Unify how the Claude Code integration components (sync script and MCP server) discover their target document. Support two identity models:

1. **Configured identity** (Jarvis/Legion): Agent knows its document_id at startup
2. **Discovered identity** (Claude Code): Session ID discovered via PID-keyed lookup

This eliminates the accumulated state files while enabling persistent agent memory across sessions.

## Problem Statement

The current integration uses state files for session discovery:

```
data/transcript-state/<session-id>.jsonl → {document_id, last_pid}
```

Problems:
1. **Accumulation**: 116+ state files build up, never cleaned
2. **Scanning**: MCP server scans all files to find matching PID
3. **Single identity model**: Only supports session-based (Claude Code) identity
4. **No persistent agents**: Can't accumulate memory across sessions (Jarvis/Legion model)

## Requirements

### 1. Environment Variable for Configured Identity

Both sync script and MCP server check `RAGZOOM_DOCUMENT_ID` first:

```python
document_id = os.environ.get("RAGZOOM_DOCUMENT_ID")
```

When set:
- Sync writes to that document (regardless of JSONL filename)
- MCP queries that document (no PID discovery needed)

**Use case**: Jarvis spawns both components with `RAGZOOM_DOCUMENT_ID=jarvis-<user-id>`. Multiple conversations write to and query the same persistent document.

### 2. PID-Keyed Temp File for Discovered Identity

When `RAGZOOM_DOCUMENT_ID` is not set (Claude Code model):

**SessionStart hook** writes:
```
/tmp/ragzoom-session-<claude-code-pid>
```

Contents: just the session_id (document_id)

**MCP server** reads:
```python
def _discover_session_id() -> str:
    pid = os.getppid()  # Claude Code's PID
    path = Path(f"/tmp/ragzoom-session-{pid}")
    return path.read_text().strip()
```

**Benefits**:
- Ephemeral (OS cleans `/tmp`)
- Direct lookup (no scanning)
- PID-scoped (no collision)

### 3. Sync Script Identity Resolution

Priority order:
1. `--document-id` CLI flag (explicit override)
2. `RAGZOOM_DOCUMENT_ID` env var (configured identity)
3. `jsonl_path.stem` (session-based, current behavior)

```python
@cli.command("sync")
@click.argument("jsonl_path", type=click.Path(exists=True, path_type=Path))
@click.option("--document-id", "-d", envvar="RAGZOOM_DOCUMENT_ID", default=None)
def sync_cmd(jsonl_path: Path, document_id: str | None) -> None:
    doc_id = document_id or jsonl_path.stem
    # ... rest of sync
```

### 4. MCP Server Identity Resolution

Priority order:
1. `RAGZOOM_DOCUMENT_ID` env var (configured identity)
2. PID temp file lookup (discovered identity)

```python
def _get_document_id() -> str:
    # Configured identity (Jarvis/Legion)
    if doc_id := os.environ.get("RAGZOOM_DOCUMENT_ID"):
        return doc_id

    # Discovered identity (Claude Code)
    return _read_pid_temp_file(os.getppid())
```

### 5. Hook Updates

The SessionStart hook writes the PID temp file instead of state file:

**Before (accumulates):**
```bash
ragzoom-claude-code set-pid "$SESSION_ID" "$PPID"
# Writes: data/transcript-state/<session-id>.jsonl
```

**After (ephemeral):**
```bash
echo "$SESSION_ID" > "/tmp/ragzoom-session-$PPID"
```

The `set-pid` CLI command can be removed or deprecated.

### 6. Remove State File Machinery

Remove from `transcript_sync.py`:
- `get_state_path()` function
- `set_session_pid()` function
- `get_session_document_id()` function
- State directory creation logic

Remove from CLI:
- `set-pid` command (or deprecate with warning)

Update `reset` command:
- Remove state file cleanup (no longer exists)

### 7. Backward Compatibility

Old state files are ignored. They can be manually deleted:
```bash
rm -rf data/transcript-state/
```

Existing RagZoom documents are unaffected - the identity change is only about discovery, not content.

## API Changes

### CLI

**sync command:**
```diff
 @cli.command("sync")
 @click.argument("jsonl_path", type=click.Path(exists=True, path_type=Path))
+@click.option("--document-id", "-d", envvar="RAGZOOM_DOCUMENT_ID", default=None,
+              help="Override document ID (default: JSONL filename stem)")
 @click.option("--server-address", ...)
-def sync_cmd(jsonl_path: Path, server_address: str) -> None:
+def sync_cmd(jsonl_path: Path, document_id: str | None, server_address: str) -> None:
```

**set-pid command:**
- Remove or deprecate (hook writes temp file directly)

### transcript_sync.py

**Remove:**
```python
def get_state_path(document_id: str) -> Path: ...
def set_session_pid(document_id: str, pid: int) -> None: ...
def get_session_document_id(pid: int) -> str | None: ...
```

### mcp_server.py

**Replace:**
```python
def _get_session_id() -> str:
    # OLD: Scan state files for matching PID
    # NEW: Check env var, then read PID temp file
    if doc_id := os.environ.get("RAGZOOM_DOCUMENT_ID"):
        return doc_id

    pid = os.getppid()
    temp_path = Path(f"/tmp/ragzoom-session-{pid}")
    if not temp_path.exists():
        raise ValueError(
            f"No session found for PID {pid}. "
            "Either set RAGZOOM_DOCUMENT_ID or ensure SessionStart hook ran."
        )
    return temp_path.read_text().strip()
```

### Hook (.claude/hooks/session-start.sh)

**Replace `ragzoom-claude-code set-pid` with:**
```bash
echo "$SESSION_ID" > "/tmp/ragzoom-session-$PPID"
```

## Implementation Outline

### Phase 1: Add Environment Variable Support

1. Add `--document-id` option to sync CLI (with `envvar="RAGZOOM_DOCUMENT_ID"`)
2. Update `execute_sync()` signature to accept explicit document_id
3. Add env var check to MCP server's `_get_session_id()`
4. Tests for env var precedence in both components

### Phase 2: PID Temp File Discovery

1. Update hook to write `/tmp/ragzoom-session-$PPID`
2. Update MCP server to read PID temp file as fallback
3. Tests for PID-based discovery

### Phase 3: Remove State File Machinery

1. Remove `get_state_path()`, `set_session_pid()`, `get_session_document_id()`
2. Remove `set-pid` CLI command
3. Update `reset` command to not reference state files
4. Delete `data/transcript-state/` directory

## Acceptance Criteria

1. ⬚ `RAGZOOM_DOCUMENT_ID` env var works for sync script
2. ⬚ `RAGZOOM_DOCUMENT_ID` env var works for MCP server
3. ⬚ `--document-id` CLI flag overrides env var and stem
4. ⬚ PID temp file discovery works for Claude Code (no env var set)
5. ⬚ No state files accumulate in `data/transcript-state/`
6. ⬚ Multiple syncs to same document work (Jarvis model)
7. ⬚ MCP queries work with configured identity (Jarvis model)
8. ⬚ `reset` command works without state file cleanup

## Identity Model Summary

| Model | sync document_id | MCP document_id |
|-------|-----------------|-----------------|
| **Jarvis** | `RAGZOOM_DOCUMENT_ID` env | `RAGZOOM_DOCUMENT_ID` env |
| **Claude Code** | `jsonl_path.stem` | PID temp file lookup |
| **Explicit** | `--document-id` flag | N/A |

## Non-Goals

- **Migration tool**: Old state files are simply ignored/deleted
- **Hybrid identity**: Each session uses one model, not a mix
- **Document creation**: This spec is about identity, not lifecycle
- **Server address discovery**: Already handled via `RAGZOOM_SERVER_ADDRESS`

## Dependencies

- `stateless-transcript-sync.md`: Sync algorithm (COMPLETE)
- `timestamped-transcript-sync.md`: Turn-based indexing (COMPLETE)

## Future Considerations

For Legion (hierarchical agent orgs), document_id may be assigned by the orchestrator when spawning agents. The env var approach supports this - the parent process sets `RAGZOOM_DOCUMENT_ID` before spawning the agent.
