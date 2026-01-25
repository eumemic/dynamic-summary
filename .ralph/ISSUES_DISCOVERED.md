# Issues Discovered During Verification

*Discovered: 2026-01-23 during implementation plan verification*

## 1. Daemon Crashes Without Cleanup

**Symptom:** After indexing completes, daemon process exits but PID/port files remain. Subsequent commands fail with "Daemon failed to start: timed out after 30.0s waiting for healthy state".

**Facts:**
- Signal handlers (SIGTERM/SIGINT) contain cleanup code that removes PID/port files
- Normal process exit doesn't trigger signal handlers
- When `server.wait_for_termination()` returns normally, process exits without cleanup
- Stale state files break auto-start detection

**Location:** `ragzoom/daemon.py:260-291`, `ragzoom/cli.py:1785-1807`

---

## 2. Temporal Documents Require Manual Config

**Symptom:** `ragzoom sync-claude-code-transcript` fails with "Temporal documents require client-controlled chunking (target_chunk_tokens=None)"

**Facts:**
- Default config has `target_chunk_tokens: 200`
- Temporal documents (with timestamps) require `target_chunk_tokens: null`
- Server must be started with `--config` pointing to a file with `{"target_chunk_tokens": null}`
- No auto-detection or per-request override exists

**Location:** `ragzoom/server/append_executor.py:481-486`, `ragzoom/default_config.json`

---

## 3. Auto-Start Ignores Server Config

**Symptom:** Even with server manually started with `--config temporal-config.json`, CLI commands trigger auto-start which ignores the config.

**Facts:**
- `ensure_server_running()` spawns daemon with `ragzoom server start --daemon`
- No mechanism to pass or persist config for auto-started daemons
- User workaround: always use `--server-address 127.0.0.1:50051` to bypass auto-start

**Location:** `ragzoom/daemon.py:423-525` (`ensure_server_running`, `start_daemon`)

---

## 4. Database Schema Migration Missing

**Symptom:** After adding `summary_system_prompt` column, old databases fail with "no such column: documents.summary_system_prompt"

**Facts:**
- New field added to Document model in `ragzoom/models.py`
- No migration script or auto-migration
- User must manually delete database to get new schema
- Affects both SQLite and PostgreSQL backends

**Location:** `ragzoom/models.py:169-171`

---

## 5. Custom Prompt Replaces Instead of Appends

**Symptom:** Users can accidentally break summarization by providing a custom prompt that doesn't include "output ONLY the compressed text, nothing else"

**Facts:**
- `summary_system_prompt` completely replaces the default system prompt
- Spec updated to change behavior to append under "# Summarization Guidance" section
- Spec also renames field to `summarization_guidance`
- Implementation not yet updated to match spec

**Location:** `specs/custom-prompt-config.md`, `ragzoom/services/summary_utils.py:196-206`

---

## 6. Lease Acquisition Failures

**Symptom:** Server logs show "Failed to acquire indexer lease after 90s (45 attempts)" then exits

**Facts:**
- Lease contention occurs when previous server didn't release lease cleanly
- Related to Issue #1 (daemon crash without cleanup)
- `sys.exit(1)` in lease failure path also doesn't cleanup PID/port files

**Location:** `ragzoom/server/app.py:179`, `ragzoom/server/lease.py`
