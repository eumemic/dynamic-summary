---
status: COMPLETE
---

# Event-Driven Daemon Tests

## Problem

Daemon lifecycle tests are the slowest in the test suite (1.4-2.3s each) due to hardcoded `time.sleep()` calls:

- `time.sleep(0.5)` after every `proc.wait()` to wait for daemon readiness
- `time.sleep(0.3)` in test scripts for daemon startup
- Polling loops with `time.sleep(0.1)` intervals

These sleeps are conservative estimates that waste time on fast machines and can still be flaky on slow CI runners. The tests are fundamentally non-deterministic.

## Goal

Eliminate all sleep-based synchronization in daemon tests by making daemon readiness event-driven. Tests should block only as long as actually needed and fail immediately on daemon crash.

## Non-Goals

- Changing production daemon behavior (only test infrastructure)
- Parallelizing daemon tests (orthogonal concern)
- Mocking the daemon entirely (we want real fork/exec coverage)

## Solution: Ready-Pipe Pattern

The daemon signals readiness by writing to a pipe. The test blocks on reading from the pipe. This is instant on success and detects crashes immediately via EOF.

### API Addition

Add an optional `ready_fd` parameter to `daemonize()`:

```python
def daemonize(
    log_file: Path,
    ready_fd: int | None = None,  # NEW: file descriptor to signal readiness
) -> None:
    """Fork to background and become a daemon.

    Args:
        log_file: Path for stdout/stderr redirection.
        ready_fd: If provided, write b"R" to this fd after daemonization
                  completes, then close it. Used for synchronization in tests.
    """
```

### Daemon Side

After completing all daemonization steps (fork, setsid, redirect stdio, write PID file):

```python
if ready_fd is not None:
    os.write(ready_fd, b"R")
    os.close(ready_fd)
```

### Test Side

```python
def test_daemonize_forks_to_background(self, tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()

    script = f"""
    ...
    daemonize(Path("{log_file}"), ready_fd={write_fd})
    ...
    """

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        pass_fds=(write_fd,),  # Pass write end to child
        ...
    )
    os.close(write_fd)  # Close our copy of write end

    # Block until daemon signals ready (or crash → EOF)
    ready = os.read(read_fd, 1)
    os.close(read_fd)

    if ready != b"R":
        raise AssertionError("Daemon crashed before signaling ready")

    # Daemon is now guaranteed ready - no sleep needed
    assert flag_file.exists()
```

### Benefits

| Aspect | Before (sleep) | After (pipe) |
|--------|---------------|--------------|
| Latency | Fixed 0.5s minimum | Actual startup time (~50ms) |
| Reliability | Can be flaky | Deterministic |
| Crash detection | Silent timeout | Immediate EOF |
| CI behavior | Same slow speed | As fast as possible |

### Test Utility

Create a helper to reduce boilerplate:

```python
# tests/conftest.py or tests/daemon_test_utils.py

@contextmanager
def daemon_ready_pipe() -> Iterator[tuple[int, int]]:
    """Context manager that creates a ready-signal pipe.

    Yields (read_fd, write_fd). The write_fd should be passed to
    the daemon via pass_fds. The read_fd blocks until daemon writes.

    Example:
        with daemon_ready_pipe() as (read_fd, write_fd):
            proc = subprocess.Popen(..., pass_fds=(write_fd,))
            os.close(write_fd)
            wait_for_daemon_ready(read_fd)
    """
    read_fd, write_fd = os.pipe()
    try:
        yield read_fd, write_fd
    finally:
        # Clean up any unclosed fds
        for fd in (read_fd, write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def wait_for_daemon_ready(read_fd: int, timeout: float = 5.0) -> None:
    """Block until daemon signals ready or timeout.

    Args:
        read_fd: Read end of ready pipe.
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If daemon doesn't signal within timeout.
        AssertionError: If daemon crashes (EOF on pipe).
    """
    import select

    ready, _, _ = select.select([read_fd], [], [], timeout)
    if not ready:
        raise TimeoutError(f"Daemon did not signal ready within {timeout}s")

    data = os.read(read_fd, 1)
    if data != b"R":
        raise AssertionError("Daemon crashed before signaling ready (got EOF)")
```

## Migration

### Phase 1: Add Infrastructure

1. Add `ready_fd` parameter to `daemonize()`
2. Add `daemon_ready_pipe()` and `wait_for_daemon_ready()` test utilities
3. Verify existing tests still pass (ready_fd=None is backward compatible)

### Phase 2: Migrate Tests

Convert each test class:

1. `TestDaemonizeFunction` - 6 tests using `time.sleep(0.5)`
2. `TestDaemonizeIntegration` - 1 test using `time.sleep(0.8)`
3. `TestSignalHandlers` - 2 tests using polling loops + `time.sleep(0.5)`
4. `TestAtexitCleanup` - 2 tests using polling loops + `time.sleep(0.5)`
5. `TestStartServerAtexitIntegration` - 3 tests using `time.sleep(1.5)`

### Phase 3: Remove Sleeps

After all tests are migrated, grep for remaining `time.sleep` in daemon tests and remove any that are now unnecessary.

## Edge Cases

### Signal Handler Tests

Tests that send signals to the daemon need to:
1. Wait for daemon ready (via pipe)
2. Send signal
3. Wait for daemon to process signal and exit

For step 3, use `os.waitpid()` with `WNOHANG` in a tight loop, or poll for the cleanup marker file with a short timeout.

### Multiple Ready Signals

Some tests verify the daemon updates state after startup (e.g., writes "started" then "still_running"). For these, use multiple marker files or a sequence of pipe writes:

```python
# Daemon writes sequence
os.write(ready_fd, b"1")  # Phase 1 complete
time.sleep(0.1)  # Do some work
os.write(ready_fd, b"2")  # Phase 2 complete

# Test reads sequence
assert os.read(read_fd, 1) == b"1"
assert os.read(read_fd, 1) == b"2"
```

### Atexit Tests

Tests that verify atexit cleanup need to wait for the daemon process to fully exit. Use `proc.wait()` on the parent, then poll briefly for the daemon to clean up, or have the daemon write a "cleanup complete" marker.

## Testing the Tests

After migration, verify:

1. **Speed improvement**: Run `pytest --durations=10` - daemon tests should drop from ~1.4s to <0.3s each
2. **No flakiness**: Run daemon tests 50x in a loop to verify determinism
3. **Crash detection works**: Modify a test to crash the daemon early, verify test fails immediately (not after timeout)

## Acceptance Criteria

- [ ] `daemonize()` accepts optional `ready_fd` parameter
- [ ] All daemon tests use ready-pipe pattern instead of `time.sleep()`
- [ ] No `time.sleep()` calls remain in `test_daemon_lifecycle.py` or `test_daemon_atexit.py`
- [ ] Daemon test suite runs in <5s total (currently ~15s)
- [ ] Tests detect daemon crashes immediately (no silent timeouts)
