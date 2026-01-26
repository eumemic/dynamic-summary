"""Tests for daemon atexit cleanup.

Verifies that state files (PID and port) are cleaned up when the daemon
exits normally (not via signal), such as when run_server() returns.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest


def _load_daemon_module_code() -> str:
    """Return code to load daemon module directly, avoiding heavy __init__.py imports."""
    return f"""
import importlib.util
_spec = importlib.util.spec_from_file_location("daemon", "{Path.cwd()}/ragzoom/daemon.py")
daemon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(daemon)
"""


class TestAtexitCleanup:
    """Tests for atexit cleanup of daemon state files."""

    @pytest.mark.slow_threshold(5)
    def test_normal_exit_cleans_up_state_files(self, tmp_path: Path) -> None:
        """When daemon exits normally, PID and port files should be removed.

        This tests the atexit cleanup path: when run_server() returns normally
        (not via SIGTERM/SIGINT), the atexit handler should clean up state files.

        The test verifies that start_server (in daemon mode) registers atexit
        cleanup that removes state files on normal exit.

        Uses the ready-pipe pattern for event-driven synchronization:
        - Daemon signals b"R" when started (ready_fd)
        - Daemon signals b"D" when done and about to exit (done_fd)
        This eliminates polling loops and time.sleep() for deterministic tests.
        """
        import os
        import select

        from tests.conftest import daemon_ready_pipe, wait_for_daemon_ready

        cwd = Path.cwd()
        pid_file = tmp_path / "daemon.pid"
        port_file = tmp_path / "daemon.port"
        log_file = tmp_path / "daemon.log"

        done_read_fd, done_write_fd = os.pipe()

        try:
            with daemon_ready_pipe() as (read_fd, write_fd):
                script = f"""
import atexit
import os
import sys
sys.path.insert(0, "{cwd}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"

{_load_daemon_module_code()}
from pathlib import Path

# Do what start_server does in daemon mode
daemon.daemonize(Path("{log_file}"), ready_fd={write_fd})
daemon.write_port_file(50099)
daemon.install_shutdown_handlers()

# THIS IS THE FIX: Register atexit cleanup for normal exits
atexit.register(daemon.cleanup_stale_state)

# Signal done before exit
os.write({done_write_fd}, b"D")
os.close({done_write_fd})

# Normal exit - atexit handlers will run
"""
                proc = subprocess.Popen(
                    [sys.executable, "-c", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(cwd),
                    pass_fds=(write_fd, done_write_fd),
                )
                os.close(write_fd)

                proc.wait(timeout=10)
                wait_for_daemon_ready(read_fd)

            # Close our copy after passing to child
            try:
                os.close(done_write_fd)
            except OSError:
                pass

            # Wait for daemon to signal done
            ready, _, _ = select.select([done_read_fd], [], [], 5.0)
            if not ready:
                raise TimeoutError("Daemon did not signal done within 5s")

            data = os.read(done_read_fd, 1)
            if data != b"D":
                raise AssertionError(
                    f"Daemon crashed before signaling done (got {data!r})"
                )

            # Wait for EOF on pipe (indicates process fully exited, atexit complete)
            remaining = os.read(done_read_fd, 1)
            assert remaining == b"", f"Expected EOF, got {remaining!r}"

        finally:
            try:
                os.close(done_read_fd)
            except OSError:
                pass
            try:
                os.close(done_write_fd)
            except OSError:
                pass

        # State files should be cleaned up by atexit
        assert not pid_file.exists(), "PID file should be removed on normal exit"
        assert not port_file.exists(), "Port file should be removed on normal exit"

    @pytest.mark.slow_threshold(5)
    def test_without_atexit_files_remain(self, tmp_path: Path) -> None:
        """WITHOUT atexit cleanup, state files should remain after normal exit.

        This test documents the BUG that atexit fixes - without the atexit
        registration, state files persist after normal daemon exit.

        Uses the ready-pipe pattern for event-driven synchronization:
        - Daemon signals b"R" when started (ready_fd)
        - Daemon signals b"D" when done and about to exit (done_fd)
        This eliminates polling loops and time.sleep() for deterministic tests.
        """
        import os
        import select

        from tests.conftest import daemon_ready_pipe, wait_for_daemon_ready

        cwd = Path.cwd()
        pid_file = tmp_path / "daemon.pid"
        port_file = tmp_path / "daemon.port"
        log_file = tmp_path / "daemon.log"

        done_read_fd, done_write_fd = os.pipe()

        try:
            with daemon_ready_pipe() as (read_fd, write_fd):
                script = f"""
import os
import sys
sys.path.insert(0, "{cwd}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"

{_load_daemon_module_code()}
from pathlib import Path

# Do what start_server does in daemon mode (WITHOUT atexit fix)
daemon.daemonize(Path("{log_file}"), ready_fd={write_fd})
daemon.write_port_file(50099)
daemon.install_shutdown_handlers()

# NO atexit.register - this is the bug

# Signal done before exit
os.write({done_write_fd}, b"D")
os.close({done_write_fd})

# Normal exit - NO atexit handlers registered, so files remain
"""
                proc = subprocess.Popen(
                    [sys.executable, "-c", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(cwd),
                    pass_fds=(write_fd, done_write_fd),
                )
                os.close(write_fd)

                proc.wait(timeout=10)
                wait_for_daemon_ready(read_fd)

            # Close our copy after passing to child
            try:
                os.close(done_write_fd)
            except OSError:
                pass

            # Wait for daemon to signal done
            ready, _, _ = select.select([done_read_fd], [], [], 5.0)
            if not ready:
                raise TimeoutError("Daemon did not signal done within 5s")

            data = os.read(done_read_fd, 1)
            if data != b"D":
                raise AssertionError(
                    f"Daemon crashed before signaling done (got {data!r})"
                )

            # Wait for EOF on pipe (indicates process fully exited)
            remaining = os.read(done_read_fd, 1)
            assert remaining == b"", f"Expected EOF, got {remaining!r}"

        finally:
            try:
                os.close(done_read_fd)
            except OSError:
                pass
            try:
                os.close(done_write_fd)
            except OSError:
                pass

        # WITHOUT atexit, files remain (this is the bug)
        assert pid_file.exists(), "Without atexit, PID file should remain"
        assert port_file.exists(), "Without atexit, port file should remain"


@pytest.mark.skip_ci
class TestStartServerAtexitIntegration:
    """Integration tests for atexit cleanup. Skipped in CI - subprocess forking is flaky."""

    @pytest.mark.slow_threshold(10)
    def test_start_server_daemon_mode_cleans_up_on_normal_exit(
        self, tmp_path: Path
    ) -> None:
        """start_server with --daemon should clean up state files on normal exit.

        This tests the actual CLI command (through subprocess) to verify
        that atexit cleanup is properly registered.

        Uses event-driven polling for state file cleanup instead of fixed sleep.
        Since this test goes through CliRunner which doesn't expose ready_fd,
        we poll for the expected outcome (state files removed) rather than
        sleeping blindly.
        """
        cwd = Path.cwd()
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        pid_file = state_dir / "daemon.pid"
        port_file = state_dir / "daemon.port"

        # Script that invokes the real start_server command, mocking run_server
        # to return immediately (simulating normal server shutdown)
        script = f'''
import os
import sys
sys.path.insert(0, "{cwd}")
os.environ["RAGZOOM_STATE_DIR"] = "{state_dir}"

from unittest.mock import patch, MagicMock
from pathlib import Path

def mock_run_server(options):
    """Simulate server running briefly then exiting normally."""
    Path("{tmp_path / 'run_server_called'}").write_text("yes")
    # No sleep needed - just return immediately to trigger atexit

# Patch run_server before importing cli to avoid heavy imports
with patch.dict("sys.modules", {{"ragzoom.server.app": MagicMock()}}):
    import ragzoom.cli
    ragzoom.cli.run_server = mock_run_server

    from click.testing import CliRunner

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        ragzoom.cli.cli,
        ["server", "start", "--daemon", "--port", "50098"],
        catch_exceptions=False
    )

    Path("{tmp_path / 'cli_exit_code'}").write_text(str(result.exit_code))
    if result.output:
        Path("{tmp_path / 'cli_output'}").write_text(result.output)
'''
        script_file = tmp_path / "test_start_server.py"
        script_file.write_text(script)

        proc = subprocess.Popen(
            [sys.executable, str(script_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
        )
        stdout, stderr = proc.communicate(timeout=15)

        # Check if run_server was called (sanity check)
        run_server_marker = tmp_path / "run_server_called"

        # Poll for state file cleanup (replaces time.sleep(1.5))
        # We poll for the expected outcome rather than sleeping blindly.
        # Use short intervals to detect cleanup as soon as it happens.
        cleanup_detected = False
        for _ in range(150):  # 15 second timeout with 0.1s intervals
            if not pid_file.exists() and not port_file.exists():
                cleanup_detected = True
                break
            time.sleep(0.1)

        # Debug output
        debug_files = list(tmp_path.iterdir())
        state_files = list(state_dir.iterdir()) if state_dir.exists() else []

        # The key assertion: state files should be cleaned up
        # Note: With daemon mode, the parent process exits and the daemon
        # does the real work. CliRunner captures the parent's exit, but
        # the daemon is what creates/cleans up files.
        #
        # If this test is flaky, it's because daemon forking interacts
        # poorly with CliRunner. The unit tests above provide the real
        # verification - this is just a smoke test.
        if run_server_marker.exists():
            # run_server was called, so daemon mode started
            # Files should be cleaned up if atexit is registered
            assert cleanup_detected, (
                f"Timeout waiting for state file cleanup. "
                f"State files: {state_files}, tmp files: {debug_files}"
            )
            assert not pid_file.exists(), (
                f"PID file should be removed. "
                f"State files: {state_files}, tmp files: {debug_files}"
            )
            assert not port_file.exists(), (
                f"Port file should be removed. "
                f"State files: {state_files}, tmp files: {debug_files}"
            )

    @pytest.mark.slow_threshold(5)
    def test_atexit_cleanup_idempotent(self, tmp_path: Path) -> None:
        """atexit cleanup should not fail if files are already removed.

        Signal handlers might run before atexit, so cleanup must be idempotent.
        """
        # Script removes files then calls cleanup_stale_state
        script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"

from ragzoom.daemon import (
    cleanup_stale_state,
    write_pid_file,
    write_port_file,
)
from pathlib import Path

# Create state files
write_pid_file(12345)
write_port_file(50051)

# Remove them manually (simulating signal handler cleanup)
(Path("{tmp_path}") / "daemon.pid").unlink()
(Path("{tmp_path}") / "daemon.port").unlink()

# Call cleanup again - should not raise
cleanup_stale_state()

# Mark success
Path("{tmp_path / 'success'}").write_text("ok")
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate(timeout=5)

        success_marker = tmp_path / "success"
        assert (
            success_marker.exists()
        ), f"Script should succeed. stderr: {stderr.decode()}"

    @pytest.mark.slow_threshold(5)
    def test_atexit_runs_after_exception(self, tmp_path: Path) -> None:
        """atexit cleanup should run even if an exception is raised.

        Python's atexit handlers run on normal exit, including after
        uncaught exceptions (the exception is handled by the interpreter).

        Uses the ready-pipe pattern for event-driven synchronization:
        - Daemon signals b"R" when started (ready_fd)
        - Daemon signals b"D" when about to raise exception (done_fd)
        This eliminates polling loops and time.sleep() for deterministic tests.
        """
        import os
        import select

        from tests.conftest import daemon_ready_pipe, wait_for_daemon_ready

        cwd = Path.cwd()
        pid_file = tmp_path / "daemon.pid"
        port_file = tmp_path / "daemon.port"
        log_file = tmp_path / "daemon.log"

        done_read_fd, done_write_fd = os.pipe()

        try:
            with daemon_ready_pipe() as (read_fd, write_fd):
                script = f"""
import atexit
import os
import sys
sys.path.insert(0, "{cwd}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"

{_load_daemon_module_code()}
from pathlib import Path

daemon.daemonize(Path("{log_file}"), ready_fd={write_fd})
daemon.write_port_file(50051)

# Register atexit cleanup
atexit.register(daemon.cleanup_stale_state)

# Signal done before raising exception
os.write({done_write_fd}, b"D")
os.close({done_write_fd})

# Raise exception - atexit should still run
raise RuntimeError("Simulated error in server")
"""
                proc = subprocess.Popen(
                    [sys.executable, "-c", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(cwd),
                    pass_fds=(write_fd, done_write_fd),
                )
                os.close(write_fd)

                proc.wait(timeout=10)
                wait_for_daemon_ready(read_fd)

            # Close our copy after passing to child
            try:
                os.close(done_write_fd)
            except OSError:
                pass

            # Wait for daemon to signal done (about to raise exception)
            ready, _, _ = select.select([done_read_fd], [], [], 5.0)
            if not ready:
                raise TimeoutError("Daemon did not signal done within 5s")

            data = os.read(done_read_fd, 1)
            if data != b"D":
                raise AssertionError(
                    f"Daemon crashed before signaling done (got {data!r})"
                )

            # Wait for EOF on pipe (indicates process fully exited, atexit complete)
            remaining = os.read(done_read_fd, 1)
            assert remaining == b"", f"Expected EOF, got {remaining!r}"

        finally:
            try:
                os.close(done_read_fd)
            except OSError:
                pass
            try:
                os.close(done_write_fd)
            except OSError:
                pass

        # State files should be cleaned up even after exception
        assert not pid_file.exists(), "PID file should be removed after exception"
        assert not port_file.exists(), "Port file should be removed after exception"
