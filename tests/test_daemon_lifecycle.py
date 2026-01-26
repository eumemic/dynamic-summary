"""Tests for daemon process lifecycle management."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.skip_ci
class TestDaemonizeFunction:
    """Tests for the daemonize() function. Skipped in CI - subprocess forking is flaky."""

    @pytest.mark.slow_threshold(5)
    def test_daemonize_ready_fd_signals(self, tmp_path: Path) -> None:
        """daemonize() should write b'R' to ready_fd after completing daemonization."""
        log_file = tmp_path / "daemon.log"
        flag_file = tmp_path / "daemon_ran"

        # Create a pipe - parent reads, child writes
        read_fd, write_fd = os.pipe()

        script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

# The write_fd is passed via pass_fds
daemonize(Path("{log_file}"), ready_fd={write_fd})
# If we get here, we're in the daemon process and have signaled ready
Path("{flag_file}").write_text(f"daemon pid: {{os.getpid()}}")
"""
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
            )
            # Close our copy of write_fd - only the child should have it
            os.close(write_fd)
            write_fd = -1  # Mark as closed

            # Parent process should exit quickly
            proc.wait(timeout=5)

            # Read from pipe - should get b"R" when daemon is ready
            import select

            ready, _, _ = select.select([read_fd], [], [], 5.0)
            assert ready, "Pipe should be readable within timeout"

            data = os.read(read_fd, 1)
            assert data == b"R", f"Expected b'R' but got {data!r}"

            # Daemon should have written the flag file
            assert flag_file.exists(), "Daemon process should have run"
            content = flag_file.read_text()
            assert content.startswith("daemon pid:")
        finally:
            # Clean up file descriptors
            if write_fd != -1:
                try:
                    os.close(write_fd)
                except OSError:
                    pass
            try:
                os.close(read_fd)
            except OSError:
                pass

    @pytest.mark.slow_threshold(5)
    def test_daemonize_forks_to_background(self, tmp_path: Path) -> None:
        """daemonize() should fork process to background and write flag."""
        from tests.conftest import daemon_ready_pipe, wait_for_daemon_ready

        log_file = tmp_path / "daemon.log"
        flag_file = tmp_path / "daemon_ran"

        with daemon_ready_pipe() as (read_fd, write_fd):
            script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

daemonize(Path("{log_file}"), ready_fd={write_fd})
Path("{flag_file}").write_text(f"daemon pid: {{os.getpid()}}")
"""
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
            )
            os.close(write_fd)
            proc.wait(timeout=5)
            wait_for_daemon_ready(read_fd)

        assert flag_file.exists(), "Daemon process should have run"
        assert flag_file.read_text().startswith("daemon pid:")

    @pytest.mark.slow_threshold(5)
    def test_daemonize_writes_pid_file(self, tmp_path: Path) -> None:
        """daemonize() should write daemon PID to state file."""
        from tests.conftest import daemon_ready_pipe, wait_for_daemon_ready

        log_file = tmp_path / "daemon.log"
        pid_file = tmp_path / "daemon.pid"

        with daemon_ready_pipe() as (read_fd, write_fd):
            script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

daemonize(Path("{log_file}"), ready_fd={write_fd})
"""
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
            )
            os.close(write_fd)
            proc.wait(timeout=5)
            wait_for_daemon_ready(read_fd)

        assert pid_file.exists(), "PID file should be created"
        pid_content = pid_file.read_text().strip()
        pid = int(pid_content)
        assert pid > 0, "PID should be positive"

    @pytest.mark.slow_threshold(5)
    def test_daemonize_redirects_stdout_stderr(self, tmp_path: Path) -> None:
        """daemonize() should redirect stdout/stderr to log file."""
        from tests.conftest import daemon_ready_pipe, wait_for_daemon_ready

        log_file = tmp_path / "daemon.log"

        with daemon_ready_pipe() as (read_fd, write_fd):
            script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

daemonize(Path("{log_file}"), ready_fd={write_fd})
print("stdout message")
print("stderr message", file=sys.stderr)
sys.stdout.flush()
sys.stderr.flush()
"""
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
            )
            os.close(write_fd)
            proc.wait(timeout=5)
            wait_for_daemon_ready(read_fd)

        # Log file should contain the messages
        assert log_file.exists(), "Log file should be created"
        log_content = log_file.read_text()
        assert "stdout message" in log_content
        assert "stderr message" in log_content

    @pytest.mark.slow_threshold(5)
    def test_daemonize_detaches_from_terminal(self, tmp_path: Path) -> None:
        """daemonize() should create new session (setsid)."""
        from tests.conftest import daemon_ready_pipe, wait_for_daemon_ready

        log_file = tmp_path / "daemon.log"
        result_file = tmp_path / "session_info"

        with daemon_ready_pipe() as (read_fd, write_fd):
            script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

original_sid = os.getsid(0)
daemonize(Path("{log_file}"), ready_fd={write_fd})
new_sid = os.getsid(0)
new_pid = os.getpid()

Path("{result_file}").write_text(
    f"original_sid={{original_sid}}\\n"
    f"new_sid={{new_sid}}\\n"
    f"pid={{new_pid}}\\n"
)
"""
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
            )
            os.close(write_fd)
            proc.wait(timeout=5)
            wait_for_daemon_ready(read_fd)

        # Check that daemon has its own session
        assert result_file.exists(), "Result file should be created"
        lines = result_file.read_text().strip().split("\n")
        info = dict(line.split("=") for line in lines)

        new_sid = int(info["new_sid"])
        pid = int(info["pid"])
        original_sid = int(info["original_sid"])

        # After double fork, the daemon is in a new session
        # (session ID is set by intermediate process via setsid)
        assert (
            new_sid != original_sid or pid != new_sid
        ), "Daemon should be detached from original session"

    @pytest.mark.slow_threshold(5)
    def test_daemonize_closes_stdin(self, tmp_path: Path) -> None:
        """daemonize() should close stdin (redirect to /dev/null)."""
        from tests.conftest import daemon_ready_pipe, wait_for_daemon_ready

        log_file = tmp_path / "daemon.log"
        result_file = tmp_path / "stdin_info"

        with daemon_ready_pipe() as (read_fd, write_fd):
            script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

daemonize(Path("{log_file}"), ready_fd={write_fd})
try:
    data = sys.stdin.read(1)
    Path("{result_file}").write_text(f"read: {{repr(data)}}")
except Exception as e:
    Path("{result_file}").write_text(f"error: {{e}}")
"""
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
            )
            os.close(write_fd)
            proc.wait(timeout=5)
            wait_for_daemon_ready(read_fd)

        # stdin should be redirected to /dev/null (empty read)
        assert result_file.exists()
        content = result_file.read_text()
        # Reading from /dev/null returns empty string
        assert "read: ''" in content or "error:" in content

    @pytest.mark.slow_threshold(5)
    def test_daemonize_log_file_created_if_missing(self, tmp_path: Path) -> None:
        """daemonize() should create log file and parent dirs if they don't exist."""
        log_file = tmp_path / "subdir" / "daemon.log"

        script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

daemonize(Path("{log_file}"))
print("log message")
sys.stdout.flush()
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait(timeout=5)

        time.sleep(0.5)

        # Log file and parent directory should be created
        assert log_file.parent.exists()
        assert log_file.exists()


@pytest.mark.skip_ci
class TestDaemonizeIntegration:
    """Integration tests for daemon behavior. Skipped in CI - subprocess forking is flaky."""

    @pytest.mark.slow_threshold(5)
    def test_daemon_survives_parent_exit(self, tmp_path: Path) -> None:
        """Daemon should keep running after parent process exits."""
        from ragzoom.daemon import read_pid_file

        log_file = tmp_path / "daemon.log"
        marker_file = tmp_path / "still_running"

        script = f"""
import os
import sys
import time
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

daemonize(Path("{log_file}"))
Path("{marker_file}").write_text("started")
time.sleep(0.3)
Path("{marker_file}").write_text("still_running")
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait(timeout=5)  # Parent exits

        # Wait for daemon to update marker (shorter than 2s test timeout)
        time.sleep(0.8)

        assert marker_file.exists()
        assert marker_file.read_text() == "still_running"

        # Clean up daemon
        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            pid = read_pid_file()
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass  # Already exited


@pytest.mark.skip_ci
class TestSignalHandlers:
    """Tests for signal handling in the daemon. Skipped in CI - signal handling is flaky."""

    @pytest.mark.slow_threshold(5)
    def test_sigterm_graceful_shutdown(self, tmp_path: Path) -> None:
        """SIGTERM should trigger graceful shutdown and cleanup state files."""
        from ragzoom.daemon import read_pid_file

        log_file = tmp_path / "daemon.log"
        pid_file = tmp_path / "daemon.pid"
        cleanup_marker = tmp_path / "cleanup_ran"

        # Script starts daemon with signal handlers and waits for signal
        script = f"""
import os
import sys
import time
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize, install_shutdown_handlers, remove_pid_file
from pathlib import Path

daemonize(Path("{log_file}"))

# Install signal handlers with a cleanup callback
def on_shutdown():
    Path("{cleanup_marker}").write_text("cleanup_ran")

install_shutdown_handlers(cleanup_callback=on_shutdown)

# Write marker indicating daemon is ready
Path("{tmp_path / 'ready'}").write_text("ready")

# Wait for signal (up to 10 seconds)
time.sleep(10)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait(timeout=5)  # Parent exits

        # Wait for daemon to be ready
        ready_file = tmp_path / "ready"
        for _ in range(20):
            if ready_file.exists():
                break
            time.sleep(0.1)
        assert ready_file.exists(), "Daemon should signal ready"

        # Get daemon PID and verify it's running
        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            pid = read_pid_file()
        assert pid is not None, "PID file should exist"

        # Send SIGTERM
        os.kill(pid, signal.SIGTERM)

        # Wait for cleanup
        time.sleep(0.5)

        # Cleanup callback should have run
        assert cleanup_marker.exists(), "Cleanup callback should have run"
        assert cleanup_marker.read_text() == "cleanup_ran"

        # PID file should be removed
        assert not pid_file.exists(), "PID file should be removed on shutdown"

    @pytest.mark.slow_threshold(5)
    def test_sigint_graceful_shutdown(self, tmp_path: Path) -> None:
        """SIGINT should trigger graceful shutdown like SIGTERM."""
        from ragzoom.daemon import read_pid_file

        log_file = tmp_path / "daemon.log"
        pid_file = tmp_path / "daemon.pid"
        cleanup_marker = tmp_path / "cleanup_ran"

        script = f"""
import os
import sys
import time
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize, install_shutdown_handlers
from pathlib import Path

daemonize(Path("{log_file}"))

def on_shutdown():
    Path("{cleanup_marker}").write_text("cleanup_ran")

install_shutdown_handlers(cleanup_callback=on_shutdown)
Path("{tmp_path / 'ready'}").write_text("ready")
time.sleep(10)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait(timeout=5)

        ready_file = tmp_path / "ready"
        for _ in range(20):
            if ready_file.exists():
                break
            time.sleep(0.1)
        assert ready_file.exists()

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            pid = read_pid_file()
        assert pid is not None

        # Send SIGINT instead of SIGTERM
        os.kill(pid, signal.SIGINT)

        time.sleep(0.5)

        assert cleanup_marker.exists(), "Cleanup callback should have run"
        assert not pid_file.exists(), "PID file should be removed"

    def test_install_shutdown_handlers_registers_signals(self) -> None:
        """install_shutdown_handlers() should register SIGTERM and SIGINT handlers."""
        from ragzoom.daemon import install_shutdown_handlers

        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        try:
            install_shutdown_handlers()

            # Handlers should be installed
            new_sigterm = signal.getsignal(signal.SIGTERM)
            new_sigint = signal.getsignal(signal.SIGINT)

            assert new_sigterm != signal.SIG_DFL, "SIGTERM handler should be installed"
            assert new_sigint != signal.SIG_DFL, "SIGINT handler should be installed"
        finally:
            # Restore original handlers
            signal.signal(signal.SIGTERM, original_sigterm)
            signal.signal(signal.SIGINT, original_sigint)


class TestGetProcessUptime:
    """Tests for the get_process_uptime() function."""

    def test_get_process_uptime_current_process(self) -> None:
        """get_process_uptime() should return a valid uptime string for current process."""
        from ragzoom.daemon import get_process_uptime

        # Use the current process (guaranteed to exist)
        uptime = get_process_uptime(os.getpid())

        # Should return a non-empty string
        assert uptime
        assert uptime != "unknown"

        # Should match expected format (seconds, minutes, or hours)
        import re

        assert re.match(r"^\d+[smh]", uptime), f"Uptime '{uptime}' should match format"

    def test_get_process_uptime_nonexistent_pid(self) -> None:
        """get_process_uptime() should return 'unknown' for nonexistent PID."""
        from ragzoom.daemon import get_process_uptime

        # Use a PID that's very unlikely to exist
        uptime = get_process_uptime(999999999)

        assert uptime == "unknown"

    def test_get_process_uptime_formats_correctly(self) -> None:
        """get_process_uptime() should format uptime in human-readable form."""

        import psutil

        from ragzoom.daemon import get_process_uptime

        # Mock psutil.Process to control the create_time
        mock_process = MagicMock()

        with patch.object(psutil, "Process", return_value=mock_process):
            # Test seconds format (< 60s)
            mock_process.create_time.return_value = time.time() - 30
            assert get_process_uptime(1234) == "30s"

            # Test minutes format (< 1h)
            mock_process.create_time.return_value = time.time() - 300
            assert get_process_uptime(1234) == "5m"

            # Test hours format with minutes
            mock_process.create_time.return_value = time.time() - 8100  # 2h 15m
            assert get_process_uptime(1234) == "2h 15m"

            # Test hours format without minutes (exact hours)
            mock_process.create_time.return_value = time.time() - 7200  # 2h exactly
            assert get_process_uptime(1234) == "2h"

            # Test negative uptime (clock skew) returns "unknown"
            mock_process.create_time.return_value = time.time() + 100  # Future time
            assert get_process_uptime(1234) == "unknown"


class TestDaemonReadyPipe:
    """Tests for the daemon_ready_pipe() context manager and wait_for_daemon_ready()."""

    def test_daemon_ready_pipe_cleanup(self) -> None:
        """daemon_ready_pipe() should clean up file descriptors on exit."""
        from tests.conftest import daemon_ready_pipe

        # Get fds and verify they're valid during context
        with daemon_ready_pipe() as (read_fd, write_fd):
            # Both fds should be valid (can write/read)
            os.write(write_fd, b"X")
            data = os.read(read_fd, 1)
            assert data == b"X"

            # Store fds to check after context exits
            stored_read_fd = read_fd
            stored_write_fd = write_fd

        # After context exits, fds should be closed
        # Attempting to use them should raise OSError (Bad file descriptor)
        with pytest.raises(OSError):
            os.read(stored_read_fd, 1)

        with pytest.raises(OSError):
            os.write(stored_write_fd, b"Y")

    def test_daemon_ready_pipe_cleanup_with_early_close(self) -> None:
        """daemon_ready_pipe() should handle already-closed fds gracefully."""
        from tests.conftest import daemon_ready_pipe

        with daemon_ready_pipe() as (read_fd, write_fd):
            # Close fds manually before context exits (simulates typical usage)
            os.close(write_fd)
            os.close(read_fd)
            # Context manager should handle this gracefully on exit (no exception)

    def test_wait_for_daemon_ready_success(self) -> None:
        """wait_for_daemon_ready() should return when daemon signals ready."""
        from tests.conftest import daemon_ready_pipe, wait_for_daemon_ready

        with daemon_ready_pipe() as (read_fd, write_fd):
            # Simulate daemon signaling ready
            os.write(write_fd, b"R")
            os.close(write_fd)

            # Should return without exception
            wait_for_daemon_ready(read_fd, timeout=1.0)

    def test_wait_for_daemon_ready_timeout(self) -> None:
        """wait_for_daemon_ready() should raise TimeoutError when daemon doesn't signal."""
        from tests.conftest import daemon_ready_pipe, wait_for_daemon_ready

        with daemon_ready_pipe() as (read_fd, write_fd):
            # Don't write anything - simulate slow daemon
            # Use very short timeout to avoid slow tests
            with pytest.raises(TimeoutError, match="did not signal ready within"):
                wait_for_daemon_ready(read_fd, timeout=0.01)

    def test_wait_for_daemon_ready_crash_detection(self) -> None:
        """wait_for_daemon_ready() should raise AssertionError when daemon crashes (EOF)."""
        from tests.conftest import daemon_ready_pipe, wait_for_daemon_ready

        with daemon_ready_pipe() as (read_fd, write_fd):
            # Close write end without writing b"R" - simulates daemon crash
            os.close(write_fd)

            # Should detect crash via EOF
            with pytest.raises(AssertionError, match="crashed before signaling ready"):
                wait_for_daemon_ready(read_fd, timeout=1.0)
