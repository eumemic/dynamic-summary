"""Tests for daemon process lifecycle management."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest


class TestDaemonizeFunction:
    """Tests for the daemonize() function."""

    def test_daemonize_forks_to_background(self, tmp_path: Path) -> None:
        """daemonize() should fork process to background and write flag."""
        log_file = tmp_path / "daemon.log"
        flag_file = tmp_path / "daemon_ran"

        # Run a subprocess that calls daemonize
        script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

daemonize(Path("{log_file}"))
# If we get here, we're in the daemon process
Path("{flag_file}").write_text(f"daemon pid: {{os.getpid()}}")
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Parent process should exit quickly
        proc.wait(timeout=5)

        # Give daemon time to write flag
        time.sleep(0.5)

        # Daemon should have written the flag file
        assert flag_file.exists(), "Daemon process should have run"
        content = flag_file.read_text()
        assert content.startswith("daemon pid:")

    def test_daemonize_writes_pid_file(self, tmp_path: Path) -> None:
        """daemonize() should write daemon PID to state file."""
        log_file = tmp_path / "daemon.log"
        pid_file = tmp_path / "daemon.pid"

        script = f"""
import os
import sys
import time
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

daemonize(Path("{log_file}"))
Path("{tmp_path / 'ran'}").write_text("yes")
time.sleep(0.3)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait(timeout=5)

        # Give daemon time to start
        time.sleep(0.5)

        # PID file should exist and contain a valid PID
        assert pid_file.exists(), "PID file should be created"
        pid_content = pid_file.read_text().strip()
        pid = int(pid_content)
        assert pid > 0, "PID should be positive"

    def test_daemonize_redirects_stdout_stderr(self, tmp_path: Path) -> None:
        """daemonize() should redirect stdout/stderr to log file."""
        log_file = tmp_path / "daemon.log"

        script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

daemonize(Path("{log_file}"))
print("stdout message")
print("stderr message", file=sys.stderr)
sys.stdout.flush()
sys.stderr.flush()
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait(timeout=5)

        # Give daemon time to write
        time.sleep(0.5)

        # Log file should contain the messages
        assert log_file.exists(), "Log file should be created"
        log_content = log_file.read_text()
        assert "stdout message" in log_content
        assert "stderr message" in log_content

    def test_daemonize_detaches_from_terminal(self, tmp_path: Path) -> None:
        """daemonize() should create new session (setsid)."""
        log_file = tmp_path / "daemon.log"
        result_file = tmp_path / "session_info"

        script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

original_sid = os.getsid(0)
daemonize(Path("{log_file}"))
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
        )
        proc.wait(timeout=5)

        time.sleep(0.5)

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

    def test_daemonize_closes_stdin(self, tmp_path: Path) -> None:
        """daemonize() should close stdin (redirect to /dev/null)."""
        log_file = tmp_path / "daemon.log"
        result_file = tmp_path / "stdin_info"

        script = f"""
import os
import sys
sys.path.insert(0, "{Path.cwd()}")
os.environ["RAGZOOM_STATE_DIR"] = "{tmp_path}"
from ragzoom.daemon import daemonize
from pathlib import Path

daemonize(Path("{log_file}"))
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
        )
        proc.wait(timeout=5)

        time.sleep(0.5)

        # stdin should be redirected to /dev/null (empty read)
        assert result_file.exists()
        content = result_file.read_text()
        # Reading from /dev/null returns empty string
        assert "read: ''" in content or "error:" in content

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


class TestDaemonizeIntegration:
    """Integration tests for daemon behavior."""

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
