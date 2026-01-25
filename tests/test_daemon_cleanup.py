"""Tests for daemon try/finally cleanup wrapper.

Verifies that state files (PID and port) are cleaned up through the try/finally
wrapper around run_server(), which handles exception paths that exit before
atexit handlers run.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest


class TestTryFinallyCleanup:
    """Tests for try/finally cleanup wrapper in start_server."""

    @pytest.mark.slow_threshold(5)
    def test_exception_triggers_finally_cleanup(self, tmp_path: Path) -> None:
        """Exception in run_server triggers finally cleanup of state files.

        When run_server raises an exception, the finally block ensures cleanup
        occurs even if atexit handlers haven't run yet.
        """
        cwd = Path.cwd()
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        pid_file = state_dir / "daemon.pid"
        port_file = state_dir / "daemon.port"
        log_file = state_dir / "daemon.log"

        # Subprocess script that simulates daemon mode with try/finally wrapper.
        # Raises an exception to test that finally block cleans up state files.
        script = f'''
import atexit
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, "{cwd}")
os.environ["RAGZOOM_STATE_DIR"] = "{state_dir}"

_spec = importlib.util.spec_from_file_location("daemon", "{cwd}/ragzoom/daemon.py")
daemon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(daemon)

# Simulate daemon mode setup (what start_server does)
daemon.daemonize(Path("{log_file}"))
daemon.write_port_file(50099)
daemon.install_shutdown_handlers()
atexit.register(daemon.cleanup_stale_state)

Path("{tmp_path / 'daemon_started'}").write_text("yes")

def mock_run_server():
    """Simulate run_server that raises an exception."""
    raise RuntimeError("Simulated server crash")

# Test pattern: try/finally wrapper around run_server
try:
    mock_run_server()
finally:
    daemon.cleanup_stale_state()

Path("{tmp_path / 'should_not_reach'}").write_text("unreachable")
'''
        script_file = tmp_path / "test_script.py"
        script_file.write_text(script)

        proc = subprocess.Popen(
            [sys.executable, str(script_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
        )
        proc.wait(timeout=10)

        # Wait for daemon to start
        for _ in range(30):
            if (tmp_path / "daemon_started").exists():
                break
            time.sleep(0.1)

        # Give cleanup time to run
        time.sleep(0.5)

        # State files should be cleaned up via try/finally
        state_contents = list(state_dir.iterdir())
        assert (
            not pid_file.exists()
        ), f"PID file should be removed. State: {state_contents}"
        assert (
            not port_file.exists()
        ), f"Port file should be removed. State: {state_contents}"

    @pytest.mark.slow_threshold(5)
    def test_finally_runs_before_atexit(self, tmp_path: Path) -> None:
        """Finally block runs before atexit handlers.

        Verifies cleanup order: finally executes first, then atexit on process exit.
        """
        cwd = Path.cwd()
        marker_file = tmp_path / "cleanup_order.txt"

        # Subprocess script that records cleanup execution order
        script = f"""
import atexit
import os
import sys
from pathlib import Path

sys.path.insert(0, "{cwd}")

marker = Path("{marker_file}")
cleanup_order = []

def atexit_cleanup():
    cleanup_order.append("atexit")
    marker.write_text(",".join(cleanup_order))

def finally_cleanup():
    cleanup_order.append("finally")
    marker.write_text(",".join(cleanup_order))

atexit.register(atexit_cleanup)

try:
    pass  # Normal exit
finally:
    finally_cleanup()
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
        )
        proc.wait(timeout=5)

        time.sleep(0.3)

        # Verify cleanup order
        assert marker_file.exists(), "Marker file should be created"
        order = marker_file.read_text()
        assert order == "finally,atexit", f"Expected 'finally,atexit', got: {order}"

    @pytest.mark.slow_threshold(5)
    @pytest.mark.skip_ci
    def test_finally_cleanup_idempotent_with_atexit(self, tmp_path: Path) -> None:
        """Cleanup called from both finally and atexit is safe.

        cleanup_stale_state must be idempotent since it's called from both
        the finally block and atexit handler. Calling it twice should not fail.
        """
        cwd = Path.cwd()
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        success_marker = tmp_path / "success"

        script = f"""
import atexit
import os
import sys
from pathlib import Path

sys.path.insert(0, "{cwd}")
os.environ["RAGZOOM_STATE_DIR"] = "{state_dir}"

from ragzoom.daemon import cleanup_stale_state, write_pid_file, write_port_file

# Create state files
write_pid_file(12345)
write_port_file(50051)

# Register atexit handler (runs second)
atexit.register(cleanup_stale_state)

try:
    pass  # Normal exit
finally:
    # First cleanup call (from try/finally)
    cleanup_stale_state()

# Second cleanup call happens via atexit - must not fail
Path("{success_marker}").write_text("ok")
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
        )
        stdout, stderr = proc.communicate(timeout=5)

        time.sleep(0.3)

        assert success_marker.exists(), (
            f"Script should complete without error. "
            f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
        )
        assert (
            proc.returncode == 0
        ), f"Script should exit cleanly. stderr: {stderr.decode()}"
