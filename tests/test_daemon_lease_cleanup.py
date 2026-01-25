"""Tests for daemon cleanup when lease acquisition fails.

Verifies that state files (PID and port) are cleaned up when the daemon
exits due to lease acquisition failure (sys.exit(1) in _run_with_lease).
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# Timeout constants
SUBPROCESS_TIMEOUT_SEC = 10
STARTUP_POLL_COUNT = 30
STARTUP_POLL_INTERVAL_SEC = 0.1
ATEXIT_COMPLETION_DELAY_SEC = 0.5


def _setup_daemon_test_script(
    tmp_path: Path,
    cwd: Path,
    async_body: str,
) -> tuple[Path, Path, Path, Path]:
    """Create daemon test environment and script.

    Returns: (state_dir, pid_file, port_file, script_file)
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    pid_file = state_dir / "daemon.pid"
    port_file = state_dir / "daemon.port"
    log_file = state_dir / "daemon.log"

    script = f"""
import atexit
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, "{cwd}")
os.environ["RAGZOOM_STATE_DIR"] = "{state_dir}"

_spec = importlib.util.spec_from_file_location("daemon", "{cwd}/ragzoom/daemon.py")
daemon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(daemon)

daemon.daemonize(Path("{log_file}"))
daemon.write_port_file(50099)
daemon.install_shutdown_handlers()
atexit.register(daemon.cleanup_stale_state)

Path("{tmp_path / 'daemon_started'}").write_text("yes")

{async_body}
"""
    script_file = tmp_path / "test_script.py"
    script_file.write_text(script)

    return state_dir, pid_file, port_file, script_file


def _run_daemon_script_and_wait(script_file: Path, tmp_path: Path, cwd: Path) -> None:
    """Execute daemon test script and wait for startup."""
    proc = subprocess.Popen(
        [sys.executable, str(script_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd),
    )
    proc.wait(timeout=SUBPROCESS_TIMEOUT_SEC)

    for _ in range(STARTUP_POLL_COUNT):
        if (tmp_path / "daemon_started").exists():
            break
        time.sleep(STARTUP_POLL_INTERVAL_SEC)

    time.sleep(ATEXIT_COMPLETION_DELAY_SEC)


def _assert_state_files_removed(
    pid_file: Path,
    port_file: Path,
    state_dir: Path,
) -> None:
    """Verify state files were cleaned up."""
    state_contents = list(state_dir.iterdir())
    assert (
        not pid_file.exists()
    ), f"PID file should be removed. State dir: {state_contents}"
    assert (
        not port_file.exists()
    ), f"Port file should be removed. State dir: {state_contents}"


class TestLeaseFailureCleanup:
    """Tests for cleanup when lease acquisition fails."""

    @pytest.mark.slow_threshold(5)
    def test_lease_failure_cleans_up_state_files(self, tmp_path: Path) -> None:
        """Atexit handler cleans up state files when sys.exit(1) is called.

        This simulates the scenario in ragzoom/server/app.py:179 where lease
        acquisition fails and sys.exit(1) is called from within asyncio.run().
        """
        cwd = Path.cwd()

        async_body = """
async def simulate_lease_failure():
    sys.exit(1)

asyncio.run(simulate_lease_failure())
"""

        state_dir, pid_file, port_file, script_file = _setup_daemon_test_script(
            tmp_path, cwd, async_body
        )
        _run_daemon_script_and_wait(script_file, tmp_path, cwd)
        _assert_state_files_removed(pid_file, port_file, state_dir)

    @pytest.mark.slow_threshold(5)
    def test_explicit_cleanup_before_sysexit_in_async(self, tmp_path: Path) -> None:
        """Explicit cleanup before sys.exit() ensures state file removal.

        Belt-and-suspenders approach: explicitly calling cleanup before exit.
        """
        cwd = Path.cwd()

        async_body = """
async def simulate_lease_failure_with_explicit_cleanup():
    daemon.cleanup_stale_state()
    sys.exit(1)

asyncio.run(simulate_lease_failure_with_explicit_cleanup())
"""

        state_dir, pid_file, port_file, script_file = _setup_daemon_test_script(
            tmp_path, cwd, async_body
        )
        _run_daemon_script_and_wait(script_file, tmp_path, cwd)
        _assert_state_files_removed(pid_file, port_file, state_dir)
