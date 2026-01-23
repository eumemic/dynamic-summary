"""Daemon lifecycle management for RagZoom gRPC server.

Provides XDG-compliant state directory management, PID file handling,
health checks, and auto-start capabilities.
"""

import errno
import os
from pathlib import Path

PID_FILENAME = "daemon.pid"


def get_daemon_state_dir() -> Path:
    """Get the daemon state directory path.

    Uses XDG Base Directory Specification:
    - Default: ~/.local/state/ragzoom/
    - Override: RAGZOOM_STATE_DIR environment variable

    Returns:
        Absolute path to the state directory (may not exist yet).
    """
    env_override = os.environ.get("RAGZOOM_STATE_DIR")
    if env_override:
        # Expand ~ and convert to absolute path
        path = Path(env_override).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    # XDG default: ~/.local/state/ragzoom/
    return Path("~/.local/state/ragzoom").expanduser()


def ensure_daemon_state_dir() -> Path:
    """Ensure the daemon state directory exists.

    Creates the directory and any parent directories if needed.

    Returns:
        Absolute path to the state directory (guaranteed to exist).
    """
    state_dir = get_daemon_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def get_pid_file_path() -> Path:
    """Get the path to the PID file.

    Returns:
        Path to daemon.pid in the state directory.
    """
    return get_daemon_state_dir() / PID_FILENAME


def write_pid_file(pid: int) -> None:
    """Write the daemon PID to the PID file.

    Creates the state directory if it doesn't exist.

    Args:
        pid: Process ID to write.
    """
    state_dir = ensure_daemon_state_dir()
    pid_file = state_dir / PID_FILENAME
    pid_file.write_text(f"{pid}\n")


def read_pid_file() -> int | None:
    """Read the daemon PID from the PID file.

    Returns:
        The PID as an integer, or None if:
        - The file doesn't exist
        - The file contents are not a valid integer
    """
    pid_file = get_pid_file_path()
    if not pid_file.exists():
        return None

    try:
        content = pid_file.read_text().strip()
        return int(content)
    except (ValueError, OSError):
        return None


def remove_pid_file() -> None:
    """Remove the PID file if it exists.

    This is idempotent - does not raise if the file doesn't exist.
    """
    pid_file = get_pid_file_path()
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass


def is_pid_stale(pid: int) -> bool:
    """Check if a PID refers to a non-existent process.

    Uses the "kill 0" trick: sending signal 0 checks if a process
    exists without actually sending a signal.

    Args:
        pid: Process ID to check.

    Returns:
        True if the process does NOT exist (stale PID).
        False if the process exists.
    """
    try:
        os.kill(pid, 0)
        return False  # Process exists
    except OSError as e:
        if e.errno == errno.ESRCH:
            # ESRCH = No such process
            return True
        if e.errno == errno.EPERM:
            # EPERM = Process exists but we don't have permission to signal it
            return False
        # Other errors - assume stale to be safe
        return True
