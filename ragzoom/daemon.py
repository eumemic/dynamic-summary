"""Daemon lifecycle management for RagZoom gRPC server.

Provides XDG-compliant state directory management, PID file handling,
health checks, and auto-start capabilities.
"""

import errno
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from types import FrameType

PID_FILENAME = "daemon.pid"
PORT_FILENAME = "daemon.port"
LOG_FILENAME = "daemon.log"


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


def get_port_file_path() -> Path:
    """Get the path to the port file.

    Returns:
        Path to daemon.port in the state directory.
    """
    return get_daemon_state_dir() / PORT_FILENAME


def write_port_file(port: int) -> None:
    """Write the daemon port to the port file.

    Creates the state directory if it doesn't exist.

    Args:
        port: Port number to write.
    """
    state_dir = ensure_daemon_state_dir()
    port_file = state_dir / PORT_FILENAME
    port_file.write_text(f"{port}\n")


def read_port_file() -> int | None:
    """Read the daemon port from the port file.

    Returns:
        The port as an integer, or None if:
        - The file doesn't exist
        - The file contents are not a valid integer
    """
    port_file = get_port_file_path()
    if not port_file.exists():
        return None

    try:
        content = port_file.read_text().strip()
        return int(content)
    except (ValueError, OSError):
        return None


def remove_port_file() -> None:
    """Remove the port file if it exists.

    This is idempotent - does not raise if the file doesn't exist.
    """
    port_file = get_port_file_path()
    try:
        port_file.unlink()
    except FileNotFoundError:
        pass


def get_log_file_path() -> Path:
    """Get the default path to the daemon log file.

    Returns:
        Path to daemon.log in the state directory.
    """
    return get_daemon_state_dir() / LOG_FILENAME


def daemonize(log_file: Path | None = None) -> None:
    """Fork the current process into a background daemon.

    Implements the standard Unix double-fork daemonization pattern:
    1. First fork: Parent exits, child continues
    2. setsid(): Child becomes session leader
    3. Second fork: Session leader exits, grandchild continues
       (prevents acquiring controlling terminal)
    4. Redirect stdin to /dev/null, stdout/stderr to log file
    5. Write daemon PID to state file

    Args:
        log_file: Path to redirect stdout/stderr. If None, uses default
                  location in state directory. Parent directories will
                  be created if they don't exist.

    Note:
        This function only returns in the final daemon process.
        Parent processes call sys.exit(0).
    """

    if log_file is None:
        log_file = get_log_file_path()

    # Flush buffered output before forking to prevent duplication
    sys.stdout.flush()
    sys.stderr.flush()

    # First fork - parent exits, child continues
    pid = os.fork()
    if pid > 0:
        # Parent - exit cleanly
        sys.exit(0)

    # Child - become session leader
    os.setsid()

    # Second fork - prevents acquiring controlling terminal
    pid = os.fork()
    if pid > 0:
        # Intermediate process - exit
        sys.exit(0)

    # Grandchild (daemon) continues from here

    # Ensure log file directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Redirect stdin to /dev/null
    devnull_fd = os.open("/dev/null", os.O_RDONLY)
    os.dup2(devnull_fd, sys.stdin.fileno())
    os.close(devnull_fd)

    # Redirect stdout and stderr to log file
    log_fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    os.close(log_fd)

    # Write daemon PID to state file
    write_pid_file(os.getpid())


def install_shutdown_handlers(
    cleanup_callback: Callable[[], None] | None = None,
) -> None:
    """Install signal handlers for graceful daemon shutdown.

    Handles SIGTERM and SIGINT by:
    1. Calling the optional cleanup callback
    2. Removing state files (PID and port files)
    3. Exiting with status 0

    Args:
        cleanup_callback: Optional function to call before cleanup.
                         Use this to stop servers, finish in-flight work, etc.
    """

    def shutdown_handler(signum: int, frame: FrameType | None) -> None:
        """Handle shutdown signals."""
        # Call cleanup callback if provided
        if cleanup_callback is not None:
            cleanup_callback()

        # Remove state files
        remove_pid_file()
        remove_port_file()

        # Exit cleanly
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
