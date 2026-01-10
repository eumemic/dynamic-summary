"""CLI for Claude Code memory integration.

This module provides CLI commands for managing Claude Code session state
used by the memory service. Called by hooks (SessionStart) and MCP server.

Usage:
    python -m memory_service.ingestion.claude set-session-pid <session-id> <pid>
    python -m memory_service.ingestion.claude get-session-pid
"""

import sys
from pathlib import Path


def cmd_set_session_pid(session_id: str, pid: int) -> None:
    """Register a Claude Code session's PID for MCP server lookup.

    Called by SessionStart hook to associate the session with its process.
    Creates the state file if it doesn't exist.
    """
    from memory_service.ingestion.claude.transcript_sync import set_session_pid

    set_session_pid(session_id, pid)
    print(f"Registered PID {pid} for session '{session_id}'", file=sys.stderr)


def cmd_get_session_pid() -> None:
    """Get the session ID for the current process (or its parent).

    Searches all session state files to find one whose registered PID
    matches this process's PID or PPID. Used by agents to discover their
    own session ID.

    Prints the session ID to stdout, or exits with error if not found.
    """
    import os

    from memory_service.ingestion.claude.transcript_sync import (
        SessionPidMapping,
        _get_state_dir,
    )

    current_pid = os.getpid()
    parent_pid = os.getppid()

    state_dir = _get_state_dir()
    if not state_dir.exists():
        print(f"No state directory found at {state_dir}", file=sys.stderr)
        sys.exit(1)

    # Search all state files for a matching PID
    for state_file in state_dir.glob("*.jsonl"):
        mapping = SessionPidMapping.load(state_file)
        if mapping is not None and mapping.last_pid in (current_pid, parent_pid):
            # Print session ID to stdout (for capture by caller)
            print(mapping.document_id)
            return

    print(
        f"No session found for PID {current_pid} or PPID {parent_pid}",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "set-session-pid":
        if len(sys.argv) != 4:
            print(
                "Usage: python -m memory_service.ingestion.claude "
                "set-session-pid <session-id> <pid>",
                file=sys.stderr,
            )
            sys.exit(1)
        session_id = sys.argv[2]
        try:
            pid = int(sys.argv[3])
        except ValueError:
            print(f"Invalid PID: {sys.argv[3]}", file=sys.stderr)
            sys.exit(1)
        cmd_set_session_pid(session_id, pid)

    elif command == "get-session-pid":
        if len(sys.argv) != 2:
            print(
                "Usage: python -m memory_service.ingestion.claude get-session-pid",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd_get_session_pid()

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
