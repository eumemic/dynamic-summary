"""CLI for Claude Code integration with RagZoom."""

from __future__ import annotations

from pathlib import Path

import click

from ragzoom_claude_code.transcript_sync import (
    execute_sync,
    get_state_path,
    set_session_pid,
)


@click.group()
def cli() -> None:
    """Claude Code integration for RagZoom memory."""
    pass


@cli.command("sync")
@click.argument("jsonl_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--server-address",
    "-s",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default="localhost:50051",
    show_default=True,
    help="RagZoom gRPC server address",
)
def sync_cmd(jsonl_path: Path, server_address: str) -> None:
    """Sync a Claude Code JSONL log to a RagZoom document.

    Incrementally transcribes new conversation records and indexes them.
    Uses UUID-based ancestry tracking to detect and handle reverts.
    Uses the session ID (JSONL filename without extension) as the document ID.
    Tracks progress via state files (configurable via RAGZOOM_STATE_DIR env var).

    The JSONL files are typically found in:
    ~/.claude/projects/<project-path>/<session-id>.jsonl

    Example:
      ragzoom-claude-code sync ~/.claude/projects/.../session.jsonl
    """
    from ragzoom.wrapper import RagZoom

    # State file uses same naming convention but with .jsonl extension
    state_path = get_state_path(jsonl_path.stem)

    client = RagZoom(server_address=server_address)

    try:
        result = execute_sync(jsonl_path, state_path, client)
        if result.truncated:
            click.echo(
                f"Reverted document '{result.document_id}' to span {result.truncate_span}"
            )
        if result.appended_uuids:
            click.echo(
                f"Synced {len(result.appended_uuids)} messages to '{result.document_id}'"
            )
        else:
            click.echo(f"No new content to sync for '{result.document_id}'")
    except Exception as e:
        click.echo(f"Error syncing Claude Code transcript: {e}", err=True)
        raise SystemExit(1) from e


@cli.command("set-pid")
@click.argument("document_id")
@click.argument("pid", type=int)
def set_pid_cmd(document_id: str, pid: int) -> None:
    """Set the PID for a session's state file.

    Called by the SessionStart hook to register the Claude Code PID
    before any tool calls. Creates the state file if needed.

    Example:
      ragzoom-claude-code set-pid my-session-123 12345
    """
    try:
        set_session_pid(document_id, pid)
        click.echo(f"Set PID {pid} for session '{document_id}'")
    except Exception as e:
        click.echo(f"Error setting session PID: {e}", err=True)
        raise SystemExit(1) from e


@cli.command("mcp-server")
def mcp_server_cmd() -> None:
    """Start the MCP server for the 'remember' tool.

    The MCP server exposes the 'remember' tool which allows Claude Code
    to query pre-compaction conversation history using RagZoom's
    hierarchical summarization.

    Example:
      ragzoom-claude-code mcp-server
    """
    from ragzoom_claude_code.mcp_server import mcp

    mcp.run(transport="stdio")


if __name__ == "__main__":
    cli()
