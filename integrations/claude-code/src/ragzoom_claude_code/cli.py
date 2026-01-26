"""CLI for Claude Code integration with RagZoom."""

from __future__ import annotations

from pathlib import Path

import click

from ragzoom_claude_code.transcript_sync import (
    execute_sync,
    get_state_path,
    set_session_pid,
)

# Note: get_state_path and set_session_pid are still used by set-pid command
# and reset command for state file management. execute_sync no longer uses state files.


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

    The JSONL files are typically found in:
    ~/.claude/projects/<project-path>/<session-id>.jsonl

    Example:
      ragzoom-claude-code sync ~/.claude/projects/.../session.jsonl
    """
    from ragzoom.wrapper import RagZoom

    document_id = jsonl_path.stem
    client = RagZoom(server_address=server_address)

    try:
        result = execute_sync(jsonl_path, document_id, client)
        if result.truncated:
            click.echo(
                f"Reverted document '{result.document_id}' "
                f"(cutoff: {result.truncate_cutoff_time})"
            )
        if result.turns_appended > 0:
            click.echo(
                f"Synced {result.turns_appended} turns to '{result.document_id}'"
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


@cli.command("reset")
@click.argument("jsonl_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--server-address",
    "-s",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default="localhost:50051",
    show_default=True,
    help="RagZoom gRPC server address",
)
@click.option(
    "--resync/--no-resync",
    default=True,
    help="Re-sync after reset (default: yes)",
)
def reset_cmd(jsonl_path: Path, server_address: str, resync: bool) -> None:
    """Reset a session by clearing both state file and document.

    Deletes the local sync state file and clears the document from RagZoom,
    then optionally re-syncs from scratch.

    Example:
      ragzoom-claude-code reset ~/.claude/projects/.../session.jsonl
      ragzoom-claude-code reset session.jsonl --no-resync
    """
    from ragzoom.client import GrpcRagzoomClient
    from ragzoom.wrapper import RagZoom

    document_id = jsonl_path.stem
    state_path = get_state_path(document_id)

    # Step 1: Delete state file
    if state_path.exists():
        state_path.unlink()
        click.echo(f"Deleted state file: {state_path}")
    else:
        click.echo(f"No state file found at: {state_path}")

    # Step 2: Clear document from RagZoom
    try:
        with GrpcRagzoomClient(server_address) as grpc_client:
            clear_result = grpc_client.clear_document(document_id)
            if clear_result.document_existed:
                click.echo(
                    f"Cleared document '{document_id}' ({clear_result.deleted_nodes} nodes)"
                )
            else:
                click.echo(f"Document '{document_id}' did not exist")
    except Exception as e:
        click.echo(f"Warning: Could not clear document: {e}", err=True)

    # Step 3: Re-sync if requested
    if resync:
        click.echo("Re-syncing from scratch...")
        ragzoom_client = RagZoom(server_address=server_address)
        try:
            sync_result = execute_sync(jsonl_path, document_id, ragzoom_client)
            if sync_result.turns_appended > 0:
                click.echo(
                    f"Synced {sync_result.turns_appended} turns to '{sync_result.document_id}'"
                )
            else:
                click.echo(f"No content to sync for '{sync_result.document_id}'")
        except Exception as e:
            click.echo(f"Error during re-sync: {e}", err=True)
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
