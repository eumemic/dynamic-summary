"""CLI for Claude Code integration with RagZoom."""

from __future__ import annotations

from pathlib import Path

import click

from ragzoom_claude_code.transcript_sync import execute_sync


@click.group()
def cli() -> None:
    """Claude Code integration for RagZoom memory."""
    pass


@cli.command("sync")
@click.argument("jsonl_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--document-id",
    "-d",
    envvar="RAGZOOM_DOCUMENT_ID",
    help="Override document ID (default: JSONL filename stem)",
)
@click.option(
    "--server-address",
    "-s",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default="localhost:50051",
    show_default=True,
    help="RagZoom gRPC server address",
)
def sync_cmd(jsonl_path: Path, document_id: str | None, server_address: str) -> None:
    """Sync a Claude Code JSONL log to a RagZoom document.

    Incrementally transcribes new conversation records and indexes them.
    Uses UUID-based ancestry tracking to detect and handle reverts.

    Document ID priority:
      1. --document-id CLI flag
      2. RAGZOOM_DOCUMENT_ID environment variable
      3. JSONL filename stem (default)

    The JSONL files are typically found in:
    ~/.claude/projects/<project-path>/<session-id>.jsonl

    Example:
      ragzoom-claude-code sync ~/.claude/projects/.../session.jsonl
      ragzoom-claude-code sync session.jsonl --document-id my-custom-id
      RAGZOOM_DOCUMENT_ID=jarvis ragzoom-claude-code sync session.jsonl
    """
    from ragzoom.wrapper import RagZoom

    doc_id = document_id or jsonl_path.stem
    client = RagZoom(server_address=server_address)

    try:
        result = execute_sync(jsonl_path, doc_id, client)
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
    """Reset a session by clearing the document and re-syncing.

    Clears the document from RagZoom, then optionally re-syncs from scratch.

    Example:
      ragzoom-claude-code reset ~/.claude/projects/.../session.jsonl
      ragzoom-claude-code reset session.jsonl --no-resync
    """
    from ragzoom.client import GrpcRagzoomClient
    from ragzoom.wrapper import RagZoom

    document_id = jsonl_path.stem

    # Step 1: Clear document from RagZoom
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

    # Step 2: Re-sync if requested
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
