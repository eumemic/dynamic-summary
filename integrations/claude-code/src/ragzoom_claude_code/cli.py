"""CLI for Claude Code integration with RagZoom."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ragzoom.output_formatters import format_tiling_spans
from ragzoom_claude_code.recall import execute_recall, execute_search
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
@click.option(
    "--append-only",
    is_flag=True,
    envvar="RAGZOOM_APPEND_ONLY",
    help="Skip revert detection, append entries after document time_end "
    "[env: RAGZOOM_APPEND_ONLY]",
)
def sync_cmd(
    jsonl_path: Path, document_id: str | None, server_address: str, append_only: bool
) -> None:
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
        result = execute_sync(jsonl_path, doc_id, client, append_only=append_only)
        if result.truncated:
            click.echo(
                f"Reverted document '{result.document_id}' "
                f"(cutoff: {result.truncate_cutoff_time})"
            )
        if result.steps_appended > 0:
            click.echo(
                f"Synced {result.steps_appended} steps to '{result.document_id}'"
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
            if sync_result.steps_appended > 0:
                click.echo(
                    f"Synced {sync_result.steps_appended} steps to '{sync_result.document_id}'"
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


@cli.command("recall")
@click.argument("query", default="")
@click.option(
    "--document-id",
    "-d",
    envvar="RAGZOOM_DOCUMENT_ID",
    required=True,
    help="Document ID to query (required)",
)
@click.option(
    "--token-budget",
    "-t",
    type=int,
    default=2000,
    show_default=True,
    help="Max tokens for returned context",
)
@click.option(
    "--time-start",
    type=str,
    default=None,
    help="ISO timestamp to start from (e.g., '2024-01-15T10:00:00')",
)
@click.option(
    "--time-end",
    type=str,
    default=None,
    help="ISO timestamp to end at (e.g., '2024-01-15T18:00:00')",
)
@click.option(
    "--server-address",
    "-s",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default="localhost:50051",
    show_default=True,
    help="RagZoom gRPC server address",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output as JSON instead of human-readable format",
)
def recall_cmd(
    query: str,
    document_id: str,
    token_budget: int,
    time_start: str | None,
    time_end: str | None,
    server_address: str,
    json_output: bool,
) -> None:
    """Search conversation history.

    Query RagZoom's hierarchical summarization to recall context from
    earlier in a conversation. Uses semantic search with token-budget
    controlled detail level.

    Document ID priority:
      1. --document-id CLI flag
      2. RAGZOOM_DOCUMENT_ID environment variable

    Example:
      ragzoom-claude-code recall "authentication bug" -d session-id
      ragzoom-claude-code recall "OAuth" -d my-session --token-budget 5000
      ragzoom-claude-code recall "" -d session --time-start 2024-01-15T10:00:00
    """
    try:
        result = execute_recall(
            query=query,
            document_id=document_id,
            token_budget=token_budget,
            time_start=time_start,
            time_end=time_end,
            server_address=server_address,
        )

        if json_output:
            retrieval = result.retrieval
            output = {
                "nodes": [
                    {
                        "text": node.text,
                        "time_start": node.time_start,
                        "time_end": node.time_end,
                        "height": node.height,
                    }
                    for node_id in retrieval.tiling_ids
                    if (node := retrieval.nodes.get(node_id)) and node.text
                ]
            }
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo(format_tiling_spans(result))

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@cli.command("search")
@click.argument("question")
@click.option(
    "--document-id",
    "-d",
    envvar="RAGZOOM_DOCUMENT_ID",
    required=True,
    help="Document ID to search (required)",
)
@click.option(
    "--server-address",
    "-s",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default="localhost:50051",
    show_default=True,
    help="RagZoom gRPC server address",
)
def search_cmd(
    question: str,
    document_id: str,
    server_address: str,
) -> None:
    """Agentic search: question in, answer out.

    The server-side search agent iteratively zooms into the document
    to find the best answer.

    Example:
      ragzoom-claude-code search "What was the auth bug?" -d session-id
    """
    try:
        result = execute_search(
            question=question,
            document_id=document_id,
            server_address=server_address,
        )
        click.echo(result.answer)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    cli()
