"""CLI for OpenClaw integration with RagZoom."""

from __future__ import annotations

from pathlib import Path

import click

from ragzoom_openclaw.transcript_sync import sync_transcript


@click.group()
def cli() -> None:
    """OpenClaw integration for RagZoom memory."""
    pass


@cli.command("sync")
@click.argument("jsonl_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--state-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=Path("data/openclaw-state"),
    show_default=True,
    help="Directory for sync state files",
)
@click.option(
    "--document-id",
    "-i",
    default=None,
    help="Override document ID (defaults to openclaw-<filename>)",
)
@click.option(
    "--server-address",
    "-s",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default="localhost:50052",
    show_default=True,
    help="RagZoom gRPC server address",
)
def sync_cmd(
    jsonl_path: Path,
    state_dir: Path,
    document_id: str | None,
    server_address: str,
) -> None:
    """Sync an OpenClaw JSONL session to a RagZoom document.

    Incrementally transcribes new conversation steps and indexes them.
    Each user/assistant message becomes a separate indexed chunk with
    its own timestamp for fine-grained temporal queries.

    Thinking blocks are preserved with 💭 marker.

    Example:
      ragzoom-openclaw sync session.jsonl
      ragzoom-openclaw sync session.jsonl --document-id jarvis-main
    """
    from ragzoom.wrapper import RagZoom

    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{jsonl_path.stem}.jsonl"

    client = RagZoom(server_address=server_address)

    try:
        result = sync_transcript(
            transcript_path=jsonl_path,
            state_path=state_path,
            client=client,
            document_id=document_id,
        )
        if result.new_steps > 0:
            click.echo(
                f"✅ Synced {result.new_steps} new steps to '{result.document_id}' "
                f"({result.total_steps} total)"
            )
        else:
            click.echo(
                f"No new steps to sync for '{result.document_id}' "
                f"({result.total_steps} total)"
            )
    except Exception as e:
        click.echo(f"Error syncing OpenClaw transcript: {e}", err=True)
        raise SystemExit(1) from e


@cli.command("recall")
@click.argument("query")
@click.option(
    "--budget",
    "-b",
    type=int,
    default=2000,
    show_default=True,
    help="Token budget for results",
)
@click.option(
    "--session",
    "-S",
    envvar="RAGZOOM_DOCUMENT_ID",
    default="agent:main:main",
    show_default=True,
    help="Session key / document ID to query",
)
@click.option(
    "--start",
    "-s",
    default=None,
    help="Start time (ISO timestamp) for time-bounded query",
)
@click.option(
    "--end",
    "-e",
    default=None,
    help="End time (ISO timestamp) for time-bounded query",
)
@click.option(
    "--server-address",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default="localhost:50052",
    show_default=True,
    help="RagZoom gRPC server address",
)
def recall_cmd(
    query: str,
    budget: int,
    session: str,
    start: str | None,
    end: str | None,
    server_address: str,
) -> None:
    """Query conversation memory with semantic search.

    Returns time-bounded spans at various summarization levels.
    Use --start/--end to zoom into specific time ranges for more detail.

    The iterative zoom workflow:
      1. Query broad to find relevant time ranges
      2. Note the time_start/time_end in results
      3. Query again with --start/--end to zoom in

    Example:
      ragzoom-openclaw recall "authentication bug"
      ragzoom-openclaw recall "auth bug" --start 2026-01-31T14:00:00Z --end 2026-01-31T15:00:00Z
      ragzoom-openclaw recall "topic" --session agent:main:signal:group:abc123
    """
    import sys
    sys.path.insert(0, "/Users/jarvis/code/dynamic-summary/integrations/claude-code/src")
    
    from ragzoom_claude_code.recall import execute_recall, format_for_cli

    try:
        result = execute_recall(
            query=query,
            document_id=session,
            token_budget=budget,
            time_start=start,
            time_end=end,
            server_address=server_address,
        )
        click.echo(format_for_cli(result))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    cli()
