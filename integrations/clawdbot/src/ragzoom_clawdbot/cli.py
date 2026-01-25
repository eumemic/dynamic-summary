"""CLI for Clawdbot integration with RagZoom."""

from __future__ import annotations

from pathlib import Path

import click

from ragzoom_clawdbot.transcript_sync import sync_transcript


@click.group()
def cli() -> None:
    """Clawdbot integration for RagZoom memory."""
    pass


@cli.command("sync")
@click.argument("jsonl_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--state-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=Path("data/clawdbot-state"),
    show_default=True,
    help="Directory for sync state files",
)
@click.option(
    "--document-id",
    "-i",
    default=None,
    help="Override document ID (defaults to clawdbot-<filename>)",
)
@click.option(
    "--server-address",
    "-s",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default="localhost:50051",
    show_default=True,
    help="RagZoom gRPC server address",
)
def sync_cmd(
    jsonl_path: Path,
    state_dir: Path,
    document_id: str | None,
    server_address: str,
) -> None:
    """Sync a Clawdbot JSONL session to a RagZoom document.

    Incrementally transcribes new conversation turns and indexes them.
    Clawdbot sessions are linear (no branching), so sync is append-only.

    Example:
      ragzoom-clawdbot sync session.jsonl
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
        if result.new_turns > 0:
            click.echo(
                f"Synced {result.new_turns} new turns to '{result.document_id}' "
                f"({result.total_turns} total)"
            )
        else:
            click.echo(
                f"No new turns to sync for '{result.document_id}' "
                f"({result.total_turns} total)"
            )
    except Exception as e:
        click.echo(f"Error syncing Clawdbot transcript: {e}", err=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    cli()
