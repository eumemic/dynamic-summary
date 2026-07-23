"""CLI interface for RagZoom."""

from __future__ import annotations

import asyncio
import atexit
import inspect
import json
import logging
import os
import random
import shutil
import signal
import socket
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol, cast

import click
from dotenv import load_dotenv

from ragzoom.client import (
    DocumentWorkStatus,
    GrpcRagzoomClient,
    WorkerRunSnapshot,
)
from ragzoom.config import (
    IndexConfig,
    OperationalConfig,
    QueryConfig,
)
from ragzoom.constants import (
    DEFAULT_GRPC_ADDRESS,
    DEFAULT_GRPC_HOST,
)
from ragzoom.daemon import (
    cleanup_stale_state,
    daemonize,
    get_config_dir,
    get_log_file_path,
    get_process_uptime,
    install_shutdown_handlers,
    is_pid_stale,
    read_pid_file,
    read_port_file,
    write_config_file,
)
from ragzoom.error_handling import handle_graceful_error
from ragzoom.exceptions import (
    ConfigurationError,
    DatabaseError,
    LLMError,
    NodeNotFoundError,
    ResourceError,
    ValidationError,
)
from ragzoom.output_formatters import build_json_error_from_exception, build_json_output
from ragzoom.progress_display import DocumentProgressTotals, WorkerProgressDisplay
from ragzoom.server.app import ServerOptions, run_server
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.store import create_store_with_docker
from ragzoom.telemetry_types import TelemetryDataDict
from ragzoom.validation import validate_document
from ragzoom.vector_factory import create_vector_index


class AppendTextCallable(Protocol):
    def __call__(
        self,
        *,
        document_id: str,
        content: bytes,
        collect_telemetry: bool,
        replace_existing: bool = ...,  # optional keyword for rebuilds
        summarization_guidance: str | None = ...,  # optional custom prompt
    ) -> IndexingResult: ...


# Load environment variables: CWD .env first (dev override), then XDG config (production)
load_dotenv()
load_dotenv(get_config_dir() / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
# Suppress noisy SQLAlchemy logs even in debug mode
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
# Suppress Chroma telemetry noise
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)

GRPC_ADDRESS_HELP = f"gRPC server address (defaults to {DEFAULT_GRPC_ADDRESS})"

# Dev/Prod port separation
PRODUCTION_PORT = 50051
DEV_PORT = 50052


def _is_dev_invocation() -> bool:
    """Detect if invoked via 'python -m ragzoom.cli' vs 'ragzoom' entry point.

    Returns True for module invocation (development), False for entry point (production).
    This enables automatic dev/prod separation without explicit flags.
    """
    argv0 = sys.argv[0] if sys.argv else ""
    # Module invocation: argv[0] ends with .py or contains ragzoom/cli.py path
    return (
        argv0.endswith(".py") or "ragzoom/cli.py" in argv0 or "ragzoom\\cli.py" in argv0
    )


def _get_default_port() -> int:
    """Get the default port based on invocation mode.

    Returns DEV_PORT for module invocation, PRODUCTION_PORT for entry point.
    """
    return DEV_PORT if _is_dev_invocation() else PRODUCTION_PORT


def _resolve_server_address(value: str | None) -> str:
    """Resolve server address, fail fast if not reachable.

    Args:
        value: Explicit server address (host:port) or None to use default.

    Returns:
        The resolved server address.

    Raises:
        click.ClickException: If the server is not reachable.
    """
    if value:
        address = value
    else:
        env_value = os.environ.get("RAGZOOM_SERVER_ADDRESS")
        if env_value:
            address = env_value
        else:
            address = f"localhost:{_get_default_port()}"

    # Quick TCP connectivity check
    host, port_str = address.rsplit(":", 1)
    port = int(port_str)
    try:
        with socket.create_connection((host, port), timeout=2):
            pass  # Connection succeeded
    except (OSError, TimeoutError):
        raise click.ClickException(
            f"Cannot connect to RagZoom server at {address}.\n"
            f"Start the server with: ragzoom server start"
        )

    return address


def _display_worker_snapshots(
    snapshots: Iterable[WorkerRunSnapshot],
    *,
    target_document_id: str | None,
) -> WorkerRunSnapshot | None:
    focus = {target_document_id} if target_document_id else None
    display = WorkerProgressDisplay(
        focus_documents=focus,
        stream=sys.stderr,
        line_printer=click.echo,
    )

    last_snapshot: WorkerRunSnapshot | None = None
    try:
        for snapshot in snapshots:
            last_snapshot = snapshot
            documents = {
                doc_id: DocumentProgressTotals(
                    inflight=progress.inflight,
                    completed=progress.completed,
                    total=progress.total,
                )
                for doc_id, progress in snapshot.documents.items()
            }
            display.update(
                queue_depth=snapshot.queue_depth,
                inflight=snapshot.inflight,
                documents=documents,
                message=snapshot.message,
            )

            if target_document_id:
                progress = snapshot.documents.get(target_document_id)
                if progress and progress.pending == 0 and progress.inflight == 0:
                    break
    finally:
        display.finish()

    return last_snapshot


def handle_cli_error(e: Exception, operation: str) -> None:
    """Handle CLI errors with appropriate user-friendly messages."""
    # Helpful guidance for optional dependencies
    msg = str(e)
    if isinstance(e, ImportError) and "chromadb" in msg.lower():
        click.echo(
            "\n❌ Missing optional dependency for Chroma.\n\n"
            "You selected the Chroma vector index, but 'chromadb' is not installed.\n"
            "Fix one of the following:\n"
            "  • pip install ragzoom[chroma]\n"
            "  • pip install chromadb\n"
            "  • Or switch to in-memory vector index: export RAGZOOM_VECTOR_BACKEND=python\n\n"
            f"Technical error: {e}",
            err=True,
        )
        sys.exit(1)
    if isinstance(e, DatabaseError):
        click.echo(
            f"\n❌ Database error during {operation}.\n\n"
            "Try these steps:\n"
            "  1. Run 'ragzoom doctor' to check your setup\n"
            "  2. Ensure Docker is running\n"
            "  3. Check README.md for setup instructions\n\n"
            f"Technical error: {e}",
            err=True,
        )
    elif isinstance(e, LLMError):
        msg_lower = str(e).lower()
        if "model" in msg_lower and (
            "not found" in msg_lower or "does not exist" in msg_lower
        ):
            click.echo(
                f"❌ Invalid model specified during {operation}.\n\n"
                "The model name may be incorrect or you may not have access to it.\n"
                "Common models: gpt-4o, gpt-4o-mini, gpt-5, gpt-5-mini\n\n"
                f"Technical error: {e}",
                err=True,
            )
        else:
            click.echo(f"❌ AI service error during {operation}: {e}", err=True)
    elif isinstance(e, ValidationError):
        click.echo(f"❌ Validation error during {operation}: {e}", err=True)
    elif isinstance(e, ConfigurationError):
        click.echo(f"❌ Configuration error during {operation}: {e}", err=True)
    elif isinstance(e, ResourceError):
        click.echo(f"❌ Resource error during {operation}: {e}", err=True)
    elif isinstance(e, NodeNotFoundError):
        click.echo(f"❌ Node not found during {operation}: {e}", err=True)
    elif isinstance(e, RuntimeError) and "currently being modified" in str(e):
        click.echo(
            "❌ Another indexing is already in progress for this document.\n"
            "   Please wait for the current run to finish and try again.",
            err=True,
        )
    else:
        click.echo(f"❌ Error during {operation}: {e}", err=True)
    sys.exit(1)


def configure_logging_level(debug: bool) -> None:
    """Configure logging level based on debug flag."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("ragzoom").setLevel(logging.DEBUG)
    else:
        logging.getLogger("ragzoom").setLevel(logging.INFO)


def setup_command_environment(
    log_level: str | None, debug: bool, validate: bool = False
) -> None:
    """Set up logging and validation for CLI commands."""
    # Configure logging level
    if log_level:
        logging.getLogger("ragzoom").setLevel(getattr(logging, log_level))
    elif debug:
        configure_logging_level(debug)

    # Set global validation flag
    from ragzoom.validate import set_validation_enabled

    set_validation_enabled(validate)


# Keep ragzoom.index at INFO to show batch progress


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """RagZoom: Incremental, hierarchical RAG memory system.

    🚀 Quick Start:
      ragzoom index document.txt
      ragzoom query "your question" -d document.txt

    🔧 Configuration:
      ragzoom config --examples
    """
    # Only create configs, not components
    ctx.ensure_object(dict)
    ctx.obj["index_config"] = IndexConfig.load()  # Load defaults
    ctx.obj["query_config"] = QueryConfig()
    # CLI defaults to Chroma for vector index; fail if not installed
    ctx.obj["operational_config"] = OperationalConfig(vector_backend="chroma")


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--document-id", help="Document ID (defaults to filename)")
@click.option(
    "--debug",
    is_flag=True,
    help="Show debug information including token usage statistics",
)
@click.option(
    "--telemetry",
    "telemetry_file",
    type=click.Path(),
    is_flag=False,
    flag_value="telemetry.json",
    default=None,
    help="Save telemetry data to JSON file",
)
@click.option(
    "--collect-telemetry",
    is_flag=True,
    help="Enable document-scoped telemetry logging without waiting for export",
)
@click.option("--no-progress", is_flag=True, help="Disable progress bar")
@click.option(
    "--append",
    is_flag=True,
    help="Append file contents to an existing document instead of rebuilding. Requires --document-id.",
)
@click.option(
    "--server-address",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default=None,
    show_default=False,
    help=GRPC_ADDRESS_HELP,
)
@click.option(
    "--await-workers/--no-await-workers",
    "await_workers",
    default=True,
    show_default=True,
    help=(
        "Wait for background summarization to finish before exiting. "
        "Disable to exit once leaf ingestion has been scheduled."
    ),
)
@click.option(
    "--summarization-guidance",
    "summarization_guidance",
    default=None,
    help="Custom guidance for summarization (appended to default prompt).",
)
@click.pass_context
def index(
    ctx: click.Context,
    file_path: str,
    document_id: str | None,
    debug: bool,
    telemetry_file: str | None,
    no_progress: bool,
    append: bool,
    server_address: str | None,
    await_workers: bool,
    collect_telemetry: bool,
    summarization_guidance: str | None,
) -> None:
    """Index a document from file.

    Examples:
      ragzoom index document.txt
      ragzoom index notes.txt --document-id my-doc --append
    """

    setup_command_environment(None, debug)

    try:
        append_document_id: str | None = None

        if append:
            if not document_id:
                raise click.UsageError("--document-id is required when using --append")
            append_document_id = document_id

        telemetry_requested = telemetry_file is not None
        collect_requested = collect_telemetry or telemetry_requested

        if telemetry_requested and not await_workers:
            raise click.UsageError(
                "--telemetry cannot be combined with --no-await-workers; the command must wait for telemetry collection."
            )

        result: IndexingResult
        resolved_address = _resolve_server_address(server_address)
        final_snapshot: WorkerRunSnapshot | None = None
        refreshed_status: DocumentWorkStatus | None = None
        refresh_error: str | None = None

        content_bytes = Path(file_path).read_bytes()
        target_document_id = append_document_id or document_id or Path(file_path).name

        click.echo(
            f"Appending {Path(file_path).name} to document '{target_document_id}'..."
            if append
            else f"Indexing {Path(file_path).name}..."
        )

        telemetry_run_id: str | None = None

        with GrpcRagzoomClient(resolved_address) as client:
            append_method = cast(AppendTextCallable, client.append_text)

            target_callable_obj = append_method
            side_effect = getattr(append_method, "side_effect", None)
            if callable(side_effect):
                target_callable_obj = side_effect
            else:
                wrapped = getattr(append_method, "__wrapped__", None)
                if callable(wrapped):
                    target_callable_obj = wrapped

            try:
                params: Mapping[str, inspect.Parameter] = inspect.signature(
                    target_callable_obj
                ).parameters
            except (
                TypeError,
                ValueError,
            ) as exc:  # pragma: no cover - dynamic callables
                params = handle_graceful_error(
                    exc, "Signature introspection failed for append method", default={}
                )

            if "replace_existing" in params:
                result = append_method(
                    document_id=target_document_id,
                    content=content_bytes,
                    collect_telemetry=collect_requested,
                    replace_existing=not append,
                    summarization_guidance=summarization_guidance,
                )
            else:
                result = append_method(
                    document_id=target_document_id,
                    content=content_bytes,
                    collect_telemetry=collect_requested,
                    summarization_guidance=summarization_guidance,
                )
            telemetry_run_id = result.telemetry_run_id
            if await_workers:
                final_snapshot = _display_worker_snapshots(
                    client.iter_worker_snapshots(),
                    target_document_id=target_document_id,
                )

                should_refresh_status = False
                if final_snapshot is None:
                    should_refresh_status = True
                else:
                    progress = final_snapshot.documents.get(target_document_id)
                    if progress:
                        should_refresh_status = (
                            progress.pending == 0 and progress.inflight == 0
                        )
                    else:
                        should_refresh_status = final_snapshot.idle

                if should_refresh_status:
                    try:
                        refreshed_status = client.get_document_work_status(
                            target_document_id
                        )
                    except Exception as exc:  # pragma: no cover - network failures
                        refresh_error = str(exc)

            else:
                click.echo(
                    "ℹ️ Leaf ingestion queued; summarization workers continue in the background."
                )
                click.echo(
                    "   Use `ragzoom status --document-id "
                    f"{target_document_id}` to monitor progress."
                )
                if collect_requested:
                    click.echo(
                        "   Run `ragzoom telemetry-export --document-id "
                        f"{target_document_id}` once workers finish to synthesize telemetry."
                    )

        document_id = target_document_id

        if await_workers:
            if refreshed_status:
                result.tree_depth = refreshed_status.tree_depth
                result.chunks_created = refreshed_status.leaf_count
            elif refresh_error:
                click.echo(
                    f"⚠️ Failed to refresh document status: {refresh_error}",
                    err=True,
                )
        success_message = (
            "✅ Document appended successfully!"
            if append
            else "✅ Document indexed successfully!"
        )

        click.echo(success_message)
        click.echo(f"   Document ID: {result.document_id}")
        click.echo(f"   Total chunks: {result.chunks_created}")
        if result.mutated_nodes is not None:
            resummarized = result.resummarized_nodes or 0
            click.echo(
                f"   Mutated nodes: {result.mutated_nodes} (resummarized {resummarized})"
            )
        if result.new_leaves is not None:
            click.echo(f"   New leaves: {result.new_leaves}")
        if result.tree_depth is not None:
            click.echo(f"   Tree height: {result.tree_depth}")
        if telemetry_run_id:
            click.echo(f"   Telemetry run ID: {telemetry_run_id}")

        if telemetry_requested and await_workers:
            telemetry_path = cast(str, telemetry_file)
            telemetry_payload: TelemetryDataDict | None = None
            telemetry_error: str | None = None

            if collect_requested and telemetry_run_id:
                try:
                    with GrpcRagzoomClient(resolved_address) as telemetry_client:
                        poll_result = telemetry_client.get_telemetry(
                            document_id=target_document_id,
                            run_id=telemetry_run_id,
                            wait=True,
                        )
                except Exception as exc:  # pragma: no cover - network failures
                    telemetry_error = str(exc)
                else:
                    if poll_result.error:
                        telemetry_error = poll_result.error
                    elif poll_result.telemetry:
                        telemetry_payload = poll_result.telemetry

            if telemetry_payload is None and telemetry_error is None:
                try:
                    with GrpcRagzoomClient(resolved_address) as export_client:
                        export_result = export_client.export_document_telemetry(
                            document_id=target_document_id
                        )
                except Exception as exc:  # pragma: no cover - network failures
                    telemetry_error = str(exc)
                else:
                    if export_result.error:
                        telemetry_error = export_result.error
                    elif export_result.telemetry:
                        telemetry_payload = export_result.telemetry
                    else:
                        telemetry_error = "Telemetry data was empty for this document."

            if telemetry_payload:
                try:
                    with open(telemetry_path, "w", encoding="utf-8") as f:
                        json.dump(telemetry_payload, f, indent=2)
                    click.echo(f"✅ Saved telemetry: {telemetry_path}")
                except OSError as exc:
                    telemetry_error = str(exc)

            if telemetry_error:
                click.echo(f"❌ Telemetry export failed: {telemetry_error}", err=True)

        # Show debug hint if enabled
        if debug:
            click.echo(
                "\n💡 Debug information (including token usage statistics) logged to stderr"
            )

    except OSError as e:
        # Clean user-friendly errors (no "Error indexing document" prefix)
        click.echo(str(e), err=True)
        sys.exit(1)
    except Exception as e:
        handle_cli_error(e, "indexing document")


@cli.command()
@click.option(
    "--server-address",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default=None,
    show_default=False,
    help=GRPC_ADDRESS_HELP,
)
@click.pass_context
def documents(ctx: click.Context, server_address: str | None) -> None:
    """List all indexed documents.

    Spec: specs/grpc-cli-architecture.md § Commands Requiring Migration
    """
    try:
        resolved_address = _resolve_server_address(server_address)

        with GrpcRagzoomClient(resolved_address) as client:
            docs = client.list_documents()

        if not docs:
            click.echo("No documents indexed yet.")
            return

        click.echo("\nIndexed Documents:")
        click.echo("-" * 60)

        for doc in docs:
            click.echo(f"\nDocument ID: {doc.document_id}")
            doc_type = "temporal" if doc.is_temporal else "non-temporal"
            click.echo(f"Type: {doc_type}")
            click.echo(f"Total nodes: {doc.node_count}")
            click.echo(f"Leaf nodes: {doc.leaf_count}")
            if doc.completion_pct is not None:
                click.echo(f"Completion: {doc.completion_pct:.1f}%")
            if doc.is_temporal and doc.time_start and doc.time_end:
                click.echo(f"Time range: {doc.time_start} to {doc.time_end}")

    except Exception as e:
        handle_cli_error(e, "listing documents")


@cli.command()
@click.option("--document-id", "document_id", required=True, help="Document ID")
@click.option("--run-id", "run_id", required=True, help="Telemetry run ID")
@click.option(
    "--wait/--no-wait",
    "wait",
    default=False,
    show_default=True,
    help="Block until the telemetry run completes.",
)
@click.option(
    "--output",
    type=str,
    help="Path to write telemetry JSON instead of stdout.",
)
@click.option(
    "--server-address",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default=None,
    show_default=False,
    help=GRPC_ADDRESS_HELP,
)
@click.pass_context
def telemetry(
    ctx: click.Context,
    document_id: str,
    run_id: str,
    wait: bool,
    output: str | None,
    server_address: str | None,
) -> None:
    """Fetch telemetry for a previous indexing run."""

    resolved_address = _resolve_server_address(server_address)

    try:
        with GrpcRagzoomClient(resolved_address) as client:
            response = client.get_telemetry(
                document_id=document_id,
                run_id=run_id,
                wait=wait,
            )
    except Exception as exc:
        handle_cli_error(exc, "fetching telemetry")
        return

    if not response.complete:
        click.echo(
            "⚠️ Telemetry run is still in progress; rerun with --wait or try later.",
            err=True,
        )
        sys.exit(1)

    if response.error:
        click.echo(f"❌ Telemetry run failed: {response.error}", err=True)
        sys.exit(1)

    if response.telemetry is None:
        click.echo(
            "⚠️ Telemetry data is unavailable for this run.",
            err=True,
        )
        return

    if output:
        try:
            with open(output, "w", encoding="utf-8") as fh:
                json.dump(response.telemetry, fh, indent=2)
            click.echo(f"✅ Saved telemetry: {output}")
        except OSError as exc:
            click.echo(f"❌ Failed to write telemetry file: {exc}", err=True)
            sys.exit(1)
    else:
        click.echo(json.dumps(response.telemetry, indent=2))


@cli.command("telemetry-export")
@click.option("--document-id", required=True, help="Document ID to export")
@click.option(
    "--output",
    type=str,
    default="telemetry.json",
    show_default=True,
    help="Path to write synthesized telemetry JSON",
)
@click.option(
    "--server-address",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default=None,
    show_default=False,
    help=GRPC_ADDRESS_HELP,
)
def telemetry_export(document_id: str, output: str, server_address: str | None) -> None:
    """Synthesize document-level telemetry from server logs."""

    resolved_address = _resolve_server_address(server_address)
    try:
        with GrpcRagzoomClient(resolved_address) as client:
            result = client.export_document_telemetry(document_id=document_id)
    except Exception as exc:
        handle_cli_error(exc, "exporting telemetry")
        return

    if result.error:
        click.echo(f"❌ Telemetry export failed: {result.error}", err=True)
        sys.exit(1)

    if result.telemetry is None:
        click.echo(
            "⚠️ Telemetry data was empty for this document; nothing was written.",
            err=True,
        )
        sys.exit(1)

    try:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(result.telemetry, fh, indent=2)
        click.echo(f"✅ Saved telemetry: {output}")
    except OSError as exc:
        click.echo(f"❌ Failed to write telemetry file: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("document_id", type=str)
@click.option(
    "--complete",
    is_flag=True,
    help="Require forest completeness: all sibling pairs have parents, all leaves have embeddings.",
)
@click.option(
    "--telemetry-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help=(
        "Path to telemetry JSON file; when provided, cross-check contents "
        "against stored nodes."
    ),
)
@click.option(
    "--fast",
    is_flag=True,
    help=(
        "Use SQL-only validation for faster results (~7x speedup). "
        "Skips: preceding_context checks, telemetry consistency, vector index checks."
    ),
)
@click.pass_context
def validate(
    ctx: click.Context,
    document_id: str,
    complete: bool,
    telemetry_file: str | None,
    fast: bool,
) -> None:
    """Validate invariants for a document tree."""

    index_config: IndexConfig = ctx.obj["index_config"]
    operational_config = OperationalConfig()

    store = create_store_with_docker(
        operational_config, embedding_model=index_config.embedding_model
    )
    vector_backend = (operational_config.vector_backend or "").strip().lower()
    vector_index = None
    if vector_backend != "python":
        vector_index = create_vector_index(
            operational_config.vector_backend,
            operational_config.database_url,
            index_config.embedding_model,
        )

    telemetry_payload: TelemetryDataDict | None = None
    if telemetry_file:
        try:
            with open(telemetry_file, encoding="utf-8") as fh:
                telemetry_json = json.load(fh)
        except OSError as exc:
            click.echo(f"❌ Failed to read telemetry file: {exc}", err=True)
            raise SystemExit(1)
        except json.JSONDecodeError as exc:
            click.echo(
                f"❌ Telemetry file is not valid JSON: {exc.msg}",
                err=True,
            )
            raise SystemExit(1)

        if not isinstance(telemetry_json, dict):
            click.echo(
                "❌ Telemetry file must contain a JSON object at the top level.",
                err=True,
            )
            raise SystemExit(1)

        telemetry_payload = cast(TelemetryDataDict, telemetry_json)

    report = validate_document(
        document_id=document_id,
        store=store,
        vector_index=vector_index,
        require_complete=complete,
        target_chunk_tokens=index_config.target_chunk_tokens,
        telemetry=telemetry_payload,
        fast=fast,
    )

    heading = (
        "✅ Document validation passed"
        if report.status == "ok"
        else "❌ Document validation failed"
    )
    if complete and report.status == "ok":
        heading += " (complete forest required)"

    click.echo(heading)
    click.echo(
        f"   Nodes: {report.metrics.get('node_count', 0)}, "
        f"Leaves: {report.metrics.get('leaf_count', 0)}, "
        f"Roots: {report.metrics.get('root_count', 0)}"
    )

    # Show pending work if any
    pending_embeddings = report.metrics.get("pending_embeddings", 0)
    pending_summaries = report.metrics.get("pending_summaries", 0)
    if pending_embeddings > 0 or pending_summaries > 0:
        parts = []
        if pending_embeddings > 0:
            parts.append(f"{pending_embeddings} embeddings")
        if pending_summaries > 0:
            parts.append(f"{pending_summaries} summaries")
        click.echo(f"   Pending: {', '.join(parts)}")

    if report.findings:
        click.echo("\nFindings:")
        ordered = sorted(
            report.findings,
            key=lambda finding: 0 if finding.severity == "error" else 1,
        )
        for finding in ordered:
            prefix = "ERROR" if finding.severity == "error" else "WARN"
            suffix = f" (node {finding.node_id})" if finding.node_id else ""
            click.echo(f" - [{prefix}] {finding.message}{suffix}")
    else:
        click.echo("\nNo issues detected.")

    if report.status == "failed":
        raise SystemExit(1)


@cli.command()
@click.argument("query_text", default="")
@click.option("--document-id", "-d", required=True, help="Document ID to query within")
@click.option(
    "--num-seeds",
    type=int,
    help="Number of seed nodes to retrieve (0 for minimal root-only summary)",
)
@click.option("--token-budget", type=int, help="Token budget for summary")
@click.option("--embedding-model", type=str, help="Embedding model for query")
@click.option(
    "--recent-verbatim-token-budget",
    type=int,
    help="Token budget for recent content to include verbatim (most recent first)",
)
@click.option(
    "--span-start",
    type=int,
    default=0,
    help="Start of document window (character position, default: 0)",
)
@click.option(
    "--span-end",
    type=int,
    default=None,
    help="End of document window (character position, default: document end)",
)
@click.option(
    "--time-start",
    type=str,
    default=None,
    help="Start of time window (ISO 8601 with timezone, e.g., 2024-01-21T14:00:00Z)",
)
@click.option(
    "--time-end",
    type=str,
    default=None,
    help="End of time window (ISO 8601 with timezone, e.g., 2024-01-21T15:00:00Z)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Show debug information including retrieval statistics",
)
@click.option(
    "--viz-width",
    type=int,
    help="Override visualization width (defaults to terminal width)",
)
@click.option(
    "--viz-coords",
    type=click.Choice(["source-chars", "output-tokens"]),
    default="source-chars",
    help="Coordinate system for tree visualization (source-chars=source position, output-tokens=output budget)",
)
@click.option(
    "--server-address",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default=None,
    show_default=False,
    help=GRPC_ADDRESS_HELP,
)
@click.option(
    "--profile",
    is_flag=True,
    help="Show detailed timing breakdown for each pipeline phase",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output results as JSON for programmatic consumption",
)
@click.option(
    "--no-bm25",
    is_flag=True,
    help="Disable BM25 hybrid search (use pure vector search)",
)
@click.pass_context
def query(
    ctx: click.Context,
    query_text: str,
    document_id: str,
    num_seeds: int | None,
    token_budget: int | None,
    embedding_model: str | None,
    recent_verbatim_token_budget: int | None,
    span_start: int,
    span_end: int | None,
    time_start: str | None,
    time_end: str | None,
    debug: bool,
    viz_width: int | None,
    viz_coords: str,
    server_address: str | None,
    profile: bool,
    json_output: bool,
    no_bm25: bool,
) -> None:
    """Query the system and get a summary."""
    # Handle query/num_seeds defaults
    if not query_text:
        # No query → minimal summary mode
        if num_seeds is not None and num_seeds > 0:
            raise click.UsageError("Cannot specify --num-seeds > 0 without a query")
        num_seeds = 0
    elif num_seeds is None and token_budget is None:
        # Query provided but no num_seeds/budget → default to 1 seed, unlimited budget
        num_seeds = 1

    setup_command_environment(None, debug)

    try:
        query_config = ctx.obj["query_config"]

        if token_budget is not None:
            query_config = query_config.replace(budget_tokens=token_budget)
        if embedding_model is not None:
            query_config = query_config.replace(embedding_model=embedding_model)
        if no_bm25:
            query_config = query_config.replace(use_bm25=False)

        ctx.obj["query_config"] = query_config

        effective_budget = token_budget or query_config.budget_tokens
        resolved_address = _resolve_server_address(server_address)

        # Calculate visualization width
        if not debug:
            actual_viz_width = viz_width or 0
        elif viz_width:
            actual_viz_width = viz_width
        else:
            terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
            actual_viz_width = max(80, terminal_width - 1)

        use_token_coords = viz_coords == "output-tokens"

        with GrpcRagzoomClient(resolved_address) as client:
            response = client.execute_query(
                query=query_text,
                document_id=document_id,
                budget_tokens=effective_budget,
                num_seeds=num_seeds,
                embedding_model=query_config.embedding_model,
                debug=debug,
                viz_width=actual_viz_width,
                use_token_coords=use_token_coords,
                recent_verbatim_token_budget=recent_verbatim_token_budget,
                profile=profile,
                span_start=span_start,
                span_end=span_end,
                time_start=time_start,
                time_end=time_end,
            )

        query_result = response.query_result
        retrieval = response.retrieval

        # JSON output mode: output structured JSON and return
        if json_output:
            json_data = build_json_output(
                response=response,
                query_text=query_text,
                document_id=document_id,
            )
            click.echo(json.dumps(json_data, indent=2))
            return

        click.echo("\n" + "=" * 60)
        click.echo("SUMMARY")
        click.echo("=" * 60)
        if debug and retrieval.tiling_ids:
            for idx, node_id in enumerate(retrieval.tiling_ids):
                node = retrieval.nodes.get(node_id)
                if node is None:
                    continue
                span = f"{node.span_start}-{node.span_end}"
                height = node.height
                is_seed = node_id in retrieval.selected_ids
                idx_str = f"{idx}{'*' if is_seed else ' '}"
                click.echo(
                    f"[{idx_str}| SPAN: {span} | HEIGHT: {height} | NODE: {node.node_id}]"
                )
                if node.text:
                    click.echo(node.text)
                    if idx < len(retrieval.tiling_ids) - 1:
                        click.echo("")
        else:
            summary_text = query_result.summary or "<no summary>"
            click.echo(summary_text)
        click.echo("")

        if debug and response.visualization:
            click.echo("=" * 60)
            click.echo("VISUALIZATION")
            click.echo("=" * 60)
            click.echo(response.visualization)
            click.echo("")

        if response.validation_warning:
            click.echo(f"⚠️ {response.validation_warning}", err=True)

        click.echo("=" * 60)
        click.echo("STATISTICS")
        click.echo("=" * 60)
        click.echo(f"  Seed nodes: {query_result.seed_count}")
        if query_result.verbatim_count > 0:
            click.echo(f"  Verbatim leaves: {query_result.verbatim_count}")
        click.echo(f"  Tiling size: {query_result.tiling_size}")
        click.echo(f"  Token count: {query_result.token_count}")
        if debug:
            click.echo(f"  Coverage: {len(retrieval.coverage_map)} nodes")

        if profile and response.profile:
            p = response.profile
            click.echo("")
            click.echo("=" * 60)
            click.echo("PROFILE")
            click.echo("=" * 60)
            click.echo("")
            click.echo("Phase Timings:")
            click.echo(f"  embedding:     {p.embedding_ms:8.2f} ms")
            click.echo(f"  search:        {p.search_ms:8.2f} ms")
            click.echo(f"  mmr:           {p.mmr_ms:8.2f} ms")
            click.echo(f"  coverage_map:  {p.coverage_map_ms:8.2f} ms")
            click.echo(f"  scoring:       {p.scoring_ms:8.2f} ms")
            click.echo(f"  tiling:        {p.tiling_ms:8.2f} ms")
            click.echo(f"  assembly:      {p.assembly_ms:8.2f} ms")
            click.echo("  ─────────────────────────")
            click.echo(f"  TOTAL:         {p.total_ms:8.2f} ms")
            click.echo("")
            click.echo("Metrics:")
            click.echo(
                f"  candidates:    {p.candidates_retrieved} retrieved, {p.candidates_filtered} after filter"
            )
            click.echo(f"  seeds:         {p.seeds_found}/{p.seeds_requested} found")
            click.echo(f"  coverage:      {p.coverage_size} nodes")
            click.echo(
                f"  tiling:        {p.tiling_size} nodes -> {p.output_tokens} tokens"
            )
            if p.embedding_model:
                click.echo(f"  model:         {p.embedding_model}")

    except Exception as e:
        if json_output:
            error_data = build_json_error_from_exception(e)
            click.echo(json.dumps(error_data, indent=2))
            sys.exit(1)
        handle_cli_error(e, "processing query")


@cli.command()
@click.option(
    "--server-address",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default=None,
    show_default=False,
    help=GRPC_ADDRESS_HELP,
)
@click.pass_context
def status(ctx: click.Context, server_address: str | None) -> None:
    """Show system status.

    Spec: specs/grpc-cli-architecture.md § Commands Requiring Migration
    """
    try:
        resolved_address = _resolve_server_address(server_address)

        with GrpcRagzoomClient(resolved_address) as client:
            system_status = client.get_system_status()

        index_config = ctx.obj["index_config"]
        query_config = ctx.obj["query_config"]

        click.echo("\nSYSTEM STATUS:")
        click.echo("=" * 40)
        click.echo(f"Total nodes: {system_status.total_nodes}")
        click.echo(f"Leaf nodes: {system_status.leaf_nodes}")
        click.echo(f"Tree height: {system_status.tree_depth}")
        click.echo("\nCONFIGURATION:")
        click.echo("=" * 40)
        click.echo(f"Budget tokens: {query_config.budget_tokens}")
        click.echo(f"Target chunk tokens: {index_config.target_chunk_tokens}")
        click.echo(f"MMR lambda: {query_config.mmr_lambda}")

    except Exception as e:
        handle_cli_error(e, "getting status")


@cli.command()
@click.argument("document_id")
@click.option(
    "--server-address",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default=None,
    show_default=False,
    help=GRPC_ADDRESS_HELP,
)
def cost(document_id: str, server_address: str | None) -> None:
    """Display cost statistics for a document.

    Shows total indexing cost, node counts, and per-node metrics.

    Spec: specs/grpc-cli-architecture.md § Commands Requiring Migration

    Examples:
      ragzoom cost my-document
      ragzoom cost e0d9b972-3bad-472f-a570-a4e02d0a1ff4
    """
    try:
        resolved_address = _resolve_server_address(server_address)

        with GrpcRagzoomClient(resolved_address) as client:
            stats_list = client.get_cost_stats(document_id)

        if not stats_list:
            click.echo(f"Document '{document_id}' not found.", err=True)
            sys.exit(1)

        stats = stats_list[0]
        click.echo(f"\nDocument: {stats.document_id}")
        click.echo(
            f"Total nodes: {stats.total_nodes:,} "
            f"({stats.leaf_nodes:,} leaves, {stats.summary_nodes:,} summaries)"
        )
        click.echo()

        if stats.total_cost > 0:
            click.echo(f"Total cost:     ${stats.total_cost:.4f}")
            if stats.total_nodes > 0:
                cost_per_node = stats.total_cost / stats.total_nodes
                click.echo(f"Per node (avg): ${cost_per_node:.6f}")
        else:
            click.echo("No cost data recorded for this document.")
            click.echo("(Cost tracking was added after this document was indexed)")

    except Exception as e:
        handle_cli_error(e, "getting cost statistics")


@cli.command("document-status")
@click.argument("document_id")
@click.option(
    "--server-address",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default=None,
    show_default=False,
    help=GRPC_ADDRESS_HELP,
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def document_status(
    ctx: click.Context,
    document_id: str,
    server_address: str | None,
    output_json: bool,
) -> None:
    """Display document status and completion metrics.

    Shows document existence, node counts, indexing completion percentage,
    and temporal range for documents with temporal metadata.

    Examples:
      ragzoom document-status my-document
      ragzoom document-status session-abc123 --json
    """
    try:
        resolved_address = _resolve_server_address(server_address)

        with GrpcRagzoomClient(resolved_address) as client:
            status = client.get_document_status(document_id)

        if output_json:
            # JSON output format matching spec
            output = {
                "document_id": status.document_id,
                "exists": status.exists,
                "is_temporal": status.is_temporal,
                "leaf_count": status.leaf_count,
                "node_count": status.node_count,
                "complete_forest_size": status.complete_forest_size,
                "completion_pct": status.completion_pct,
                "time_start": status.time_start,
                "time_end": status.time_end,
            }
            click.echo(json.dumps(output, indent=2))
        else:
            # Human-readable output format matching spec
            click.echo(f"Document: {status.document_id}")
            if not status.exists:
                click.echo("Status: does not exist")
            else:
                doc_type = "temporal" if status.is_temporal else "non-temporal"
                click.echo(f"Type: {doc_type}")
                click.echo(f"Leaves: {status.leaf_count}")
                click.echo(
                    f"Nodes: {status.node_count} / {status.complete_forest_size} "
                    f"({status.completion_pct:.1f}% complete)"
                )
                if status.is_temporal and status.time_start and status.time_end:
                    click.echo(f"Time range: {status.time_start} to {status.time_end}")

    except Exception as e:
        handle_cli_error(e, "getting document status")


@cli.command()
@click.option(
    "--host", default="127.0.0.1", help="Host to bind to"
)  # nosec B104 - secure default
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the REST API server."""
    try:
        import uvicorn

        click.echo(f"Starting RagZoom API server on {host}:{port}...")
        uvicorn.run(
            "ragzoom.api:app",
            host=host,
            port=port,
            reload=reload,
        )
    except Exception as e:
        handle_cli_error(e, "starting server")


@cli.command()
@click.option("--document-id", "-d", help="Clear only a specific document")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def clear(ctx: click.Context, document_id: str | None, confirm: bool) -> None:
    """Clear data from the database via the gRPC server."""

    try:
        resolved_address = _resolve_server_address(ctx.obj.get("server_address"))

        with GrpcRagzoomClient(resolved_address) as client:
            if document_id:
                if not confirm:
                    click.confirm(
                        f"⚠️  This will delete document '{document_id}' and all its data. Are you sure?",
                        abort=True,
                    )

                result = client.clear_document(document_id)
                if result.document_existed:
                    click.echo(
                        f"✅ Cleared document '{document_id}' ({result.deleted_nodes} nodes deleted)"
                    )
                else:
                    click.echo(f"⚠️ Document '{document_id}' does not exist")
            else:
                if not confirm:
                    click.confirm(
                        "⚠️  This will delete ALL data. Are you sure?", abort=True
                    )

                results = client.clear_all_documents()
                existing = [entry for entry in results if entry.document_existed]
                deleted_count = sum(entry.deleted_nodes for entry in existing)
                click.echo(f"✅ Cleared {deleted_count} nodes from the database")

    except click.Abort:
        click.echo("❌ Clear operation cancelled")
    except Exception as e:
        handle_cli_error(e, "clearing database")


@cli.command("inspect")
@click.argument("node_id")
@click.pass_context
def inspect_node(
    ctx: click.Context,
    node_id: str,
) -> None:
    """Inspect a node's summary, source text, and preceding context.

    NODE_ID can be a prefix (e.g. first 8 characters of the UUID).
    If multiple nodes match the prefix, they will be listed.

    Examples:
      ragzoom inspect 73d5aa10
      ragzoom inspect 73d5aa10-1234-5678-abcd-ef1234567890
    """
    try:
        operational_config = ctx.obj["operational_config"]
        index_config: IndexConfig = ctx.obj["index_config"]

        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )

        # Use document-agnostic store to find nodes by prefix
        global_store = store.for_document(None)
        matches = global_store.nodes.get_by_prefix(node_id)

        if not matches:
            click.echo(f"No nodes found matching prefix '{node_id}'.", err=True)
            sys.exit(1)

        if len(matches) > 1:
            click.echo(f"Multiple nodes match prefix '{node_id}':")
            for match in matches:
                click.echo(
                    f"  {match.id} (height={match.height}, level={match.level_index})"
                )
            click.echo("\nPlease provide a more specific prefix.")
            sys.exit(1)

        # Single match - display full node details
        node = matches[0]

        # Create doc_store for this node's document
        doc_store = store.for_document(node.document_id)

        # Get children
        left_child, right_child = doc_store.tree.get_children(node.id)

        # Reconstruct source text from children
        source_parts: list[str] = []
        if left_child:
            source_parts.append(left_child.text)
        if right_child:
            source_parts.append(right_child.text)
        source_text = "\n".join(source_parts) if source_parts else "(no children)"

        # Reconstruct preceding context from tiling node IDs
        preceding_text: str | None = None
        if node.preceding_context:
            tiling_ids: list[str] = json.loads(node.preceding_context)
            if tiling_ids:
                tiling_texts: list[str] = []
                for tiling_id in tiling_ids:
                    tiling_node = doc_store.nodes.get(tiling_id)
                    if tiling_node and tiling_node.text:
                        tiling_texts.append(tiling_node.text)
                if tiling_texts:
                    preceding_text = "\n".join(tiling_texts)

        # Print formatted output
        click.echo(f"\nNODE: {node.id}")
        click.echo(
            f"Height: {node.height} | Level Index: {node.level_index} | "
            f"Span: {node.span_start}-{node.span_end}"
        )

        click.echo("\n── PRECEDING CONTEXT " + "─" * 55)
        if preceding_text:
            click.echo(preceding_text)
        else:
            click.echo("(none)")

        if node.preceding_context_summary:
            click.echo("\n── PRECEDING CONTEXT SUMMARY " + "─" * 47)
            click.echo(node.preceding_context_summary)

        click.echo("\n── SOURCE TEXT (children concatenated) " + "─" * 37)
        click.echo(source_text)

        click.echo("\n── SUMMARY (this node's text) " + "─" * 45)
        click.echo(node.text or "(empty)")

    except Exception as e:
        handle_cli_error(e, "inspecting node")


@cli.command()
@click.argument("output_file", type=click.Path())
@click.option("--format", type=click.Choice(["json", "text"]), default="text")
@click.option(
    "--stream/--no-stream",
    default=False,
    help="Stream output to avoid loading all nodes in memory",
)
@click.pass_context
def export(ctx: click.Context, output_file: str, format: str, stream: bool) -> None:
    """Export tree structure to file."""
    try:
        # Create services for this command
        operational_config = ctx.obj["operational_config"]
        index_config = ctx.obj["index_config"]
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )

        output_path = Path(output_file)

        if stream:
            # Streaming mode: write incrementally without building large lists
            if format == "json":
                with output_path.open("w", encoding="utf-8") as f:
                    f.write("[")
                    first = True
                    for doc in store.list_documents():
                        ds = store.for_document(doc.id)
                        batches = ds.nodes.get_all_paginated(page_size=1000)
                        for batch in batches:
                            for node in batch:
                                node_dict = {
                                    "id": node.id,
                                    "parent_id": node.parent_id,
                                    "height": node.height,
                                    "span_start": node.span_start,
                                    "span_end": node.span_end,
                                    "is_leaf": (
                                        node.left_child_id is None
                                        and node.right_child_id is None
                                    ),
                                    "text_preview": (
                                        node.text[:100] + "..."
                                        if len(node.text) > 100
                                        else node.text
                                    ),
                                }
                                if not first:
                                    f.write(",\n")
                                f.write(json.dumps(node_dict))
                                first = False
                    f.write("]\n")
            else:  # text
                with output_path.open("w", encoding="utf-8") as f:
                    for doc in store.list_documents():
                        ds = store.for_document(doc.id)
                        batches = ds.nodes.get_all_paginated(page_size=1000)
                        for batch in batches:
                            for node in batch:
                                # Coerce to ints to satisfy type checker and ensure stability
                                height_val = getattr(node, "height", 0)
                                try:
                                    height = int(height_val)
                                except Exception:
                                    height = 0
                                indent = "  " * height
                                leaf_marker = (
                                    "🍃"
                                    if (
                                        node.left_child_id is None
                                        and node.right_child_id is None
                                    )
                                    else "📁"
                                )
                                node_id_short = (
                                    node.id[:8]
                                    if isinstance(node.id, str) and len(node.id) > 8
                                    else str(node.id)
                                )
                                # Coerce spans to ints defensively
                                try:
                                    span_start = int(getattr(node, "span_start", 0))
                                except Exception:
                                    span_start = 0
                                try:
                                    span_end = int(getattr(node, "span_end", 0))
                                except Exception:
                                    span_end = 0
                                preview = (
                                    node.text[:100] + "..."
                                    if len(node.text) > 100
                                    else node.text
                                )
                                f.write(
                                    f"{indent}{leaf_marker} {node_id_short} [{span_start},{span_end}) {preview}\n"
                                )
            click.echo(f"✅ Exported data to {output_file} (streaming mode)")
        else:
            # Legacy collect-and-write mode
            nodes_data = []
            for doc in store.list_documents():
                ds = store.for_document(doc.id)
                for node in ds.nodes.get_all():
                    node_dict = {
                        "id": node.id,
                        "parent_id": node.parent_id,
                        "height": node.height,
                        "span_start": node.span_start,
                        "span_end": node.span_end,
                        "is_leaf": (
                            node.left_child_id is None and node.right_child_id is None
                        ),
                        "text_preview": (
                            node.text[:100] + "..."
                            if len(node.text) > 100
                            else node.text
                        ),
                    }
                    nodes_data.append(node_dict)

            if format == "json":
                output_path.write_text(json.dumps(nodes_data, indent=2))
            else:
                lines = []
                for node_dict in sorted(
                    nodes_data, key=lambda x: (x["height"], x["span_start"])
                ):
                    height_val = node_dict.get("height", 0)
                    height = int(height_val) if isinstance(height_val, int) else 0
                    indent = "  " * height
                    leaf_marker = "🍃" if node_dict.get("is_leaf") else "📁"
                    node_id = node_dict.get("id", "")
                    node_id_short = (
                        node_id[:8]
                        if isinstance(node_id, str) and len(node_id) > 8
                        else str(node_id)
                    )
                    ss = node_dict.get("span_start", 0)
                    se = node_dict.get("span_end", 0)
                    span_start = int(ss) if isinstance(ss, int) else 0
                    span_end = int(se) if isinstance(se, int) else 0
                    lines.append(
                        f"{indent}{leaf_marker} {node_id_short} [{span_start},{span_end}) {node_dict.get('text_preview','')}"
                    )
                output_path.write_text("\n".join(lines))

            click.echo(f"✅ Exported {len(nodes_data)} nodes to {output_file}")

    except Exception as e:
        handle_cli_error(e, "exporting data")


@cli.command()
@click.option(
    "--examples", is_flag=True, help="Show configuration examples and common use cases"
)
@click.option(
    "--create",
    "output_file",
    type=click.Path(),
    help="Create a sample configuration file at the specified path",
)
def config(examples: bool, output_file: str | None) -> None:
    """Manage configuration files and show examples."""

    if examples:
        click.echo("\n🔧 RAGZOOM CONFIGURATION EXAMPLES\n")
        click.echo("=" * 50)

        click.echo("\n📄 Basic Configuration (development.json):")
        click.echo(
            json.dumps(
                {
                    "target_chunk_tokens": 150,
                    "embedding_model": "text-embedding-3-small",
                    "max_retries": 0,
                    "budget_tokens": 4000,
                    "mmr_lambda": 0.7,
                },
                indent=2,
            )
        )

        click.echo("\n🚀 Production Configuration (production.json):")
        click.echo(
            json.dumps(
                {
                    "target_chunk_tokens": 300,
                    "embedding_model": "text-embedding-3-large",
                    "retry_threshold": 0.15,
                    "max_retries": 2,
                    "embedding_batch_size": 50,
                    "budget_tokens": 8000,
                    "mmr_lambda": 0.8,
                    "mmr_k_multiplier": 2.5,
                },
                indent=2,
            )
        )

        click.echo("\n⚡ Fast Configuration (fast.json):")
        click.echo(
            json.dumps(
                {
                    "target_chunk_tokens": 100,
                    "embedding_model": "text-embedding-3-small",
                    "max_retries": 0,
                    "embedding_batch_size": 200,
                    "budget_tokens": 2000,
                    "mmr_lambda": 0.6,
                },
                indent=2,
            )
        )

        click.echo("\n💰 Cost-Optimized Configuration (budget.json):")
        click.echo(
            json.dumps(
                {
                    "target_chunk_tokens": 200,
                    "embedding_model": "text-embedding-3-small",
                    "max_retries": 0,
                    "budget_tokens": 4000,
                    "mmr_lambda": 0.7,
                },
                indent=2,
            )
        )

        click.echo("\n📖 Usage:")
        click.echo("  ragzoom index doc.txt --config development.json")
        click.echo("  ragzoom query 'question' -d doc.txt --config production.json")
        click.echo("  ragzoom config --create my-config.json")

        click.echo("\n💡 Tips:")
        click.echo("  • Start with development.json for initial testing")
        click.echo("  • Use production.json for high-quality results")
        click.echo("  • Use fast.json for rapid iteration")
        click.echo("  • Use budget.json to minimize API costs")
        click.echo("  • CLI options override config file settings")

    elif output_file:
        # Create a sample config file
        sample_config = {
            "target_chunk_tokens": 200,
            "embedding_model": "text-embedding-3-small",
            "retry_threshold": 0.2,
            "max_retries": 0,
            "embedding_batch_size": 100,
            "budget_tokens": 8000,
            "mmr_lambda": 0.7,
            "mmr_k_multiplier": 2.0,
        }

        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(sample_config, f, indent=2)

        click.echo(f"✅ Created sample configuration file: {output_file}")
        click.echo("\n💡 Edit this file to customize your settings.")
        click.echo("   Use 'ragzoom config --examples' to see common configurations.")

    else:
        click.echo("\n🔧 Configuration Management")
        click.echo("\nOptions:")
        click.echo("  ragzoom config --examples     Show configuration examples")
        click.echo("  ragzoom config --create FILE  Create a sample config file")
        click.echo("\nFor detailed help: ragzoom config --help")


@cli.group()
def server() -> None:
    """Manage the RagZoom gRPC server."""


def _persist_daemon_config(config_path: Path) -> None:
    """Persist relevant config fields for auto-start.

    Reads the config file and extracts fields that affect daemon behavior,
    saving them to daemon.config.json for use by ensure_server_running().

    Args:
        config_path: Path to the config file to read.
    """
    from ragzoom.config import IndexConfig

    # Load the config to get resolved values
    index_cfg = IndexConfig.load(config_path=config_path)

    # Extract fields that affect daemon behavior
    daemon_config: dict[str, str | int | float | bool | None] = {}

    # target_chunk_tokens is critical for temporal documents
    daemon_config["target_chunk_tokens"] = index_cfg.target_chunk_tokens

    # summarization_guidance affects summary quality
    if index_cfg.summarization_guidance is not None:
        daemon_config["summarization_guidance"] = index_cfg.summarization_guidance

    # Persist the config
    write_config_file(daemon_config)


@server.command("start")
@click.option(
    "--host",
    default=DEFAULT_GRPC_HOST,
    show_default=True,
    help="Host to bind",
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="Port to bind (default: 50051 for prod, 50052 for dev)",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional indexing config file",
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option(
    "--collect-telemetry",
    is_flag=True,
    help="Persist document-scoped telemetry logs on the server",
)
@click.option(
    "--telemetry-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to store telemetry events (defaults near data dir)",
)
@click.option(
    "--max-parallelism",
    type=int,
    help="Maximum concurrent indexing jobs (default: 30)",
)
@click.option(
    "--preceding-context-leaf-num-seeds",
    "preceding_context_leaf_num_seeds",
    type=int,
    help="Number of seeds for leaf node preceding context",
)
@click.option(
    "--preceding-context-leaf-verbatim-tokens",
    "preceding_context_leaf_verbatim_tokens",
    type=int,
    help="Verbatim token budget for leaf node preceding context",
)
@click.option(
    "--preceding-context-leaf-min-forest-completeness",
    "preceding_context_leaf_min_forest_completeness",
    type=float,
    help="Min forest completeness for leaf nodes (0.0-1.0)",
)
@click.option(
    "--preceding-context-leaf-token-cap",
    "preceding_context_leaf_token_cap",
    type=int,
    help="Token cap for leaf node preceding context (select rightmost N tokens)",
)
@click.option(
    "--preceding-context-inner-num-seeds",
    "preceding_context_inner_num_seeds",
    type=int,
    help="Number of seeds for inner node preceding context",
)
@click.option(
    "--preceding-context-inner-verbatim-tokens",
    "preceding_context_inner_verbatim_tokens",
    type=int,
    help="Verbatim token budget for inner node preceding context",
)
@click.option(
    "--preceding-context-inner-min-forest-completeness",
    "preceding_context_inner_min_forest_completeness",
    type=float,
    help="Min forest completeness for inner nodes (0.0-1.0)",
)
@click.option(
    "--preceding-context-inner-token-cap",
    "preceding_context_inner_token_cap",
    type=int,
    help="Token cap for inner node preceding context (select rightmost N tokens)",
)
@click.option(
    "--daemon",
    is_flag=True,
    help="Run as background daemon (fork to background, redirect output to log file)",
)
@click.option(
    "--http-port",
    "http_port",
    default=None,
    type=int,
    help="Enable HTTP REST API on this port (for sandboxed clients using curl)",
)
@click.option(
    "--summary-model",
    "summary_model",
    type=str,
    default=None,
    help=(
        "LLM model for hierarchical summarization "
        "(overrides RAGZOOM_SUMMARY_MODEL and config file)"
    ),
)
@click.option(
    "--summary-api-base",
    "summary_api_base",
    type=str,
    default=None,
    help=(
        "Endpoint override for the summary model, e.g. a proxy URL "
        "(overrides RAGZOOM_SUMMARY_API_BASE and config file)"
    ),
)
@click.option(
    "--search-agent-model",
    "search_agent_model",
    type=str,
    default=None,
    help="LLM model for the agentic search agent (default: gpt-5-mini)",
)
@click.option(
    "--search-max-iterations",
    "search_max_iterations",
    type=int,
    default=None,
    help="Maximum recall iterations for search agent (default: 5)",
)
@click.option(
    "--search-max-budget",
    "search_max_budget",
    type=int,
    default=None,
    help="Maximum token budget per recall call in search (default: 4000)",
)
@click.option(
    "--search-profiling",
    "search_profiling",
    is_flag=True,
    help="Enable search profiling (iteration traces, retrospective, cost)",
)
def start_server(
    host: str,
    port: int | None,
    config_path: Path | None,
    debug: bool,
    collect_telemetry: bool,
    telemetry_dir: Path | None,
    max_parallelism: int | None,
    preceding_context_leaf_num_seeds: int | None,
    preceding_context_leaf_verbatim_tokens: int | None,
    preceding_context_leaf_min_forest_completeness: float | None,
    preceding_context_leaf_token_cap: int | None,
    preceding_context_inner_num_seeds: int | None,
    preceding_context_inner_verbatim_tokens: int | None,
    preceding_context_inner_min_forest_completeness: float | None,
    preceding_context_inner_token_cap: int | None,
    daemon: bool,
    http_port: int | None,
    summary_model: str | None,
    summary_api_base: str | None,
    search_agent_model: str | None,
    search_max_iterations: int | None,
    search_max_budget: int | None,
    search_profiling: bool,
) -> None:
    """Start the RagZoom gRPC server."""

    # Resolve port: use explicit value or fall back to dev/prod default
    is_dev = _is_dev_invocation()
    resolved_port = port if port is not None else _get_default_port()

    # Log which mode we're running in
    mode_str = "dev" if is_dev else "production"
    logger.info(f"Starting server in {mode_str} mode on port {resolved_port}")

    # If daemon mode, fork to background before starting server
    if daemon:
        daemonize()
        # Note: Port file is written in app.py AFTER lease acquisition.
        # This ensures clients only connect once the daemon is truly ready.
        # Install signal handlers for graceful shutdown (SIGTERM/SIGINT)
        install_shutdown_handlers()
        # Register atexit cleanup for normal exits (when run_server returns)
        # This ensures state files are cleaned up even without signals
        atexit.register(cleanup_stale_state)

    # Persist config for auto-start if a config file is provided.
    # This runs for both daemon and foreground modes so auto-start
    # works regardless of how the server was originally started.
    if config_path:
        _persist_daemon_config(config_path)

    # Configure logging AFTER daemonization (if any) so log output
    # goes to the daemon process, not the parent that exits
    setup_command_environment(None, debug)

    options = ServerOptions(
        host=host,
        port=resolved_port,
        http_port=http_port,
        config_path=str(config_path) if config_path else None,
        collect_telemetry=collect_telemetry,
        telemetry_dir=str(telemetry_dir) if telemetry_dir else None,
        max_parallelism=max_parallelism,
        preceding_context_leaf_num_seeds=preceding_context_leaf_num_seeds,
        preceding_context_leaf_verbatim_tokens=preceding_context_leaf_verbatim_tokens,
        preceding_context_leaf_min_forest_completeness=preceding_context_leaf_min_forest_completeness,
        preceding_context_leaf_token_cap=preceding_context_leaf_token_cap,
        preceding_context_inner_num_seeds=preceding_context_inner_num_seeds,
        preceding_context_inner_verbatim_tokens=preceding_context_inner_verbatim_tokens,
        preceding_context_inner_min_forest_completeness=preceding_context_inner_min_forest_completeness,
        preceding_context_inner_token_cap=preceding_context_inner_token_cap,
        summary_model=summary_model,
        summary_api_base=summary_api_base,
        search_agent_model=search_agent_model,
        search_max_iterations=search_max_iterations,
        search_max_budget=search_max_budget,
        search_profiling=search_profiling,
    )

    # In daemon mode, wrap run_server in try/finally for belt-and-suspenders cleanup.
    # This ensures state files are cleaned up when run_server raises an exception,
    # complementing the atexit handler which covers normal exits.
    # cleanup_stale_state is idempotent, so calling it from both paths is safe.
    if daemon:
        try:
            run_server(options)
        finally:
            cleanup_stale_state()
    else:
        run_server(options)


# Constants for stop command
STOP_TIMEOUT_SECONDS = 10.0
STOP_POLL_INTERVAL = 0.2


@server.command("stop")
def stop_server() -> None:
    """Stop the RagZoom daemon.

    Sends SIGTERM to the daemon process for graceful shutdown,
    then waits for the process to terminate. Cleans up state files
    (PID and port files) after shutdown.

    If no daemon is running, this command does nothing (idempotent).
    """
    pid = read_pid_file()

    # No PID file means no daemon
    if pid is None:
        click.echo("Daemon is not running")
        return

    # Check if process is already dead (stale PID file)
    if is_pid_stale(pid):
        click.echo("Daemon is not running (cleaning up stale state)")
        cleanup_stale_state()
        return

    # Send SIGTERM for graceful shutdown
    click.echo(f"Stopping daemon (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        # Process died between check and kill - that's fine
        pass

    # Wait for process to terminate
    deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if is_pid_stale(pid):
            # Process terminated
            cleanup_stale_state()
            click.echo("Stopped")
            return
        time.sleep(STOP_POLL_INTERVAL)

    # Timeout - process didn't terminate gracefully
    click.echo("Timeout waiting for daemon to stop (forcing cleanup)")
    cleanup_stale_state()


@server.command("status")
def server_status() -> None:
    """Show daemon status.

    Displays whether the daemon is running, and if so, shows:
    - PID (process ID)
    - Port (if known)
    - Uptime (how long the daemon has been running)

    Does NOT auto-start the daemon - just reports current state.
    """
    pid = read_pid_file()
    if pid is None or is_pid_stale(pid):
        click.echo("Not running")
        return

    # Daemon is running - get details
    port = read_port_file()
    uptime = get_process_uptime(pid)

    # Build status line
    parts = [f"Running: PID {pid}"]
    if port is not None:
        parts.append(f"port {port}")
    parts.append(f"uptime {uptime}")

    click.echo(", ".join(parts))


# Constants for logs command
DEFAULT_LOG_LINES = 50


@server.command("logs")
@click.option(
    "-n",
    "--lines",
    "num_lines",
    default=DEFAULT_LOG_LINES,
    type=int,
    help=f"Number of lines to show (default: {DEFAULT_LOG_LINES})",
)
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    help="Follow log output (like tail -f)",
)
def server_logs(num_lines: int, follow: bool) -> None:
    """Show daemon logs.

    Displays contents of the daemon log file. By default shows the
    last 50 lines. Use -n to change the number of lines, or -f to
    continuously follow new output.
    """
    log_file = get_log_file_path()

    if not log_file.exists():
        click.echo("No log file found (daemon may not have run yet)")
        return

    if follow:
        import subprocess

        try:
            subprocess.run(
                ["tail", "-n", str(num_lines), "-f", str(log_file)],
                check=False,
            )
        except KeyboardInterrupt:
            pass
        return

    try:
        content = log_file.read_text()
    except OSError as e:
        click.echo(f"Error reading log file: {e}")
        return

    if not content:
        return

    for line in content.splitlines()[-num_lines:]:
        click.echo(line)


@cli.command()
def doctor() -> None:
    """Check system setup and diagnose potential issues."""
    import subprocess

    click.echo("🏥 RagZoom System Check")
    click.echo("=" * 24)

    issues_found = False

    # Check Python environment
    click.echo(f"✅ Python: {sys.version.split()[0]} ({sys.executable})")

    # Check for virtual environment
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        click.echo("✅ Virtual environment: Active")
    else:
        click.echo("⚠️  Virtual environment: None (consider using one)")

    # Check Docker availability (only if using postgres backend)
    from ragzoom.config import OperationalConfig

    chroma_available = True
    try:
        _cfg = OperationalConfig(vector_backend="chroma")
    except ImportError:
        chroma_available = False
        # Fall back to python vector backend for diagnostics so doctor can continue
        _cfg = OperationalConfig(vector_backend="python")

    if _cfg.backend == "sqlite":
        click.echo("✅ Backend: SQLite (file-backed)")
        click.echo("   Skipping Docker checks")
        # Report vector index availability
        if chroma_available:
            click.echo("✅ Vector index: Chroma available")
        else:
            click.echo("⚠️  Vector index: Chroma not installed")
            click.echo("   Install with: pip install chromadb")
    else:
        try:
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                click.echo(f"✅ Docker: {version}")

                # Check Docker daemon
                try:
                    subprocess.run(
                        ["docker", "ps"], capture_output=True, check=True, timeout=5
                    )
                    click.echo("✅ Docker daemon: Running")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    click.echo("❌ Docker daemon: Not running")
                    click.echo(
                        "   Start Docker Desktop or run: sudo systemctl start docker"
                    )
                    issues_found = True
            else:
                raise subprocess.CalledProcessError(result.returncode, "docker")

        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            click.echo("❌ Docker: Not found or not working")
            click.echo("   Install Docker Desktop from https://docker.com")
            issues_found = True

    # Check PostgreSQL container status
    if not issues_found and _cfg.backend != "sqlite":  # Only if using postgres
        try:
            from ragzoom.docker_postgres import DockerPostgres

            docker_pg = DockerPostgres()
            status = docker_pg.get_status()

            if status["container_exists"]:
                if status["container_running"]:
                    if status["postgres_ready"]:
                        click.echo("✅ PostgreSQL: Running and ready")
                        click.echo(f"   Container: {docker_pg.container_name}")
                        click.echo(f"   Connection: {status['connection_url']}")
                    else:
                        click.echo("⚠️  PostgreSQL: Container running but not ready")
                        click.echo(
                            "   Container may be starting up, wait a moment and try again"
                        )
                else:
                    click.echo("⚠️  PostgreSQL: Container exists but not running")
                    click.echo(f"   Run: docker start {docker_pg.container_name}")
            else:
                click.echo("⚠️  PostgreSQL: No container found")
                click.echo("   Will be created automatically on first use")

        except ImportError:
            click.echo("❌ PostgreSQL management: Not available")
            issues_found = True

    # Test database connection
    if not issues_found:
        try:
            click.echo("\n🔗 Testing database connection...")

            # Create operational config
            operational_config = OperationalConfig()

            # Try to create a store (auto-start only if postgres)
            store = create_store_with_docker(operational_config)

            # Test basic operation without exposing sessions
            # Attempt a lightweight repository call
            _ = store.list_documents()

            click.echo("✅ Database connection: Working")
            store.close()

        except Exception as e:
            click.echo("❌ Database connection: Failed")
            click.echo(f"   Error: {e}")
            issues_found = True

    # Check OpenAI API key
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        if api_key.startswith("sk-") and len(api_key) > 20:
            click.echo("✅ OpenAI API key: Set")
        else:
            click.echo("⚠️  OpenAI API key: Set but format looks invalid")
            click.echo("   Should start with 'sk-' and be longer than 20 characters")
    else:
        click.echo("⚠️  OpenAI API key: Not set")
        click.echo("   Set OPENAI_API_KEY environment variable or add to .env file")

    # Summary
    click.echo("\n📋 Summary")
    if not issues_found:
        click.echo("🎉 System looks good! You're ready to use RagZoom.")
        click.echo("\nTry: ragzoom index document.txt")
    else:
        click.echo("⚠️  Issues found. Please address the problems above.")
        click.echo(
            "\nFor help, see: https://github.com/eumemic/dynamic-summary#installation"
        )


# Telemetry commands are available via optional dependencies
# Install with: pip install ragzoom[telemetry]
# Usage: ragzoom-telemetry analyze|compare|visualize


@cli.group()
def eval() -> None:
    """Summary quality evaluation commands."""


@eval.command("measure")
@click.argument("document_id")
@click.option(
    "--num-samples",
    "-n",
    default=100,
    type=int,
    help="Number of nodes to sample (-1 for all)",
)
@click.option(
    "--model",
    "-m",
    default="gpt-5-nano",
    help="Model for evaluation",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for evaluation JSON (default: <document_id>.eval.json)",
)
@click.pass_context
def measure(
    ctx: click.Context,
    document_id: str,
    num_samples: int,
    model: str,
    output: str | None,
) -> None:
    """Evaluate summary quality for a document using LLM-as-judge.

    Evaluates summaries on four dimensions:
      - Retention: Keeps most important details
      - Isolation: Doesn't bleed facts from preceding context
      - Faithfulness: No hallucination or knowledge contamination
      - Continuity: Flows smoothly from preceding context

    Saves evaluations to JSON for later analysis with 'ragzoom eval report'.

    Examples:
      ragzoom eval measure my-document
      ragzoom eval measure my-document -n 50 --model gpt-4o
      ragzoom eval measure my-document -n -1  # All nodes
    """
    try:
        operational_config = ctx.obj["operational_config"]
        index_config: IndexConfig = ctx.obj["index_config"]

        eval_model = model

        # Get store and document
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )
        doc_store = store.for_document(document_id)

        # Verify document exists
        doc_meta = doc_store.get_metadata()
        if doc_meta is None:
            click.echo(f"Document '{document_id}' not found.", err=True)
            sys.exit(1)

        # Get all inner nodes (height > 0)
        all_nodes = doc_store.nodes.get_all()
        inner_nodes = [n for n in all_nodes if n.height > 0]

        if not inner_nodes:
            click.echo(
                "No inner nodes found (document may only have leaves).", err=True
            )
            sys.exit(1)

        # Filter to valid inner nodes (those with both children)
        valid_inner_nodes = []
        for node in inner_nodes:
            left_child, right_child = doc_store.tree.get_children(node.id)
            if left_child is not None and right_child is not None:
                valid_inner_nodes.append((node, left_child, right_child))

        total_inner = len(valid_inner_nodes)

        if total_inner == 0:
            click.echo(
                "No valid inner nodes found (all nodes missing children).", err=True
            )
            sys.exit(1)

        # Sample nodes
        if num_samples == -1 or num_samples >= total_inner:
            selected_nodes = valid_inner_nodes
        else:
            selected_nodes = random.sample(valid_inner_nodes, num_samples)

        # Prepare node data for evaluation
        # Tuple format: (node_id, summary, source_text, preceding_context,
        #                height, level_index, span_start, compression_ratio)
        node_data: list[tuple[str, str, str, str | None, int, int, int, float]] = []
        for node, left_child, right_child in selected_nodes:

            # Reconstruct the preceding context that was used during summarization
            # by fetching and concatenating the tiling node texts
            preceding_text: str | None = None
            if node.preceding_context:
                import json

                tiling_ids: list[str] = json.loads(node.preceding_context)
                if tiling_ids:
                    tiling_texts: list[str] = []
                    for tiling_id in tiling_ids:
                        tiling_node = doc_store.nodes.get(tiling_id)
                        if tiling_node and tiling_node.text:
                            tiling_texts.append(tiling_node.text)
                    if tiling_texts:
                        preceding_text = "\n\n".join(tiling_texts)

            # Concatenate children texts as the summarizer sees them
            source_text = left_child.text + right_child.text

            # Compression ratio: children tokens / summary tokens
            children_tokens = left_child.token_count + right_child.token_count
            compression = (
                children_tokens / node.token_count if node.token_count > 0 else 1.0
            )

            node_data.append(
                (
                    node.id,
                    node.text,
                    source_text,
                    preceding_text,
                    node.height,
                    node.level_index,
                    node.span_start,
                    compression,
                )
            )

        if not node_data:
            click.echo("No valid inner nodes to evaluate.", err=True)
            sys.exit(1)

        # Run evaluation
        click.echo(f"\nEvaluating {len(node_data)} of {total_inner} inner nodes...")
        click.echo(f"Model: {eval_model}")

        from tqdm import tqdm

        from ragzoom.adapters.chat_model_factory import build_chat_model
        from ragzoom.evaluation import evaluate_nodes
        from ragzoom.evaluation.types import NodeEvaluation

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            click.echo("OPENAI_API_KEY environment variable not set.", err=True)
            sys.exit(1)

        async def run_evaluation() -> list[NodeEvaluation]:
            chat_model = build_chat_model(eval_model, api_key=api_key)

            def update_progress() -> None:
                pbar.update(1)

            with tqdm(total=len(node_data), unit="node", leave=False) as pbar:
                return await evaluate_nodes(
                    nodes=node_data,
                    chat_model=chat_model,
                    max_concurrent=30,
                    on_progress=update_progress,
                )

        evaluations = asyncio.run(run_evaluation())

        # Save to JSON
        output_path = output or f"{document_id}.eval.json"
        eval_data = {
            "document_id": document_id,
            "evaluations": [e.to_dict() for e in evaluations],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2)

        click.echo(f"\n✅ Saved {len(evaluations)} evaluations to {output_path}")

    except Exception as e:
        handle_cli_error(e, "evaluating document")


@eval.command("report")
@click.argument("eval_file", type=click.Path(exists=True))
@click.option(
    "--threshold",
    "-t",
    default=3.0,
    type=float,
    help="Minimum mean score to pass (1-5)",
)
@click.option(
    "--model",
    "-m",
    default="gpt-5-nano",
    help="Model for issue synthesis",
)
def report(
    eval_file: str,
    threshold: float,
    model: str,
) -> None:
    """Generate a quality report from evaluation JSON.

    Loads evaluations from a JSON file created by 'ragzoom eval measure',
    runs issue synthesis, and prints a formatted report.

    Examples:
      ragzoom eval report my-document.eval.json
      ragzoom eval report my-document.eval.json --threshold 4.0
    """
    try:
        # Load evaluations from JSON
        with open(eval_file, encoding="utf-8") as f:
            data = json.load(f)

        document_id = data["document_id"]
        eval_dicts = data["evaluations"]

        from ragzoom.evaluation import (
            EvaluationReport,
            NodeEvaluation,
            generate_issue_summary,
        )
        from ragzoom.evaluation import (
            print_report as print_eval_report,
        )

        evaluations = [NodeEvaluation.from_dict(e) for e in eval_dicts]

        click.echo(f"Loaded {len(evaluations)} evaluations for '{document_id}'")

        # Build report
        eval_report = EvaluationReport(
            document_id=document_id,
            total_inner_nodes=len(evaluations),
            nodes_evaluated=len(evaluations),
            evaluations=evaluations,
        )

        # Generate issue summary
        from tqdm import tqdm

        from ragzoom.adapters.chat_model_factory import build_chat_model
        from ragzoom.evaluation.issue_summary import RecurringIssue

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            click.echo("OPENAI_API_KEY environment variable not set.", err=True)
            sys.exit(1)

        # 10 parallel theme identification + 1 synthesis = 11 LLM calls
        num_parallel = 10

        async def run_synthesis() -> list[RecurringIssue]:
            chat_model = build_chat_model(model, api_key=api_key)

            def update_progress() -> None:
                pbar.update(1)

            with tqdm(
                total=num_parallel + 1, unit="call", desc="Analyzing", leave=False
            ) as pbar:
                return await generate_issue_summary(
                    eval_report, chat_model, num_parallel, update_progress
                )

        issues = asyncio.run(run_synthesis())

        # Print report
        print_eval_report(eval_report, threshold, issues)

        # Exit with appropriate code
        if not eval_report.passed(threshold):
            sys.exit(1)

    except json.JSONDecodeError as e:
        click.echo(f"Invalid JSON in evaluation file: {e}", err=True)
        sys.exit(1)
    except KeyError as e:
        click.echo(f"Missing required field in evaluation file: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        handle_cli_error(e, "generating report")


@cli.command("reset-session")
@click.argument("session_id")
@click.option(
    "--user-id",
    envvar="RAGZOOM_USER_ID",
    required=True,
    help="User ID for multi-tenant isolation (or set RAGZOOM_USER_ID)",
)
@click.option(
    "--server",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default="localhost:50051",
    help="gRPC server address",
)
def reset_session_cmd(session_id: str, user_id: str, server: str) -> None:
    """Reset a session's cursor to force full re-sync.

    Clears the sync state on the server, causing the next sync to
    re-process the entire transcript from scratch.

    Example:
      ragzoom reset-session 7cdd0798-4f29-4ce6-bfc9-6dc3b7bb2153
    """
    try:
        with GrpcRagzoomClient(server) as client:
            success, message = client.reset_session_cursor(
                session_id=session_id, user_id=user_id
            )
            if success:
                click.echo(f"✅ {message}")
            else:
                click.echo(f"❌ {message}", err=True)
                raise SystemExit(1)
    except Exception as e:
        handle_cli_error(e, "resetting session")


if __name__ == "__main__":
    cli()
