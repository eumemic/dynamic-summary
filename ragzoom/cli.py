"""CLI interface for RagZoom."""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol, cast

import click
from dotenv import load_dotenv

from ragzoom.client import (
    DocumentStatusView,
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
    DEFAULT_GRPC_PORT,
)
from ragzoom.error_handling import handle_graceful_error
from ragzoom.exceptions import (
    ConfigurationError,
    DatabaseError,
    InvalidOperationError,
    LLMError,
    NodeNotFoundError,
    ResourceError,
    ValidationError,
)
from ragzoom.progress_display import DocumentProgressTotals, WorkerProgressDisplay
from ragzoom.server.app import ServerOptions, run_server
from ragzoom.services.document_service import DocumentService
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
    ) -> IndexingResult: ...


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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


def _resolve_server_address(value: str | None) -> str:
    """Resolve the gRPC server address with sensible defaults."""
    if value:
        return value
    env_value = os.environ.get("RAGZOOM_SERVER_ADDRESS")
    if env_value:
        return env_value
    return DEFAULT_GRPC_ADDRESS


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
                    pending=progress.pending,
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
        refreshed_status: DocumentStatusView | None = None
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
                )
            else:
                result = append_method(
                    document_id=target_document_id,
                    content=content_bytes,
                    collect_telemetry=collect_requested,
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
                        refreshed_status = client.get_document_status(
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
@click.pass_context
def documents(ctx: click.Context) -> None:
    """List all indexed documents."""
    try:
        # Create services for this command
        operational_config = ctx.obj["operational_config"]
        index_config = ctx.obj["index_config"]
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )
        document_service = DocumentService(store)

        # Get all documents
        docs = document_service.list_documents()

        if not docs:
            click.echo("No documents indexed yet.")
            return

        click.echo("\nIndexed Documents:")
        click.echo("-" * 60)

        for doc in docs:
            click.echo(f"\nDocument ID: {doc.document_id}")
            if doc.file_path:
                click.echo(f"File: {doc.file_path}")
            click.echo(f"Indexed: {doc.indexed_at}")
            click.echo(f"Total nodes: {doc.node_count}")

            # Calculate leaf count via DocumentStore
            doc_store = store.for_document(doc.document_id)
            leaf_count = len(doc_store.nodes.get_leaves())
            click.echo(f"Leaf nodes: {leaf_count}")

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
    help="Require the document tree to be fully summarized (single root).",
)
@click.option(
    "--telemetry-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help=(
        "Path to telemetry JSON file; when provided, cross-check contents "
        "against stored nodes."
    ),
)
@click.pass_context
def validate(
    ctx: click.Context,
    document_id: str,
    complete: bool,
    telemetry_file: str | None,
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
    )

    heading = (
        "✅ Document validation passed"
        if report.status == "ok"
        else "❌ Document validation failed"
    )
    if complete and report.status == "ok":
        heading += " (complete tree required)"

    click.echo(heading)
    click.echo(
        f"   Nodes: {report.metrics.get('node_count', 0)}, "
        f"Leaves: {report.metrics.get('leaf_count', 0)}, "
        f"Parentless: {report.metrics.get('parentless_count', 0)}"
    )

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
@click.argument("query_text")
@click.option("--document-id", "-d", required=True, help="Document ID to query within")
@click.option("--num-seeds", type=int, help="Number of seed nodes to retrieve")
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
    default="output-tokens",
    help="Coordinate system for tree visualization (source-chars=source position, output-tokens=output budget)",
)
@click.option(
    "--tiling-strategy",
    type=click.Choice(["dp", "greedy"]),
    default=None,
    help="Tiling algorithm to use (dp=dynamic programming, greedy=frontier roll-up)",
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
    debug: bool,
    viz_width: int | None,
    viz_coords: str,
    tiling_strategy: str | None,
    server_address: str | None,
    profile: bool,
) -> None:
    """Query the system and get a summary."""

    setup_command_environment(None, debug)

    try:
        query_config = ctx.obj["query_config"]

        if token_budget is not None:
            query_config = query_config.replace(budget_tokens=token_budget)
        if embedding_model is not None:
            query_config = query_config.replace(embedding_model=embedding_model)
        if tiling_strategy is not None:
            query_config = query_config.replace(tiling_strategy=tiling_strategy)

        ctx.obj["query_config"] = query_config

        effective_budget = token_budget or query_config.budget_tokens
        resolved_address = _resolve_server_address(server_address)

        if debug:
            if viz_width:
                actual_viz_width = viz_width
            else:
                terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
                actual_viz_width = max(80, terminal_width - 1)
        else:
            actual_viz_width = viz_width or 0

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
                tiling_strategy=query_config.tiling_strategy,
                recent_verbatim_token_budget=recent_verbatim_token_budget,
                profile=profile,
                span_start=span_start,
                span_end=span_end,
            )

        query_result = response.query_result
        retrieval = response.retrieval

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
        handle_cli_error(e, "processing query")


@cli.command()
@click.argument("node_id")
@click.option(
    "--document-id",
    help="Document ID (optional - will be auto-detected from node)",
)
@click.pass_context
def pin(ctx: click.Context, node_id: str, document_id: str | None) -> None:
    """Pin a node to always include it.

    The node must belong to a document and be within the allowed pinning depth.
    Document ID is optional - it will be auto-detected from the node if not provided.
    """
    try:
        # Create services for this command
        operational_config = ctx.obj["operational_config"]
        index_config = ctx.obj["index_config"]
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )

        document_service = DocumentService(store)
        # If a document was provided, verify the node belongs to it
        if document_id:
            ds = store.for_document(document_id)
            if not ds.nodes.get_node(node_id):
                click.echo(
                    f"❌ Node {node_id} not found"
                    + (f" in document {document_id}" if document_id else "")
                )
                sys.exit(1)
        # Delegate to service to find and pin the node
        document_service.pin_node(node_id)

        click.echo(f"✅ Node {node_id} pinned successfully!")

    except NodeNotFoundError:
        click.echo(f"❌ Node {node_id} not found")
        sys.exit(1)
    except InvalidOperationError as e:
        click.echo(f"❌ Failed to pin node {node_id}: {e}")
        sys.exit(1)

    except Exception as e:
        handle_cli_error(e, "pinning node")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status."""
    try:
        # Get configs and create services
        index_config = ctx.obj["index_config"]
        query_config = ctx.obj["query_config"]
        operational_config = ctx.obj["operational_config"]
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )
        document_service = DocumentService(store)

        # Get system status
        status = document_service.get_system_status()

        click.echo("\nSYSTEM STATUS:")
        click.echo("=" * 40)
        click.echo(f"Total nodes: {status.total_nodes}")
        click.echo(f"Leaf nodes: {status.leaf_nodes}")
        click.echo(f"Tree height: {status.tree_depth}")
        click.echo(f"Pinned nodes: {status.pinned_nodes}")
        click.echo("\nCONFIGURATION:")
        click.echo("=" * 40)
        click.echo(f"Budget tokens: {query_config.budget_tokens}")
        click.echo(f"Target chunk tokens: {index_config.target_chunk_tokens}")
        click.echo(f"MMR lambda: {query_config.mmr_lambda}")

    except Exception as e:
        handle_cli_error(e, "getting status")


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
                    "preceding_context_tokens": 100,
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
            "preceding_context_tokens": 75,
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


@server.command("start")
@click.option(
    "--host",
    default=DEFAULT_GRPC_HOST,
    show_default=True,
    help="Host to bind",
)
@click.option(
    "--port",
    default=DEFAULT_GRPC_PORT,
    type=int,
    show_default=True,
    help="Port to bind",
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
def start_server(
    host: str,
    port: int,
    config_path: Path | None,
    debug: bool,
    collect_telemetry: bool,
    telemetry_dir: Path | None,
) -> None:
    """Start the RagZoom gRPC server."""

    setup_command_environment(None, debug)
    options = ServerOptions(
        host=host,
        port=port,
        config_path=str(config_path) if config_path else None,
        collect_telemetry=collect_telemetry,
        telemetry_dir=str(telemetry_dir) if telemetry_dir else None,
    )
    run_server(options)


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


if __name__ == "__main__":
    cli()
