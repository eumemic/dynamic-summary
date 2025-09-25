"""CLI interface for RagZoom."""

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from ragzoom.client import GrpcRagzoomClient, WorkerRunSnapshot
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
from ragzoom.exceptions import (
    ConfigurationError,
    DatabaseError,
    InvalidOperationError,
    LLMError,
    NodeNotFoundError,
    ResourceError,
    ValidationError,
)
from ragzoom.server.app import ServerOptions, run_server
from ragzoom.services.document_service import DocumentService
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.store import create_store_with_docker
from ragzoom.worktree_utils import DEFAULT_DATA_DIR_NAME

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


def _display_worker_snapshots(snapshots: list[WorkerRunSnapshot]) -> None:
    if not snapshots:
        return

    for snapshot in snapshots:
        if snapshot.documents:
            details = ", ".join(
                f"{doc_id}(pending={pending}, inflight={inflight})"
                for doc_id, (pending, inflight) in sorted(snapshot.documents.items())
            )
            suffix = f" [{details}]"
        else:
            suffix = ""

        click.echo(snapshot.message)
        click.echo(
            f"   Workers: queue={snapshot.queue_depth}, inflight={snapshot.inflight}{suffix}"
        )


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
        logging.getLogger("ragzoom").setLevel(logging.DEBUG)
    else:
        logging.getLogger("ragzoom").setLevel(logging.INFO)


def setup_command_environment(
    log_level: str | None, debug: bool, validate: bool
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
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Load configuration from JSON file",
)
@click.option(
    "--target-chunk-tokens", type=int, help="Target size for leaf chunks in tokens"
)
@click.option(
    "--preceding-context-tokens", type=int, help="Context tokens from adjacent chunks"
)
@click.option("--summary-model", "-m", type=str, help="Model for summarization")
@click.option("--embedding-model", type=str, help="Model for embeddings")
@click.option(
    "--retry-threshold", type=float, help="Max deviation before retry (0.0-1.0)"
)
@click.option("--max-retries", type=int, help="Maximum summary retries")
@click.option("--embedding-batch-size", type=int, help="Batch size for embeddings")
@click.option(
    "--data-dir",
    type=click.Path(),
    help="Directory for data storage (default: current directory)",
)
@click.option(
    "--database",
    type=str,
    help="Database URL (sqlite:///path/to.db or postgresql+psycopg://host/db)",
)
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
@click.option("--validate", is_flag=True, help="Enable validation checks")
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
@click.pass_context
def index(
    ctx: click.Context,
    file_path: str,
    document_id: str | None,
    config_path: Path | None,
    target_chunk_tokens: int | None,
    preceding_context_tokens: int | None,
    summary_model: str | None,
    embedding_model: str | None,
    retry_threshold: float | None,
    max_retries: int | None,
    embedding_batch_size: int | None,
    data_dir: str | None,
    database: str | None,
    debug: bool,
    telemetry_file: str | None,
    validate: bool,
    no_progress: bool,
    append: bool,
    server_address: str | None,
) -> None:
    """Index a document from file.

    Configuration can be set via:
    1. CLI options (highest priority)
    2. Config file specified with --config
    3. Default values from internal configuration

    Examples:
      ragzoom index document.txt --target-chunk-tokens 300 --summary-model gpt-5-nano
      ragzoom index document.txt --config myconfig.json
    """

    setup_command_environment(None, debug, validate)

    try:
        # Load indexing configuration with CLI overrides
        # Create config object with merged values
        index_config = IndexConfig.load(
            config_path,
            target_chunk_tokens=target_chunk_tokens,
            preceding_context_tokens=preceding_context_tokens,
            summary_model=summary_model,
            embedding_model=embedding_model,
            retry_threshold=retry_threshold,
            max_retries=max_retries,
            embedding_batch_size=embedding_batch_size,
        )
        query_config = QueryConfig()  # Use defaults for indexing command
        # Prefer Chroma by default in CLI; fail loudly if unavailable
        operational_config = OperationalConfig(vector_backend="chroma")

        # Override storage location if provided
        if data_dir:
            data_path = Path(data_dir)
            # Choose URL based on backend
            if operational_config.backend == "sqlite":
                from ragzoom.worktree_utils import get_default_sqlite_url

                operational_config = operational_config.replace(
                    database_url=get_default_sqlite_url(data_path)
                )
            else:
                # For postgres, form a sensible DB name under base_dir
                dbname = (data_path / DEFAULT_DATA_DIR_NAME / "ragzoom").name
                operational_config = operational_config.replace(
                    database_url=f"postgresql+psycopg://localhost/{dbname}"
                )

        if database:
            operational_config = operational_config.replace(database_url=database)

        # Update context with new configs
        ctx.obj["index_config"] = index_config
        ctx.obj["query_config"] = query_config
        ctx.obj["operational_config"] = operational_config

        append_document_id: str | None = None

        if append:
            if not document_id:
                raise click.UsageError("--document-id is required when using --append")
            append_document_id = document_id

        result: IndexingResult
        resolved_address = _resolve_server_address(server_address)

        if append:
            assert append_document_id is not None
            click.echo(
                f"Appending {Path(file_path).name} to document '{append_document_id}'..."
            )
            content_bytes = Path(file_path).read_bytes()
            with GrpcRagzoomClient(resolved_address) as client:
                result = client.append_text(
                    document_id=append_document_id,
                    content=content_bytes,
                    collect_telemetry=bool(telemetry_file),
                )
                _display_worker_snapshots(client.run_workers_once())
            success_message = "✅ Document appended successfully!"
        else:
            click.echo(f"Indexing {Path(file_path).name}...")
            path = Path(file_path)
            content_bytes = path.read_bytes()
            target_document_id = document_id or path.name
            with GrpcRagzoomClient(resolved_address) as client:
                result = client.index_document(
                    document_id=target_document_id,
                    content=content_bytes,
                    collect_telemetry=bool(telemetry_file),
                )
                _display_worker_snapshots(client.run_workers_once())
            document_id = target_document_id
            success_message = "✅ Document indexed successfully!"

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

        telemetry_data = result.telemetry

        if telemetry_file and telemetry_data:
            try:
                with open(telemetry_file, "w", encoding="utf-8") as f:
                    json.dump(telemetry_data, f, indent=2)
                click.echo(f"✅ Saved telemetry: {telemetry_file}")
            except OSError as exc:
                click.echo(f"❌ Failed to write telemetry file: {exc}", err=True)

        if validate:
            click.echo(
                "⚠️ Validation checks are not yet supported when using the gRPC server.",
                err=True,
            )

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
@click.argument("query_text")
@click.option("--document-id", "-d", required=True, help="Document ID to query within")
@click.option("--num-seeds", type=int, help="Number of seed nodes to retrieve")
@click.option("--token-budget", type=int, help="Token budget for summary")
@click.option("--embedding-model", type=str, help="Embedding model for query")
@click.option(
    "--debug",
    is_flag=True,
    help="Show debug information including retrieval statistics",
)
@click.option("--validate", is_flag=True, help="Enable validation checks")
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
    "--server-address",
    envvar="RAGZOOM_SERVER_ADDRESS",
    default=None,
    show_default=False,
    help=GRPC_ADDRESS_HELP,
)
@click.pass_context
def query(
    ctx: click.Context,
    query_text: str,
    document_id: str,
    num_seeds: int | None,
    token_budget: int | None,
    embedding_model: str | None,
    debug: bool,
    validate: bool,
    viz_width: int | None,
    viz_coords: str,
    server_address: str | None,
) -> None:
    """Query the system and get a summary."""

    setup_command_environment(None, debug, validate)

    try:
        query_config = ctx.obj["query_config"]

        if token_budget is not None:
            query_config = query_config.replace(budget_tokens=token_budget)
        if embedding_model is not None:
            query_config = query_config.replace(embedding_model=embedding_model)

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
            )

        query_result = response.query_result
        retrieval = response.retrieval

        if validate:
            click.echo(
                "⚠️ Query validation is not yet supported when using the gRPC server.",
                err=True,
            )

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
        click.echo(f"  Nodes retrieved: {query_result.nodes_retrieved}")
        click.echo(f"  Tiling size: {query_result.tiling_size}")
        click.echo(f"  Token count: {query_result.token_count}")
        if debug:
            click.echo(f"  Coverage: {len(retrieval.coverage_map)} nodes")

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
    """Clear data from the database.

    Without --document-id, clears all data.
    With --document-id, clears only the specified document.
    """
    try:
        # Create services for this command
        operational_config = ctx.obj["operational_config"]
        index_config = ctx.obj["index_config"]
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )
        document_service = DocumentService(store)

        if document_id:
            # Clear specific document
            if not confirm:
                click.confirm(
                    f"⚠️  This will delete document '{document_id}' and all its data. Are you sure?",
                    abort=True,
                )

            deleted_count = document_service.clear_document(document_id)
            # Clear vectors for this document (fail-soft)
            try:
                from ragzoom.vector_factory import create_vector_index

                # Use the document's embedding model if available
                doc_row = store.get_document_by_id(document_id)
                emb_model = (
                    doc_row.embedding_model
                    if doc_row and doc_row.embedding_model
                    else index_config.embedding_model
                )
                vindex = create_vector_index(
                    operational_config.vector_backend,
                    operational_config.database_url,
                    emb_model,
                )
                vindex.delete(filter={"document_id": document_id})
            except Exception:
                pass
            click.echo(
                f"✅ Cleared document '{document_id}' ({deleted_count} nodes deleted)"
            )
        else:
            # Clear all data
            if not confirm:
                click.confirm("⚠️  This will delete ALL data. Are you sure?", abort=True)

            deleted_count = document_service.clear_all_documents()
            # Also clear all vectors for all documents
            try:
                from ragzoom.vector_factory import create_vector_index

                # Attempt to clear vectors by iterating all documents first
                docs = store.list_documents()
                if docs:
                    # Use embedding model from first doc, or default
                    emb_model = (
                        docs[0].embedding_model
                        if hasattr(docs[0], "embedding_model")
                        else index_config.embedding_model
                    )
                    vindex = create_vector_index(
                        operational_config.vector_backend,
                        operational_config.database_url,
                        emb_model,
                    )
                    for doc in docs:
                        try:
                            vindex.delete(filter={"document_id": doc.id})
                        except Exception:
                            continue
            except Exception:
                pass
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
@click.option("--validate", is_flag=True, help="Enable validation checks")
def start_server(
    host: str,
    port: int,
    config_path: Path | None,
    debug: bool,
    validate: bool,
) -> None:
    """Start the RagZoom gRPC server."""

    setup_command_environment(None, debug, validate)
    options = ServerOptions(
        host=host,
        port=port,
        config_path=str(config_path) if config_path else None,
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
