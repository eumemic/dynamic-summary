"""CLI interface for RagZoom."""

import json
import logging
import shutil
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from ragzoom.assemble import Assembler
from ragzoom.config import (
    IndexConfig,
    OperationalConfig,
    QueryConfig,
)
from ragzoom.exceptions import InvalidOperationError, NodeNotFoundError
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import TreeNode, create_store_with_docker
from ragzoom.tree_viz import build_ascii_tree

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
    ctx.obj["operational_config"] = OperationalConfig()


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
    help="PostgreSQL database URL (default: postgresql://localhost/ragzoom)",
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
        operational_config = OperationalConfig()

        # Override database URL if provided
        if data_dir:
            data_path = Path(data_dir)
            operational_config = operational_config.replace(
                database_url=f"postgresql:///{data_path / 'ragzoom'}",
            )

        if database:
            operational_config = operational_config.replace(database_url=database)

        # Update context with new configs
        ctx.obj["index_config"] = index_config
        ctx.obj["query_config"] = query_config
        ctx.obj["operational_config"] = operational_config

        # Create components for this command
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )
        tree_builder = TreeBuilder(
            index_config,
            store,
            api_key=operational_config.openai_api_key,
        )

        # Read file
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")

        # Determine document ID (use provided ID or filename)
        if not document_id:
            document_id = path.name

        # Always clear existing data for the document (handles both complete and interrupted indexing)
        deleted_count = store.clear_document(document_id)
        if deleted_count > 0:
            click.echo(f"Clearing existing data for '{document_id}'...")
            click.echo(f"   Cleared {deleted_count} nodes")

        click.echo(f"Indexing {path.name}...")

        # Index with telemetry if requested
        if telemetry_file:
            doc_id, telemetry = tree_builder.add_document_with_telemetry(
                text,
                document_id=document_id,
                file_path=str(path.absolute()),
                show_progress=not no_progress,
            )
        else:
            doc_id = tree_builder.add_document(
                text,
                document_id=document_id,
                file_path=str(path.absolute()),
                show_progress=not no_progress,
            )

        # Get stats

        # Get leaf nodes for this specific document
        with store.SessionLocal() as session:
            doc_leaves = (
                session.query(TreeNode)
                .filter_by(document_id=doc_id)
                .filter(
                    TreeNode.left_child_id.is_(None), TreeNode.right_child_id.is_(None)
                )
                .all()
            )

        # Get root node for this document
        with store.SessionLocal() as session:
            root = (
                session.query(TreeNode)
                .filter_by(document_id=doc_id, parent_id=None)
                .first()
            )

        click.echo("✅ Document indexed successfully!")
        click.echo(f"   Document ID: {doc_id}")
        click.echo(f"   Chunks created: {len(doc_leaves)}")
        tree_height = root.height if root else 0
        click.echo(f"   Tree height: {tree_height}")

        # Run validation checks
        from ragzoom.validate import (
            validate as run_validate,
        )
        from ragzoom.validate import (
            validate_chunk_sizes,
            validate_document_coverage,
            validate_equal_leaf_depth,
            validate_tree_structure,
        )

        # Validations will run only if --validate was passed
        run_validate(
            lambda: validate_document_coverage(text, doc_leaves), "document coverage"
        )

        run_validate(
            lambda: validate_chunk_sizes(doc_leaves, index_config.target_chunk_tokens),
            "chunk sizes",
        )

        run_validate(
            lambda: validate_tree_structure(store, doc_id, text), "tree structure"
        )

        run_validate(
            lambda: validate_equal_leaf_depth(store, doc_id), "equal leaf depth"
        )

        # Show debug hint if enabled
        if debug:
            click.echo(
                "\n💡 Debug information (including token usage statistics) logged to stderr"
            )

        # Save telemetry if requested
        if telemetry_file:
            # telemetry_file will be either the flag_value or the user-provided path
            output_file = telemetry_file

            click.echo(f"\n📁 Saving telemetry to {output_file}...")

            # Telemetry data is already flat - just save it directly
            # The telemetry data from finalize() already contains all necessary information
            telemetry_data = telemetry

            with open(output_file, "w") as f:
                json.dump(telemetry_data, f, indent=2)

            click.echo(f"✅ Telemetry saved to {output_file}")

    except OSError as e:
        # Clean user-friendly errors (no "Error indexing document" prefix)
        click.echo(str(e), err=True)
        sys.exit(1)
    except Exception as e:
        # Handle database and other unexpected errors
        error_msg = str(e).lower()
        if "connection" in error_msg or "postgresql" in error_msg:
            click.echo(
                "\n❌ Database connection failed.\n\n"
                "Try these steps:\n"
                "  1. Run 'ragzoom doctor' to check your setup\n"
                "  2. Ensure Docker is running\n"
                "  3. Check README.md for setup instructions\n\n"
                f"Technical error: {type(e).__name__}",
                err=True,
            )
        else:
            click.echo(f"❌ Error indexing document: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def documents(ctx: click.Context) -> None:
    """List all indexed documents."""
    try:
        # Create components for this command
        operational_config = ctx.obj["operational_config"]
        index_config = ctx.obj["index_config"]
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )

        # Get all unique documents
        with store.SessionLocal() as session:
            from ragzoom.store import Document

            docs = session.query(Document).all()

        if not docs:
            click.echo("No documents indexed yet.")
            return

        click.echo("\nIndexed Documents:")
        click.echo("-" * 60)

        for doc in docs:
            # Get document stats
            with store.SessionLocal() as session:
                from ragzoom.store import TreeNode

                node_count = (
                    session.query(TreeNode).filter_by(document_id=doc.id).count()
                )
                leaf_count = (
                    session.query(TreeNode)
                    .filter_by(document_id=doc.id)
                    .filter(
                        TreeNode.left_child_id.is_(None),
                        TreeNode.right_child_id.is_(None),
                    )
                    .count()
                )

            click.echo(f"\nDocument ID: {doc.id}")
            if doc.file_path:
                click.echo(f"File: {doc.file_path}")
            click.echo(f"Indexed: {doc.indexed_at}")
            click.echo(f"Chunks: {doc.chunk_count}")
            click.echo(f"Total nodes: {node_count}")
            click.echo(f"Leaf nodes: {leaf_count}")

    except Exception as e:
        click.echo(f"❌ Error listing documents: {e}", err=True)
        sys.exit(1)


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
) -> None:
    """Query the system and get a summary."""

    setup_command_environment(None, debug, validate)

    try:
        # Get configs from context (set up during CLI initialization)
        query_config = ctx.obj["query_config"]
        operational_config = ctx.obj["operational_config"]

        # Override with CLI parameters if provided
        if token_budget is not None:
            query_config = query_config.replace(budget_tokens=token_budget)
        if embedding_model is not None:
            query_config = query_config.replace(embedding_model=embedding_model)

        # Create components for this command
        store = create_store_with_docker(
            operational_config, embedding_model=query_config.embedding_model
        )
        retriever = Retriever(
            query_config,
            store,
            api_key=operational_config.openai_api_key,
        )
        assembler = Assembler(store)

        # Retrieve with CLI parameters
        result = retriever.retrieve(
            query_text,
            budget_tokens=query_config.budget_tokens,
            document_id=document_id,
            num_seeds=num_seeds,
        )

        # Assemble
        summary = assembler.assemble(result)
        token_count = assembler.get_token_count(summary)

        # Tiling validation (validation now simplified - always off unless debug)
        if debug and getattr(result, "tiling", None) and result.tiling:
            from ragzoom.validate import validate_tiling

            error = validate_tiling(
                result.tiling,
                store,
                document_id,
                budget_tokens=query_config.budget_tokens,
            )
            if error:
                click.echo(f"⚠️ Tiling validation warning: {error}", err=True)

        # Output summary
        click.echo("\n" + "=" * 60)
        click.echo("SUMMARY")
        click.echo("=" * 60)
        if debug and getattr(result, "tiling", None) and result.tiling:
            for idx, node_id in enumerate(result.tiling):
                node = store.get_node(node_id)
                if node:
                    # Node span is always the full span
                    span_start, span_end = node.span_start, node.span_end
                    span = f"{span_start}-{span_end}"
                    height = node.height
                    # Add asterisk to index if this is a seed node
                    is_seed = node_id in result.node_ids
                    idx_str = f"{idx}{'*' if is_seed else ' '}"
                    click.echo(
                        f"[{idx_str}| SPAN: {span} | HEIGHT: {height} | NODE: {node.id}]"
                    )
                    # Get the node text
                    text = node.text
                    click.echo(text)
                    if idx < len(result.tiling) - 1:
                        click.echo("")
        else:
            click.echo(summary)
        click.echo("")

        # Show debug info if requested
        if debug:
            # Show ASCII tree visualization first
            if result.tiling:
                # Get terminal width with fallback, or use CLI override
                if viz_width:
                    actual_viz_width = viz_width
                else:
                    terminal_width = shutil.get_terminal_size(
                        fallback=(120, 24)
                    ).columns
                    # Use width minus 1 to prevent wrapping on exact terminal width
                    actual_viz_width = max(80, terminal_width - 1)

                click.echo("=" * 60)
                click.echo("VISUALIZATION")
                click.echo("=" * 60)

                tree_viz = build_ascii_tree(
                    result.tiling,
                    store,
                    document_id,
                    width=actual_viz_width,
                    coverage_map=result.coverage_map,
                    seed_node_ids=set(result.node_ids),
                    use_token_coords=(viz_coords == "output-tokens"),
                    preloaded_nodes=result.nodes,
                )
                click.echo(tree_viz)
                click.echo("")

            # Show statistics after tree visualization
            click.echo("=" * 60)
            click.echo("STATISTICS")
            click.echo("=" * 60)
            click.echo(f"  Nodes retrieved: {len(result.node_ids)}")
            tiling_size = len(result.tiling) if result.tiling else 0
            click.echo(f"  Tiling size: {tiling_size}")
            click.echo(f"  Token count: {token_count}")
            click.echo(f"  Coverage: {len(result.coverage_map)} nodes")

    except Exception as e:
        click.echo(f"❌ Error processing query: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("node_id")
@click.pass_context
def pin(ctx: click.Context, node_id: str) -> None:
    """Pin a node to always include it."""
    try:
        # Create components for this command
        operational_config = ctx.obj["operational_config"]
        index_config = ctx.obj["index_config"]
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )
        store.pin_node(node_id)
        click.echo(f"✅ Node {node_id} pinned successfully!")
    except NodeNotFoundError:
        click.echo(f"❌ Node {node_id} not found")
        sys.exit(1)
    except InvalidOperationError as e:
        click.echo(f"❌ Failed to pin node {node_id}: {e}")
        sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Error pinning node: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status."""
    try:
        # Get configs and create components
        index_config = ctx.obj["index_config"]
        query_config = ctx.obj["query_config"]
        operational_config = ctx.obj["operational_config"]
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )
        # Gather stats
        with store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            all_nodes = session.query(TreeNode).count()
        leaf_nodes = store.get_leaf_nodes()
        root = store.get_root_node()
        pinned = store.get_pinned_nodes()

        click.echo("\nSYSTEM STATUS:")
        click.echo("=" * 40)
        click.echo(f"Total nodes: {all_nodes}")
        click.echo(f"Leaf nodes: {len(leaf_nodes)}")
        tree_height = root.height if root else 0
        click.echo(f"Tree height: {tree_height}")
        click.echo(f"Pinned nodes: {len(pinned)}")
        click.echo("\nCONFIGURATION:")
        click.echo("=" * 40)
        click.echo(f"Budget tokens: {query_config.budget_tokens}")
        click.echo(f"Target chunk tokens: {index_config.target_chunk_tokens}")
        click.echo(f"MMR lambda: {query_config.mmr_lambda}")

    except Exception as e:
        click.echo(f"❌ Error getting status: {e}", err=True)
        sys.exit(1)


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
        click.echo(f"❌ Error starting server: {e}", err=True)
        sys.exit(1)


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
        # Create components for this command
        operational_config = ctx.obj["operational_config"]
        index_config = ctx.obj["index_config"]
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )

        if document_id:
            # Clear specific document
            if not confirm:
                click.confirm(
                    f"⚠️  This will delete document '{document_id}' and all its data. Are you sure?",
                    abort=True,
                )

            # Clear document (handles both complete documents and orphaned nodes)
            deleted_count = store.clear_document(document_id)

            click.echo(
                f"✅ Cleared document '{document_id}' ({deleted_count} nodes deleted)"
            )
        else:
            # Clear all data
            if not confirm:
                click.confirm("⚠️  This will delete ALL data. Are you sure?", abort=True)

            # Clear database data
            with store.SessionLocal() as session:
                # Import models
                from ragzoom.store import Document, TreeNode

                # Delete all nodes
                deleted_count = session.query(TreeNode).count()
                session.query(TreeNode).delete()
                session.query(Document).delete()
                session.commit()

            # Clear the cache
            store.node_cache.clear()
            store.cache_order.clear()

            click.echo(f"✅ Cleared {deleted_count} nodes from the database")

    except click.Abort:
        click.echo("❌ Clear operation cancelled")
    except Exception as e:
        click.echo(f"❌ Error clearing database: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("output_file", type=click.Path())
@click.option("--format", type=click.Choice(["json", "text"]), default="text")
@click.pass_context
def export(ctx: click.Context, output_file: str, format: str) -> None:
    """Export tree structure to file."""
    try:
        # Create components for this command
        operational_config = ctx.obj["operational_config"]
        index_config = ctx.obj["index_config"]
        store = create_store_with_docker(
            operational_config, embedding_model=index_config.embedding_model
        )

        # Get all nodes
        nodes_data = []
        with store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            nodes = session.query(TreeNode).all()
            for node in nodes:
                node_dict = {
                    "id": node.id,
                    "parent_id": node.parent_id,
                    "height": node.height,
                    "span_start": node.span_start,
                    "span_end": node.span_end,
                    "is_leaf": store.is_leaf_node(node.id),
                    "text_preview": (
                        node.text[:100] + "..." if len(node.text) > 100 else node.text
                    ),
                }
                nodes_data.append(node_dict)

        # Write output
        output_path = Path(output_file)
        if format == "json":
            output_path.write_text(json.dumps(nodes_data, indent=2))
        else:
            # Text format
            lines = []
            for node_dict in sorted(
                nodes_data, key=lambda x: (x["height"], x["span_start"])
            ):
                height = node_dict.get("height", 0)
                if isinstance(height, int):
                    indent = "  " * height
                else:
                    indent = ""
                leaf_marker = "🍃" if node_dict.get("is_leaf") else "📁"
                node_id = node_dict.get("id", "")
                if isinstance(node_id, str):
                    node_id_short = node_id[:8] if len(node_id) > 8 else node_id
                else:
                    node_id_short = str(node_id)[:8]
                span_start = node_dict.get("span_start", 0)
                span_end = node_dict.get("span_end", 0)
                lines.append(
                    f"{indent}{leaf_marker} {node_id_short}... [{span_start}-{span_end}]"
                )
            output_path.write_text("\n".join(lines))

        click.echo(f"✅ Exported {len(nodes_data)} nodes to {output_file}")

    except Exception as e:
        click.echo(f"❌ Error exporting: {e}", err=True)
        sys.exit(1)


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

    # Check Docker availability
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
    if not issues_found:  # Only check if Docker is working
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

            # Try to create a store (this will auto-start PostgreSQL if needed)
            store = create_store_with_docker(operational_config)

            # Test basic operation
            with store.SessionLocal() as session:
                # Simple query to test connection
                from sqlalchemy import text

                session.execute(text("SELECT 1"))

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
