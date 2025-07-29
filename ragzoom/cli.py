"""CLI interface for RagZoom."""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.metrics import IndexingMetrics
from ragzoom.retrieve import Retriever
from ragzoom.store import Store, TreeNode
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
# Keep ragzoom.index at INFO to show batch progress


def display_metrics(metrics: IndexingMetrics) -> None:
    """Display detailed performance metrics in a user-friendly format."""
    click.echo("\n📊 PERFORMANCE METRICS")
    click.echo(f"{'='*50}")

    click.echo("\n⏱️  Timing:")
    click.echo(f"  Total duration: {metrics.total_duration_seconds:.2f} seconds")
    click.echo(f"  Throughput: {metrics.tokens_per_second:.1f} tokens/sec")
    click.echo(f"  Time per 1K tokens: {metrics.time_per_1k_tokens:.2f} seconds")

    click.echo("\n📄 Document:")
    click.echo(f"  Source tokens: {metrics.source_document_tokens:,}")
    click.echo(f"  Chunks created: {metrics.chunks_created}")
    click.echo(f"  Tree height: {metrics.tree_height}")
    click.echo(f"  Nodes per level: {metrics.nodes_per_level}")

    click.echo("\n🔌 API Usage:")
    click.echo(f"  Total API calls: {metrics.total_api_calls}")
    click.echo(f"  Embedding calls: {metrics.embedding_api_calls}")
    click.echo(f"  Summary calls: {metrics.summary_api_calls}")
    click.echo(f"  Avg embedding batch size: {metrics.avg_embedding_batch_size:.1f}")

    click.echo("\n🪙 Token Usage:")
    click.echo(f"  Embedding tokens: {metrics.total_embedding_tokens:,}")
    click.echo(f"  Summary prompt tokens: {metrics.total_summary_prompt_tokens:,}")
    click.echo(
        f"  Summary completion tokens: {metrics.total_summary_completion_tokens:,}"
    )
    click.echo(
        f"  Embedding tokens per 1K source: {metrics.embedding_tokens_per_1k:.1f}"
    )
    click.echo(f"  Summary tokens per 1K source: {metrics.summary_tokens_per_1k:.1f}")

    click.echo("\n💰 Cost Analysis:")
    click.echo(f"  Cost per 1K source tokens: ${metrics.cost_per_1k_tokens:.4f}")
    total_cost = metrics.cost_per_1k_tokens * metrics.source_document_tokens / 1000
    click.echo(f"  Total estimated cost: ${total_cost:.4f}")

    click.echo("\n💾 Memory Usage:")
    click.echo(f"  Peak memory: {metrics.peak_memory_mb:.1f} MB")
    click.echo(f"  Memory growth: {metrics.memory_usage_mb:.1f} MB")

    if metrics.summary_stats:
        click.echo("\n📏 Summary Accuracy:")
        for target_size, stats in sorted(metrics.summary_stats.items()):
            click.echo(f"\n  Target {target_size} tokens:")
            click.echo(f"    Count: {stats.count}")
            click.echo(f"    Average size: {stats.avg_tokens:.1f} tokens")
            click.echo(f"    Average deviation: {stats.avg_deviation_percent:.1f}%")
            click.echo(
                f"    Over target: {stats.percent_over_target:.1f}% (max: {stats.max_overage_percent:.1f}%)"
            )
            click.echo(
                f"    Under target: {stats.percent_under_target:.1f}% (max: {stats.max_underage_percent:.1f}%)"
            )


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """RagZoom: Incremental, hierarchical RAG memory system."""
    # Initialize shared components
    # RagZoomConfig will read from environment automatically due to pydantic_settings
    config = RagZoomConfig()  # Will use RAGZOOM_OPENAI_API_KEY from env
    store = Store(config)

    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["store"] = store
    tree_builder = TreeBuilder(config, store)
    ctx.obj["tree_builder"] = tree_builder
    ctx.obj["retriever"] = Retriever(config, store, tree_builder)
    ctx.obj["assembler"] = Assembler(config, store)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--document-id", help="Optional document ID")
@click.option("--clear", is_flag=True, help="Clear existing document before indexing")
@click.option("--no-progress", is_flag=True, help="Disable progress bar")
@click.option(
    "--max-concurrent",
    type=int,
    default=10,
    help="Maximum concurrent API requests (default: 10)",
)
@click.option("--validate", is_flag=True, help="Enable validation checks")
@click.option("--benchmark", is_flag=True, help="Enable performance metrics collection")
@click.option(
    "--benchmark-output", type=click.Path(), help="Save benchmark results to JSON file"
)
@click.pass_context
def index(
    ctx: click.Context,
    file_path: str,
    document_id: Optional[str],
    clear: bool,
    no_progress: bool,
    max_concurrent: int,
    validate: bool,
    benchmark: bool,
    benchmark_output: Optional[str],
) -> None:
    """Index a document from file."""
    # Set global validation flag
    from ragzoom.validate import set_validation_enabled

    set_validation_enabled(validate)

    try:
        # Read file
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")

        # Determine document ID (use provided ID or filename)
        if not document_id:
            document_id = path.name

        # Clear existing document if requested
        if clear:
            store = ctx.obj["store"]
            # Check if document exists
            with store.SessionLocal() as session:
                from ragzoom.store import Document

                existing_doc = session.query(Document).filter_by(id=document_id).first()
                if existing_doc:
                    click.echo(f"Clearing existing document '{document_id}'...")
                    # Delete document nodes
                    deleted_count = store.delete_document_nodes(document_id)
                    # Delete document record
                    session.query(Document).filter_by(id=document_id).delete()
                    session.commit()
                    click.echo(f"   Cleared {deleted_count} nodes")

        click.echo(f"Indexing {path.name}...")

        # Create tree builder with specified concurrency
        config = ctx.obj["config"]
        store = ctx.obj["store"]
        tree_builder = TreeBuilder(config, store, max_concurrent=max_concurrent)

        # Index with or without metrics based on benchmark flag
        if benchmark:
            doc_id, metrics = tree_builder.add_document_with_metrics(
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
        store = ctx.obj["store"]

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
        tree_height = store.get_node_height(root.id) if root else 0
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
            lambda: validate_chunk_sizes(doc_leaves, config.leaf_tokens), "chunk sizes"
        )

        run_validate(
            lambda: validate_tree_structure(store, doc_id, text), "tree structure"
        )

        run_validate(
            lambda: validate_equal_leaf_depth(store, doc_id), "equal leaf depth"
        )

        # Display and save metrics if benchmark mode
        if benchmark:
            display_metrics(metrics)

            # Save to JSON if requested
            if benchmark_output:
                click.echo(f"\n📁 Saving metrics to {benchmark_output}...")
                metrics_dict = metrics.to_dict()
                metrics_dict["document_id"] = doc_id
                metrics_dict["file_path"] = str(path.absolute())

                with open(benchmark_output, "w") as f:
                    json.dump(metrics_dict, f, indent=2)
                click.echo(f"✅ Metrics saved to {benchmark_output}")

    except Exception as e:
        click.echo(f"❌ Error indexing document: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def documents(ctx: click.Context) -> None:
    """List all indexed documents."""
    try:
        store = ctx.obj["store"]

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
@click.option("--n-max", type=int, help="Max nodes to retrieve")
@click.option("--token-budget", type=int, help="Token budget for summary")
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
    n_max: Optional[int],
    token_budget: Optional[int],
    debug: bool,
    validate: bool,
    viz_width: Optional[int],
    viz_coords: str,
) -> None:
    """Query the system and get a summary."""
    # Set global validation flag
    from ragzoom.validate import set_validation_enabled

    set_validation_enabled(validate)

    try:
        retriever = ctx.obj["retriever"]
        assembler = ctx.obj["assembler"]

        # Retrieve - Pass both n_max and budget_tokens to support all three modes
        result = retriever.retrieve(
            query_text,
            n_max=n_max,
            budget_tokens=token_budget,
            document_id=document_id,
        )

        # Assemble
        summary = assembler.assemble(result)
        token_count = assembler.get_token_count(summary)

        # Tiling validation
        if validate and getattr(result, "tiling", None):
            from ragzoom.validate import validate_tiling

            error = validate_tiling(
                result.tiling, ctx.obj["store"], document_id, budget_tokens=token_budget
            )
            if error:
                click.echo(f"❌ Tiling validation failed: {error}", err=True)
                sys.exit(1)

        # Output summary
        click.echo("\n" + "=" * 60)
        click.echo("SUMMARY")
        click.echo("=" * 60)
        if debug and getattr(result, "tiling", None):
            store = ctx.obj["store"]
            for idx, node_id in enumerate(result.tiling):
                node = store.get_node(node_id)
                if node:
                    # Node span is always the full span
                    span_start, span_end = node.span_start, node.span_end
                    span = f"{span_start}-{span_end}"
                    height = store.get_node_height(node.id)
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
                # Use provided width or detect terminal width
                if viz_width:
                    terminal_width = viz_width
                    actual_viz_width = viz_width
                else:
                    # Get terminal width, with fallback to 120
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
                    ctx.obj["store"],
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
        store = ctx.obj["store"]
        success = store.pin_node(node_id)

        if success:
            click.echo(f"✅ Node {node_id} pinned successfully!")
        else:
            click.echo(f"❌ Failed to pin node {node_id} (doesn't exist or too deep)")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Error pinning node: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status."""
    try:
        store = ctx.obj["store"]
        config = ctx.obj["config"]

        # Gather stats
        all_nodes = store.collection.count()
        leaf_nodes = store.get_leaf_nodes()
        root = store.get_root_node()
        pinned = store.get_pinned_nodes()

        click.echo("\nSYSTEM STATUS:")
        click.echo("=" * 40)
        click.echo(f"Total nodes: {all_nodes}")
        click.echo(f"Leaf nodes: {len(leaf_nodes)}")
        tree_height = store.get_node_height(root.id) if root else 0
        click.echo(f"Tree height: {tree_height}")
        click.echo(f"Pinned nodes: {len(pinned)}")
        click.echo("\nCONFIGURATION:")
        click.echo("=" * 40)
        click.echo(f"Budget tokens: {config.budget_tokens}")
        click.echo(f"Leaf tokens: {config.leaf_tokens}")
        click.echo(f"MMR lambda: {config.mmr_lambda}")
        click.echo(f"Slope cap: {config.slope_cap}")
        click.echo(f"Smoothing enabled: {config.smoothing_pass_enabled}")

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
def clear(ctx: click.Context, document_id: Optional[str], confirm: bool) -> None:
    """Clear data from the database.

    Without --document-id, clears all data.
    With --document-id, clears only the specified document.
    """
    try:
        store = ctx.obj["store"]

        if document_id:
            # Clear specific document
            if not confirm:
                click.confirm(
                    f"⚠️  This will delete document '{document_id}' and all its data. Are you sure?",
                    abort=True,
                )

            # Check if document exists
            with store.SessionLocal() as session:
                from ragzoom.store import Document

                doc = session.query(Document).filter_by(id=document_id).first()
                if not doc:
                    click.echo(f"❌ Document '{document_id}' not found")
                    sys.exit(1)

            # Delete document nodes
            deleted_count = store.delete_document_nodes(document_id)

            # Delete document record
            with store.SessionLocal() as session:
                session.query(Document).filter_by(id=document_id).delete()
                session.commit()

            click.echo(
                f"✅ Cleared document '{document_id}' ({deleted_count} nodes deleted)"
            )
        else:
            # Clear all data
            if not confirm:
                click.confirm("⚠️  This will delete ALL data. Are you sure?", abort=True)

            # Clear SQLite data
            with store.SessionLocal() as session:
                # Import models
                from ragzoom.store import Document, TreeNode

                # Delete all nodes
                deleted_count = session.query(TreeNode).count()
                session.query(TreeNode).delete()
                session.query(Document).delete()
                session.commit()

            # Clear Chroma collection - delete all documents
            # Get all IDs first
            results = store.collection.get()
            if results["ids"]:
                store.collection.delete(ids=results["ids"])

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
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--format", type=click.Choice(["json", "text"]), default="text")
@click.pass_context
def export(ctx: click.Context, input_file: str, output_file: str, format: str) -> None:
    """Export tree structure to file."""
    try:
        store = ctx.obj["store"]

        # Get all nodes
        nodes_data = []
        with store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            nodes = session.query(TreeNode).all()
            for node in nodes:
                node_dict = {
                    "id": node.id,
                    "parent_id": node.parent_id,
                    "height": store.get_node_height(node.id),
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
            for node in sorted(nodes_data, key=lambda x: (x["depth"], x["span_start"])):
                indent = "  " * node["height"]
                leaf_marker = "🍃" if node["is_leaf"] else "📁"
                lines.append(
                    f"{indent}{leaf_marker} {node['id'][:8]}... [{node['span_start']}-{node['span_end']}]"
                )
            output_path.write_text("\n".join(lines))

        click.echo(f"✅ Exported {len(nodes_data)} nodes to {output_file}")

    except Exception as e:
        click.echo(f"❌ Error exporting: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
