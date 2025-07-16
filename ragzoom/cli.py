"""CLI interface for RagZoom."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import Store

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


@click.group()
@click.pass_context
def cli(ctx):
    """RagZoom: Incremental, hierarchical RAG memory system."""
    # Initialize shared components
    config = RagZoomConfig()
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
@click.pass_context
def index(
    ctx,
    file_path: str,
    document_id: Optional[str],
    clear: bool,
    no_progress: bool,
    max_concurrent: int,
    validate: bool,
):
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
            from ragzoom.store import TreeNode

            doc_leaves = (
                session.query(TreeNode)
                .filter_by(
                    document_id=doc_id, summary=None  # Leaf nodes have no summary
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
        click.echo(f"   Tree depth: {root.depth if root else 0}")

        # Run validation checks
        from ragzoom.validate import (
            validate,
            validate_chunk_sizes,
            validate_document_coverage,
            validate_tree_structure,
        )

        # Validations will run only if --validate was passed
        validate(
            lambda: validate_document_coverage(text, doc_leaves), "document coverage"
        )

        validate(
            lambda: validate_chunk_sizes(doc_leaves, config.leaf_tokens), "chunk sizes"
        )

        validate(lambda: validate_tree_structure(store, doc_id, text), "tree structure")

    except Exception as e:
        click.echo(f"❌ Error indexing document: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def documents(ctx):
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
                    .filter_by(document_id=doc.id, summary=None)
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
@click.option("--use-eviction", is_flag=True, help="Use sliding queue eviction")
@click.option("--show-stats", is_flag=True, help="Show retrieval statistics")
@click.option("--validate", is_flag=True, help="Enable validation checks")
@click.pass_context
def query(
    ctx,
    query_text: str,
    document_id: str,
    n_max: Optional[int],
    token_budget: Optional[int],
    use_eviction: bool,
    show_stats: bool,
    validate: bool,
):
    """Query the system and get a summary."""
    # Set global validation flag
    from ragzoom.validate import set_validation_enabled

    set_validation_enabled(validate)

    try:
        retriever = ctx.obj["retriever"]
        assembler = ctx.obj["assembler"]

        # Retrieve
        if use_eviction:
            result = retriever.retrieve_with_eviction(
                query_text, token_budget, document_id=document_id
            )
        else:
            # Pass both n_max and budget_tokens to support all three modes
            result = retriever.retrieve(
                query_text,
                n_max=n_max,
                budget_tokens=token_budget,
                document_id=document_id,
            )

        # Assemble
        summary, token_count = assembler.assemble_with_budget(result, token_budget)

        # Output summary
        click.echo("\n" + "=" * 60)
        click.echo("SUMMARY:")
        click.echo("=" * 60)
        click.echo(summary)
        click.echo("=" * 60 + "\n")

        # Show stats if requested
        if show_stats:
            click.echo("STATISTICS:")
            click.echo(f"  Nodes retrieved: {len(result.node_ids)}")
            click.echo(f"  Frontier size: {len(result.frontier_nodes)}")
            click.echo(f"  Token count: {token_count}")
            click.echo(f"  Coverage: {len(result.coverage_map)} nodes")

    except Exception as e:
        click.echo(f"❌ Error processing query: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("node_id")
@click.pass_context
def pin(ctx, node_id: str):
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
def status(ctx):
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
        click.echo(f"Tree depth: {root.depth if root else 0}")
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
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
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
def clear(ctx, document_id: Optional[str], confirm: bool):
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
def export(ctx, input_file: str, output_file: str, format: str):
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
                    "depth": node.depth,
                    "span_start": node.span_start,
                    "span_end": node.span_end,
                    "is_leaf": node.summary is None,
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
                indent = "  " * node["depth"]
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
