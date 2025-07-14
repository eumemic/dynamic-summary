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
    ctx.obj["tree_builder"] = TreeBuilder(config, store)
    ctx.obj["retriever"] = Retriever(config, store)
    ctx.obj["assembler"] = Assembler(config, store)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--document-id", help="Optional document ID")
@click.pass_context
def index(ctx, file_path: str, document_id: Optional[str]):
    """Index a document from file."""
    try:
        # Read file
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        
        click.echo(f"Indexing {path.name}...")
        
        # Index document
        tree_builder = ctx.obj["tree_builder"]
        doc_id = tree_builder.add_document(text, document_id)
        
        # Get stats
        store = ctx.obj["store"]
        leaf_nodes = store.get_leaf_nodes()
        root = store.get_root_node()
        
        click.echo(f"✅ Document indexed successfully!")
        click.echo(f"   Document ID: {doc_id}")
        click.echo(f"   Chunks created: {len(leaf_nodes)}")
        click.echo(f"   Tree depth: {root.depth if root else 0}")
        
    except Exception as e:
        click.echo(f"❌ Error indexing document: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query_text")
@click.option("--n-max", type=int, help="Max nodes to retrieve")
@click.option("--token-budget", type=int, help="Token budget for summary")
@click.option("--use-eviction", is_flag=True, help="Use sliding queue eviction")
@click.option("--show-stats", is_flag=True, help="Show retrieval statistics")
@click.pass_context
def query(
    ctx,
    query_text: str,
    n_max: Optional[int],
    token_budget: Optional[int],
    use_eviction: bool,
    show_stats: bool,
):
    """Query the system and get a summary."""
    try:
        retriever = ctx.obj["retriever"]
        assembler = ctx.obj["assembler"]
        
        # Retrieve
        if use_eviction:
            result = retriever.retrieve_with_eviction(query_text, token_budget)
        else:
            result = retriever.retrieve(query_text, n_max)
        
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
            nodes = session.query(store.TreeNode).all()
            for node in nodes:
                node_dict = {
                    "id": node.id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "span_start": node.span_start,
                    "span_end": node.span_end,
                    "is_leaf": node.summary is None,
                    "text_preview": node.text[:100] + "..." if len(node.text) > 100 else node.text,
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
                lines.append(f"{indent}{leaf_marker} {node['id'][:8]}... [{node['span_start']}-{node['span_end']}]")
            output_path.write_text("\n".join(lines))
        
        click.echo(f"✅ Exported {len(nodes_data)} nodes to {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Error exporting: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()