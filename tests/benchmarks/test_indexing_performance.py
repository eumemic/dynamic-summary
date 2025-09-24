"""Performance benchmarks for indexing."""

import json
import os
from pathlib import Path

import pytest

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.index import TreeBuilder

# Skip benchmarks by default unless explicitly requested
pytestmark = pytest.mark.benchmark


def get_test_document(document_path: str) -> tuple[str, str]:
    """Get a test document from the provided path.

    Args:
        document_path: Path to the document to load

    Returns:
        Tuple of (document_text, document_name)
    """
    # Read the document
    try:
        path = Path(document_path)
        if not path.is_absolute():
            # If relative path, try from current directory and from test root
            if not path.exists():
                path = Path(__file__).parent.parent.parent / document_path

        text = path.read_text(encoding="utf-8")
        # Use the filename as the document name
        name = path.stem.replace("_", " ").title()
        return text, name
    except Exception as e:
        pytest.skip(f"Could not load test document {document_path}: {e}")


@pytest.mark.parametrize("leaf_tokens", [100, 200, 400])
def test_indexing_performance(
    storage_backend: StorageBackend,
    leaf_tokens: int,
    document_path: str = "test_data/the_hobbit_chapter_1.txt",
) -> None:
    """Benchmark indexing performance with configurable document and chunk size.

    Args:
        leaf_tokens: Target tokens per leaf node
        document_path: Path to document to index (default: Hobbit Chapter 1)
    """
    # Create config for this specific test
    api_key = os.getenv("OPENAI_API_KEY", "test-key")

    # Skip if no API key
    if api_key == "test-key" or api_key == "test-key-for-tests":
        pytest.skip("OPENAI_API_KEY not set")

    index_config = IndexConfig.load(
        target_chunk_tokens=leaf_tokens,
    )

    # Get test document
    test_doc, doc_name = get_test_document(document_path)

    # Run indexing with metrics
    # Create a temporary document store for indexing
    doc_store = storage_backend.add_document(
        document_id="indexing_perf_test",
        file_path="indexing_perf_test.txt",
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
    )
    from ragzoom.vector_factory import create_vector_index

    vi = create_vector_index(
        "python", "sqlite:///:memory:", index_config.embedding_model
    )
    builder = TreeBuilder(index_config, doc_store, vi, api_key)

    # Warm up tokenizer
    _ = builder.splitter.tokenizer.encode("warmup")

    # Run indexing
    # Run indexing without document_id parameter (it's generated internally)
    doc_id, telemetry = builder.add_document_with_telemetry(
        test_doc, show_progress=False
    )

    # Save telemetry data for comparison
    # Note: telemetry is already in v3.0 format from add_document_with_telemetry()
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"telemetry_{leaf_tokens}_tokens.json"
    with open(output_file, "w") as f:
        json.dump(telemetry, f, indent=2)

    # Compute metrics from telemetry for display
    from ragzoom.telemetry_analysis import (
        compute_metrics_from_telemetry,
        compute_simplified_metrics,
    )

    # Get basic metrics for memory and timing info
    basic_metrics = compute_metrics_from_telemetry(telemetry)

    # Get simplified metrics for the new metrics system
    simplified = compute_simplified_metrics(telemetry)

    # Get metrics for this chunk size
    chunk_metrics = simplified.metrics_by_chunk_size.get(leaf_tokens)

    # Print summary
    print(f"\n=== Performance Summary ({doc_name}, leaf_tokens={leaf_tokens}) ===")
    print(f"Document: {basic_metrics.source_document_tokens:,} tokens")
    print(f"Total duration: {basic_metrics.total_duration_seconds:.2f}s")
    print(f"Total cost: ${basic_metrics.total_cost:.4f}")
    print(f"Peak memory: {basic_metrics.peak_memory_mb:.1f} MB")
    print(f"Memory growth: {basic_metrics.memory_usage_mb:.1f} MB")

    # Print new simplified metrics if available
    if chunk_metrics:
        print("\n--- Simplified Metrics ---")
        # ChunkMetrics is now a dataclass with typed attributes
        if hasattr(chunk_metrics, "target_fit"):
            print("\nTarget Fit:")
            tf = chunk_metrics.target_fit
            print(f"  median_error: {tf.median_error:.2f}")
            print(f"  p95_error: {tf.p95_error:.2f}")
            print(f"  percent_within_10: {tf.percent_within_10:.2f}")

        if hasattr(chunk_metrics, "retries"):
            print("\nRetries:")
            r = chunk_metrics.retries
            print(f"  retry_rate: {r.retry_rate:.2f}")
            print(f"  max_retries: {r.max_retries:.0f}")

        if hasattr(chunk_metrics, "latency"):
            print("\nLatency:")
            latency = chunk_metrics.latency
            print(f"  median_seconds: {latency.median_seconds:.2f}")
            print(f"  total_indexing_seconds: {latency.total_indexing_seconds:.2f}")

        if hasattr(chunk_metrics, "cost"):
            print("\nCost:")
            c = chunk_metrics.cost
            print(f"  usd_per_node: ${c.usd_per_node:.4f}")
            print(f"  total_tokens: {c.total_tokens:,}")

        if hasattr(chunk_metrics, "pipeline_efficiency"):
            print("\nPipeline Efficiency:")
            eff = chunk_metrics.pipeline_efficiency
            print(f"  pipeline_efficiency: {eff:.1f}%")
            if eff >= 60:
                print("  🚀 High parallelism utilization")
            elif eff >= 20:
                print("  ✅ Moderate parallelism utilization")
            else:
                print("  ⚠️  Low parallelism utilization")

    # Summary accuracy (simplified - using only available properties)
    if hasattr(basic_metrics, "summary_stats") and basic_metrics.summary_stats:
        for target, stats in basic_metrics.summary_stats.items():
            print(f"\nSummary accuracy (target={target}):")
            print(f"  Count: {stats.count}")
            print(f"  Average size: {stats.avg_tokens:.1f} tokens")
            print(f"  Average deviation: {stats.avg_deviation:.1f} tokens")
            print(f"  Std deviation: {stats.std_deviation:.1f} tokens")
            print(f"  Over target: {stats.over_target_count}")
            print(f"  Under target: {stats.under_target_count}")


def test_performance_comparison() -> None:
    """Compare benchmark results if available."""
    output_dir = Path("benchmark_results")
    if not output_dir.exists():
        pytest.skip("No benchmark results to compare")

    # Import needed for computing metrics from telemetry
    from ragzoom.telemetry_analysis import (
        compute_metrics_from_telemetry,
        compute_simplified_metrics,
    )

    # Load all results
    results = {}
    for file in output_dir.glob("telemetry_*_tokens.json"):
        with open(file) as f:
            telemetry = json.load(f)
            # Extract chunk size from telemetry data (v3.0 format)
            chunk_size = telemetry.get("chunk_size", 0)

            # Skip if no chunk size found
            if not chunk_size:
                continue

            # Compute metrics from telemetry data
            basic_metrics = compute_metrics_from_telemetry(telemetry)
            simplified = compute_simplified_metrics(telemetry)

            # Get chunk-specific metrics
            chunk_metrics = simplified.metrics_by_chunk_size.get(chunk_size)

            # Store metrics for comparison
            results[chunk_size] = {
                "duration_seconds": basic_metrics.total_duration_seconds,
                "total_cost": basic_metrics.total_cost,
                "source_tokens": basic_metrics.source_document_tokens,
                "chunk_metrics": chunk_metrics,
            }

    if len(results) < 2:
        pytest.skip("Need at least 2 benchmark results to compare")

    # Print comparison table
    print("\n=== Performance Comparison ===")
    print(
        f"{'Chunk Size':<12} {'Duration':<12} {'Total Cost':<12} {'Target-fit':<15} {'Retry Rate':<12}"
    )
    print("-" * 70)

    for chunk_size in sorted(results.keys()):
        m = results[chunk_size]
        chunk_m = m["chunk_metrics"]

        # Extract key metrics with proper typing
        target_fit = 0.0
        retry_rate = 0.0
        if (
            chunk_m is not None
            and hasattr(chunk_m, "target_fit")
            and hasattr(chunk_m.target_fit, "percent_within_10")
        ):
            target_fit = chunk_m.target_fit.percent_within_10
        if (
            chunk_m is not None
            and hasattr(chunk_m, "retries")
            and hasattr(chunk_m.retries, "retry_rate")
        ):
            retry_rate = chunk_m.retries.retry_rate

        print(
            f"{chunk_size:<12} "
            f"{m['duration_seconds']:<12.2f} "
            f"${m['total_cost']:<11.4f} "
            f"{target_fit:<15.1f} "
            f"{retry_rate:<12.1f}"
        )


if __name__ == "__main__":
    # Run specific benchmark
    import sys

    if len(sys.argv) > 1:
        leaf_tokens = int(sys.argv[1])
        # Config is created inside test_indexing_performance function

        # Note: For standalone execution, this would require a storage backend.
        # Use pytest fixtures in practice: `pytest tests/benchmarks/test_indexing_performance.py`
        print(
            "Error: Standalone execution requires refactoring to use pytest fixtures."
        )
        print(
            "Run: pytest tests/benchmarks/test_indexing_performance.py::test_indexing_performance"
        )
        import sys

        sys.exit(1)
    else:
        print("Usage: python test_indexing_performance.py <leaf_tokens>")
        print("Example: python test_indexing_performance.py 200")
