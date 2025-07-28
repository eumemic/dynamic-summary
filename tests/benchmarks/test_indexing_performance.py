"""Performance benchmarks for indexing."""

import json
import os
import time
from pathlib import Path

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.store import Store

# Skip benchmarks by default unless explicitly requested
pytestmark = pytest.mark.benchmark


def get_test_document(document_type: str = "narrative") -> tuple[str, str]:
    """Get a test document from the test_data directory.

    Args:
        document_type: Type of document to load
            - "narrative": The Hobbit Chapter 1 (~7K tokens)
            - "technical": Technical documentation sample
            - "classic": Moby Dick sample

    Returns:
        Tuple of (document_text, document_name)
    """
    documents = {
        "narrative": ("test_data/the_hobbit_chapter_1.txt", "The Hobbit Ch1"),
        "technical": ("test_data/smoke_test_larger.txt", "Technical Doc"),
        "classic": ("test_data/moby_dick_sample.txt", "Moby Dick Sample"),
    }

    if document_type not in documents:
        document_type = "narrative"  # Default

    file_path, name = documents[document_type]

    # Read the document
    try:
        path = Path(file_path)
        if not path.exists():
            # If running from different directory, try from root
            path = Path(__file__).parent.parent.parent / file_path

        text = path.read_text(encoding="utf-8")
        return text, name
    except Exception as e:
        pytest.skip(f"Could not load test document {file_path}: {e}")


@pytest.fixture
def benchmark_config():
    """Config for benchmarks with real API calls."""
    return RagZoomConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
        embedding_batch_size=100,
    )


@pytest.mark.parametrize("leaf_tokens", [100, 200, 400])
@pytest.mark.parametrize("document_type", ["narrative"])
def test_indexing_performance(benchmark_config, leaf_tokens, document_type):
    """Benchmark indexing performance at different chunk sizes with real documents."""
    # Skip if no API key
    if benchmark_config.openai_api_key == "test-key":
        pytest.skip("OPENAI_API_KEY not set")

    # Update config with chunk size
    benchmark_config.leaf_tokens = leaf_tokens

    # Get test document
    test_doc, doc_name = get_test_document(document_type)

    # Run indexing with metrics
    with Store.temporary() as store:
        builder = TreeBuilder(benchmark_config, store)

        # Warm up tokenizer
        _ = builder.splitter.tokenizer.encode("warmup")

        # Run indexing
        start_time = time.time()
        doc_id, metrics = builder.add_document_with_metrics(
            test_doc, document_id=f"{document_type}_{leaf_tokens}", show_progress=False
        )
        end_time = time.time()

        # Verify metrics match actual timing
        actual_duration = end_time - start_time
        assert abs(metrics.total_duration_seconds - actual_duration) < 0.1

        # Save metrics to file for comparison
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"metrics_{leaf_tokens}_tokens.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "config": {
                        "leaf_tokens": leaf_tokens,
                        "embedding_model": benchmark_config.embedding_model,
                        "summary_model": benchmark_config.summary_model,
                        "document": doc_name,
                    },
                    "metrics": metrics.to_dict(),
                    "timestamp": time.time(),
                },
                f,
                indent=2,
            )

        # Print summary
        print(f"\n=== Performance Summary ({doc_name}, leaf_tokens={leaf_tokens}) ===")
        print(f"Document: {metrics.source_document_tokens:,} tokens")
        print(f"Tokens/second: {metrics.tokens_per_second:.1f}")
        print(f"Time per 1K tokens: {metrics.time_per_1k_tokens:.2f}s")
        print(f"Embedding tokens per 1K: {metrics.embedding_tokens_per_1k:.1f}")
        print(f"Summary tokens per 1K: {metrics.summary_tokens_per_1k:.1f}")
        print(f"API calls per 1K: {metrics.api_calls_per_1k:.1f}")
        print(f"Cost per 1K tokens: ${metrics.cost_per_1k_tokens:.4f}")
        print(f"Peak memory: {metrics.peak_memory_mb:.1f} MB")
        print(f"Memory growth: {metrics.memory_usage_mb:.1f} MB")

        # Summary accuracy
        if metrics.summary_stats:
            for target, stats in metrics.summary_stats.items():
                print(f"\nSummary accuracy (target={target}):")
                print(f"  Average size: {stats.avg_tokens:.1f} tokens")
                print(f"  Average deviation: {stats.avg_deviation_percent:.1f}%")
                print(f"  Over target: {stats.percent_over_target:.1f}%")
                print(f"  Under target: {stats.percent_under_target:.1f}%")


def test_performance_comparison():
    """Compare benchmark results if available."""
    output_dir = Path("benchmark_results")
    if not output_dir.exists():
        pytest.skip("No benchmark results to compare")

    # Load all results
    results = {}
    for file in output_dir.glob("metrics_*_tokens.json"):
        with open(file) as f:
            data = json.load(f)
            chunk_size = data["config"]["leaf_tokens"]
            results[chunk_size] = data["metrics"]

    if len(results) < 2:
        pytest.skip("Need at least 2 benchmark results to compare")

    # Print comparison table
    print("\n=== Performance Comparison ===")
    print(
        f"{'Chunk Size':<12} {'Tokens/sec':<12} {'Embed/1K':<10} {'Summary/1K':<12} {'Cost/1K':<10}"
    )
    print("-" * 60)

    for chunk_size in sorted(results.keys()):
        m = results[chunk_size]
        print(
            f"{chunk_size:<12} "
            f"{m['timing']['tokens_per_second']:<12.1f} "
            f"{m['efficiency']['embedding_tokens_per_1k']:<10.1f} "
            f"{m['efficiency']['summary_tokens_per_1k']:<12.1f} "
            f"${m['efficiency']['cost_per_1k_tokens']:<9.4f}"
        )


if __name__ == "__main__":
    # Run specific benchmark
    import sys

    if len(sys.argv) > 1:
        leaf_tokens = int(sys.argv[1])
        config = RagZoomConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            leaf_tokens=leaf_tokens,
        )
        test_indexing_performance(config, leaf_tokens)
    else:
        print("Usage: python test_indexing_performance.py <leaf_tokens>")
        print("Example: python test_indexing_performance.py 200")
