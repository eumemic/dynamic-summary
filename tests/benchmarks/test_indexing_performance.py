"""Performance benchmarks for indexing."""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.store import Store

# Skip benchmarks by default unless explicitly requested
pytestmark = pytest.mark.benchmark


def generate_test_document(size_tokens: int = 10000) -> str:
    """Generate a test document with approximately the specified number of tokens.

    Creates realistic narrative text that's good for testing summarization.
    """
    # Each sentence is roughly 15-20 tokens
    tokens_per_sentence = 17
    sentences_needed = size_tokens // tokens_per_sentence

    # Generate a story-like document
    paragraphs = []
    sentences_written = 0

    # Opening
    opening = [
        "In the early morning light, Sarah walked through the bustling city streets.",
        "The coffee shop on the corner was already filled with the usual crowd of commuters.",
        "She ordered her regular cappuccino and found a quiet table by the window.",
        "Outside, the city was coming to life with its familiar rhythm of honking cars and hurried footsteps.",
    ]
    paragraphs.append(" ".join(opening))
    sentences_written += len(opening)

    # Middle sections - vary the content
    topics = [
        "technology conference",
        "medical research",
        "environmental project",
        "educational initiative",
        "business venture",
        "scientific discovery",
    ]

    for i, topic in enumerate(topics):
        if sentences_written >= sentences_needed:
            break

        if topic == "technology conference":
            section = [
                f"The {topic} was scheduled to begin at nine o'clock sharp.",
                "Speakers from around the world had gathered to share their latest innovations.",
                "Sarah's presentation on distributed systems was scheduled for the afternoon session.",
                "She had spent months preparing the slides and rehearsing her key points.",
                "The demo would showcase a new approach to handling large-scale data processing.",
            ]
        elif topic == "medical research":
            section = [
                f"Her colleague Dr. Chen was leading a groundbreaking {topic} project.",
                "The team had been working on developing new treatments for rare diseases.",
                "Initial trials showed promising results with minimal side effects.",
                "The research facility was equipped with state-of-the-art laboratory equipment.",
                "Funding for the next phase had just been approved by the board.",
            ]
        elif topic == "environmental project":
            section = [
                f"The {topic} aimed to restore the local wetlands ecosystem.",
                "Volunteers from the community had been working tirelessly every weekend.",
                "Native plants were being reintroduced to support wildlife habitats.",
                "The project had already shown positive impacts on water quality.",
                "Local schools were using it as an educational opportunity for students.",
            ]
        else:
            section = [
                f"Meanwhile, the {topic} was gaining momentum across the region.",
                "Stakeholders from various sectors had expressed strong interest.",
                "The implementation phase would require careful coordination.",
                "Success metrics had been clearly defined and agreed upon.",
                "Regular progress reports would be submitted to all participants.",
            ]

        paragraphs.append(" ".join(section))
        sentences_written += len(section)

    # Closing
    closing = [
        "As the day drew to a close, Sarah reflected on all that had been accomplished.",
        "The journey ahead would be challenging, but the foundation was solid.",
        "Tomorrow would bring new opportunities and fresh perspectives.",
        "She finished her coffee and prepared for the next chapter.",
    ]
    paragraphs.append(" ".join(closing))

    return "\n\n".join(paragraphs)


class BenchmarkStore:
    """Wrapper for Store that provides cleanup after benchmarks."""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = RagZoomConfig(
            sqlite_database_url=f"sqlite:///{self.temp_dir}/test.db",
            chroma_persist_directory=f"{self.temp_dir}/chroma",
        )
        self.store = Store(self.config)

    def cleanup(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __enter__(self):
        return self.store

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


@pytest.fixture
def benchmark_config():
    """Config for benchmarks with real API calls."""
    return RagZoomConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
        embedding_batch_size=100,
    )


@pytest.mark.parametrize("leaf_tokens", [100, 200, 400, 800])
def test_indexing_performance(benchmark_config, leaf_tokens):
    """Benchmark indexing performance at different chunk sizes."""
    # Skip if no API key
    if benchmark_config.openai_api_key == "test-key":
        pytest.skip("OPENAI_API_KEY not set")

    # Update config with chunk size
    benchmark_config.leaf_tokens = leaf_tokens

    # Generate test document
    test_doc = generate_test_document(10000)  # 10K tokens

    # Run indexing with metrics
    with BenchmarkStore() as store:
        builder = TreeBuilder(benchmark_config, store)

        # Warm up tokenizer
        _ = builder.splitter.tokenizer.encode("warmup")

        # Run indexing
        start_time = time.time()
        doc_id, metrics = builder.add_document_with_metrics(
            test_doc, document_id=f"benchmark_{leaf_tokens}", show_progress=False
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
                    },
                    "metrics": metrics.to_dict(),
                    "timestamp": time.time(),
                },
                f,
                indent=2,
            )

        # Print summary
        print(f"\n=== Performance Summary (leaf_tokens={leaf_tokens}) ===")
        print(f"Tokens/second: {metrics.tokens_per_second:.1f}")
        print(f"Time per 1K tokens: {metrics.time_per_1k_tokens:.2f}s")
        print(f"Embedding tokens per 1K: {metrics.embedding_tokens_per_1k:.1f}")
        print(f"Summary tokens per 1K: {metrics.summary_tokens_per_1k:.1f}")
        print(f"API calls per 1K: {metrics.api_calls_per_1k:.1f}")
        print(f"Cost per 1K tokens: ${metrics.cost_per_1k_tokens:.4f}")

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
