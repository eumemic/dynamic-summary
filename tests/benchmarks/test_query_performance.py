"""Performance benchmarks for query operations."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, QueryConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
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


def setup_test_document(store: Store, api_key: str) -> str:
    """Index a test document for query benchmarking.

    Returns the document ID of the indexed document.
    """
    # Use consistent chunk size for baseline
    index_config = IndexConfig.load(target_chunk_tokens=200)

    # Get test document
    test_doc, doc_name = get_test_document("narrative")

    # Index the document
    builder = TreeBuilder(index_config, store, api_key)
    doc_id = builder.add_document(test_doc, document_id="query_benchmark_doc")

    return doc_id


@pytest.mark.parametrize("num_seeds", [5, 10, 20])
@pytest.mark.parametrize("budget_tokens", [1000, 2000, 4000])
@pytest.mark.parametrize("query_type", ["specific", "broad", "complex"])
def test_query_performance(num_seeds, budget_tokens, query_type):
    """Benchmark query performance at different parameter combinations."""
    # Create config for this specific test
    api_key = os.getenv("OPENAI_API_KEY", "test-key")

    # Skip if no API key
    if api_key == "test-key":
        pytest.skip("OPENAI_API_KEY not set")

    # Query configurations
    queries = {
        "specific": "What is Bilbo's opinion about adventures at the beginning?",
        "broad": "Describe the overall setting and atmosphere of the story",
        "complex": "How do the dwarves convince Bilbo to join their quest, and what are their motivations?",
    }

    query_config = QueryConfig.load(
        budget_tokens=budget_tokens,
    )

    # Run query with metrics
    with Store.temporary() as store:
        # Index document first (or reuse if exists)
        doc_id = setup_test_document(store, api_key)

        # Create retriever and assembler
        retriever = Retriever(
            query_config,
            store,
            api_key=api_key,
        )
        assembler = Assembler(store)

        # Run query with telemetry
        result, telemetry = asyncio.run(
            retriever.retrieve_with_telemetry(
                queries[query_type],
                num_seeds=num_seeds,
                budget_tokens=budget_tokens,
                document_id=doc_id,
            )
        )

        # Assemble the result and measure assembly time
        import time

        start = time.perf_counter()
        summary = assembler.assemble(result)
        telemetry.assembly_time = time.perf_counter() - start

        # Update end time after assembly
        telemetry.end_time = time.perf_counter()

        # Get actual token count
        actual_tokens = assembler.get_token_count(summary)
        telemetry.output_tokens = actual_tokens

        # Save telemetry data for comparison
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)

        output_file = (
            output_dir
            / f"query_telemetry_{num_seeds}seeds_{budget_tokens}tokens_{query_type}.json"
        )

        # Save telemetry
        telemetry_data = {
            "format_version": "1.0",  # Query telemetry format version
            "telemetry": telemetry.to_dict(),
            "config": {
                "num_seeds": num_seeds,
                "budget_tokens": budget_tokens,
                "query_type": query_type,
            },
            "summary_preview": (
                summary[:500] if summary else ""
            ),  # First 500 chars for validation
        }

        with open(output_file, "w") as f:
            json.dump(telemetry_data, f, indent=2)

        # Print summary
        print(
            f"\n=== Query Performance ({query_type}, seeds={num_seeds}, budget={budget_tokens}) ==="
        )
        print(f"Total time: {telemetry.total_time:.3f}s")
        print("Phase breakdown:")
        print(f"  - Embedding: {telemetry.embedding_time:.3f}s")
        print(f"  - Search: {telemetry.search_time:.3f}s")
        print(f"  - MMR: {telemetry.mmr_time:.3f}s")
        print(f"  - Coverage map: {telemetry.coverage_map_time:.3f}s")
        print(f"  - Scoring: {telemetry.scoring_time:.3f}s")
        print(f"  - DP tiling: {telemetry.dp_time:.3f}s")
        print(f"  - Assembly: {telemetry.assembly_time:.3f}s")
        print("\nResults:")
        print(f"  - Seeds found: {telemetry.seeds_found}/{telemetry.seeds_requested}")
        print(f"  - Coverage size: {telemetry.coverage_size} nodes")
        print(f"  - Tiling size: {telemetry.tiling_size} nodes")
        print(f"  - Output tokens: {telemetry.output_tokens}/{budget_tokens}")

        # Basic validation
        assert (
            telemetry.output_tokens <= budget_tokens
        ), f"Output exceeded budget: {telemetry.output_tokens} > {budget_tokens}"
        assert telemetry.tiling_size > 0, "No nodes in tiling"
        assert telemetry.total_time > 0, "Invalid timing"


def test_query_performance_comparison():
    """Compare query benchmark results if available."""
    output_dir = Path("benchmark_results")
    if not output_dir.exists():
        pytest.skip("No benchmark results to compare")

    # Load all query telemetry files
    query_results = []
    for file in output_dir.glob("query_telemetry_*.json"):
        with open(file) as f:
            data = json.load(f)
            telemetry = data["telemetry"]
            config = data["config"]

            query_results.append(
                {
                    "config": config,
                    "telemetry": telemetry,
                }
            )

    if len(query_results) < 2:
        pytest.skip("Need at least 2 query benchmark results to compare")

    # Group by configuration for comparison
    by_config = {}
    for result in query_results:
        key = (
            result["config"]["num_seeds"],
            result["config"]["budget_tokens"],
            result["config"]["query_type"],
        )
        by_config[key] = result["telemetry"]

    # Print comparison table
    print("\n=== Query Performance Comparison ===")
    print(
        f"{'Config':<40} {'Total':<10} {'Embed':<10} {'Search':<10} {'DP':<10} {'Efficiency':<10}"
    )
    print("-" * 90)

    for (seeds, budget, qtype), telemetry in sorted(by_config.items()):
        config_str = f"{qtype[:8]}, {seeds} seeds, {budget} tokens"
        timings = telemetry["timings"]
        metrics = telemetry["metrics"]

        efficiency = metrics["output_tokens"] / budget if budget > 0 else 0

        print(
            f"{config_str:<40} "
            f"{timings['total_time']:<10.3f} "
            f"{timings['embedding_time']:<10.3f} "
            f"{timings['search_time']:<10.3f} "
            f"{timings['dp_time']:<10.3f} "
            f"{efficiency:<10.2%}"
        )

    # Identify potential bottlenecks
    print("\n=== Phase Analysis ===")
    total_times = [t["timings"]["total_time"] for t in by_config.values()]
    avg_total = sum(total_times) / len(total_times)

    phase_times = {}
    for phase in [
        "embedding_time",
        "search_time",
        "mmr_time",
        "coverage_map_time",
        "scoring_time",
        "dp_time",
        "assembly_time",
    ]:
        phase_times[phase] = sum(t["timings"][phase] for t in by_config.values()) / len(
            by_config
        )

    print(f"Average total time: {avg_total:.3f}s")
    print("\nPhase contribution to total time:")
    for phase, avg_time in sorted(
        phase_times.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (avg_time / avg_total) * 100
        print(
            f"  - {phase.replace('_time', ''):<15} {avg_time:.3f}s ({percentage:.1f}%)"
        )


if __name__ == "__main__":
    # Run specific benchmark
    import sys

    if len(sys.argv) > 3:
        num_seeds = int(sys.argv[1])
        budget_tokens = int(sys.argv[2])
        query_type = sys.argv[3]
        test_query_performance(num_seeds, budget_tokens, query_type)
    else:
        print(
            "Usage: python test_query_performance.py <num_seeds> <budget_tokens> <query_type>"
        )
        print("Example: python test_query_performance.py 10 2000 specific")
        print("Query types: specific, broad, complex")
