"""Performance benchmarks for query operations."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, QueryConfig
from ragzoom.document_store import DocumentStore
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import StoreManager
from ragzoom.telemetry_query import QueryMetricsDict

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


def setup_test_document(store: StoreManager, api_key: str) -> str:
    """Get or reuse the test document for query benchmarking.

    Returns the document ID of the indexed document.
    """
    # Use the same document ID as the 200-token indexing benchmark
    doc_id = "test_200_tokens"

    # Check if document already exists from earlier benchmark
    with store.SessionLocal() as session:
        from ragzoom.models import Document

        doc = session.query(Document).filter_by(id=doc_id).first()
        if doc:
            print(f"  Using existing index: {doc_id}")
            return doc_id

    # Only index if not found (shouldn't happen in CI since we index earlier)
    print(f"  Document {doc_id} not found, indexing...")
    index_config = IndexConfig.load(target_chunk_tokens=200)
    test_doc, doc_name = get_test_document("narrative")
    builder = TreeBuilder(index_config, cast(DocumentStore, store), api_key)
    return builder.add_document(test_doc)


@pytest.mark.slow
@pytest.mark.parametrize("num_seeds", [5, 10, 20])
@pytest.mark.parametrize("budget_tokens", [1000, 2000, 4000])
@pytest.mark.parametrize("query_type", ["specific", "broad", "complex"])
def test_query_performance(num_seeds: int, budget_tokens: int, query_type: str) -> None:
    """Benchmark query performance at different parameter combinations."""
    # Create config for this specific test
    api_key = os.getenv("OPENAI_API_KEY", "test-key")

    # Skip if no API key
    if api_key == "test-key" or api_key == "test-key-for-tests":
        pytest.skip("OPENAI_API_KEY not set")

    # Number of runs to reduce API variance (can be overridden via env var)
    num_runs = int(os.getenv("QUERY_BENCHMARK_RUNS", "3"))

    # Query configurations
    queries = {
        "specific": "What is Bilbo's opinion about adventures at the beginning?",
        "broad": "Describe the overall setting and atmosphere of the story",
        "complex": "How do the dwarves convince Bilbo to join their quest, and what are their motivations?",
    }

    query_config = QueryConfig(
        budget_tokens=budget_tokens,
    )

    # Run query with metrics
    with StoreManager.temporary() as store:
        # Index document first (or reuse if exists)
        doc_id = setup_test_document(store, api_key)

        # Create retriever and assembler using the proper pattern
        from openai import OpenAI

        from ragzoom.retrieval.budget_planner import BudgetPlanner
        from ragzoom.retrieval.embedding_service import EmbeddingService

        client = OpenAI(api_key=api_key)
        document_store = store.for_document(doc_id)
        embedding_service = EmbeddingService(
            client, document_store, "text-embedding-3-small"
        )
        index_cfg = IndexConfig.load()
        budget_planner = BudgetPlanner(document_store, index_cfg.target_chunk_tokens)
        retriever = Retriever(
            query_config,
            document_store,
            embedding_service,
            budget_planner,
        )
        assembler = Assembler(document_store)

        # Collect telemetry from multiple runs
        all_telemetries = []
        all_summaries = []

        for run_idx in range(num_runs):
            print(f"  Run {run_idx + 1}/{num_runs}")

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

            all_telemetries.append(telemetry.to_dict())
            all_summaries.append(summary)

        # Calculate statistics from multiple runs
        import statistics

        # All timing phases to calculate statistics for
        timing_phases = [
            "embedding_time",
            "search_time",
            "mmr_time",
            "coverage_map_time",
            "scoring_time",
            "dp_time",
            "assembly_time",
            "total_time",
        ]

        statistics_summary: dict[str, Any] = {"num_runs": num_runs}

        # Calculate statistics for each phase
        for phase in timing_phases:
            phase_times = []
            for t in all_telemetries:
                if phase in t["timings"]:
                    # Use cast to access timing values with dynamic keys
                    timings_dict = cast(dict[str, float], t["timings"])
                    phase_times.append(timings_dict[phase])

            if phase_times:
                # Calculate basic statistics
                median_time = statistics.median(phase_times)
                mean_time = statistics.mean(phase_times)
                std_dev = statistics.stdev(phase_times) if len(phase_times) > 1 else 0.0

                # Calculate MAD for robust variance measurement
                absolute_deviations = [abs(t - median_time) for t in phase_times]
                mad = (
                    statistics.median(absolute_deviations)
                    if absolute_deviations
                    else 0.0
                )

                statistics_summary[phase] = {
                    "median": median_time,
                    "mean": mean_time,
                    "std_dev": std_dev,
                    "mad": mad,  # Median Absolute Deviation for robust variance
                    "min": min(phase_times),
                    "max": max(phase_times),
                }

        # Save telemetry data for comparison
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)

        output_file = (
            output_dir
            / f"query_telemetry_{num_seeds}seeds_{budget_tokens}tokens_{query_type}.json"
        )

        # Save telemetry (updated format with multiple runs)
        telemetry_data = {
            "format_version": "1.1",  # Updated format version for multiple runs
            "telemetries": all_telemetries,  # All individual run telemetries
            "statistics": statistics_summary,  # Aggregated statistics
            "config": {
                "num_seeds": num_seeds,
                "budget_tokens": budget_tokens,
                "query_type": query_type,
                "num_runs": num_runs,
            },
            "summary_preview": (
                all_summaries[0][:500] if all_summaries else ""
            ),  # First summary for validation
        }

        with open(output_file, "w") as f:
            json.dump(telemetry_data, f, indent=2)

        # Print summary with statistics
        print(
            f"\n=== Query Performance ({query_type}, seeds={num_seeds}, budget={budget_tokens}) ==="
        )
        print(f"Runs: {num_runs}")
        total_stats = statistics_summary["total_time"]
        embedding_stats = statistics_summary["embedding_time"]
        print(f"Total time: {total_stats['median']:.3f}s (median)")
        print(f"  ± {total_stats['std_dev']:.3f}s std dev")
        print(f"  Range: {total_stats['min']:.3f}s - {total_stats['max']:.3f}s")
        print(f"Embedding time: {embedding_stats['median']:.3f}s (median)")
        print(f"  ± {embedding_stats['std_dev']:.3f}s std dev")

        # Show median phase breakdown from middle run
        median_run_idx = len(all_telemetries) // 2
        median_telemetry = all_telemetries[median_run_idx]
        print("Phase breakdown (median run):")
        print(f"  - Embedding: {median_telemetry['timings']['embedding_time']:.3f}s")
        print(f"  - Search: {median_telemetry['timings']['search_time']:.3f}s")
        print(f"  - MMR: {median_telemetry['timings']['mmr_time']:.3f}s")
        print(
            f"  - Coverage map: {median_telemetry['timings']['coverage_map_time']:.3f}s"
        )
        print(f"  - Scoring: {median_telemetry['timings']['scoring_time']:.3f}s")
        print(f"  - DP tiling: {median_telemetry['timings']['dp_time']:.3f}s")
        print(f"  - Assembly: {median_telemetry['timings']['assembly_time']:.3f}s")
        print("\nResults (median run):")
        print(
            f"  - Seeds found: {median_telemetry['metrics']['seeds_found']}/{median_telemetry['metrics']['seeds_requested']}"
        )
        print(
            f"  - Coverage size: {median_telemetry['metrics']['coverage_size']} nodes"
        )
        print(f"  - Tiling size: {median_telemetry['metrics']['tiling_size']} nodes")
        print(
            f"  - Output tokens: {median_telemetry['metrics']['output_tokens']}/{budget_tokens}"
        )

        # Basic validation using median run
        median_metrics: QueryMetricsDict = median_telemetry["metrics"]
        assert (
            median_metrics["output_tokens"] <= budget_tokens
        ), f"Output exceeded budget: {median_metrics['output_tokens']} > {budget_tokens}"
        assert median_metrics["tiling_size"] > 0, "No nodes in tiling"
        assert total_stats["median"] > 0, "Invalid timing"


@pytest.mark.slow
def test_query_performance_comparison() -> None:
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
