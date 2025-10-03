"""Performance benchmarks for query operations."""

import asyncio
import json
import os
from pathlib import Path
from typing import cast

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol
from ragzoom.retrieve import Retriever
from ragzoom.telemetry_query import QueryMetricsDict
from tests.benchmarks.runtime_helpers import append_document


# Skip benchmarks by default unless explicitly requested
def test_query_performance_placeholder() -> None:
    """Placeholder so marker exclusion doesn't yield exit code 5 when deselected."""
    pytest.skip("Benchmark tests require explicit opt-in")


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
        max_chars = int(os.getenv("QUERY_BENCHMARK_DOC_CHARS", "0"))
        if max_chars > 0:
            text = text[:max_chars]
        return text, name
    except Exception as e:
        pytest.skip(f"Could not load test document {file_path}: {e}")


def setup_test_document(
    storage_backend: StorageBackend, api_key: str, vector_index: _VectorIndexProtocol
) -> str:
    """Get or reuse the test document for query benchmarking.

    Returns the document ID of the indexed document.
    """
    # Use the same document ID as the 200-token indexing benchmark
    doc_id = "test_200_tokens"

    # Check if document already exists from earlier benchmark
    existing_doc = storage_backend.get_document_by_id(doc_id)
    if existing_doc:
        print(f"  Using existing index: {doc_id}")
        return doc_id

    # Only index if not found (shouldn't happen in CI since we index earlier)
    print(f"  Document {doc_id} not found, indexing...")
    index_config = IndexConfig.load(target_chunk_tokens=200)
    test_doc, _ = get_test_document("narrative")

    try:
        vector_index.delete(filter={"document_id": doc_id})
    except Exception:
        pass

    storage_backend.clear_document(doc_id)

    _result, _telemetry = append_document(
        storage_backend=storage_backend,
        index_config=index_config,
        vector_index=vector_index,
        document_id=doc_id,
        text=test_doc,
        api_key=api_key,
        file_path="test_narrative.txt",
        collect_telemetry=False,
    )

    return doc_id


@pytest.mark.benchmark
@pytest.mark.slow_threshold(float(os.getenv("QUERY_BENCHMARK_SLOW_THRESHOLD", "600")))
@pytest.mark.parametrize("num_seeds", [5, 10, 20])
@pytest.mark.parametrize("budget_tokens", [1000, 2000, 4000])
@pytest.mark.parametrize("query_type", ["specific", "broad", "complex"])
def test_query_performance(
    storage_backend: StorageBackend,
    num_seeds: int,
    budget_tokens: int,
    query_type: str,
    vector_index: _VectorIndexProtocol,
) -> None:
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
    # Index document first (or reuse if exists)
    doc_id = setup_test_document(storage_backend, api_key, vector_index)

    # Create services for Retriever
    from openai import OpenAI

    from ragzoom.retrieval.budget_planner import BudgetPlanner
    from ragzoom.retrieval.embedding_service import EmbeddingService

    client = OpenAI(api_key=api_key)

    # Get DocumentStore for the test document
    doc_store = storage_backend.for_document(doc_id)

    embedding_service = EmbeddingService(
        client, doc_store, query_config.embedding_model
    )
    budget_planner = BudgetPlanner(
        doc_store, IndexConfig.load(target_chunk_tokens=200).target_chunk_tokens
    )

    # Create retriever and assembler
    retriever = Retriever(
        query_config,
        doc_store,
        embedding_service,
        budget_planner,
        vector_index,
    )
    assembler = Assembler(doc_store)

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

    statistics_summary: dict[str, object] = {"num_runs": num_runs}

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
            mad = statistics.median(absolute_deviations) if absolute_deviations else 0.0

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
    total_stats = cast(dict[str, float], statistics_summary["total_time"])
    embedding_stats = cast(dict[str, float], statistics_summary["embedding_time"])
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
    print(f"  - Coverage map: {median_telemetry['timings']['coverage_map_time']:.3f}s")
    print(f"  - Scoring: {median_telemetry['timings']['scoring_time']:.3f}s")
    print(f"  - DP tiling: {median_telemetry['timings']['dp_time']:.3f}s")
    print(f"  - Assembly: {median_telemetry['timings']['assembly_time']:.3f}s")
    print("\nResults (median run):")
    print(
        f"  - Seeds found: {median_telemetry['metrics']['seeds_found']}/{median_telemetry['metrics']['seeds_requested']}"
    )
    print(f"  - Coverage size: {median_telemetry['metrics']['coverage_size']} nodes")
    print(f"  - Tiling size: {median_telemetry['metrics']['tiling_size']} nodes")
    print(
        f"  - Output tokens: {median_telemetry['metrics']['output_tokens']}/{budget_tokens}"
    )

    # Basic validation using median run
    median_metrics: QueryMetricsDict = median_telemetry["metrics"]
    output_tokens_val = median_metrics["output_tokens"]
    output_tokens = (
        output_tokens_val
        if isinstance(output_tokens_val, int)
        else int(output_tokens_val)
    )
    assert (
        output_tokens <= budget_tokens
    ), f"Output exceeded budget: {output_tokens} > {budget_tokens}"
    assert median_metrics["tiling_size"] > 0, "No nodes in tiling"
    assert total_stats["median"] > 0, "Invalid timing"


@pytest.mark.benchmark
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

        # Note: For standalone execution, this would require a storage backend.
        # Use pytest fixtures in practice: `pytest tests/benchmarks/test_query_performance.py`
        print(
            "Error: Standalone execution requires refactoring to use pytest fixtures."
        )
        print(
            "Run: pytest tests/benchmarks/test_query_performance.py::test_query_performance"
        )
        import sys

        sys.exit(1)
    else:
        print(
            "Usage: python test_query_performance.py <num_seeds> <budget_tokens> <query_type>"
        )
        print("Example: python test_query_performance.py 10 2000 specific")
        print("Query types: specific, broad, complex")
