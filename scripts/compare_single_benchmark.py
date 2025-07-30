#!/usr/bin/env python3
"""Compare two individual benchmark JSON files from ragzoom CLI.

Usage:
    python compare_single_benchmark.py baseline.json current.json

Example:
    # Run on two different branches
    ragzoom index document.txt --benchmark --benchmark-output master.json
    git checkout feature-branch
    ragzoom index document.txt --benchmark --benchmark-output feature.json
    python scripts/compare_single_benchmark.py master.json feature.json
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import from compare_benchmarks
sys.path.insert(0, str(Path(__file__).parent))

from compare_benchmarks import generate_comparison_table

# Default chunk size if unable to determine from benchmark data
DEFAULT_CHUNK_SIZE = 200


def load_single_benchmark(filepath: Path) -> tuple[int, dict]:
    """Load a single benchmark file and extract chunk size and metrics.

    Returns:
        Tuple of (chunk_size, metrics_dict)
    """
    with open(filepath) as f:
        data = json.load(f)

    # Handle new telemetry format
    if "telemetry" in data and "config" in data:
        # New format with telemetry
        chunk_size = data["config"]["leaf_tokens"]

        # Import telemetry analysis functions
        from ragzoom.config import RagZoomConfig
        from ragzoom.telemetry import (
            analyze_retry_patterns,
            compute_amplification_metrics,
            compute_batch_efficiency,
        )

        # Create config for analysis
        config = RagZoomConfig(
            openai_api_key="dummy",  # Not needed for analysis
            leaf_tokens=chunk_size,
            summary_input_cost_per_1k=0.0025,
            summary_output_cost_per_1k=0.01,
        )

        # Compute metrics from telemetry
        telemetry = data["telemetry"]
        amplification = compute_amplification_metrics(telemetry, config)
        batch_efficiency = compute_batch_efficiency(telemetry)
        retry_patterns = analyze_retry_patterns(telemetry)

        # Build metrics dict in old format for compatibility
        metrics = {
            "timing": {
                "total_duration_seconds": 0,  # Not available in telemetry
                "tokens_per_second": 0,
                "time_per_1k_tokens": 0,
            },
            "document": data.get("document", {}),
            "api_usage": {
                "total_calls": batch_efficiency["total_batches"] + retry_patterns["total_attempts"],
                "embedding_calls": batch_efficiency["total_batches"],
                "summary_calls": retry_patterns["total_attempts"],
                "embedding_tokens": batch_efficiency["total_embeddings"],
                "summary_prompt_tokens": 0,  # Would need computation
                "summary_completion_tokens": 0,  # Would need computation
            },
            "efficiency": {
                "avg_embedding_batch_size": batch_efficiency["avg_batch_size"],
                "batch_utilization": batch_efficiency["batch_utilization"],
                "embedding_tokens_per_1k": 0,  # Would need computation
                "summary_tokens_per_1k": 0,  # Would need computation
                "api_calls_per_1k": 0,  # Would need computation
                "cost_per_1k_tokens": 0,  # Would need computation
            },
            "amplification": {
                "median_cost": amplification["median_cost"],
                "cost_p90": amplification["cost_p90"],
                "cost_p95": amplification["cost_p95"],
                "median_input": amplification["median_input"],
                "median_output": amplification["median_output"],
            },
            "summary_accuracy": {},  # Would need more complex analysis
        }

        return chunk_size, metrics

    # Old format - the data IS the metrics
    # Try to infer chunk size from summary_accuracy keys
    chunk_size = None

    if "summary_accuracy" in data and data["summary_accuracy"]:
        # Get the first key which should be the target size (same as chunk size)
        chunk_size = int(list(data["summary_accuracy"].keys())[0])
    elif "document" in data and "chunks_created" in data["document"]:
        # This is approximate but better than nothing
        source_tokens = data["document"]["source_tokens"]
        chunks = data["document"]["chunks_created"]
        chunk_size = source_tokens // chunks if chunks > 0 else DEFAULT_CHUNK_SIZE

    if chunk_size is None:
        print(f"Warning: Could not determine chunk size, defaulting to {DEFAULT_CHUNK_SIZE}", file=sys.stderr)
        chunk_size = DEFAULT_CHUNK_SIZE

    return chunk_size, data


def main() -> None:
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python compare_single_benchmark.py baseline.json current.json")
        print("Example: python compare_single_benchmark.py master.json feature.json")
        sys.exit(1)

    baseline_file = Path(sys.argv[1])
    current_file = Path(sys.argv[2])

    # Check files exist
    if not baseline_file.exists():
        print(f"Error: Baseline file not found: {baseline_file}", file=sys.stderr)
        sys.exit(1)

    if not current_file.exists():
        print(f"Error: Current file not found: {current_file}", file=sys.stderr)
        sys.exit(1)

    # Load benchmarks
    try:
        baseline_chunk_size, baseline_metrics = load_single_benchmark(baseline_file)
        current_chunk_size, current_metrics = load_single_benchmark(current_file)
    except Exception as e:
        print(f"Error loading benchmark files: {e}", file=sys.stderr)
        sys.exit(1)

    # Warn if chunk sizes don't match
    if baseline_chunk_size != current_chunk_size:
        print(f"Warning: Chunk sizes differ - baseline: {baseline_chunk_size}, current: {current_chunk_size}", file=sys.stderr)
        print("Using baseline chunk size for comparison\n", file=sys.stderr)

    # Wrap in expected format for generate_comparison_table
    # The function expects: {chunk_size: {"metrics": {...}}}
    # But ragzoom CLI output IS the metrics directly, so we wrap it
    baseline_wrapped = {baseline_chunk_size: {"metrics": baseline_metrics}}
    current_wrapped = {baseline_chunk_size: {"metrics": current_metrics}}

    # Get thresholds from environment
    summary_threshold = float(os.getenv("PERF_SUMMARY_TOKEN_REGRESSION_THRESHOLD", "10.0"))
    avg_deviation_threshold = float(os.getenv("PERF_AVG_DEVIATION_REGRESSION_THRESHOLD", "20.0"))
    median_deviation_threshold = float(os.getenv("PERF_MEDIAN_DEVIATION_REGRESSION_THRESHOLD", "20.0"))
    std_deviation_threshold = float(os.getenv("PERF_STD_DEVIATION_REGRESSION_THRESHOLD", "30.0"))
    p95_threshold = float(os.getenv("PERF_P95_REGRESSION_THRESHOLD", "25.0"))

    # Generate comparison report
    report, has_summary_regression, has_accuracy_regression = generate_comparison_table(
        baseline_wrapped,
        current_wrapped,
        summary_token_regression_threshold=summary_threshold,
        avg_deviation_regression_threshold=avg_deviation_threshold,
        median_deviation_regression_threshold=median_deviation_threshold,
        std_deviation_regression_threshold=std_deviation_threshold,
        p95_regression_threshold=p95_threshold,
    )

    # Add file info header
    header = "Comparing benchmarks:\n"
    header += f"  Baseline: {baseline_file.name}\n"
    header += f"  Current:  {current_file.name}\n"

    print(header)
    print(report)

    # Exit with error code if regressions detected
    if has_summary_regression or has_accuracy_regression:
        sys.exit(1)


if __name__ == "__main__":
    main()
