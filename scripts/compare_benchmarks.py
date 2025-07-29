#!/usr/bin/env python3
"""Compare benchmark results between branches and generate markdown report."""

import json
import os
import sys
from pathlib import Path


def load_benchmark_results(results_dir: Path) -> dict[int, dict]:
    """Load benchmark results from JSON files."""
    results = {}

    for file in results_dir.glob("metrics_*_tokens.json"):
        try:
            with open(file) as f:
                data = json.load(f)
                chunk_size = data["config"]["leaf_tokens"]
                results[chunk_size] = data
        except Exception as e:
            print(f"Error loading {file}: {e}", file=sys.stderr)

    return results


def calculate_change(old_value: float, new_value: float) -> tuple[float, str]:
    """Calculate percentage change and return (percentage, emoji)."""
    if old_value == 0:
        return 0, ""

    change = ((new_value - old_value) / old_value) * 100

    # Determine emoji based on metric type and direction
    if abs(change) < 1:
        emoji = ""
    elif change > 0:
        # For cost and time metrics, increase is bad
        emoji = "⚠️"
    else:
        # For cost and time metrics, decrease is good
        emoji = "✅"

    return change, emoji


def format_value(value: float, metric_type: str) -> str:
    """Format value based on metric type."""
    if metric_type == "cost":
        return f"${value:.4f}"
    elif metric_type == "percent":
        return f"{value:.1f}%"
    elif metric_type == "time":
        return f"{value:.2f}s"
    else:
        return f"{value:.1f}"


def generate_comparison_table(
    baseline: dict[int, dict],
    current: dict[int, dict],
    output_format: str = "markdown",
    summary_token_regression_threshold: float = 10.0,
) -> tuple[str, bool]:
    """Generate comparison table between baseline and current results.

    Args:
        baseline: Baseline benchmark results by chunk size
        current: Current benchmark results by chunk size
        output_format: Output format (only 'markdown' supported)
        summary_token_regression_threshold: Percentage increase to trigger regression warning

    Returns:
        Tuple of (markdown report, has_summary_regression)
    """

    # Get all chunk sizes present in both sets
    chunk_sizes = sorted(set(baseline.keys()) & set(current.keys()))

    if not chunk_sizes:
        return "❌ No matching chunk sizes found between baseline and current results", False

    lines = []

    # Header
    lines.append("## 📊 Performance Report\n")

    # Token usage comparison
    lines.append("\n### Token Usage (per 1K source tokens)")
    lines.append("| Chunk Size | Metric | Baseline | Current | Change |")
    lines.append("|------------|--------|----------|---------|--------|")

    summary_regression = False

    for size in chunk_sizes:
        base_m = baseline[size]["metrics"]["efficiency"]
        curr_m = current[size]["metrics"]["efficiency"]

        # Summary tokens
        base_summary = base_m["summary_tokens_per_1k"]
        curr_summary = curr_m["summary_tokens_per_1k"]
        change, emoji = calculate_change(base_summary, curr_summary)

        if change > summary_token_regression_threshold:
            summary_regression = True
            emoji = "❌"
        elif abs(change) > 1:
            emoji = "⚠️" if change > 0 else "✅"
        else:
            emoji = ""

        lines.append(
            f"| | Summary | {base_summary:.1f} | {curr_summary:.1f} | "
            f"{emoji} {change:+.1f}% |"
        )

        # Total cost (informational only)
        base_cost = base_m["cost_per_1k_tokens"]
        curr_cost = curr_m["cost_per_1k_tokens"]
        change, emoji = calculate_change(base_cost, curr_cost)

        # Just show warning/success for cost, don't use for regression
        if change > 10:
            emoji = "⚠️"
        elif change < -5:
            emoji = "✅"
        else:
            emoji = ""

        lines.append(
            f"| | **Total Cost** | ${base_cost:.4f} | ${curr_cost:.4f} | "
            f"{emoji} {change:+.1f}% |"
        )

    # Summary accuracy if available
    if any("summary_accuracy" in current[size]["metrics"] for size in chunk_sizes):
        lines.append("\n### Summary Size Accuracy")
        lines.append("| Chunk Size | Avg Deviation | Over Target | Under Target |")
        lines.append("|------------|---------------|-------------|--------------|")

        for size in chunk_sizes:
            if "summary_accuracy" not in current[size]["metrics"]:
                continue

            # Get summary stats for the chunk size (same as target)
            stats_dict = current[size]["metrics"]["summary_accuracy"]
            if str(size) in stats_dict:
                stats = stats_dict[str(size)]
                lines.append(
                    f"| {size} tokens | {stats['avg_deviation_percent']:.1f}% | "
                    f"{stats['percent_over_target']:.1f}% | "
                    f"{stats['percent_under_target']:.1f}% |"
                )

    # Summary
    lines.append("\n### Summary")

    if summary_regression:
        lines.append(f"❌ Summary token regression detected (>{summary_token_regression_threshold}% increase)")
    else:
        lines.append("✅ No significant regressions detected")

    # Show thresholds used
    lines.append(f"\n*Regression threshold: summary tokens >{summary_token_regression_threshold}% increase*")
    lines.append("*Cost changes are shown for informational purposes but do not trigger regression detection.*")

    return "\n".join(lines), summary_regression


def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 3:
        print("Usage: python compare_benchmarks.py <baseline_dir> <current_dir> [output_file]")
        print("Example: python compare_benchmarks.py baseline_results/ current_results/ report.md")
        sys.exit(1)

    baseline_dir = Path(sys.argv[1])
    current_dir = Path(sys.argv[2])
    output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    # Get threshold from environment or use default
    summary_threshold = float(os.getenv("PERF_SUMMARY_TOKEN_REGRESSION_THRESHOLD", "10.0"))

    # Load results
    baseline_results = load_benchmark_results(baseline_dir)
    current_results = load_benchmark_results(current_dir)

    if not baseline_results:
        print(f"Error: No benchmark results found in {baseline_dir}", file=sys.stderr)
        sys.exit(1)

    if not current_results:
        print(f"Error: No benchmark results found in {current_dir}", file=sys.stderr)
        sys.exit(1)

    # Generate comparison with configurable threshold
    report, has_regression = generate_comparison_table(
        baseline_results,
        current_results,
        summary_token_regression_threshold=summary_threshold,
    )

    # Output
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report written to {output_file}")
    else:
        print(report)

    # Exit with error code if regressions detected
    if has_regression:
        sys.exit(1)


if __name__ == "__main__":
    main()

