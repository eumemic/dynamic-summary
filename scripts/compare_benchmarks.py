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
    avg_deviation_regression_threshold: float = 20.0,
    median_deviation_regression_threshold: float = 10.0,
    std_deviation_regression_threshold: float = 30.0,
    p95_regression_threshold: float = 25.0,
) -> tuple[str, bool, bool]:
    """Generate comparison table between baseline and current results.

    Args:
        baseline: Baseline benchmark results by chunk size
        current: Current benchmark results by chunk size
        output_format: Output format (only 'markdown' supported)
        summary_token_regression_threshold: Percentage increase to trigger regression warning
        avg_deviation_regression_threshold: Percentage increase for avg deviation regression
        median_deviation_regression_threshold: Percentage increase for median deviation regression
        std_deviation_regression_threshold: Percentage increase for std deviation regression
        p95_regression_threshold: Percentage increase for P95 regression

    Returns:
        Tuple of (markdown report, has_summary_regression, has_accuracy_regression)
    """

    # Get all chunk sizes present in both sets
    chunk_sizes = sorted(set(baseline.keys()) & set(current.keys()))

    if not chunk_sizes:
        return "❌ No matching chunk sizes found between baseline and current results", False, False

    lines = []

    # Header
    lines.append("## 📊 Performance Report\n")

    # Token usage comparison
    lines.append("\n### Token Usage (per 1K source tokens)")
    lines.append("| Chunk Size | Metric | Baseline | Current | Change |")
    lines.append("|------------|--------|----------|---------|--------|")

    summary_regression = False
    accuracy_regression = False

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

    # Summary accuracy comparison
    if any("summary_accuracy" in current[size]["metrics"] for size in chunk_sizes):
        lines.append("\n### Summary Size Accuracy")
        lines.append("| Chunk Size | Metric | Baseline | Current | Change |")
        lines.append("|------------|--------|----------|---------|--------|")

        for size in chunk_sizes:
            if "summary_accuracy" not in current[size]["metrics"]:
                continue

            # Get current stats
            curr_stats_dict = current[size]["metrics"]["summary_accuracy"]
            if str(size) not in curr_stats_dict:
                continue
            curr_stats = curr_stats_dict[str(size)]

            # Check if baseline has summary accuracy
            has_baseline_accuracy = (
                "summary_accuracy" in baseline[size]["metrics"] and
                str(size) in baseline[size]["metrics"]["summary_accuracy"]
            )

            if has_baseline_accuracy:
                base_stats = baseline[size]["metrics"]["summary_accuracy"][str(size)]

                # Average deviation (always present in old metrics)
                base_avg_dev = base_stats.get("avg_deviation_percent", 0)
                curr_avg_dev = curr_stats["avg_deviation_percent"]
                change, emoji = calculate_change(base_avg_dev, curr_avg_dev)

                # Only warning level for average deviation
                if abs(change) > 10:
                    emoji = "⚠️" if change > 0 else "✅"
                else:
                    emoji = ""

                lines.append(
                    f"| {size} tokens | Avg Deviation | {base_avg_dev:.1f}% | "
                    f"{curr_avg_dev:.1f}% | {emoji} {change:+.1f}% |"
                )

                # Check for new distribution metrics
                has_distribution_metrics = "median_deviation_percent" in base_stats

                if has_distribution_metrics:
                    # Median deviation - the only metric that triggers accuracy regression
                    base_median = base_stats["median_deviation_percent"]
                    curr_median = curr_stats["median_deviation_percent"]
                    change, emoji = calculate_change(base_median, curr_median)

                    if change > median_deviation_regression_threshold:
                        accuracy_regression = True
                        emoji = "❌"
                    elif abs(change) > 5:
                        emoji = "⚠️" if change > 0 else "✅"
                    else:
                        emoji = ""

                    lines.append(
                        f"| | Median Deviation | {base_median:.1f}% | "
                        f"{curr_median:.1f}% | {emoji} {change:+.1f}% |"
                    )

                    # Standard deviation - warning only
                    base_std = base_stats["std_deviation_percent"]
                    curr_std = curr_stats["std_deviation_percent"]
                    change, emoji = calculate_change(base_std, curr_std)

                    # Only warning level
                    if abs(change) > 20:
                        emoji = "⚠️" if change > 0 else "✅"
                    else:
                        emoji = ""

                    lines.append(
                        f"| | Std Deviation | {base_std:.1f}% | "
                        f"{curr_std:.1f}% | {emoji} {change:+.1f}% |"
                    )

                    # P95 Deviation - warning only
                    base_p95 = base_stats["percentile_95"]
                    curr_p95 = curr_stats["percentile_95"]
                    change, emoji = calculate_change(base_p95, curr_p95)

                    # Only warning level
                    if abs(change) > 20:
                        emoji = "⚠️" if change > 0 else "✅"
                    else:
                        emoji = ""

                    lines.append(
                        f"| | P95 Deviation | {base_p95:.1f}% | "
                        f"{curr_p95:.1f}% | {emoji} {change:+.1f}% |"
                    )
                else:
                    # Show new metrics without comparison
                    lines.append(
                        f"| | Median Deviation | - | {curr_stats['median_deviation_percent']:.1f}% | 📊 New |"
                    )
                    lines.append(
                        f"| | Std Deviation | - | {curr_stats['std_deviation_percent']:.1f}% | 📊 New |"
                    )
                    lines.append(
                        f"| | P95 Deviation | - | {curr_stats['percentile_95']:.1f}% | 📊 New |"
                    )
            else:
                # No baseline at all - show current values
                lines.append(
                    f"| {size} tokens | Avg Deviation | - | "
                    f"{curr_stats['avg_deviation_percent']:.1f}% | 📊 New |"
                )
                lines.append(
                    f"| | Median Deviation | - | {curr_stats['median_deviation_percent']:.1f}% | 📊 New |"
                )
                lines.append(
                    f"| | Std Deviation | - | {curr_stats['std_deviation_percent']:.1f}% | 📊 New |"
                )
                lines.append(
                    f"| | P95 Deviation | - | {curr_stats['percentile_95']:.1f}% | 📊 New |"
                )

    # Token Amplification comparison
    if any("amplification" in current[size]["metrics"] for size in chunk_sizes):
        lines.append("\n### Token Efficiency (Amplification Factors)")
        lines.append("| Chunk Size | Metric | Baseline | Current | Change |")
        lines.append("|------------|--------|----------|---------|--------|")

        amplification_regression = False

        for size in chunk_sizes:
            if "amplification" not in current[size]["metrics"]:
                continue

            curr_amp = current[size]["metrics"]["amplification"]

            # Check if baseline has amplification metrics
            has_baseline_amp = (
                "amplification" in baseline[size]["metrics"]
            )

            if has_baseline_amp:
                base_amp = baseline[size]["metrics"]["amplification"]

                # Cost amplification (main regression metric)
                base_cost_amp = base_amp.get("median_cost", 0)
                curr_cost_amp = curr_amp.get("median_cost", 0)

                if base_cost_amp > 0:
                    change = ((curr_cost_amp - base_cost_amp) / base_cost_amp) * 100

                    if change > 10:  # 10% threshold for cost amplification
                        amplification_regression = True
                        emoji = "❌"
                    elif abs(change) > 5:
                        emoji = "⚠️" if change > 0 else "✅"
                    else:
                        emoji = ""

                    lines.append(
                        f"| {size} tokens | Cost Amplification | {base_cost_amp:.2f}x | "
                        f"{curr_cost_amp:.2f}x | {emoji} {change:+.1f}% |"
                    )

                    # Input amplification (informational)
                    base_input = base_amp.get("median_input", 0)
                    curr_input = curr_amp.get("median_input", 0)
                    if base_input > 0:
                        change = ((curr_input - base_input) / base_input) * 100
                        emoji = "⚠️" if abs(change) > 10 else ""
                        lines.append(
                            f"| | ├─ Input | {base_input:.2f}x | {curr_input:.2f}x | "
                            f"{emoji} {change:+.1f}% |"
                        )

                    # Output amplification (informational)
                    base_output = base_amp.get("median_output", 0)
                    curr_output = curr_amp.get("median_output", 0)
                    if base_output > 0:
                        change = ((curr_output - base_output) / base_output) * 100
                        emoji = "⚠️" if abs(change) > 20 else ""
                        lines.append(
                            f"| | └─ Output | {base_output:.2f}x | {curr_output:.2f}x | "
                            f"{emoji} {change:+.1f}% |"
                        )
            else:
                # No baseline - show current values
                lines.append(
                    f"| {size} tokens | Cost Amplification | - | "
                    f"{curr_amp.get('median_cost', 0):.2f}x | 📊 New |"
                )
                lines.append(
                    f"| | ├─ Input | - | {curr_amp.get('median_input', 0):.2f}x | 📊 New |"
                )
                lines.append(
                    f"| | └─ Output | - | {curr_amp.get('median_output', 0):.2f}x | 📊 New |"
                )

    # Summary
    lines.append("\n### Summary")

    issues = []
    if summary_regression:
        issues.append(f"❌ Summary token regression detected (>{summary_token_regression_threshold}% increase)")
    if accuracy_regression:
        issues.append("❌ Summary accuracy regression detected")
    if "amplification_regression" in locals() and amplification_regression:
        issues.append("❌ Cost amplification regression detected (>10% increase)")

    if issues:
        lines.extend(issues)
    else:
        lines.append("✅ No significant regressions detected")

    # Check if we have new metrics without baseline
    has_new_metrics = any(
        "summary_accuracy" in current[size]["metrics"] and
        (
            "summary_accuracy" not in baseline[size]["metrics"] or
            "median_deviation_percent" not in baseline[size]["metrics"]["summary_accuracy"].get(str(size), {})
        )
        for size in chunk_sizes
    )

    if has_new_metrics:
        lines.append("\nℹ️ Summary accuracy tracking enhanced - baseline will update on next master merge")

    # Show thresholds used
    lines.append("\n*Regression thresholds (failures):*")
    lines.append(f"- Summary tokens: >{summary_token_regression_threshold}% increase")

    # Only show median threshold if we have baseline accuracy data with distribution metrics
    if any(
        "summary_accuracy" in baseline.get(size, {}).get("metrics", {}) and
        "median_deviation_percent" in baseline.get(size, {}).get("metrics", {}).get("summary_accuracy", {}).get(str(size), {})
        for size in chunk_sizes
    ):
        lines.append(f"- Median deviation: >{median_deviation_regression_threshold}% increase")

    lines.append("\n*Cost changes are shown for informational purposes but do not trigger regression detection.*")

    # Include amplification regression in the return
    has_amplification_regression = "amplification_regression" in locals() and amplification_regression

    return "\n".join(lines), summary_regression or has_amplification_regression, accuracy_regression


def main() -> None:
    """Main entry point for CLI usage."""
    if len(sys.argv) < 3:
        print("Usage: python compare_benchmarks.py <baseline_dir> <current_dir> [output_file]")
        print("Example: python compare_benchmarks.py baseline_results/ current_results/ report.md")
        sys.exit(1)

    baseline_dir = Path(sys.argv[1])
    current_dir = Path(sys.argv[2])
    output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    # Get thresholds from environment or use defaults
    summary_threshold = float(os.getenv("PERF_SUMMARY_TOKEN_REGRESSION_THRESHOLD", "10.0"))
    avg_deviation_threshold = float(os.getenv("PERF_AVG_DEVIATION_REGRESSION_THRESHOLD", "20.0"))
    median_deviation_threshold = float(os.getenv("PERF_MEDIAN_DEVIATION_REGRESSION_THRESHOLD", "10.0"))
    std_deviation_threshold = float(os.getenv("PERF_STD_DEVIATION_REGRESSION_THRESHOLD", "30.0"))
    p95_threshold = float(os.getenv("PERF_P95_REGRESSION_THRESHOLD", "25.0"))

    # Load results
    baseline_results = load_benchmark_results(baseline_dir)
    current_results = load_benchmark_results(current_dir)

    if not baseline_results:
        print(f"Error: No benchmark results found in {baseline_dir}", file=sys.stderr)
        sys.exit(1)

    if not current_results:
        print(f"Error: No benchmark results found in {current_dir}", file=sys.stderr)
        sys.exit(1)

    # Generate comparison with configurable thresholds
    report, has_summary_regression, has_accuracy_regression = generate_comparison_table(
        baseline_results,
        current_results,
        summary_token_regression_threshold=summary_threshold,
        avg_deviation_regression_threshold=avg_deviation_threshold,
        median_deviation_regression_threshold=median_deviation_threshold,
        std_deviation_regression_threshold=std_deviation_threshold,
        p95_regression_threshold=p95_threshold,
    )

    # Output
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report written to {output_file}")
    else:
        print(report)

    # Exit with error code if regressions detected
    if has_summary_regression or has_accuracy_regression:
        sys.exit(1)


if __name__ == "__main__":
    main()

