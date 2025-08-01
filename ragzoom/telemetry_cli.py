"""CLI interface for RagZoom Telemetry - Developer tools for analyzing telemetry data."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import click

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry_analysis import (
    analyze_retry_patterns,
    compute_amplification_metrics,
    compute_batch_efficiency,
    compute_metrics_from_telemetry,
)
from ragzoom.telemetry_config import (
    CHANGE_SIGNIFICANCE_THRESHOLD,
    EMOJI_THRESHOLD_MINOR,
)

# Type aliases for complex dictionaries
# Note: More specific typing would require extensive refactoring due to mixed types
MetricsDict = dict[str, Any]
TelemetryDict = dict[str, Any]
ThresholdsDict = dict[str, float]

# Check for optional telemetry dependencies
# Note: telemetry_viz.py also imports these but this check provides
# user-friendly error messages before attempting to use visualization features
try:
    import matplotlib  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    import seaborn as sns  # noqa: F401
    from matplotlib.gridspec import GridSpec  # noqa: F401

    TELEMETRY_DEPS_AVAILABLE = True
except ImportError as e:
    TELEMETRY_DEPS_AVAILABLE = False
    MISSING_DEPS = str(e)


def _check_telemetry_deps() -> None:
    """Check if telemetry dependencies are available, exit with helpful message if not."""
    if not TELEMETRY_DEPS_AVAILABLE:
        click.echo("❌ Error: Missing required telemetry dependencies.", err=True)
        click.echo("", err=True)
        click.echo(
            "The ragzoom-telemetry commands require additional dependencies for visualization and data analysis.",
            err=True,
        )
        click.echo("", err=True)
        click.echo("Please install them with:", err=True)
        click.echo("  pip install ragzoom[telemetry]", err=True)
        click.echo("", err=True)
        click.echo("Or install individual packages:", err=True)
        click.echo("  pip install matplotlib seaborn pandas numpy", err=True)
        click.echo("", err=True)
        click.echo(f"Missing dependency details: {MISSING_DEPS}", err=True)
        sys.exit(1)


@click.group()
def cli() -> None:
    """RagZoom Telemetry: Developer tools for analyzing telemetry data."""
    pass


def _write_error_report(error_msg: str, output: str | None) -> None:
    """Write error report to output file or stdout."""
    report = f"""## ❌ Performance Comparison Failed

**Error**: {error_msg}

The benchmark comparison could not be completed. Please check:
- File paths are correct and files exist
- Files contain valid telemetry data
- ragzoom package is properly installed and importable

For debugging, try running the commands manually to see detailed error messages.
"""

    if output:
        Path(output).write_text(report)
        click.echo(f"❌ Error report saved to {output}")
    else:
        click.echo(report)


def load_single_benchmark(filepath: Path) -> tuple[int, MetricsDict]:
    """Load a telemetry benchmark file and extract chunk size and computed metrics.

    Returns:
        Tuple of (chunk_size, computed_metrics_dict)
    """
    with open(filepath) as f:
        data = json.load(f)

    # Only support telemetry format
    if "telemetry" not in data or "config" not in data:
        raise ValueError(
            f"File {filepath} is not in telemetry format. "
            "Only telemetry format is supported. "
            "Expected structure: {config, document, telemetry}"
        )

    chunk_size = data["config"]["leaf_tokens"]

    # Create config for analysis
    api_key = os.getenv("RAGZOOM_OPENAI_API_KEY", "not-needed-for-analysis")
    config = RagZoomConfig(
        openai_api_key=api_key,
        leaf_tokens=chunk_size,
        summary_input_cost_per_1k=0.0025,
        summary_output_cost_per_1k=0.01,
    )

    # Compute metrics from telemetry
    telemetry = data["telemetry"]
    amplification = compute_amplification_metrics(telemetry, config)
    batch_efficiency = compute_batch_efficiency(telemetry)
    retry_patterns = analyze_retry_patterns(telemetry)

    # Compute full metrics to extract summary stats
    full_metrics = compute_metrics_from_telemetry(telemetry, config)

    # Extract summary accuracy stats
    summary_accuracy = {}
    for target_size, stats in full_metrics.summary_stats.items():
        summary_accuracy[target_size] = {
            "avg_deviation": stats.avg_deviation_percent,
            "median_deviation": stats.median_deviation_percent,
            "std_deviation": stats.std_deviation_percent,
            "p95_deviation": stats.percentile_95,
            "count": stats.count,
            "deviations": stats.deviations,  # Raw data for visualization
        }

    # Return computed metrics (not in old format - this is the new way)
    metrics = {
        "amplification": {
            "median_cost": amplification["median_cost"],
            "cost_p90": amplification["cost_p90"],
            "cost_p95": amplification["cost_p95"],
            "median_input": amplification["median_input"],
            "median_output": amplification["median_output"],
        },
        "efficiency": {
            "avg_embedding_batch_size": batch_efficiency["avg_batch_size"],
            "batch_utilization": batch_efficiency["batch_utilization"],
        },
        "retry_patterns": {
            "retry_rate": retry_patterns["retry_rate"],
            "retry_success_rate": retry_patterns["retry_success_rate"],
        },
        "summary_accuracy": summary_accuracy,
        "document": data.get("document", {}),
    }

    return chunk_size, metrics


def calculate_change(old_value: float, new_value: float) -> tuple[float, str]:
    """Calculate percentage change and return (percentage, emoji)."""
    if old_value == 0:
        return 0, ""

    change = ((new_value - old_value) / old_value) * 100

    # Determine emoji based on metric type and direction
    # Only show emojis for significant changes to reduce noise
    if abs(change) < EMOJI_THRESHOLD_MINOR:
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


def format_metric_rows_with_chunk_label(
    chunk_size: int,
    chunk_rows: list[tuple[str, float, float, float, str]],
    format_baseline: str = "{baseline:.1f}%",
    format_current: str = "{current:.1f}%",
    special_formatting: dict[str, tuple[str, str]] | None = None,
) -> list[str]:
    """Format metric rows with chunk size label on the first row.

    Args:
        chunk_size: The chunk size in tokens
        chunk_rows: List of (metric_name, baseline, current, change, emoji) tuples
        format_baseline: Format string for baseline values
        format_current: Format string for current values
        special_formatting: Dict mapping metric names to (baseline_fmt, current_fmt) tuples

    Returns:
        List of formatted table rows
    """
    rows = []
    for i, (metric_name, baseline, current, change, emoji) in enumerate(chunk_rows):
        # Check for special formatting for this metric
        if special_formatting and metric_name in special_formatting:
            baseline_fmt, current_fmt = special_formatting[metric_name]
            baseline_str = baseline_fmt.format(baseline=baseline)
            current_str = current_fmt.format(current=current)
        else:
            baseline_str = format_baseline.format(baseline=baseline)
            current_str = format_current.format(current=current)

        if i == 0:
            rows.append(
                f"| {chunk_size} tokens | {metric_name} | {baseline_str} | {current_str} | {change:+.1f}% {emoji} |"
            )
        else:
            rows.append(
                f"| | {metric_name} | {baseline_str} | {current_str} | {change:+.1f}% {emoji} |"
            )
    return rows


def check_regression(
    change_pct: float, metric_name: str, thresholds: ThresholdsDict
) -> bool:
    """Check if a metric change represents a regression."""
    # Map metric names to threshold types
    threshold_map = {
        "Median Cost Amplification": "summary_token",
        "90th Percentile Cost": "summary_token",
        "95th Percentile Cost": "summary_token",
        "Median Input Amplification": "avg_deviation",
        "Median Output Amplification": "avg_deviation",
    }

    threshold_type = threshold_map.get(metric_name, "default")
    threshold = thresholds.get(threshold_type, 10.0)

    # For amplification metrics, increase is bad
    return change_pct > threshold


def _format_summary_accuracy_section(
    baseline_metrics: MetricsDict,
    current_metrics: MetricsDict,
    thresholds: ThresholdsDict,
    chunk_size: int | None = None,
) -> tuple[list[str], bool]:
    """Format summary accuracy section for reports.

    Args:
        baseline_metrics: Baseline metrics dict
        current_metrics: Current metrics dict
        thresholds: Regression thresholds
        chunk_size: Chunk size to look for in summary_accuracy data

    Returns:
        Tuple of (report lines, has_regression)
    """
    report: list[str] = []
    has_regression = False

    baseline_summary = baseline_metrics.get("summary_accuracy", {})
    current_summary = current_metrics.get("summary_accuracy", {})

    if not baseline_summary or not current_summary:
        return report, has_regression

    # For single file comparison, try to find the chunk size
    if chunk_size is None:
        # Use the first available chunk size
        chunk_sizes = set(baseline_summary.keys()) & set(current_summary.keys())
        if chunk_sizes:
            chunk_size = sorted(chunk_sizes)[0]
        else:
            return report, has_regression

    baseline_stats = baseline_summary.get(chunk_size, {})
    current_stats = current_summary.get(chunk_size, {})

    if not baseline_stats or not current_stats:
        return report, has_regression

    report.append("### 📏 Summary Size Accuracy")
    report.append("")
    report.append("| Metric | Baseline | Current | Change |")
    report.append("|--------|----------|---------|--------|")

    # Average Deviation
    baseline_val = baseline_stats.get("avg_deviation", 0)
    current_val = current_stats.get("avg_deviation", 0)
    if baseline_val > 0:
        change_pct, emoji = calculate_change(baseline_val, current_val)
        # For deviations, increase is bad
        if change_pct > 10:
            emoji = "⚠️"
        if change_pct > 30:
            has_regression = True
            emoji += " ❌"
        change_str = f"{change_pct:+.1f}% {emoji}"
        report.append(
            f"| Average Deviation | {baseline_val:.1f}% | {current_val:.1f}% | {change_str} |"
        )

    # Median Deviation
    baseline_val = baseline_stats.get("median_deviation", 0)
    current_val = current_stats.get("median_deviation", 0)
    if baseline_val > 0:
        change_pct, emoji = calculate_change(baseline_val, current_val)
        if change_pct > 10:
            emoji = "⚠️"
        if change_pct > 30:
            has_regression = True
            emoji += " ❌"
        change_str = f"{change_pct:+.1f}% {emoji}"
        report.append(
            f"| Median Deviation | {baseline_val:.1f}% | {current_val:.1f}% | {change_str} |"
        )

    # P95 Deviation
    baseline_val = baseline_stats.get("p95_deviation", 0)
    current_val = current_stats.get("p95_deviation", 0)
    if baseline_val > 0:
        change_pct, emoji = calculate_change(baseline_val, current_val)
        if change_pct > 20:
            emoji = "⚠️"
        change_str = f"{change_pct:+.1f}% {emoji}"
        report.append(
            f"| P95 Deviation | {baseline_val:.1f}% | {current_val:.1f}% | {change_str} |"
        )

    report.append("")
    return report, has_regression


def _format_amplification_section(
    baseline_metrics: MetricsDict,
    current_metrics: MetricsDict,
    thresholds: ThresholdsDict,
) -> tuple[list[str], bool]:
    """Format amplification metrics section.

    Returns:
        Tuple of (report lines, has_regression)
    """
    report: list[str] = []
    has_regression = False

    baseline_amp = baseline_metrics.get("amplification", {})
    current_amp = current_metrics.get("amplification", {})

    if not baseline_amp or not current_amp:
        return report, has_regression

    report.append("### 📈 Amplification Metrics")
    report.append("")
    report.append("| Metric | Baseline | Current | Change |")
    report.append("|--------|----------|---------|--------|")

    metrics = [
        ("Median Cost Amplification", "median_cost", ""),
        ("90th Percentile Cost", "cost_p90", ""),
        ("95th Percentile Cost", "cost_p95", ""),
        ("Median Input Amplification", "median_input", ""),
        ("Median Output Amplification", "median_output", ""),
    ]

    for display_name, key, unit in metrics:
        baseline_val = baseline_amp.get(key, 0)
        current_val = current_amp.get(key, 0)

        if baseline_val > 0:
            change_pct, emoji = calculate_change(baseline_val, current_val)
            change_str = f"{change_pct:+.1f}% {emoji}"

            # Check for regression
            if check_regression(change_pct, display_name, thresholds):
                has_regression = True
                change_str += " ❌"
        else:
            change_str = "N/A"

        report.append(
            f"| {display_name} | {baseline_val:.2f}x | {current_val:.2f}x | {change_str} |"
        )

    report.append("")
    return report, has_regression


def _format_efficiency_section(
    baseline_metrics: MetricsDict,
    current_metrics: MetricsDict,
) -> list[str]:
    """Format efficiency metrics section."""
    report: list[str] = []

    baseline_eff = baseline_metrics.get("efficiency", {})
    current_eff = current_metrics.get("efficiency", {})

    if not baseline_eff or not current_eff:
        return report

    report.append("### 📦 Efficiency Metrics")
    report.append("")
    report.append("| Metric | Baseline | Current | Change |")
    report.append("|--------|----------|---------|--------|")

    metrics = [
        ("Avg Embedding Batch Size", "avg_embedding_batch_size", ""),
        ("Batch Utilization", "batch_utilization", "percent"),
    ]

    for display_name, key, metric_type in metrics:
        baseline_val = baseline_eff.get(key, 0)
        current_val = current_eff.get(key, 0)

        if baseline_val > 0:
            change_pct, emoji = calculate_change(baseline_val, current_val)
            # For efficiency metrics, decrease might be bad
            if "Utilization" in display_name and change_pct < 0:
                emoji = "⚠️" if abs(change_pct) > EMOJI_THRESHOLD_MINOR else ""
            change_str = f"{change_pct:+.1f}% {emoji}"
        else:
            change_str = "N/A"

        baseline_fmt = format_value(baseline_val, metric_type)
        current_fmt = format_value(current_val, metric_type)

        report.append(
            f"| {display_name} | {baseline_fmt} | {current_fmt} | {change_str} |"
        )

    report.append("")
    return report


def _format_retry_patterns_section(
    baseline_metrics: MetricsDict,
    current_metrics: MetricsDict,
) -> list[str]:
    """Format retry patterns section if there are retries."""
    report: list[str] = []

    baseline_retry = baseline_metrics.get("retry_patterns", {})
    current_retry = current_metrics.get("retry_patterns", {})

    if not baseline_retry or not current_retry:
        return report

    # Only show if there are retries
    baseline_rate = baseline_retry.get("retry_rate", 0)
    current_rate = current_retry.get("retry_rate", 0)

    if baseline_rate == 0 and current_rate == 0:
        return report

    report.append("### 🔄 Retry Patterns")
    report.append("")
    report.append("| Metric | Baseline | Current | Change |")
    report.append("|--------|----------|---------|--------|")

    # Retry rate
    if baseline_rate > 0 or current_rate > 0:
        if baseline_rate > 0:
            change_pct, emoji = calculate_change(baseline_rate, current_rate)
            if change_pct > 50:
                emoji = "⚠️"
            change_str = f"{change_pct:+.1f}% {emoji}"
        else:
            change_str = "New retries ⚠️"

        report.append(
            f"| Retry Rate | {baseline_rate:.1f}% | {current_rate:.1f}% | {change_str} |"
        )

    # Success rate
    baseline_success = baseline_retry.get("retry_success_rate", 0)
    current_success = current_retry.get("retry_success_rate", 0)

    if baseline_rate > 0 or current_rate > 0:
        if baseline_success > 0:
            change_pct, emoji = calculate_change(baseline_success, current_success)
            # For success rate, decrease is bad
            if change_pct < -10:
                emoji = "⚠️"
            change_str = f"{change_pct:+.1f}% {emoji}"
        else:
            change_str = "N/A"

        report.append(
            f"| Retry Success Rate | {baseline_success:.1f}% | {current_success:.1f}% | {change_str} |"
        )

    report.append("")
    return report


def generate_comparison_report(
    baseline_metrics: MetricsDict,
    current_metrics: MetricsDict,
    baseline_name: str,
    current_name: str,
    thresholds: ThresholdsDict | None = None,
) -> tuple[str, bool]:
    """Generate comparison report between two benchmarks.

    Returns:
        Tuple of (report_text, has_regression)
    """
    if thresholds is None:
        thresholds = _load_thresholds()

    report = []
    has_regression = False

    report.append("## 📊 Performance Comparison Report")
    report.append("")
    report.append(f"**Baseline:** {baseline_name}")
    report.append(f"**Current:** {current_name}")
    report.append("")

    # Summary accuracy section (NEW - was computed but not shown)
    summary_lines, summary_regression = _format_summary_accuracy_section(
        baseline_metrics, current_metrics, thresholds
    )
    report.extend(summary_lines)
    has_regression = has_regression or summary_regression

    # Amplification metrics section (using new helper)
    amp_lines, amp_regression = _format_amplification_section(
        baseline_metrics, current_metrics, thresholds
    )
    report.extend(amp_lines)
    has_regression = has_regression or amp_regression

    # Efficiency metrics section (using new helper)
    eff_lines = _format_efficiency_section(baseline_metrics, current_metrics)
    report.extend(eff_lines)

    # Retry patterns section (NEW)
    retry_lines = _format_retry_patterns_section(baseline_metrics, current_metrics)
    report.extend(retry_lines)

    # Summary
    if has_regression:
        report.append("### ❌ Regression Detected")
        report.append("")
        report.append(
            "Performance regressions were detected. Please review the metrics above."
        )
    else:
        report.append("### ✅ No Regressions")
        report.append("")
        report.append("All metrics are within acceptable thresholds.")

    return "\n".join(report), has_regression


def _generate_unified_comparison_report(
    chunk_metrics: dict[int, dict[str, MetricsDict]],
    baseline_name: str,
    current_name: str,
    thresholds: ThresholdsDict,
) -> tuple[str, bool]:
    """Generate a unified comparison report for multiple chunk sizes.

    Args:
        chunk_metrics: Dict mapping chunk_size -> {"baseline": metrics, "current": metrics}
        baseline_name: Name for baseline (e.g., "baseline_results")
        current_name: Name for current (e.g., "current_results")
        thresholds: Regression thresholds

    Returns:
        Tuple of (report_text, has_regression)
    """
    report = []
    has_regression = False

    # Get the change significance threshold
    significance_threshold = thresholds["change_significance"]

    report.append("# 📊 Performance Report")
    report.append("")

    # Sort chunk sizes once for consistent ordering across all sections
    sorted_chunks = sorted(chunk_metrics.keys())

    # Summary Size Accuracy Table (first, most important)
    report.append("## 📏 Summary Size Accuracy")
    report.append("")

    summary_rows = []
    for chunk_size in sorted_chunks:
        baseline_summary = chunk_metrics[chunk_size]["baseline"].get(
            "summary_accuracy", {}
        )
        current_summary = chunk_metrics[chunk_size]["current"].get(
            "summary_accuracy", {}
        )

        # Get the summary stats for the chunk size (target size)
        baseline_stats = baseline_summary.get(chunk_size, {})
        current_stats = current_summary.get(chunk_size, {})

        if baseline_stats and current_stats:
            # Collect all significant changes for this chunk size
            chunk_rows = []

            # Average Deviation
            baseline_val = baseline_stats.get("avg_deviation", 0)
            current_val = current_stats.get("avg_deviation", 0)
            if baseline_val > 0:
                change_pct, emoji = calculate_change(baseline_val, current_val)
                # For deviations, increase is bad
                if change_pct > 10:
                    emoji = "⚠️"
                if change_pct > 30:
                    has_regression = True
                    emoji += " ❌"
                if abs(change_pct) >= significance_threshold:
                    chunk_rows.append(
                        ("Avg Deviation", baseline_val, current_val, change_pct, emoji)
                    )

            # Median Deviation
            baseline_val = baseline_stats.get("median_deviation", 0)
            current_val = current_stats.get("median_deviation", 0)
            if baseline_val > 0:
                change_pct, emoji = calculate_change(baseline_val, current_val)
                if change_pct > 10:
                    emoji = "⚠️"
                if change_pct > 30:
                    has_regression = True
                    emoji += " ❌"
                if abs(change_pct) >= significance_threshold:
                    chunk_rows.append(
                        (
                            "Median Deviation",
                            baseline_val,
                            current_val,
                            change_pct,
                            emoji,
                        )
                    )

            # Standard Deviation
            baseline_val = baseline_stats.get("std_deviation", 0)
            current_val = current_stats.get("std_deviation", 0)
            if baseline_val > 0:
                change_pct, emoji = calculate_change(baseline_val, current_val)
                if change_pct > 20:
                    emoji = "⚠️"
                # Don't flag regression for std deviation (too volatile)
                if abs(change_pct) >= significance_threshold:
                    chunk_rows.append(
                        ("Std Deviation", baseline_val, current_val, change_pct, emoji)
                    )

            # P95 Deviation
            baseline_val = baseline_stats.get("p95_deviation", 0)
            current_val = current_stats.get("p95_deviation", 0)
            if baseline_val > 0:
                change_pct, emoji = calculate_change(baseline_val, current_val)
                if change_pct > 20:
                    emoji = "⚠️"
                # Don't flag regression for P95 (too volatile)
                if abs(change_pct) >= significance_threshold:
                    chunk_rows.append(
                        (
                            "P95 Deviation",
                            baseline_val,
                            current_val,
                            change_pct,
                            emoji,
                        )
                    )

            # Add rows with chunk size label on the first row
            summary_rows.extend(
                format_metric_rows_with_chunk_label(chunk_size, chunk_rows)
            )

    if summary_rows:
        report.append("| Chunk Size | Metric | Baseline | Current | Change |")
        report.append("|------------|--------|----------|---------|--------|")
        report.extend(summary_rows)
    else:
        report.append(
            f"No significant changes (all metrics within ±{significance_threshold:.0f}%)"
        )

    report.append("")

    # Amplification Metrics Table
    report.append("## 📈 Amplification Metrics")
    report.append("")

    amplification_rows = []
    for chunk_size in sorted_chunks:
        baseline_metrics = chunk_metrics[chunk_size]["baseline"]["amplification"]
        current_metrics = chunk_metrics[chunk_size]["current"]["amplification"]

        # Collect all significant changes for this chunk size
        chunk_rows = []

        # Cost amplification (main metric)
        baseline_val = baseline_metrics.get("median_cost", 0)
        current_val = current_metrics.get("median_cost", 0)
        if baseline_val > 0:
            change_pct, emoji = calculate_change(baseline_val, current_val)
            if check_regression(change_pct, "Median Cost Amplification", thresholds):
                has_regression = True
                emoji += " ❌"
            if abs(change_pct) >= significance_threshold:
                chunk_rows.append(
                    (
                        "Median Cost Amplification",
                        baseline_val,
                        current_val,
                        change_pct,
                        emoji,
                    )
                )

        # Input amplification (sub-metric)
        baseline_val = baseline_metrics.get("median_input", 0)
        current_val = current_metrics.get("median_input", 0)
        if baseline_val > 0:
            change_pct, emoji = calculate_change(baseline_val, current_val)
            if abs(change_pct) >= significance_threshold:
                chunk_rows.append(
                    (
                        "Input Amplification",
                        baseline_val,
                        current_val,
                        change_pct,
                        emoji,
                    )
                )

        # Output amplification (sub-metric)
        baseline_val = baseline_metrics.get("median_output", 0)
        current_val = current_metrics.get("median_output", 0)
        if baseline_val > 0:
            change_pct, emoji = calculate_change(baseline_val, current_val)
            if abs(change_pct) >= significance_threshold:
                chunk_rows.append(
                    (
                        "Output Amplification",
                        baseline_val,
                        current_val,
                        change_pct,
                        emoji,
                    )
                )

        # Add rows with chunk size label on the first row
        amplification_rows.extend(
            format_metric_rows_with_chunk_label(
                chunk_size,
                chunk_rows,
                format_baseline="{baseline:.2f}x",
                format_current="{current:.2f}x",
            )
        )

    if amplification_rows:
        report.append("| Chunk Size | Metric | Baseline | Current | Change |")
        report.append("|------------|--------|----------|---------|--------|")
        report.extend(amplification_rows)
    else:
        report.append(
            f"No significant changes (all metrics within ±{significance_threshold:.0f}%)"
        )

    report.append("")

    # Embedding Efficiency Table
    report.append("## 📦 Embedding Efficiency")
    report.append("")

    efficiency_rows = []
    for chunk_size in sorted_chunks:
        baseline_metrics = chunk_metrics[chunk_size]["baseline"]["efficiency"]
        current_metrics = chunk_metrics[chunk_size]["current"]["efficiency"]

        # Collect all significant changes for this chunk size
        chunk_rows = []

        # Batch size
        baseline_val = baseline_metrics.get("avg_embedding_batch_size", 0)
        current_val = current_metrics.get("avg_embedding_batch_size", 0)
        if baseline_val > 0:
            change_pct, emoji = calculate_change(baseline_val, current_val)
            if abs(change_pct) >= significance_threshold:
                chunk_rows.append(
                    (
                        "Avg Embedding Batch Size",
                        baseline_val,
                        current_val,
                        change_pct,
                        emoji,
                    )
                )

        # Batch utilization
        baseline_val = baseline_metrics.get("batch_utilization", 0)
        current_val = current_metrics.get("batch_utilization", 0)
        if baseline_val > 0:
            change_pct, emoji = calculate_change(baseline_val, current_val)
            # For utilization, decrease might be bad
            if change_pct < 0 and abs(change_pct) > EMOJI_THRESHOLD_MINOR:
                emoji = "⚠️"
            if abs(change_pct) >= significance_threshold:
                chunk_rows.append(
                    ("Batch Utilization", baseline_val, current_val, change_pct, emoji)
                )

        # Add rows with chunk size label on the first row
        efficiency_rows.extend(
            format_metric_rows_with_chunk_label(
                chunk_size,
                chunk_rows,
                format_baseline="{baseline:.1f}",
                format_current="{current:.1f}",
                special_formatting={
                    "Batch Utilization": ("{baseline:.1f}%", "{current:.1f}%")
                },
            )
        )

    if efficiency_rows:
        report.append("| Chunk Size | Metric | Baseline | Current | Change |")
        report.append("|------------|--------|----------|---------|--------|")
        report.extend(efficiency_rows)
    else:
        report.append(
            f"No significant changes (all metrics within ±{significance_threshold:.0f}%)"
        )

    report.append("")

    # Summary
    if has_regression:
        report.append("## ❌ Regression Detected")
        report.append("")
        report.append(
            "Performance regressions were detected in one or more chunk sizes. Please review the metrics above."
        )
        report.append("")  # Extra newline for better formatting
    else:
        report.append("## ✅ No Regressions")
        report.append("")
        report.append(
            "All metrics across all chunk sizes are within acceptable thresholds."
        )
        report.append("")  # Extra newline for better formatting

    return "\n".join(report), has_regression


def _load_thresholds() -> ThresholdsDict:
    """Load regression thresholds from environment variables."""
    return {
        "summary_token": float(
            os.getenv("PERF_SUMMARY_TOKEN_REGRESSION_THRESHOLD", "10.0")
        ),
        "avg_deviation": float(
            os.getenv("PERF_AVG_DEVIATION_REGRESSION_THRESHOLD", "20.0")
        ),
        "median_deviation": float(
            os.getenv("PERF_MEDIAN_DEVIATION_REGRESSION_THRESHOLD", "20.0")
        ),
        "std_deviation": float(
            os.getenv("PERF_STD_DEVIATION_REGRESSION_THRESHOLD", "30.0")
        ),
        "p95": float(os.getenv("PERF_P95_REGRESSION_THRESHOLD", "25.0")),
        "change_significance": float(
            os.getenv(
                "PERF_CHANGE_SIGNIFICANCE_THRESHOLD", str(CHANGE_SIGNIFICANCE_THRESHOLD)
            )
        ),
    }


def _compare_files(baseline_file: Path, current_file: Path, output: str | None) -> None:
    """Compare two telemetry files and generate a report."""
    # Load benchmarks
    try:
        baseline_chunk_size, baseline_metrics = load_single_benchmark(baseline_file)
        current_chunk_size, current_metrics = load_single_benchmark(current_file)
    except Exception as e:
        error_msg = f"Error loading benchmark files: {e}"
        _write_error_report(error_msg, output)
        sys.exit(1)

    # Warn if chunk sizes don't match
    if baseline_chunk_size != current_chunk_size:
        click.echo(
            f"⚠️  Warning: Chunk sizes differ - baseline: {baseline_chunk_size}, current: {current_chunk_size}",
            err=True,
        )
        click.echo("Using baseline chunk size for comparison\n", err=True)

    # Get thresholds from environment
    thresholds = _load_thresholds()

    # Generate comparison report
    report, has_regression = generate_comparison_report(
        baseline_metrics,
        current_metrics,
        baseline_file.name,
        current_file.name,
        thresholds,
    )

    # Output report
    if output:
        Path(output).write_text(report)
        click.echo(f"✅ Comparison report saved to {output}")
    else:
        click.echo(report)

    # Exit with appropriate code
    if has_regression:
        sys.exit(1)
    else:
        sys.exit(0)


def _match_telemetry_files(dir1: Path, dir2: Path) -> list[tuple[Path, Path]]:
    """Match telemetry files between two directories by token count.

    Returns list of (baseline_file, current_file) tuples.
    """
    # Find telemetry files in both directories
    dir1_files = list(dir1.glob("telemetry_*_tokens.json"))
    dir2_files = list(dir2.glob("telemetry_*_tokens.json"))

    # Also support generic telemetry.json files
    dir1_files.extend(dir1.glob("telemetry.json"))
    dir2_files.extend(dir2.glob("telemetry.json"))

    # Create mapping by filename pattern
    dir1_map = {f.name: f for f in dir1_files}
    dir2_map = {f.name: f for f in dir2_files}

    # Find matching pairs
    matches = []
    for filename, file1 in dir1_map.items():
        if filename in dir2_map:
            matches.append((file1, dir2_map[filename]))

    return sorted(matches, key=lambda x: x[0].name)


def _compare_directories(
    baseline_dir: Path, current_dir: Path, output: str | None
) -> None:
    """Compare all matching telemetry files between two directories."""
    # Find matching files
    matches = _match_telemetry_files(baseline_dir, current_dir)

    if not matches:
        click.echo(
            f"❌ No matching telemetry files found between {baseline_dir} and {current_dir}",
            err=True,
        )
        sys.exit(1)

    click.echo(f"📊 Found {len(matches)} matching file pairs to compare\n")

    # Track overall results and collect metrics by chunk size
    chunk_metrics: dict[int, dict[str, MetricsDict]] = {}
    any_error = False
    error_messages = []

    for baseline_file, current_file in matches:
        click.echo(f"Comparing {baseline_file.name}...", err=True)

        try:
            # Load benchmarks
            baseline_chunk_size, baseline_metrics = load_single_benchmark(baseline_file)
            current_chunk_size, current_metrics = load_single_benchmark(current_file)

            # Warn if chunk sizes don't match
            if baseline_chunk_size != current_chunk_size:
                click.echo(
                    f"  ⚠️  Warning: Chunk sizes differ - baseline: {baseline_chunk_size}, current: {current_chunk_size}",
                    err=True,
                )

            # Use baseline chunk size as the key
            chunk_size = baseline_chunk_size
            chunk_metrics[chunk_size] = {
                "baseline": baseline_metrics,
                "current": current_metrics,
            }

            click.echo("  ✅ Loaded successfully", err=True)

        except Exception as e:
            click.echo(f"  ❌ Error: {e}", err=True)
            error_messages.append(f"Error loading {baseline_file.name}: {e}")
            any_error = True

    # Generate unified report if we have any successful comparisons
    if chunk_metrics:
        thresholds = _load_thresholds()
        combined_report, any_regression = _generate_unified_comparison_report(
            chunk_metrics,
            baseline_dir.name,
            current_dir.name,
            thresholds,
        )

        # Append error messages if any
        if error_messages:
            combined_report += "\n\n## ⚠️ Errors Encountered\n\n"
            for error_msg in error_messages:
                combined_report += f"- {error_msg}\n"
    else:
        # All files failed to load
        combined_report = "# ❌ Directory Comparison Failed\n\n"
        combined_report += (
            f"**Baseline:** {baseline_dir}\n**Current:** {current_dir}\n\n"
        )
        combined_report += "No files could be successfully compared:\n\n"
        for error_msg in error_messages:
            combined_report += f"- {error_msg}\n"
        any_regression = True  # Treat total failure as regression

    # Output report
    if output:
        Path(output).write_text(combined_report)
        click.echo(f"\n✅ Combined comparison report saved to {output}")
    else:
        click.echo(f"\n{'='*60}\n")
        click.echo(combined_report)

    # Exit with appropriate code
    # Exit with code 1 if any regressions were detected OR any errors occurred during comparison
    # This ensures CI fails if either performance degrades or comparison process fails
    if any_regression or any_error:
        sys.exit(1)
    else:
        sys.exit(0)


@cli.command("compare")
@click.argument("path1", type=click.Path(exists=True))
@click.argument("path2", type=click.Path(exists=True))
@click.option(
    "--output",
    type=click.Path(),
    help="Output file for comparison report (defaults to stdout)",
)
def compare(path1: str, path2: str, output: str | None) -> None:
    """Compare telemetry data between two benchmark files or directories.

    Examples:
        Compare two files:
            ragzoom-telemetry compare baseline.json current.json

        Compare directories:
            ragzoom-telemetry compare baseline_results/ current_results/
    """
    try:
        path1_obj = Path(path1)
        path2_obj = Path(path2)

        # Check if both are directories
        if path1_obj.is_dir() and path2_obj.is_dir():
            _compare_directories(path1_obj, path2_obj, output)
        elif path1_obj.is_file() and path2_obj.is_file():
            # Single file comparison (existing logic)
            _compare_files(path1_obj, path2_obj, output)
        else:
            click.echo(
                "❌ Error: Both arguments must be either files or directories", err=True
            )
            sys.exit(1)

    except Exception as e:
        error_msg = f"Unexpected error during comparison: {e}"
        _write_error_report(error_msg, output)
        sys.exit(1)


@cli.command("visualize")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    type=click.Path(),
    default="telemetry_reports",
    help="Output directory for visualizations",
)
@click.option(
    "--format",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    help="Output format (default: png)",
)
@click.option(
    "--compare",
    is_flag=True,
    help="Generate comparison visualizations when input is a directory",
)
def visualize(input_path: str, output_dir: str, format: str, compare: bool) -> None:
    """Generate visualizations from telemetry data."""
    # Check dependencies first
    _check_telemetry_deps()

    try:
        from ragzoom.telemetry_viz import TelemetryVisualizer

        visualizer = TelemetryVisualizer(Path(output_dir))
        input_path_obj = Path(input_path)

        if input_path_obj.is_file():
            # Single file visualization
            visualizer.visualize_single_benchmark(input_path_obj, format)
        elif input_path_obj.is_dir():
            # Directory of benchmarks
            json_files = list(input_path_obj.glob("telemetry_*_tokens.json"))
            # Also support new telemetry.json files
            json_files.extend(input_path_obj.glob("telemetry*.json"))

            if not json_files:
                click.echo(f"❌ No benchmark files found in {input_path}")
                sys.exit(1)

            # Visualize each file
            for file in json_files:
                visualizer.visualize_single_benchmark(file, format)

            # Generate comparison if requested
            if compare and len(json_files) >= 2:
                visualizer.visualize_comparison(input_path_obj, format)
        else:
            click.echo(f"❌ Error: {input_path} not found")
            sys.exit(1)

        click.echo("\n✅ Visualization complete!")

    except Exception as e:
        click.echo(f"❌ Error generating visualizations: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
