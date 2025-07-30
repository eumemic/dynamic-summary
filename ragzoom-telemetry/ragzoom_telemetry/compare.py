"""Telemetry comparison command."""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import click

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry import (
    TelemetryAnalysisError,
    analyze_retry_patterns,
    compute_amplification_metrics,
    compute_batch_efficiency,
)

# Default chunk size if unable to determine from benchmark data
DEFAULT_CHUNK_SIZE = 200

# Emoji display thresholds - these control when to show warning/success indicators
# They don't trigger regression failures, just visual feedback
EMOJI_THRESHOLD_NEGLIGIBLE = 1.0   # Changes below this are not highlighted
EMOJI_THRESHOLD_COST_WARN = 10.0   # Cost increase above this shows warning
EMOJI_THRESHOLD_COST_GOOD = 5.0    # Cost decrease above this shows success
EMOJI_THRESHOLD_MINOR = 5.0        # Minor changes worth noting
EMOJI_THRESHOLD_MODERATE = 10.0    # Moderate changes that warrant attention
EMOJI_THRESHOLD_MAJOR = 20.0       # Major changes that are concerning


def load_single_benchmark(filepath: Path) -> Tuple[int, Dict]:
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

        # Create config for analysis
        # Use environment variable if available, otherwise use placeholder
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

    # Handle old format (metrics at top level)
    elif "metrics" in data:
        # Old benchmark format
        metrics = data["metrics"]
        
        # Try to extract chunk size
        if "config" in data:
            chunk_size = data["config"]["leaf_tokens"]
        elif "document" in metrics:
            # Estimate from document stats
            chunks = metrics["document"].get("chunks_created", 1)
            tokens = metrics["document"].get("source_document_tokens", chunks * DEFAULT_CHUNK_SIZE)
            chunk_size = tokens // chunks if chunks > 0 else DEFAULT_CHUNK_SIZE
        else:
            chunk_size = DEFAULT_CHUNK_SIZE
            
        return chunk_size, metrics
    else:
        raise ValueError(f"Unrecognized benchmark format in {filepath}")


def calculate_change(old_value: float, new_value: float) -> Tuple[float, str]:
    """Calculate percentage change and return (percentage, emoji)."""
    if old_value == 0:
        return 0, ""

    change = ((new_value - old_value) / old_value) * 100

    # Determine emoji based on metric type and direction
    if abs(change) < EMOJI_THRESHOLD_NEGLIGIBLE:
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


def check_regression(
    change_pct: float,
    metric_name: str,
    thresholds: Dict[str, float]
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


def generate_comparison_report(
    baseline_metrics: Dict,
    current_metrics: Dict,
    baseline_name: str,
    current_name: str,
    thresholds: Optional[Dict[str, float]] = None
) -> Tuple[str, bool]:
    """Generate comparison report between two benchmarks.
    
    Returns:
        Tuple of (report_text, has_regression)
    """
    if thresholds is None:
        thresholds = {
            "summary_token": 10.0,
            "avg_deviation": 20.0,
            "median_deviation": 20.0,
            "std_deviation": 30.0,
            "p95": 25.0,
        }
    
    report = []
    has_regression = False
    
    report.append("## 📊 Performance Comparison Report")
    report.append("")
    report.append(f"**Baseline:** {baseline_name}")
    report.append(f"**Current:** {current_name}")
    report.append("")
    
    # Compare amplification metrics
    baseline_amp = baseline_metrics.get("amplification", {})
    current_amp = current_metrics.get("amplification", {})
    
    if baseline_amp and current_amp:
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
    
    # Compare efficiency metrics
    baseline_eff = baseline_metrics.get("efficiency", {})
    current_eff = current_metrics.get("efficiency", {})
    
    if baseline_eff and current_eff:
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
    
    # Summary
    if has_regression:
        report.append("### ❌ Regression Detected")
        report.append("")
        report.append("Performance regressions were detected. Please review the metrics above.")
    else:
        report.append("### ✅ No Regressions")
        report.append("")
        report.append("All metrics are within acceptable thresholds.")
    
    return "\n".join(report), has_regression


@click.command("compare")
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.option(
    "--output",
    type=click.Path(),
    help="Output file for comparison report (defaults to stdout)",
)
def compare(file1: str, file2: str, output: Optional[str]) -> None:
    """Compare telemetry data between two benchmark files."""
    try:
        baseline_file = Path(file1)
        current_file = Path(file2)

        # Load benchmarks
        try:
            baseline_chunk_size, baseline_metrics = load_single_benchmark(baseline_file)
            current_chunk_size, current_metrics = load_single_benchmark(current_file)
        except Exception as e:
            click.echo(f"❌ Error loading benchmark files: {e}", err=True)
            sys.exit(1)

        # Warn if chunk sizes don't match
        if baseline_chunk_size != current_chunk_size:
            click.echo(
                f"⚠️  Warning: Chunk sizes differ - baseline: {baseline_chunk_size}, current: {current_chunk_size}",
                err=True,
            )
            click.echo("Using baseline chunk size for comparison\n", err=True)

        # Get thresholds from environment
        thresholds = {
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
        }

        # Generate comparison report
        report, has_regression = generate_comparison_report(
            baseline_metrics,
            current_metrics,
            baseline_file.name,
            current_file.name,
            thresholds
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

    except Exception as e:
        click.echo(f"❌ Error running comparison: {e}", err=True)
        sys.exit(1)