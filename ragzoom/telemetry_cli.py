"""Telemetry CLI with simplified metrics."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry_analysis import compute_simplified_metrics


@dataclass
class DynamicThreshold:
    """Represents a threshold computed from baseline variance."""

    absolute_value: float
    baseline_variance: float
    k_factors: tuple[float, float]  # (k1_between_run, k2_baseline_uncertainty)
    metric_name: str
    is_computed: bool = True  # False if using static fallback


@dataclass
class ThresholdConfig:
    """Configuration for dynamic threshold calculation.

    The threshold formula is: threshold = (k1 + k2) × baseline_variance

    Default values are based on statistical principles and empirical observations:
    - k1=3.0: Covers 99.7% of normal distribution (3-sigma rule) for between-run variance
    - k2=2.0: Additional margin for baseline measurement uncertainty from limited samples
    - Total 5-sigma: Ensures <0.01% false positive rate for regression detection
    - ci_multiplier=1.5: Based on empirical observation that CI environments show ~1.5x higher variance

    These values mean a metric must exceed 5 standard deviations from the baseline's
    internal variance to be flagged as a regression, effectively eliminating false
    positives from natural LLM non-determinism while still catching real issues.
    """

    # K-factors for threshold calculation
    k1_between_run: float = 3.0  # Expected variance between runs (3-sigma)
    k2_baseline_uncertainty: float = 2.0  # Baseline uncertainty margin (2-sigma)

    # Minimum thresholds (floors) to prevent too-tight bounds
    min_thresholds: dict[str, float] | None = None

    # Whether to detect CI environment and adjust k-factors
    adjust_for_ci: bool = True
    ci_multiplier: float = 1.5  # Additional multiplier for CI environments

    def __post_init__(self) -> None:
        if self.min_thresholds is None:
            self.min_thresholds = {
                "median_error": 15.0,  # tokens
                "p95_error": 30.0,  # tokens
                "latency": 0.5,  # seconds
                "cost": 0.0002,  # USD
                "mad": 10.0,  # tokens
                "retry_rate": 0.2,  # ratio
                "percent_within_10": 10.0,  # percentage points
            }


def compute_dynamic_threshold(
    baseline_metrics: dict[str, Any],
    metric_name: str,
    variance_key: str,
    config: ThresholdConfig,
    is_ci: bool = False,
) -> DynamicThreshold:
    """Compute dynamic threshold based on baseline's internal variance.

    Args:
        baseline_metrics: Metrics dictionary containing variance data
        metric_name: Name of the metric (e.g., "median_error")
        variance_key: Key to extract variance (e.g., "error_mad")
        config: Threshold configuration
        is_ci: Whether running in CI environment

    Returns:
        DynamicThreshold with computed absolute threshold
    """
    # Extract variance from the appropriate metric group
    if metric_name in ["median_error", "p95_error", "percent_within_10"]:
        variance = baseline_metrics["target_fit"].get(variance_key, 0.0)
    elif metric_name in ["median_seconds"]:
        variance = baseline_metrics["latency"].get(variance_key, 0.0)
    elif metric_name == "mad":
        variance = baseline_metrics["dispersion"].get("mad", 0.0)
    elif metric_name == "retry_rate":
        variance = baseline_metrics["retries"].get(variance_key, 0.0)
    elif metric_name in ["usd_per_node", "cost"]:
        variance = baseline_metrics["cost"].get(variance_key, 0.0)
    else:
        # For metrics without variance data, fall back to static threshold
        min_value = 0.1
        if config.min_thresholds is not None:
            min_value = config.min_thresholds.get(metric_name, 0.1)
        return DynamicThreshold(
            absolute_value=min_value,
            baseline_variance=0.0,
            k_factors=(0.0, 0.0),
            metric_name=metric_name,
            is_computed=False,
        )

    # Apply k-factors
    k1 = config.k1_between_run
    k2 = config.k2_baseline_uncertainty

    # Adjust for CI if needed
    if is_ci and config.adjust_for_ci:
        k1 *= config.ci_multiplier
        k2 *= config.ci_multiplier

    # Calculate threshold
    threshold = (k1 + k2) * variance

    # Apply minimum floor
    min_threshold = 0.0
    if config.min_thresholds is not None:
        min_threshold = config.min_thresholds.get(metric_name, 0.0)
    threshold = max(threshold, min_threshold)

    return DynamicThreshold(
        absolute_value=threshold,
        baseline_variance=variance,
        k_factors=(k1, k2),
        metric_name=metric_name,
        is_computed=True,
    )


def _match_telemetry_files(dir1: Path, dir2: Path) -> list[tuple[Path, Path]]:
    """Match telemetry files between two directories by filename.

    Returns list of (baseline_file, current_file) tuples.
    """
    # Find telemetry files in both directories
    dir1_files = list(dir1.glob("telemetry_*_tokens.json"))
    dir2_files = list(dir2.glob("telemetry_*_tokens.json"))

    # Also support generic telemetry.json files
    dir1_files.extend(dir1.glob("telemetry.json"))
    dir2_files.extend(dir2.glob("telemetry.json"))

    # Create mapping by filename
    dir1_map = {f.name: f for f in dir1_files}
    dir2_map = {f.name: f for f in dir2_files}

    # Find matching pairs
    matches = []
    for filename, file1 in dir1_map.items():
        if filename in dir2_map:
            matches.append((file1, dir2_map[filename]))

    return sorted(matches, key=lambda x: x[0].name)


def _check_telemetry_deps() -> None:
    """Check if telemetry visualization dependencies are installed."""
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
    except ImportError:
        click.echo(
            "❌ Telemetry visualization requires optional dependencies.\n"
            "Install with: pip install ragzoom[telemetry]",
            err=True,
        )
        sys.exit(1)


@click.group()
def cli() -> None:
    """Telemetry analysis with simplified metrics."""
    pass


@cli.command()
@click.argument("telemetry_file", type=click.Path(exists=True, path_type=Path))
def analyze(telemetry_file: Path) -> None:
    """Analyze telemetry data and display simplified metrics."""

    # Load telemetry data
    with open(telemetry_file) as f:
        telemetry_data = json.load(f)

    # Handle wrapped telemetry format (with config/document/telemetry fields)
    # If data has 'telemetry' field but no 'documents' field, it's wrapped
    if "telemetry" in telemetry_data and "documents" not in telemetry_data:
        telemetry_data = telemetry_data["telemetry"]

    # Compute metrics
    config = RagZoomConfig()
    metrics = compute_simplified_metrics(telemetry_data, config)

    # Display metrics for each chunk size
    for chunk_size in sorted(metrics.metrics_by_chunk_size.keys()):
        chunk_metrics = metrics.metrics_by_chunk_size[chunk_size]

        click.echo(f"\n{'='*60}")
        click.echo(f"  Chunk Size: {chunk_size} tokens")
        click.echo(f"{'='*60}")

        # Target-fit metrics
        target_fit = chunk_metrics["target_fit"]
        click.echo("\n📏 Target-fit Accuracy")
        click.echo(f"  Median error:        {target_fit['median_error']:+.1f} tokens")
        click.echo(f"  p95 error:           {target_fit['p95_error']:+.1f} tokens")
        click.echo(f"  Within ±10 tokens:   {target_fit['percent_within_10']:.1f}%")
        click.echo(f"  Max overshoot:       {target_fit['max_overshoot']:+.0f} tokens")
        click.echo(f"  Max undershoot:      {target_fit['max_undershoot']:+.0f} tokens")

        # Retry metrics
        retries = chunk_metrics["retries"]
        click.echo("\n🔄 Retry Efficiency")
        click.echo(
            f"  Retry rate:          {retries['retry_rate']:.2f} extra attempts/node"
        )
        click.echo(f"  Max retries:         {retries['max_retries']:.0f}")

        # Latency metrics
        latency = chunk_metrics["latency"]
        click.echo("\n⏱️  Latency")
        click.echo(f"  Median time/node:    {latency['median_seconds']:.2f}s")
        click.echo(f"  p95 time/node:       {latency['p95_seconds']:.2f}s")
        click.echo(f"  Total indexing:      {latency['total_indexing_seconds']:.1f}s")

        # Cost metrics
        cost = chunk_metrics["cost"]
        click.echo("\n💰 Cost & Tokens")
        click.echo(f"  Prompt tokens:       {cost['total_prompt_tokens']:,}")
        click.echo(f"  Completion tokens:   {cost['total_completion_tokens']:,}")
        click.echo(f"  Total tokens:        {cost['total_tokens']:,}")
        click.echo(f"  USD per node:        ${cost['usd_per_node']:.4f}")

        # Dispersion metrics
        dispersion = chunk_metrics["dispersion"]
        click.echo("\n📊 Consistency")
        click.echo(f"  MAD:                 {dispersion['mad']:.1f} tokens")

    click.echo(f"\n{'='*60}\n")


def check_regression_with_dynamic_threshold(
    baseline_val: float,
    current_val: float,
    threshold: DynamicThreshold,
    higher_is_better: bool = False,
) -> tuple[bool, float]:
    """Check for regression using absolute dynamic threshold.

    Args:
        baseline_val: Baseline metric value
        current_val: Current metric value
        threshold: Dynamic threshold computed from baseline variance
        higher_is_better: If True, decrease is regression; if False, increase is regression

    Returns:
        Tuple of (is_regression, absolute_change)
    """
    absolute_change = current_val - baseline_val

    if higher_is_better:
        # For metrics where higher is better, regression is a decrease beyond threshold
        is_regression = absolute_change < -threshold.absolute_value
    else:
        # For metrics where lower is better, regression is an increase beyond threshold
        is_regression = absolute_change > threshold.absolute_value

    return is_regression, absolute_change


def _load_and_compute_metrics(file_path: Path) -> tuple[dict, Any]:
    """Load telemetry file and compute simplified metrics.

    Returns:
        Tuple of (telemetry_data, simplified_metrics)
    """
    with open(file_path) as f:
        data = json.load(f)

    # Handle wrapped telemetry format (with config/document/telemetry fields)
    # If data has 'telemetry' field but no 'documents' field, it's wrapped
    if "telemetry" in data and "documents" not in data:
        telemetry_data = data["telemetry"]
    else:
        telemetry_data = data

    # Compute metrics
    config = RagZoomConfig()
    metrics = compute_simplified_metrics(telemetry_data, config)

    return telemetry_data, metrics


def _compare_files(baseline_file: Path, current_file: Path, output: str) -> bool:
    """Compare two telemetry files.

    Returns:
        True if regression detected
    """
    # Load and compute metrics for both files
    baseline_data, baseline_metrics = _load_and_compute_metrics(baseline_file)
    current_data, current_metrics = _load_and_compute_metrics(current_file)

    # Detect if running in CI
    is_ci = current_data.get("environment", {}).get("ci", False)

    # Find common chunk sizes
    baseline_sizes = set(baseline_metrics.metrics_by_chunk_size.keys())
    current_sizes = set(current_metrics.metrics_by_chunk_size.keys())
    common_sizes = baseline_sizes & current_sizes

    if not common_sizes:
        click.echo("No common chunk sizes found between files", err=True)
        return False  # Return False, don't exit here

    # Check for regressions with dynamic thresholds
    has_regression, thresholds_by_chunk = (
        _check_metrics_for_regressions_with_thresholds(
            baseline_metrics, current_metrics, common_sizes, is_ci
        )
    )

    # Format comparison with thresholds
    if output == "markdown":
        _format_markdown_comparison_with_thresholds(
            baseline_metrics, current_metrics, common_sizes, thresholds_by_chunk
        )
    else:
        _format_text_comparison_with_thresholds(
            baseline_metrics, current_metrics, common_sizes, thresholds_by_chunk
        )

    return has_regression


def _check_metrics_for_regressions_with_thresholds(
    baseline: Any,
    current: Any,
    chunk_sizes: set[int],
    is_ci: bool = False,
) -> tuple[bool, dict[int, dict[str, DynamicThreshold]]]:
    """Check if metrics show regressions using dynamic thresholds.

    Returns:
        Tuple of (has_regression, thresholds_by_chunk)
    """
    has_regression = False
    config = ThresholdConfig()
    thresholds_by_chunk = {}

    for chunk_size in chunk_sizes:
        base_metrics = baseline.metrics_by_chunk_size[chunk_size]
        curr_metrics = current.metrics_by_chunk_size[chunk_size]
        chunk_thresholds = {}

        # Check median error regression
        threshold = compute_dynamic_threshold(
            base_metrics, "median_error", "error_mad", config, is_ci
        )
        chunk_thresholds["median_error"] = threshold
        is_regressed, _ = check_regression_with_dynamic_threshold(
            abs(base_metrics["target_fit"]["median_error"]),
            abs(curr_metrics["target_fit"]["median_error"]),
            threshold,
        )
        if is_regressed:
            has_regression = True

        # Check p95 error regression
        threshold = compute_dynamic_threshold(
            base_metrics, "p95_error", "error_mad", config, is_ci
        )
        chunk_thresholds["p95_error"] = threshold
        is_regressed, _ = check_regression_with_dynamic_threshold(
            abs(base_metrics["target_fit"]["p95_error"]),
            abs(curr_metrics["target_fit"]["p95_error"]),
            threshold,
        )
        if is_regressed:
            has_regression = True

        # Check latency regression
        threshold = compute_dynamic_threshold(
            base_metrics, "median_seconds", "latency_mad", config, is_ci
        )
        chunk_thresholds["latency"] = threshold
        is_regressed, _ = check_regression_with_dynamic_threshold(
            base_metrics["latency"]["median_seconds"],
            curr_metrics["latency"]["median_seconds"],
            threshold,
        )
        if is_regressed:
            has_regression = True

        # Check MAD regression
        threshold = compute_dynamic_threshold(base_metrics, "mad", "mad", config, is_ci)
        chunk_thresholds["mad"] = threshold
        is_regressed, _ = check_regression_with_dynamic_threshold(
            base_metrics["dispersion"]["mad"],
            curr_metrics["dispersion"]["mad"],
            threshold,
        )
        if is_regressed:
            has_regression = True

        # Check retry rate regression with dynamic threshold
        threshold = compute_dynamic_threshold(
            base_metrics, "retry_rate", "retry_mad", config, is_ci
        )
        chunk_thresholds["retry_rate"] = threshold
        is_regressed, _ = check_regression_with_dynamic_threshold(
            base_metrics["retries"]["retry_rate"],
            curr_metrics["retries"]["retry_rate"],
            threshold,
        )
        if is_regressed:
            has_regression = True

        # Check cost regression with dynamic threshold
        threshold = compute_dynamic_threshold(
            base_metrics, "usd_per_node", "cost_mad", config, is_ci
        )
        chunk_thresholds["cost"] = threshold
        is_regressed, _ = check_regression_with_dynamic_threshold(
            base_metrics["cost"]["usd_per_node"],
            curr_metrics["cost"]["usd_per_node"],
            threshold,
        )
        if is_regressed:
            has_regression = True

        # Check percent_within_10 regression with dynamic threshold
        threshold = compute_dynamic_threshold(
            base_metrics, "percent_within_10", "percent_within_10_mad", config, is_ci
        )
        chunk_thresholds["percent_within_10"] = threshold
        is_regressed, _ = check_regression_with_dynamic_threshold(
            base_metrics["target_fit"]["percent_within_10"],
            curr_metrics["target_fit"]["percent_within_10"],
            threshold,
            higher_is_better=True,  # Higher percentage within ±10 is better
        )
        if is_regressed:
            has_regression = True

        thresholds_by_chunk[chunk_size] = chunk_thresholds

    return has_regression, thresholds_by_chunk


def _compare_directories(baseline_dir: Path, current_dir: Path, output: str) -> bool:
    """Compare all matching telemetry files between two directories.

    Returns:
        True if any regression detected
    """
    # Find matching files
    matches = _match_telemetry_files(baseline_dir, current_dir)

    if not matches:
        click.echo(
            f"No matching telemetry files found between {baseline_dir} and {current_dir}",
            err=True,
        )
        sys.exit(1)

    # Collect all metrics from all files
    all_chunk_metrics = {}  # chunk_size -> (baseline_metrics, current_metrics)

    for baseline_file, current_file in matches:
        try:
            # Load and compute metrics for both files
            _, baseline_metrics = _load_and_compute_metrics(baseline_file)
            _, current_metrics = _load_and_compute_metrics(current_file)

            # Store metrics for each chunk size
            for chunk_size in baseline_metrics.metrics_by_chunk_size:
                if chunk_size in current_metrics.metrics_by_chunk_size:
                    all_chunk_metrics[chunk_size] = (
                        baseline_metrics.metrics_by_chunk_size[chunk_size],
                        current_metrics.metrics_by_chunk_size[chunk_size],
                    )
        except Exception as e:
            click.echo(f"Error loading {baseline_file.name}: {e}", err=True)

    if not all_chunk_metrics:
        click.echo("No valid metrics found to compare", err=True)
        sys.exit(1)

    # Generate unified comparison table
    # First, reorganize the data into the format expected by the existing comparison functions
    from ragzoom.telemetry_analysis import SimplifiedMetrics

    # Create pseudo SimplifiedMetrics objects with all chunk sizes
    baseline_combined_dict = {}
    current_combined_dict = {}

    for chunk_size in sorted(all_chunk_metrics.keys()):
        baseline_metrics, current_metrics = all_chunk_metrics[chunk_size]
        baseline_combined_dict[chunk_size] = baseline_metrics
        current_combined_dict[chunk_size] = current_metrics

    baseline_combined = SimplifiedMetrics(metrics_by_chunk_size=baseline_combined_dict)
    current_combined = SimplifiedMetrics(metrics_by_chunk_size=current_combined_dict)

    # Use existing comparison formatting functions
    chunk_sizes = set(all_chunk_metrics.keys())

    # Check for regressions with dynamic thresholds
    # Detect CI from any of the files (use False as default)
    is_ci = False
    for _, _ in matches:
        # Could check environment from files, but for now default to False
        pass

    has_regression, thresholds_by_chunk = (
        _check_metrics_for_regressions_with_thresholds(
            baseline_combined, current_combined, chunk_sizes, is_ci
        )
    )

    # Format output with thresholds
    if output == "markdown":
        _format_markdown_comparison_with_thresholds(
            baseline_combined, current_combined, chunk_sizes, thresholds_by_chunk
        )
    else:
        _format_text_comparison_with_thresholds(
            baseline_combined, current_combined, chunk_sizes, thresholds_by_chunk
        )

    return has_regression


@cli.command()
@click.argument("baseline_path", type=click.Path(exists=True))
@click.argument("current_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Choice(["text", "markdown"]), default="text")
def compare(baseline_path: str, current_path: str, output: str) -> None:
    """Compare telemetry data between files or directories.

    Examples:
        Compare two files:
            ragzoom-telemetry compare baseline.json current.json

        Compare directories:
            ragzoom-telemetry compare baseline_results/ current_results/
    """
    baseline = Path(baseline_path)
    current = Path(current_path)

    has_regression = False

    # Check if both are directories or both are files
    if baseline.is_dir() and current.is_dir():
        has_regression = _compare_directories(baseline, current, output)
    elif baseline.is_file() and current.is_file():
        has_regression = _compare_files(baseline, current, output)
    else:
        click.echo(
            "Error: Both arguments must be either files or directories", err=True
        )
        sys.exit(1)

    # Exit with code 1 if regression detected
    if has_regression:
        click.echo("\n❌ Performance regression detected!", err=True)
        sys.exit(1)
    else:
        click.echo("\n✅ No regressions detected")


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


def _format_metrics_for_chunk_with_thresholds(
    chunk_label: str,
    base_metrics: dict,
    curr_metrics: dict,
    thresholds: dict[str, DynamicThreshold],
    output_format: str,
) -> None:
    """Format all metrics for a single chunk size with dynamic thresholds."""
    # Target-fit metrics - include chunk size in first row
    _format_comparison_row_with_threshold(
        chunk_label,
        "Median error",
        base_metrics["target_fit"]["median_error"],
        curr_metrics["target_fit"]["median_error"],
        thresholds["median_error"],
        output_format=output_format,
        signed=True,
        is_error_metric=True,
    )
    _format_comparison_row_with_threshold(
        "",
        "p95 error",
        base_metrics["target_fit"]["p95_error"],
        curr_metrics["target_fit"]["p95_error"],
        thresholds["p95_error"],
        output_format=output_format,
        signed=True,
        is_error_metric=True,
    )

    # Percent within ±10 tokens (now with dynamic threshold)
    _format_comparison_row_with_threshold(
        "",
        "Within ±10 tokens",
        base_metrics["target_fit"]["percent_within_10"],
        curr_metrics["target_fit"]["percent_within_10"],
        thresholds["percent_within_10"],
        output_format=output_format,
        higher_is_better=True,
    )

    # Retry metrics
    _format_comparison_row_with_threshold(
        "",
        "Avg retries/node",
        base_metrics["retries"]["retry_rate"],
        curr_metrics["retries"]["retry_rate"],
        thresholds["retry_rate"],
        output_format=output_format,
    )

    # Latency metrics
    _format_comparison_row_with_threshold(
        "",
        "Median time/node",
        base_metrics["latency"]["median_seconds"],
        curr_metrics["latency"]["median_seconds"],
        thresholds["latency"],
        output_format=output_format,
    )

    # Cost metrics
    _format_comparison_row_with_threshold(
        "",
        "USD per node",
        base_metrics["cost"]["usd_per_node"],
        curr_metrics["cost"]["usd_per_node"],
        thresholds["cost"],
        output_format=output_format,
        is_cost=True,
    )

    # Dispersion metrics
    _format_comparison_row_with_threshold(
        "",
        "MAD",
        base_metrics["dispersion"]["mad"],
        curr_metrics["dispersion"]["mad"],
        thresholds["mad"],
        output_format=output_format,
    )


def _format_text_comparison_with_thresholds(
    baseline: Any,
    current: Any,
    chunk_sizes: set[int],
    thresholds_by_chunk: dict[int, dict[str, DynamicThreshold]],
) -> None:
    """Format comparison as plain text table with dynamic thresholds."""

    # Build table header
    click.echo("\n" + "=" * 100)
    click.echo("Performance Comparison Report (with Dynamic Thresholds)")
    click.echo("=" * 100)

    # Show threshold configuration
    config = ThresholdConfig()
    click.echo("\nThreshold Configuration:")
    click.echo(f"  Between-run variance factor (k1): {config.k1_between_run}")
    click.echo(f"  Baseline uncertainty factor (k2): {config.k2_baseline_uncertainty}")
    click.echo(
        f"  Total multiplier: {config.k1_between_run + config.k2_baseline_uncertainty}x baseline variance"
    )

    # Table headers
    header = f"{'Chunk Size':<12} | {'Metric':<20} | {'Baseline':>12} | {'Current':>12} | {'Change':>20} | {'Threshold':>15}"
    click.echo("\n" + header)
    click.echo("-" * len(header))

    for chunk_size in sorted(chunk_sizes):
        base_metrics = baseline.metrics_by_chunk_size[chunk_size]
        curr_metrics = current.metrics_by_chunk_size[chunk_size]
        thresholds = thresholds_by_chunk[chunk_size]

        chunk_label = f"{chunk_size} tokens"
        _format_metrics_for_chunk_with_thresholds(
            chunk_label, base_metrics, curr_metrics, thresholds, "text"
        )

        # Add separator between chunk sizes (except for last one)
        if chunk_size != max(chunk_sizes):
            click.echo("-" * len(header))

    # Add footer with legend
    click.echo("\n" + "=" * 100)
    click.echo("\nLegend:")
    click.echo("  * = Threshold computed dynamically from baseline variance")
    click.echo("  ❌ = Regression detected (exceeds threshold)")
    click.echo("  ✅ = Meaningful improvement (>1σ baseline variance)")
    click.echo("  ⚠️ = Meaningful degradation but within threshold (>1σ but <5σ)")
    click.echo("  (no emoji) = Change within normal variance (<1σ)")


def _prepare_row_data(
    baseline: float,
    current: float,
    unit: str,
    signed: bool = False,
    higher_is_better: bool = False,
    is_cost: bool = False,
    is_integer: bool = False,
    regression_threshold: float | None = None,
    is_error_metric: bool = False,
    for_table: bool = False,
) -> tuple[str, str, str]:
    """Prepare formatted strings for a comparison row."""
    # For backward compatibility, create a fake metric name from unit
    metric_name = {
        "tokens": "median_error",
        "s": "median_seconds",
        "$": "cost",
        "%": "percent",
    }.get(unit, "unknown")

    base_str = _format_value(baseline, metric_name, is_cost, is_integer, signed)
    curr_str = _format_value(current, metric_name, is_cost, is_integer, signed)
    change_str = _calculate_change(
        baseline,
        current,
        higher_is_better,
        regression_threshold=regression_threshold,
        is_error_metric=is_error_metric,
        for_table=for_table,
    )
    return base_str, curr_str, change_str


def _format_comparison_row_with_threshold(
    category: str,
    metric: str,
    baseline: float,
    current: float,
    threshold: DynamicThreshold,
    output_format: str = "text",
    signed: bool = False,
    higher_is_better: bool = False,
    is_cost: bool = False,
    is_integer: bool = False,
    is_error_metric: bool = False,
) -> None:
    """Format a single row in the comparison table with dynamic threshold."""
    for_table = output_format == "markdown"

    # Format baseline and current values
    base_str = _format_value(
        baseline, threshold.metric_name, is_cost, is_integer, signed
    )
    curr_str = _format_value(
        current, threshold.metric_name, is_cost, is_integer, signed
    )

    # Calculate change with threshold
    change_str = _calculate_change_with_threshold(
        baseline, current, threshold, higher_is_better, is_error_metric, for_table
    )

    # Format threshold value
    unit = _get_unit_for_metric(threshold.metric_name)
    if unit == "$":
        threshold_str = f"±{unit}{threshold.absolute_value:.4f}"
    elif unit:
        threshold_str = f"±{threshold.absolute_value:.1f} {unit}"
    else:
        threshold_str = f"±{threshold.absolute_value:.2f}"

    # Add asterisk if computed dynamically
    if threshold.is_computed:
        threshold_str += "*"

    if output_format == "markdown":
        click.echo(
            f"| {category} | {metric} | {base_str} | {curr_str} | {change_str} | {threshold_str} |"
        )
    else:
        # Text format - category is empty for data rows
        click.echo(
            f"{category:<12} | {metric:<20} | {base_str:>12} | {curr_str:>12} | {change_str:<20} | {threshold_str:>15}"
        )


def _format_comparison_row(
    category: str,
    metric: str,
    baseline: float,
    current: float,
    unit: str,
    output_format: str = "text",
    signed: bool = False,
    higher_is_better: bool = False,
    is_cost: bool = False,
    is_integer: bool = False,
    regression_threshold: float | None = None,
    is_error_metric: bool = False,
) -> None:
    """Format a single row in the comparison table (text or markdown)."""
    for_table = output_format == "markdown"
    base_str, curr_str, change_str = _prepare_row_data(
        baseline,
        current,
        unit,
        signed,
        higher_is_better,
        is_cost,
        is_integer,
        regression_threshold,
        is_error_metric,
        for_table=for_table,
    )

    if output_format == "markdown":
        click.echo(
            f"| {category} | {metric} | {base_str} | {curr_str} | {change_str} |"
        )
    else:
        # Text format - category is empty for data rows
        click.echo(
            f"{category:<12} | {metric:<20} | {base_str:>12} | {curr_str:>12} | {change_str:>12}"
        )


def _format_value(
    value: float,
    metric_name: str,
    is_cost: bool = False,
    is_integer: bool = False,
    signed: bool = False,
) -> str:
    """Format a metric value with appropriate precision and units."""
    unit = _get_unit_for_metric(metric_name)

    if is_cost or unit == "$":
        formatted = f"${value:.4f}"
    elif is_integer:
        if signed:
            formatted = f"{value:+.0f}"
        else:
            formatted = f"{value:.0f}"
    elif signed:
        formatted = f"{value:+.1f}"
    else:
        formatted = f"{value:.2f}"

    if unit and unit != "$":
        formatted += f" {unit}"

    return formatted


def _calculate_change(
    baseline: float,
    current: float,
    higher_is_better: bool = False,
    regression_threshold: float | None = None,
    is_error_metric: bool = False,
    for_table: bool = False,
) -> str:
    """Legacy percentage-based change calculation for backward compatibility."""
    # For error metrics, we compare absolute values
    if is_error_metric:
        baseline_val = abs(baseline)
        current_val = abs(current)
    else:
        baseline_val = baseline
        current_val = current

    if baseline_val == 0:
        return "—" if for_table else "N/A"

    change_pct = ((current_val - baseline_val) / abs(baseline_val)) * 100

    # Determine if change is good or bad
    if higher_is_better:
        is_improvement = current_val > baseline_val
        is_regression = regression_threshold and change_pct < -regression_threshold
    else:
        is_improvement = current_val < baseline_val
        is_regression = regression_threshold and change_pct > regression_threshold

    # Add emoji for significant changes
    if is_regression:
        emoji = " ❌"  # Regression detected
    elif abs(change_pct) < 5:
        emoji = ""
    elif is_improvement:
        emoji = " ✅"
    else:
        emoji = " ⚠️"

    return f"{change_pct:+.1f}%{emoji}"


def _determine_significance_emoji(
    absolute_change: float,
    threshold: DynamicThreshold,
    higher_is_better: bool,
) -> str:
    """Determine emoji based on change significance.

    Args:
        absolute_change: The absolute change value
        threshold: Dynamic threshold with variance information
        higher_is_better: If True, higher values are better

    Returns:
        Emoji string: ❌ for regression, ✅ for improvement, ⚠️ for degradation, empty for normal variance
    """
    # Use 1-sigma (baseline variance) for meaningful changes, full threshold for regression
    significance_threshold = (
        threshold.baseline_variance
        if threshold.is_computed
        else threshold.absolute_value * 0.2  # 20% of threshold as fallback
    )

    # Check regression using full threshold
    if higher_is_better:
        is_regression = absolute_change < -threshold.absolute_value
        is_improvement = absolute_change > significance_threshold
        is_degradation = absolute_change < -significance_threshold
    else:
        is_regression = absolute_change > threshold.absolute_value
        is_improvement = absolute_change < -significance_threshold
        is_degradation = absolute_change > significance_threshold

    if is_regression:
        return " ❌"  # Regression detected
    elif is_improvement:
        return " ✅"  # Meaningful improvement
    elif is_degradation:
        return " ⚠️"  # Meaningful degradation (but not regression)
    else:
        return ""  # Change within normal variance


def _format_absolute_change(
    absolute_change: float,
    metric_name: str,
) -> str:
    """Format absolute change with appropriate units and precision.

    Args:
        absolute_change: The absolute change value
        metric_name: Name of the metric for unit lookup

    Returns:
        Formatted string with sign, value, and unit
    """
    unit = _get_unit_for_metric(metric_name)

    # Format based on unit type
    if unit == "$":
        abs_str = f"{unit}{abs(absolute_change):.4f}"
    elif unit == "%":
        # For percentage metrics, show as percentage points (pp)
        abs_str = f"{abs(absolute_change):.1f} pp"
    elif unit:
        abs_str = f"{abs(absolute_change):.1f} {unit}"
    else:
        abs_str = f"{abs(absolute_change):.2f}"

    # Add sign
    if absolute_change >= 0:
        return "+" + abs_str
    else:
        return "-" + abs_str


def _calculate_change_with_threshold(
    baseline: float,
    current: float,
    threshold: DynamicThreshold,
    higher_is_better: bool = False,
    is_error_metric: bool = False,
    for_table: bool = False,
) -> str:
    """Calculate and format the change between baseline and current values.

    Args:
        baseline: Baseline value
        current: Current value
        threshold: Dynamic threshold with variance information
        higher_is_better: If True, higher values are better
        is_error_metric: If True, compare absolute values (for error metrics)
        for_table: If True, use table-friendly formatting

    Returns:
        Formatted string showing absolute and percentage change with emoji
    """
    # For error metrics, we compare absolute values
    if is_error_metric:
        baseline_val = abs(baseline)
        current_val = abs(current)
        absolute_change = current_val - baseline_val
    else:
        baseline_val = baseline
        current_val = current
        absolute_change = current - baseline

    if baseline_val == 0:
        return "—" if for_table else "N/A"

    # Calculate percentage for display
    change_pct = (absolute_change / abs(baseline_val)) * 100

    # Determine significance and get emoji
    emoji = _determine_significance_emoji(absolute_change, threshold, higher_is_better)

    # Format absolute change
    abs_str = _format_absolute_change(absolute_change, threshold.metric_name)

    return f"{abs_str} ({change_pct:+.1f}%){emoji}"


def _get_unit_for_metric(metric_name: str) -> str:
    """Get the appropriate unit for a metric."""
    units = {
        "median_error": "tokens",
        "p95_error": "tokens",
        "median_seconds": "s",
        "p95_seconds": "s",
        "cost": "$",
        "usd_per_node": "$",
        "mad": "tokens",
        "retry_rate": "",  # Ratio, no unit
        "max_retries": "",  # Count, no unit
        "percent_within_10": "%",
        "percent": "%",  # For backward compatibility with _prepare_row_data
        "total_tokens": "tokens",
        "total_prompt_tokens": "tokens",
        "total_completion_tokens": "tokens",
    }
    return units.get(metric_name, "")


def _compare_metric(
    name: str,
    baseline: float,
    current: float,
    unit: str,
    signed: bool = False,
    higher_is_better: bool = False,
    is_cost: bool = False,
    is_integer: bool = False,
    regression_threshold: float | None = None,
) -> None:
    """Compare a single metric and display with appropriate formatting."""
    base_str = _format_value(baseline, unit, is_cost, is_integer, signed)
    curr_str = _format_value(current, unit, is_cost, is_integer, signed)

    # Remove unit from formatting since _format_value already adds it
    change_str = _calculate_change(
        baseline, current, higher_is_better, regression_threshold
    )

    # Display
    click.echo(f"  {name:15} {base_str:>12} → {curr_str:>12}  ({change_str})")


def _format_markdown_comparison_with_thresholds(
    baseline: Any,
    current: Any,
    chunk_sizes: set[int],
    thresholds_by_chunk: dict[int, dict[str, DynamicThreshold]],
) -> None:
    """Format comparison as markdown table with dynamic thresholds."""

    click.echo("# Performance Comparison Report (with Dynamic Thresholds)\n")

    # Show threshold configuration
    config = ThresholdConfig()
    click.echo("## Threshold Configuration")
    click.echo(f"- Between-run variance factor (k1): {config.k1_between_run}")
    click.echo(f"- Baseline uncertainty factor (k2): {config.k2_baseline_uncertainty}")
    click.echo(
        f"- Total multiplier: {config.k1_between_run + config.k2_baseline_uncertainty}x baseline variance"
    )
    click.echo("- \\* = dynamically computed from baseline variance\n")

    # Create unified table
    click.echo("| Chunk Size | Metric | Baseline | Current | Change | Threshold |")
    click.echo("|------------|--------|----------|---------|--------|-----------|")

    for chunk_size in sorted(chunk_sizes):
        base_metrics = baseline.metrics_by_chunk_size[chunk_size]
        curr_metrics = current.metrics_by_chunk_size[chunk_size]
        thresholds = thresholds_by_chunk[chunk_size]

        chunk_label = f"**{chunk_size} tokens**"
        _format_metrics_for_chunk_with_thresholds(
            chunk_label, base_metrics, curr_metrics, thresholds, "markdown"
        )

    click.echo("")


if __name__ == "__main__":
    cli()
