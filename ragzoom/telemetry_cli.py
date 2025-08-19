"""Telemetry CLI with simplified metrics."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from ragzoom.telemetry_analysis import (
    ChunkMetrics,
    SimplifiedMetrics,
    compute_simplified_metrics,
)


# Metric name constants for consistency
class MetricNames:
    """Constants for metric names used throughout the telemetry system."""

    # Target-fit metrics
    MEDIAN_ERROR = "median_error"
    P95_ERROR = "p95_error"
    PERCENT_WITHIN_10 = "percent_within_10"

    # Latency metrics
    MEDIAN_SECONDS = "median_seconds"
    P95_SECONDS = "p95_seconds"

    # Cost metrics
    USD_PER_NODE = "usd_per_node"
    COST = "cost"

    # Retry metrics
    RETRY_RATE = "retry_rate"
    MAX_RETRIES = "max_retries"

    # Dispersion metrics
    MAD = "mad"

    # Token metrics
    TOTAL_TOKENS = "total_tokens"
    TOTAL_PROMPT_TOKENS = "total_prompt_tokens"
    TOTAL_COMPLETION_TOKENS = "total_completion_tokens"

    # Threshold keys (for consistency in dictionary access)
    MEDIAN_ERROR_KEY = "median_error"
    P95_ERROR_KEY = "p95_error"
    PERCENT_WITHIN_10_KEY = "percent_within_10"
    RETRY_RATE_KEY = "retry_rate"
    LATENCY_KEY = "latency"
    COST_KEY = "cost"
    MAD_KEY = "mad"


@dataclass
class DynamicThreshold:
    """Represents a threshold computed from baseline variance."""

    absolute_value: float | None  # None means no threshold enforcement
    baseline_variance: float
    k_factors: tuple[float, float]  # (k1_between_run, k2_baseline_uncertainty)
    metric_name: str
    is_computed: bool = True  # False if using static fallback
    emoji_significance_sigma: float = 1.0  # Sigma threshold for emoji display


@dataclass
class ThresholdConfig:
    """Configuration for dynamic threshold calculation.

    The threshold formula is: threshold = (k1 + k2) × baseline_variance

    Default values are based on statistical principles and empirical observations:
    - k1=3.0: Covers 99.7% of normal distribution (3-sigma rule) for between-run variance
    - k2=2.0: Additional margin for baseline measurement uncertainty from limited samples
    - Total 5-sigma: Ensures <0.01% false positive rate for regression detection
    - ci_multiplier=1.5: Based on empirical observation that CI environments show ~1.5x higher variance
    - emoji_significance_sigma=1.0: Changes beyond 1σ are considered statistically significant
      for emoji display (✅/⚠️), while regression (❌) still requires exceeding the full threshold

    These values mean a metric must exceed 5 standard deviations from the baseline's
    internal variance to be flagged as a regression, effectively eliminating false
    positives from natural LLM non-determinism while still catching real issues.
    """

    # K-factors for threshold calculation
    k1_between_run: float = 3.0  # Expected variance between runs (3-sigma)
    k2_baseline_uncertainty: float = 2.0  # Baseline uncertainty margin (2-sigma)

    # Whether to detect CI environment and adjust k-factors
    adjust_for_ci: bool = True
    ci_multiplier: float = 1.5  # Additional multiplier for CI environments

    # Emoji display configuration
    emoji_significance_sigma: float = 1.0  # Sigma threshold for showing ✅/⚠️ emojis


def get_change_emoji(
    absolute_change: float,
    higher_is_better: bool,
    threshold: DynamicThreshold | None = None,
) -> str:
    """Get emoji for metric change based on significance and direction.

    Args:
        absolute_change: Absolute change in the metric
        higher_is_better: If True, positive change is good
        threshold: Dynamic threshold with variance information

    Returns:
        Emoji indicating change significance and direction:
        - 🔴 = Regression detected (exceeds full threshold)
        - 🟡 = Significant undesirable change (>1σ in bad direction)
        - 🟢 = Significant improvement (>1σ in good direction)
        - ⚪ = Insignificant change (<1σ)
    """
    if threshold is None or threshold.baseline_variance == 0:
        # Fallback to simple direction-based logic if no threshold
        return "⚪"

    # Check for regression (exceeds full threshold)
    if threshold.absolute_value is not None:
        if higher_is_better:
            if absolute_change < -threshold.absolute_value:
                return "🔴"  # Regression: significant decrease when higher is better
        else:
            if absolute_change > threshold.absolute_value:
                return "🔴"  # Regression: significant increase when lower is better

    # Check for significance (>1σ baseline variance)
    significance_threshold = (
        threshold.baseline_variance * threshold.emoji_significance_sigma
    )

    if abs(absolute_change) < significance_threshold:
        return "⚪"  # Change within normal variance

    # Determine if change is desirable
    is_positive_change = absolute_change > 0
    is_desirable = (is_positive_change and higher_is_better) or (
        not is_positive_change and not higher_is_better
    )

    return "🟢" if is_desirable else "🟡"  # Significant change


def get_variance_emoji(
    variance_change: float,
    baseline_variance: float,
    significance_factor: float = 0.5,
) -> str:
    """Get emoji for variance change based on dynamic significance.

    Args:
        variance_change: Absolute change in variance/MAD
        baseline_variance: Baseline variance value
        significance_factor: Multiplier for baseline to determine significance (default 0.5 = 50% change)

    Returns:
        Emoji indicating variance change significance:
        - 🟡 = Significant variance increase (notable but not a regression)
        - 🟢 = Significant variance decrease (improved stability)
        - ⚪ = Insignificant variance change
    """
    if baseline_variance == 0:
        # Special case: any increase from zero variance is significant
        if variance_change > 0:
            return "🟡"  # Notable increase from perfect stability
        else:
            return "⚪"

    # Use relative change threshold (e.g., 50% of baseline variance)
    significance_threshold = baseline_variance * significance_factor

    if abs(variance_change) < significance_threshold:
        return "⚪"  # Change within normal fluctuation

    # Variance increase is notable (yellow), decrease is good (green)
    if variance_change > 0:
        return "🟡"  # Significant increase in variance (notable, not a regression)
    else:
        return "🟢"  # Significant decrease in variance (improved stability)


def compute_dynamic_threshold(
    baseline_metrics: ChunkMetrics,
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
    if metric_name in [
        MetricNames.MEDIAN_ERROR,
        MetricNames.P95_ERROR,
        MetricNames.PERCENT_WITHIN_10,
    ]:
        variance = getattr(baseline_metrics.target_fit, variance_key, 0.0)
    elif metric_name in [MetricNames.MEDIAN_SECONDS]:
        variance = getattr(baseline_metrics.latency, variance_key, 0.0)
    elif metric_name == MetricNames.MAD:
        variance = baseline_metrics.dispersion.mad
    elif metric_name == MetricNames.RETRY_RATE:
        variance = getattr(baseline_metrics.retries, variance_key, 0.0)
    elif metric_name in [MetricNames.USD_PER_NODE, MetricNames.COST]:
        variance = getattr(baseline_metrics.cost, variance_key, 0.0)
    else:
        # For metrics without variance data, don't enforce any threshold
        return DynamicThreshold(
            absolute_value=None,
            baseline_variance=0.0,
            k_factors=(0.0, 0.0),
            metric_name=metric_name,
            is_computed=False,
            emoji_significance_sigma=config.emoji_significance_sigma,
        )

    # If variance is zero, don't enforce any threshold
    # This handles cases like retry_rate when retries aren't implemented yet
    if variance == 0.0:
        return DynamicThreshold(
            absolute_value=None,  # No regression possible
            baseline_variance=0.0,
            k_factors=(config.k1_between_run, config.k2_baseline_uncertainty),
            metric_name=metric_name,
            is_computed=False,  # Not computed from variance
            emoji_significance_sigma=config.emoji_significance_sigma,
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

    return DynamicThreshold(
        absolute_value=threshold,
        baseline_variance=variance,
        k_factors=(k1, k2),
        metric_name=metric_name,
        is_computed=True,
        emoji_significance_sigma=config.emoji_significance_sigma,
    )


def _match_telemetry_files(
    dir1: Path, dir2: Path
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Match telemetry files between two directories by filename.

    Returns tuple of (indexing_matches, query_matches) where each is a list of (baseline_file, current_file) tuples.
    """
    # Find indexing telemetry files in both directories
    dir1_indexing = list(dir1.glob("telemetry_*_tokens.json"))
    dir2_indexing = list(dir2.glob("telemetry_*_tokens.json"))

    # Also support generic telemetry.json files for indexing
    dir1_indexing.extend(dir1.glob("telemetry.json"))
    dir2_indexing.extend(dir2.glob("telemetry.json"))

    # Find query telemetry files in both directories
    dir1_query = list(dir1.glob("query_telemetry_*.json"))
    dir2_query = list(dir2.glob("query_telemetry_*.json"))

    # Create mappings by filename for indexing files
    dir1_indexing_map = {f.name: f for f in dir1_indexing}
    dir2_indexing_map = {f.name: f for f in dir2_indexing}

    # Create mappings by filename for query files
    dir1_query_map = {f.name: f for f in dir1_query}
    dir2_query_map = {f.name: f for f in dir2_query}

    # Find matching pairs for indexing telemetry
    indexing_matches = []
    for filename, file1 in dir1_indexing_map.items():
        if filename in dir2_indexing_map:
            indexing_matches.append((file1, dir2_indexing_map[filename]))

    # Find matching pairs for query telemetry
    query_matches = []
    for filename, file1 in dir1_query_map.items():
        if filename in dir2_query_map:
            query_matches.append((file1, dir2_query_map[filename]))

    return (
        sorted(indexing_matches, key=lambda x: x[0].name),
        sorted(query_matches, key=lambda x: x[0].name),
    )


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

    # Handle standard format 4.2 telemetry data

    # Compute metrics
    metrics = compute_simplified_metrics(telemetry_data)

    # Display metrics for each chunk size
    for chunk_size in sorted(metrics.metrics_by_chunk_size.keys()):
        chunk_metrics = metrics.metrics_by_chunk_size[chunk_size]

        click.echo(f"\n{'='*60}")
        click.echo(f"  Chunk Size: {chunk_size} tokens")
        click.echo(f"{'='*60}")

        # Target-fit metrics
        target_fit = chunk_metrics.target_fit
        click.echo("\n📏 Target-fit Accuracy")
        click.echo(f"  Median error:        {target_fit.median_error:+.1f} tokens")
        click.echo(f"  p95 error:           {target_fit.p95_error:+.1f} tokens")
        click.echo(f"  Within ±10 tokens:   {target_fit.percent_within_10:.1f}%")
        click.echo(f"  Max overshoot:       {target_fit.max_overshoot:+.0f} tokens")
        click.echo(f"  Max undershoot:      {target_fit.max_undershoot:+.0f} tokens")

        # Retry metrics
        retries = chunk_metrics.retries
        click.echo("\n🔄 Retry Efficiency")
        click.echo(
            f"  Retry rate:          {retries.retry_rate:.2f} extra attempts/node"
        )
        click.echo(f"  Max retries:         {retries.max_retries:.0f}")

        # Latency metrics
        latency = chunk_metrics.latency
        click.echo("\n⏱️  Latency")
        click.echo(f"  Median time/node:    {latency.median_seconds:.2f}s")
        click.echo(f"  p95 time/node:       {latency.p95_seconds:.2f}s")
        click.echo(f"  Total indexing:      {latency.total_indexing_seconds:.1f}s")

        # Cost metrics
        cost = chunk_metrics.cost
        click.echo("\n💰 Cost & Tokens")
        click.echo(f"  Prompt tokens:       {cost.total_prompt_tokens:,}")
        click.echo(f"  Completion tokens:   {cost.total_completion_tokens:,}")
        click.echo(f"  Total tokens:        {cost.total_tokens:,}")
        click.echo(f"  USD per node:        ${cost.usd_per_node:.4f}")

        # Outlier detection message
        click.echo("\n💡 To analyze problematic summaries:")
        click.echo(
            "   python scripts/analyze-outlier-nodes.py --db benchmarks/latest/ragzoom.db"
        )

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

    # If no threshold is set, no regression is possible
    if threshold.absolute_value is None:
        return False, absolute_change

    if higher_is_better:
        # For metrics where higher is better, regression is a decrease beyond threshold
        is_regression = absolute_change < -threshold.absolute_value
    else:
        # For metrics where lower is better, regression is an increase beyond threshold
        is_regression = absolute_change > threshold.absolute_value

    return is_regression, absolute_change


def _load_and_compute_metrics(file_path: Path) -> tuple[dict[str, Any], Any]:
    """Load telemetry file and compute simplified metrics.

    Returns:
        Tuple of (full_data_with_config, simplified_metrics)
    """
    with open(file_path) as f:
        data = json.load(f)

    # The telemetry data can be either:
    # 1. Direct format: has 'nodes' at root level (standard telemetry file)
    # 2. Wrapped format: has 'telemetry' field containing the nodes
    if "telemetry" in data:
        # Wrapped format - extract telemetry but keep full data for config
        telemetry_data = data["telemetry"]
        full_data = data
    else:
        # Direct format - data is already the telemetry
        telemetry_data = data
        full_data = data

    # Compute metrics
    metrics = compute_simplified_metrics(telemetry_data)

    return full_data, metrics


def _compare_files(baseline_file: Path, current_file: Path) -> bool:
    """Compare two telemetry files.

    Returns:
        True if regression detected
    """
    # Check if this is query telemetry by looking for format_version or telemetry field
    with open(baseline_file) as f:
        baseline_data = json.load(f)

    # Detect query telemetry files (v1.0 or v1.1 format)
    format_version = baseline_data.get("format_version")
    if format_version in ["1.0", "1.1"] and (
        (
            "telemetry" in baseline_data
            and "timings" in baseline_data.get("telemetry", {})
        )
        or ("telemetries" in baseline_data)  # v1.1 format with multiple runs
    ):
        # This is query telemetry
        return _compare_query_telemetry_files(baseline_file, current_file)

    # Otherwise, it's indexing telemetry
    # Load and compute metrics for both files
    baseline_data, baseline_metrics = _load_and_compute_metrics(baseline_file)
    current_data, current_metrics = _load_and_compute_metrics(current_file)

    # Extract config if available
    baseline_config = baseline_data.get("config")
    current_config = current_data.get("config")

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
    _format_markdown_comparison_with_thresholds(
        baseline_metrics,
        current_metrics,
        common_sizes,
        thresholds_by_chunk,
        baseline_config,
        current_config,
    )

    return has_regression


def _compare_query_files_silent(
    baseline_file: Path, current_file: Path
) -> tuple[bool, dict]:
    """Compare two query telemetry files silently and return results.

    Returns:
        Tuple of (has_regression, report_dict)
    """
    from ragzoom.telemetry_analysis import compare_query_performance

    # Load query telemetry files
    with open(baseline_file) as f:
        baseline_data = json.load(f)
    with open(current_file) as f:
        current_data = json.load(f)

    # Use 50% regression threshold for query benchmarks due to API variance
    # (vs 20% for deterministic indexing operations)
    has_regression, report = compare_query_performance(
        baseline_data, current_data, regression_threshold=0.5
    )

    return has_regression, report


def _compare_query_telemetry_files(baseline_file: Path, current_file: Path) -> bool:
    """Compare two query telemetry files.

    Returns:
        True if regression detected
    """
    # Use silent comparison and output verbose results
    has_regression, report = _compare_query_files_silent(baseline_file, current_file)

    # Format output
    click.echo("\n### Query Performance Comparison\n")

    # Show summary
    summary = report["summary"]
    click.echo(f"**Baseline P50 Latency:** {summary['baseline_p50']:.3f}s")
    click.echo(f"**Current P50 Latency:** {summary['current_p50']:.3f}s")
    click.echo(f"**Query Count:** {summary['current_queries']} queries\n")

    # Show regressions if any
    if report["regressions"]:
        click.echo("#### ❌ Performance Regressions\n")
        click.echo("| Phase | Baseline | Current | Change |")
        click.echo("|-------|----------|---------|--------|")
        for reg in report["regressions"]:
            phase = reg["phase"]
            baseline = reg["baseline"]
            current = reg["current"]
            change = reg["change_percent"]
            click.echo(
                f"| {phase} | {baseline:.3f}s | {current:.3f}s | +{change:.1f}% |"
            )
        click.echo("")

    # Show improvements if any
    if report["improvements"]:
        click.echo("#### ✅ Performance Improvements\n")
        click.echo("| Phase | Baseline | Current | Change |")
        click.echo("|-------|----------|---------|--------|")
        for imp in report["improvements"]:
            phase = imp["phase"]
            baseline = imp["baseline"]
            current = imp["current"]
            change = imp["change_percent"]
            click.echo(
                f"| {phase} | {baseline:.3f}s | {current:.3f}s | {change:.1f}% |"
            )
        click.echo("")

    # Show phase breakdown
    current_metrics = report["current_metrics"]
    phase_breakdown = current_metrics["phase_breakdown"]

    click.echo("#### Phase Breakdown\n")
    click.echo("| Phase | Time (s) | % of Total |")
    click.echo("|-------|----------|------------|")

    total_time = phase_breakdown.get("total_time", 1.0)
    for phase, time_val in sorted(
        phase_breakdown.items(), key=lambda x: x[1], reverse=True
    ):
        if phase != "total_time" and time_val > 0:
            percentage = (time_val / total_time) * 100
            phase_name = phase.replace("_time", "").replace("_", " ").title()
            click.echo(f"| {phase_name} | {time_val:.3f} | {percentage:.1f}% |")

    click.echo("")

    # Show efficiency metrics
    click.echo("#### Efficiency Metrics\n")
    efficiency = current_metrics["efficiency"]
    click.echo(f"- **Seeds Utilization:** {efficiency['seeds_utilization']:.1%}")
    click.echo(f"- **Budget Utilization:** {efficiency['budget_utilization']:.1%}")
    click.echo(f"- **Coverage Efficiency:** {efficiency['coverage_efficiency']:.1%}")

    return has_regression


@dataclass
class MetricCheckConfig:
    """Configuration for checking a single metric."""

    metric_name: str
    variance_key: str
    metric_group: str
    metric_field: str
    threshold_key: str
    use_absolute: bool = False
    higher_is_better: bool = False


def _check_single_metric_regression(
    base_metrics: ChunkMetrics,
    curr_metrics: ChunkMetrics,
    metric_config: MetricCheckConfig,
    threshold_config: ThresholdConfig,
    is_ci: bool,
) -> tuple[bool, DynamicThreshold]:
    """Check regression for a single metric.

    Returns:
        Tuple of (is_regressed, threshold)
    """
    # Compute threshold
    threshold = compute_dynamic_threshold(
        base_metrics,
        metric_config.metric_name,
        metric_config.variance_key,
        threshold_config,
        is_ci,
    )

    # Get values
    base_group = getattr(base_metrics, metric_config.metric_group)
    curr_group = getattr(curr_metrics, metric_config.metric_group)
    base_val = getattr(base_group, metric_config.metric_field)
    curr_val = getattr(curr_group, metric_config.metric_field)

    # Apply absolute if needed
    if metric_config.use_absolute:
        base_val = abs(base_val)
        curr_val = abs(curr_val)

    # Check regression
    is_regressed, _ = check_regression_with_dynamic_threshold(
        base_val, curr_val, threshold, metric_config.higher_is_better
    )

    return is_regressed, threshold


def _check_metrics_for_regressions_with_thresholds(
    baseline: SimplifiedMetrics,
    current: SimplifiedMetrics,
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

    # Define metric configurations
    metric_configs = [
        MetricCheckConfig(
            MetricNames.MEDIAN_ERROR,
            "error_mad",
            "target_fit",
            "median_error",
            MetricNames.MEDIAN_ERROR_KEY,
            use_absolute=True,
        ),
        MetricCheckConfig(
            MetricNames.P95_ERROR,
            "error_mad",
            "target_fit",
            "p95_error",
            MetricNames.P95_ERROR_KEY,
            use_absolute=True,
        ),
        MetricCheckConfig(
            MetricNames.MEDIAN_SECONDS,
            "latency_mad",
            "latency",
            "median_seconds",
            MetricNames.LATENCY_KEY,
        ),
        MetricCheckConfig(
            MetricNames.MAD, "mad", "dispersion", "mad", MetricNames.MAD_KEY
        ),
        MetricCheckConfig(
            MetricNames.RETRY_RATE,
            "retry_mad",
            "retries",
            "retry_rate",
            MetricNames.RETRY_RATE_KEY,
        ),
        MetricCheckConfig(
            MetricNames.USD_PER_NODE,
            "cost_mad",
            "cost",
            "usd_per_node",
            MetricNames.COST_KEY,
        ),
        MetricCheckConfig(
            MetricNames.PERCENT_WITHIN_10,
            "percent_within_10_mad",
            "target_fit",
            "percent_within_10",
            MetricNames.PERCENT_WITHIN_10_KEY,
            higher_is_better=True,
        ),
    ]

    for chunk_size in chunk_sizes:
        base_metrics = baseline.metrics_by_chunk_size[chunk_size]
        curr_metrics = current.metrics_by_chunk_size[chunk_size]
        chunk_thresholds = {}

        # Check each metric
        for metric_config in metric_configs:
            is_regressed, threshold = _check_single_metric_regression(
                base_metrics, curr_metrics, metric_config, config, is_ci
            )
            chunk_thresholds[metric_config.threshold_key] = threshold
            if is_regressed:
                has_regression = True

        thresholds_by_chunk[chunk_size] = chunk_thresholds

    return has_regression, thresholds_by_chunk


def _compare_directories(baseline_dir: Path, current_dir: Path) -> bool:
    """Compare all matching telemetry files between two directories.

    Returns:
        True if any regression detected
    """
    # Find matching files for both indexing and query telemetry
    indexing_matches, query_matches = _match_telemetry_files(baseline_dir, current_dir)

    if not indexing_matches and not query_matches:
        click.echo(
            f"No matching telemetry files found between {baseline_dir} and {current_dir}",
            err=True,
        )
        sys.exit(1)

    has_regression = False

    # Process indexing telemetry files first
    if indexing_matches:
        # Collect all metrics from indexing files
        all_chunk_metrics = {}  # chunk_size -> (baseline_metrics, current_metrics)
        baseline_config = None
        current_config = None

        for baseline_file, current_file in indexing_matches:
            try:
                # Load and compute metrics for both files
                baseline_data, baseline_metrics = _load_and_compute_metrics(
                    baseline_file
                )
                current_data, current_metrics = _load_and_compute_metrics(current_file)

                # Extract config from first file (they should all be the same)
                if baseline_config is None and "config" in baseline_data:
                    baseline_config = baseline_data["config"]
                if current_config is None and "config" in current_data:
                    current_config = current_data["config"]

                # Store metrics for each chunk size
                for chunk_size in baseline_metrics.metrics_by_chunk_size:
                    if chunk_size in current_metrics.metrics_by_chunk_size:
                        all_chunk_metrics[chunk_size] = (
                            baseline_metrics.metrics_by_chunk_size[chunk_size],
                            current_metrics.metrics_by_chunk_size[chunk_size],
                        )
            except Exception as e:
                click.echo(f"Error loading {baseline_file.name}: {e}", err=True)

        if all_chunk_metrics:
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

            baseline_combined = SimplifiedMetrics(
                metrics_by_chunk_size=baseline_combined_dict
            )
            current_combined = SimplifiedMetrics(
                metrics_by_chunk_size=current_combined_dict
            )

            # Use existing comparison formatting functions
            chunk_sizes = set(all_chunk_metrics.keys())

            # Check for regressions with dynamic thresholds
            # Detect CI from any of the files (use False as default)
            is_ci = False
            for _, _ in indexing_matches:
                # Could check environment from files, but for now default to False
                pass

            indexing_has_regression, thresholds_by_chunk = (
                _check_metrics_for_regressions_with_thresholds(
                    baseline_combined, current_combined, chunk_sizes, is_ci
                )
            )

            # Format output with thresholds
            _format_markdown_comparison_with_thresholds(
                baseline_combined,
                current_combined,
                chunk_sizes,
                thresholds_by_chunk,
                baseline_config,
                current_config,
            )

            has_regression = has_regression or indexing_has_regression
        else:
            click.echo("### Indexing Performance")
            click.echo("⚠️ No indexing benchmark files found for comparison")

    # Process query telemetry files
    if query_matches:
        query_has_regression = _process_query_matches(query_matches)
        has_regression = has_regression or query_has_regression
    else:
        # Check if current directory has query files but baseline doesn't (new feature case)
        current_query_files = list(current_dir.glob("query_telemetry_*.json"))
        baseline_query_files = list(baseline_dir.glob("query_telemetry_*.json"))

        click.echo("\n### Query Performance")

        if current_query_files and not baseline_query_files:
            click.echo("📊 **New feature: Query benchmarks added**")
            click.echo(
                f"Found {len(current_query_files)} query benchmark files in current version."
            )
            click.echo(
                "Baseline comparison unavailable (query benchmarks are a new feature)."
            )

            # Show current query performance metrics without comparison
            for query_file in current_query_files[:3]:  # Show first 3 for brevity
                click.echo(f"\n**Configuration:** {query_file.name}")
                # Could add basic metrics display here if needed

        elif not current_query_files and not baseline_query_files:
            click.echo(
                "⚠️ No query benchmark files found in either baseline or current version"
            )
        else:
            click.echo("⚠️ No matching query benchmark files found for comparison")

    return has_regression


def _aggregate_phase_data(
    query_matches: list[tuple[Path, Path]],
) -> tuple[dict[str, float], dict[str, float], int, bool]:
    """Aggregate phase timing data across all query configurations.

    Returns:
        Tuple of (baseline_phases, current_phases, total_samples, has_any_regression)
    """
    from ragzoom.telemetry_analysis import analyze_query_telemetry

    # Known phases in query telemetry
    phases = [
        "embedding_time",
        "search_time",
        "mmr_time",
        "coverage_map_time",
        "scoring_time",
        "dp_time",
        "assembly_time",
        "total_time",
    ]

    baseline_all_phases: dict[str, list[float]] = {phase: [] for phase in phases}
    current_all_phases: dict[str, list[float]] = {phase: [] for phase in phases}
    total_samples = 0
    has_any_regression = False

    for baseline_file, current_file in query_matches:
        try:
            # Get comparison results
            query_has_regression, report = _compare_query_files_silent(
                baseline_file, current_file
            )
            has_any_regression = has_any_regression or query_has_regression

            # Load raw telemetry to extract all runs
            with open(baseline_file) as f:
                baseline_data = json.load(f)
            with open(current_file) as f:
                current_data = json.load(f)

            # Analyze telemetries to get phase breakdowns
            baseline_metrics = analyze_query_telemetry(baseline_data)
            current_metrics = analyze_query_telemetry(current_data)

            # Add phase data to aggregation
            for phase in phases:
                baseline_time = baseline_metrics.phase_breakdown.get(phase, 0.0)
                current_time = current_metrics.phase_breakdown.get(phase, 0.0)

                baseline_all_phases[phase].append(baseline_time)
                current_all_phases[phase].append(current_time)

            # Count samples based on telemetry format
            if baseline_data.get("format_version") == "1.1":
                total_samples += len(baseline_data.get("telemetries", []))
            else:
                total_samples += 1

        except Exception as e:
            click.echo(
                f"Warning: Could not process {baseline_file.name}: {e}", err=True
            )

    # Calculate averages across all configurations
    baseline_phases = {}
    current_phases = {}

    for phase in phases:
        baseline_phases[phase] = (
            sum(baseline_all_phases[phase]) / len(baseline_all_phases[phase])
            if baseline_all_phases[phase]
            else 0.0
        )
        current_phases[phase] = (
            sum(current_all_phases[phase]) / len(current_all_phases[phase])
            if current_all_phases[phase]
            else 0.0
        )

    return baseline_phases, current_phases, total_samples, has_any_regression


def _process_query_matches(query_matches: list[tuple[Path, Path]]) -> bool:
    """Process query telemetry file matches and output phase breakdown comparison.

    Returns:
        True if any regression detected
    """
    if not query_matches:
        return False

    # Aggregate phase data across all configurations
    baseline_phases, current_phases, total_samples, has_any_regression = (
        _aggregate_phase_data(query_matches)
    )

    if not baseline_phases:
        click.echo("\n### Query Performance\n❌ No valid telemetry data found")
        return False

    # Calculate configuration count
    config_count = len(query_matches)
    runs_per_config = total_samples // config_count if config_count > 0 else 0

    click.echo("\n### Query Performance Phase Breakdown")
    click.echo(
        f"Averaged across {config_count} configurations × {runs_per_config} runs each ({total_samples} samples total)"
    )

    # Build phase breakdown table
    click.echo("\n| Phase | Baseline | Current | Change | % of Total |")
    click.echo("|-------|----------|---------|--------|------------|")

    # Calculate total time for percentage calculation
    baseline_total = baseline_phases.get("total_time", 0.0)
    current_total = current_phases.get("total_time", 0.0)

    # Show individual phases (excluding total_time for now)
    phase_order = [
        ("Embedding", "embedding_time"),
        ("Search", "search_time"),
        ("MMR", "mmr_time"),
        ("Coverage Map", "coverage_map_time"),
        ("Scoring", "scoring_time"),
        ("DP Tiling", "dp_time"),
        ("Assembly", "assembly_time"),
    ]

    configs_with_regressions = []

    for display_name, phase_key in phase_order:
        baseline_time = baseline_phases.get(phase_key, 0.0)
        current_time = current_phases.get(phase_key, 0.0)

        # Calculate change percentage
        change_percent = (
            ((current_time - baseline_time) / baseline_time) * 100
            if baseline_time > 0
            else 0
        )

        # Calculate percentage of total time
        percent_of_total = (
            (current_time / current_total) * 100 if current_total > 0 else 0
        )

        # Check for significant regression in this phase
        if baseline_time > 0 and (current_time - baseline_time) / baseline_time > 0.5:
            configs_with_regressions.append((display_name, change_percent))

        click.echo(
            f"| {display_name} | {baseline_time:.3f}s | {current_time:.3f}s | "
            f"{change_percent:+.1f}% | {percent_of_total:.1f}% |"
        )

    # Show total row
    total_change_percent = (
        ((current_total - baseline_total) / baseline_total) * 100
        if baseline_total > 0
        else 0
    )

    click.echo(
        f"| **Total** | **{baseline_total:.3f}s** | **{current_total:.3f}s** | "
        f"**{total_change_percent:+.1f}%** | **100%** |"
    )

    # Show overall status and regression details
    if has_any_regression:
        click.echo(
            "\n❌ Performance regression detected (threshold: 50% for API variance)"
        )

        # Show which phases had regressions
        if configs_with_regressions:
            click.echo("\n**Regressed phases:**")
            for phase_name, change in configs_with_regressions:
                click.echo(f"  - {phase_name}: {change:+.1f}%")
    else:
        click.echo("\n✅ No regressions detected (threshold: 50% for API variance)")

    return has_any_regression


@cli.command()
@click.argument("baseline_path", type=click.Path(exists=True))
@click.argument("current_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path for visualization (PNG/PDF/SVG)",
)
def compare(baseline_path: str, current_path: str, output: str | None) -> None:
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
        has_regression = _compare_directories(baseline, current)
    elif baseline.is_file() and current.is_file():
        has_regression = _compare_files(baseline, current)
    else:
        click.echo(
            "Error: Both arguments must be either files or directories", err=True
        )
        sys.exit(1)

    # Generate visualization if output path specified
    if output:
        # Check dependencies
        _check_telemetry_deps()

        try:
            from ragzoom.telemetry_viz import TelemetryVisualizer

            # Determine format from extension or default to PNG
            output_path = Path(output)
            supported_formats = ["png", "pdf", "svg"]

            if output_path.suffix:
                format = output_path.suffix[1:].lower()
                if format not in supported_formats:
                    click.echo(
                        f"⚠️ Warning: Unsupported format '.{format}'. Using PNG instead.",
                        err=True,
                    )
                    format = "png"
                    output_path = output_path.with_suffix(".png")
            else:
                format = "png"
                output_path = output_path.with_suffix(".png")

            # Create visualizer and generate comparison
            visualizer = TelemetryVisualizer(output_path)

            if baseline.is_file() and current.is_file():
                visualizer.visualize_side_by_side(baseline, current, format)
                click.echo(f"\n✅ Generated comparison visualization: {output_path}")
            else:
                click.echo(
                    "⚠️ Warning: Visualization only supported for file comparisons, not directories",
                    err=True,
                )
        except Exception as e:
            click.echo(f"❌ Error generating visualization: {e}", err=True)

    # Exit with code 1 if regression detected
    if has_regression:
        click.echo("\n❌ Performance regression detected!", err=True)
        sys.exit(1)
    else:
        click.echo("\n✅ No regressions detected")


@cli.command("visualize")
@click.argument("input_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path (default: visualization.<format>)",
)
@click.option(
    "--format",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    help="Output format when -o is not specified or has no extension (default: png)",
)
def visualize(input_paths: tuple[str, ...], output: str | None, format: str) -> None:
    """Generate visualizations from one or two telemetry files.

    Examples:
        ragzoom-telemetry visualize baseline.json
        ragzoom-telemetry visualize baseline.json current.json
        ragzoom-telemetry visualize baseline.json -o analysis.png
        ragzoom-telemetry visualize baseline.json current.json -o comparison.pdf
    """
    # Check dependencies first
    _check_telemetry_deps()

    try:
        from ragzoom.telemetry_viz import TelemetryVisualizer

        # Determine output path and format
        supported_formats = ["png", "pdf", "svg"]

        if output:
            output_path = Path(output)
            # Infer format from extension if present
            if output_path.suffix:
                inferred_format = output_path.suffix[1:].lower()
                if inferred_format in supported_formats:
                    format = inferred_format
                else:
                    # Warn about unsupported format and use --format parameter
                    click.echo(
                        f"⚠️ Warning: Unsupported format '.{inferred_format}'. "
                        f"Using --format={format} instead.",
                        err=True,
                    )
                    # Replace the extension with the correct one
                    output_path = output_path.with_suffix(f".{format}")
            else:
                # No extension, add format
                output_path = output_path.with_suffix(f".{format}")
        else:
            # Default output path in current directory
            output_path = Path(f"visualization.{format}")

        # Create visualizer with output path
        visualizer = TelemetryVisualizer(output_path)

        if len(input_paths) == 1:
            # Single file visualization
            file_path = Path(input_paths[0])
            if not file_path.is_file():
                click.echo(f"❌ Error: {input_paths[0]} is not a file")
                sys.exit(1)
            visualizer.visualize_single_benchmark(file_path, format)
            click.echo(f"✅ Generated visualization: {output_path}")

        elif len(input_paths) == 2:
            # Side-by-side comparison
            file1 = Path(input_paths[0])
            file2 = Path(input_paths[1])

            if not file1.is_file() or not file2.is_file():
                click.echo("❌ Error: Both inputs must be files")
                sys.exit(1)

            visualizer.visualize_side_by_side(file1, file2, format)
            click.echo(f"✅ Generated side-by-side comparison: {output_path}")

        else:
            click.echo("❌ Error: Please provide 1 or 2 telemetry JSON files")
            click.echo(
                "  Usage: ragzoom-telemetry visualize <file1> [file2] [-o output.png]"
            )
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Error generating visualizations: {e}", err=True)
        sys.exit(1)


def _format_metrics_for_chunk_with_thresholds(
    chunk_label: str,
    base_metrics: ChunkMetrics,
    curr_metrics: ChunkMetrics,
    thresholds: dict[str, DynamicThreshold],
) -> None:
    """Format all metrics for a single chunk size with dynamic thresholds."""
    # Target-fit metrics - include chunk size in first row
    _format_comparison_row_with_threshold(
        chunk_label,
        "Median error",
        base_metrics.target_fit.median_error,
        curr_metrics.target_fit.median_error,
        thresholds[MetricNames.MEDIAN_ERROR_KEY],
        signed=True,
        is_error_metric=True,
        baseline_variance=base_metrics.target_fit.error_mad,
        current_variance=curr_metrics.target_fit.error_mad,
    )
    _format_comparison_row_with_threshold(
        "",
        "p95 error",
        base_metrics.target_fit.p95_error,
        curr_metrics.target_fit.p95_error,
        thresholds[MetricNames.P95_ERROR_KEY],
        signed=True,
        is_error_metric=True,
        baseline_variance=base_metrics.target_fit.error_mad,
        current_variance=curr_metrics.target_fit.error_mad,
    )

    # Percent within ±10 tokens (now with dynamic threshold)
    _format_comparison_row_with_threshold(
        "",
        "Within ±10 tokens",
        base_metrics.target_fit.percent_within_10,
        curr_metrics.target_fit.percent_within_10,
        thresholds[MetricNames.PERCENT_WITHIN_10_KEY],
        higher_is_better=True,
        baseline_variance=base_metrics.target_fit.percent_within_10_mad,
        current_variance=curr_metrics.target_fit.percent_within_10_mad,
    )

    # Retry metrics
    _format_comparison_row_with_threshold(
        "",
        "Avg retries/node",
        base_metrics.retries.retry_rate,
        curr_metrics.retries.retry_rate,
        thresholds[MetricNames.RETRY_RATE_KEY],
        baseline_variance=base_metrics.retries.retry_mad,
        current_variance=curr_metrics.retries.retry_mad,
    )

    # Latency metrics
    _format_comparison_row_with_threshold(
        "",
        "Median time/node",
        base_metrics.latency.median_seconds,
        curr_metrics.latency.median_seconds,
        thresholds[MetricNames.LATENCY_KEY],
        baseline_variance=base_metrics.latency.latency_mad,
        current_variance=curr_metrics.latency.latency_mad,
    )

    # Cost metrics
    _format_comparison_row_with_threshold(
        "",
        "USD per node",
        base_metrics.cost.usd_per_node,
        curr_metrics.cost.usd_per_node,
        thresholds[MetricNames.COST_KEY],
        is_cost=True,
        baseline_variance=base_metrics.cost.cost_mad,
        current_variance=curr_metrics.cost.cost_mad,
    )


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
    signed: bool = False,
    higher_is_better: bool = False,
    is_cost: bool = False,
    is_integer: bool = False,
    is_error_metric: bool = False,
    baseline_variance: float | None = None,
    current_variance: float | None = None,
) -> None:
    """Format a single row in the comparison table with dynamic threshold."""
    for_table = True

    # Format baseline and current values with variance
    base_str = _format_value(
        baseline, threshold.metric_name, is_cost, is_integer, signed, baseline_variance
    )
    curr_str = _format_value(
        current, threshold.metric_name, is_cost, is_integer, signed, current_variance
    )

    # Calculate change with threshold and variance
    change_str = _calculate_change_with_threshold(
        baseline,
        current,
        threshold,
        higher_is_better,
        is_error_metric,
        for_table,
        baseline_variance,
        current_variance,
    )

    # Format threshold value
    if threshold.absolute_value is None:
        threshold_str = "—"  # No threshold enforced
    else:
        unit = _get_unit_for_metric(threshold.metric_name)
        if unit == "$":
            threshold_str = f"±{unit}{threshold.absolute_value:.4f}"
        elif unit:
            threshold_str = f"±{threshold.absolute_value:.1f} {unit}"
        else:
            threshold_str = f"±{threshold.absolute_value:.2f}"

    # For markdown, replace newlines with <br> for proper rendering
    change_str_md = change_str.replace("\n", "<br>")
    click.echo(
        f"| {category} | {metric} | {base_str} | {curr_str} | {change_str_md} | {threshold_str} |"
    )


def _format_comparison_row(
    category: str,
    metric: str,
    baseline: float,
    current: float,
    unit: str,
    signed: bool = False,
    higher_is_better: bool = False,
    is_cost: bool = False,
    is_integer: bool = False,
    regression_threshold: float | None = None,
    is_error_metric: bool = False,
) -> None:
    """Format a single row in the comparison table (markdown only)."""
    for_table = True
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

    click.echo(f"| {category} | {metric} | {base_str} | {curr_str} | {change_str} |")


def _format_value(
    value: float,
    metric_name: str,
    is_cost: bool = False,
    is_integer: bool = False,
    signed: bool = False,
    variance: float | None = None,
) -> str:
    """Format a metric value with appropriate precision and units, optionally with variance.

    Args:
        value: The metric value
        metric_name: Name of the metric for unit lookup
        is_cost: Whether this is a cost metric
        is_integer: Whether to format as integer
        signed: Whether to show sign
        variance: Optional variance/MAD value to show as ±

    Returns:
        Formatted string like "50.0 ±2.0 tokens" or "$0.0010 ±0.0001"
    """
    unit = _get_unit_for_metric(metric_name)

    if is_cost or unit == "$":
        formatted = f"${value:.4f}"
        if variance is not None:
            formatted += f" ±{variance:.4f}"
    elif is_integer:
        if signed:
            formatted = f"{value:+.0f}"
        else:
            formatted = f"{value:.0f}"
        if variance is not None:
            formatted += f" ±{variance:.0f}"
    elif signed:
        formatted = f"{value:+.1f}"
        if variance is not None:
            formatted += f" ±{variance:.1f}"
    else:
        formatted = f"{value:.2f}"
        if variance is not None:
            formatted += f" ±{variance:.2f}"

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
    baseline_variance: float | None = None,
    current_variance: float | None = None,
) -> str:
    """Calculate and format the change between baseline and current values.

    Args:
        baseline: Baseline value
        current: Current value
        threshold: Dynamic threshold with variance information
        higher_is_better: If True, higher values are better
        is_error_metric: If True, compare absolute values (for error metrics)
        for_table: If True, use table-friendly formatting
        baseline_variance: Optional baseline variance/MAD value
        current_variance: Optional current variance/MAD value

    Returns:
        Formatted string showing absolute and percentage change with emojis
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

    # Get emoji based on significance and direction
    metric_emoji = get_change_emoji(absolute_change, higher_is_better, threshold)

    # Format absolute change
    abs_str = _format_absolute_change(absolute_change, threshold.metric_name)

    # Format first line: emoji + absolute + percentage (no extra significance emoji)
    line1 = f"{metric_emoji} {abs_str} ({change_pct:+.1f}%)"

    # Format variance change if both variances provided
    if baseline_variance is not None and current_variance is not None:
        variance_change = current_variance - baseline_variance

        # Get variance emoji based on dynamic significance
        variance_emoji = get_variance_emoji(variance_change, baseline_variance)

        # Calculate percentage for display
        if baseline_variance == 0:
            if current_variance > 0:
                variance_pct_str = " (+∞%)"
            else:
                variance_pct_str = " (±0%)"
        else:
            variance_change_pct = (variance_change / baseline_variance) * 100
            variance_pct_str = f" ({variance_change_pct:+.0f}%)"

        # Format variance absolute change based on metric type
        unit = _get_unit_for_metric(threshold.metric_name)
        if unit == "$":
            variance_abs_str = f"σ{variance_change:+.4f}"
        elif unit == "%":
            # For percentage metrics, variance is in percentage points
            variance_abs_str = f"σ{variance_change:+.1f}"
        elif unit:
            variance_abs_str = f"σ{variance_change:+.1f}"
        else:
            variance_abs_str = f"σ{variance_change:+.1f}"

        line2 = f"\n{variance_emoji} {variance_abs_str}{variance_pct_str}"
        return line1 + line2
    else:
        return line1


def _get_unit_for_metric(metric_name: str) -> str:
    """Get the appropriate unit for a metric."""
    units = {
        MetricNames.MEDIAN_ERROR: "tok",
        MetricNames.P95_ERROR: "tok",
        MetricNames.MEDIAN_SECONDS: "s",
        MetricNames.P95_SECONDS: "s",
        MetricNames.COST: "$",
        MetricNames.USD_PER_NODE: "$",
        MetricNames.MAD: "tok",
        MetricNames.RETRY_RATE: "",  # Ratio, no unit
        MetricNames.MAX_RETRIES: "",  # Count, no unit
        MetricNames.PERCENT_WITHIN_10: "%",
        "percent": "%",  # For backward compatibility with _prepare_row_data
        MetricNames.TOTAL_TOKENS: "tok",
        MetricNames.TOTAL_PROMPT_TOKENS: "tok",
        MetricNames.TOTAL_COMPLETION_TOKENS: "tok",
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
    baseline: SimplifiedMetrics,
    current: SimplifiedMetrics,
    chunk_sizes: set[int],
    thresholds_by_chunk: dict[int, dict[str, DynamicThreshold]],
    baseline_config: dict[str, Any] | None = None,
    current_config: dict[str, Any] | None = None,
) -> None:
    """Format comparison as markdown table with dynamic thresholds."""

    click.echo("# Performance Comparison Report\n")

    # Add configuration comparison if available
    if baseline_config is not None and current_config is not None:
        click.echo("## Configuration\n")
        click.echo("| Parameter | Baseline | Current |")
        click.echo("|-----------|----------|---------|")

        # Get all config keys from both configs
        all_keys = sorted(set(baseline_config.keys()) | set(current_config.keys()))

        for key in all_keys:
            baseline_val = baseline_config.get(key, "—")
            current_val = current_config.get(key, "—")

            # Format the key for display
            display_key = key.replace("_", " ").title()

            # Highlight differences
            if baseline_val != current_val:
                click.echo(
                    f"| **{display_key}** | {baseline_val} | **{current_val}** |"
                )
            else:
                click.echo(f"| {display_key} | {baseline_val} | {current_val} |")

        click.echo("\n## Performance Metrics\n")

    # Create unified table
    click.echo("| Chunk Size | Metric | Baseline | Current | Change | Threshold |")
    click.echo("|------------|--------|----------|---------|--------|-----------|")

    for chunk_size in sorted(chunk_sizes):
        base_metrics = baseline.metrics_by_chunk_size[chunk_size]
        curr_metrics = current.metrics_by_chunk_size[chunk_size]
        thresholds = thresholds_by_chunk[chunk_size]

        chunk_label = f"**{chunk_size} tokens**"
        _format_metrics_for_chunk_with_thresholds(
            chunk_label, base_metrics, curr_metrics, thresholds
        )

    click.echo("")


if __name__ == "__main__":
    cli()
