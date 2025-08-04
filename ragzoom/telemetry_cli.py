"""Telemetry CLI with simplified metrics."""

import json
import sys
from pathlib import Path
from typing import Any

import click

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry_analysis import compute_simplified_metrics


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
    if "telemetry" in telemetry_data and "format_version" not in telemetry_data:
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


def _check_for_regression(
    baseline_val: float,
    current_val: float,
    threshold_pct: float = 10.0,
    higher_is_better: bool = False,
) -> bool:
    """Check if a metric change indicates a regression.

    Args:
        baseline_val: Baseline metric value
        current_val: Current metric value
        threshold_pct: Percentage change threshold for regression
        higher_is_better: If True, decrease is regression; if False, increase is regression

    Returns:
        True if regression detected
    """
    if baseline_val == 0:
        return False

    change_pct = ((current_val - baseline_val) / abs(baseline_val)) * 100

    if higher_is_better:
        # For metrics where higher is better, regression is a decrease
        return change_pct < -threshold_pct
    else:
        # For metrics where lower is better, regression is an increase
        return change_pct > threshold_pct


def _compare_files(baseline_file: Path, current_file: Path, output: str) -> bool:
    """Compare two telemetry files.

    Returns:
        True if regression detected
    """
    # Load telemetry data
    with open(baseline_file) as f:
        baseline_data = json.load(f)

    with open(current_file) as f:
        current_data = json.load(f)

    # Handle wrapped telemetry format (with config/document/telemetry fields)
    if "telemetry" in baseline_data and "format_version" not in baseline_data:
        baseline_data = baseline_data["telemetry"]
    if "telemetry" in current_data and "format_version" not in current_data:
        current_data = current_data["telemetry"]

    # Compute metrics
    config = RagZoomConfig()
    baseline_metrics = compute_simplified_metrics(baseline_data, config)
    current_metrics = compute_simplified_metrics(current_data, config)

    # Find common chunk sizes
    baseline_sizes = set(baseline_metrics.metrics_by_chunk_size.keys())
    current_sizes = set(current_metrics.metrics_by_chunk_size.keys())
    common_sizes = baseline_sizes & current_sizes

    if not common_sizes:
        click.echo("No common chunk sizes found between files", err=True)
        return False  # Return False, don't exit here

    # Check for regressions
    has_regression = _check_metrics_for_regressions(
        baseline_metrics, current_metrics, common_sizes
    )

    # Format comparison
    if output == "markdown":
        _format_markdown_comparison(baseline_metrics, current_metrics, common_sizes)
    else:
        _format_text_comparison(baseline_metrics, current_metrics, common_sizes)

    return has_regression


def _check_metrics_for_regressions(
    baseline: Any, current: Any, chunk_sizes: set[int]
) -> bool:
    """Check if metrics show regressions.

    Returns:
        True if any regression detected
    """
    has_regression = False

    for chunk_size in chunk_sizes:
        base_metrics = baseline.metrics_by_chunk_size[chunk_size]
        curr_metrics = current.metrics_by_chunk_size[chunk_size]

        # Check target-fit regression (median error increase > 30%)
        if _check_for_regression(
            abs(base_metrics["target_fit"]["median_error"]),
            abs(curr_metrics["target_fit"]["median_error"]),
            threshold_pct=30.0,
        ):
            has_regression = True

        # Check retry rate regression (> 50% increase)
        if _check_for_regression(
            base_metrics["retries"]["retry_rate"],
            curr_metrics["retries"]["retry_rate"],
            threshold_pct=50.0,
        ):
            has_regression = True

        # Check latency regression (> 20% increase)
        if _check_for_regression(
            base_metrics["latency"]["median_seconds"],
            curr_metrics["latency"]["median_seconds"],
            threshold_pct=20.0,
        ):
            has_regression = True

        # Check cost regression (> 10% increase)
        if _check_for_regression(
            base_metrics["cost"]["usd_per_node"],
            curr_metrics["cost"]["usd_per_node"],
            threshold_pct=10.0,
        ):
            has_regression = True

    return has_regression


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

    click.echo(f"Found {len(matches)} matching file pairs to compare\n")

    # Process each file pair
    any_regression = False
    for baseline_file, current_file in matches:
        click.echo(f"\n{'='*70}")
        click.echo(f"Comparing {baseline_file.name}")
        click.echo(f"{'='*70}")

        try:
            has_regression = _compare_files(baseline_file, current_file, output)
            if has_regression:
                any_regression = True
                click.echo(f"⚠️  Regression detected in {baseline_file.name}", err=True)
        except Exception as e:
            click.echo(f"Error comparing {baseline_file.name}: {e}", err=True)

    return any_regression


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


def _format_text_comparison(baseline: Any, current: Any, chunk_sizes: set[int]) -> None:
    """Format comparison as plain text table with all chunk sizes."""

    # Build table header
    click.echo("\n" + "=" * 80)
    click.echo("Performance Comparison Report")
    click.echo("=" * 80)

    # Table headers
    header = f"{'Chunk Size':<12} | {'Metric':<20} | {'Baseline':>12} | {'Current':>12} | {'Change':>12}"
    click.echo("\n" + header)
    click.echo("-" * len(header))

    for chunk_size in sorted(chunk_sizes):
        base_metrics = baseline.metrics_by_chunk_size[chunk_size]
        curr_metrics = current.metrics_by_chunk_size[chunk_size]

        # Chunk size header row
        chunk_label = f"{chunk_size} tokens"
        click.echo(f"{chunk_label:<12} |{' '*21}|{' '*13}|{' '*13}|")

        # Target-fit metrics
        _format_table_row(
            "",
            "Median error",
            base_metrics["target_fit"]["median_error"],
            curr_metrics["target_fit"]["median_error"],
            "tokens",
            signed=True,
            regression_threshold=30.0,
        )
        _format_table_row(
            "",
            "p95 error",
            base_metrics["target_fit"]["p95_error"],
            curr_metrics["target_fit"]["p95_error"],
            "tokens",
            signed=True,
        )
        _format_table_row(
            "",
            "Within ±10 tokens",
            base_metrics["target_fit"]["percent_within_10"],
            curr_metrics["target_fit"]["percent_within_10"],
            "%",
            higher_is_better=True,
        )

        # Retry metrics
        _format_table_row(
            "",
            "Retry rate",
            base_metrics["retries"]["retry_rate"],
            curr_metrics["retries"]["retry_rate"],
            "",
            regression_threshold=50.0,
        )

        # Latency metrics
        _format_table_row(
            "",
            "Median time/node",
            base_metrics["latency"]["median_seconds"],
            curr_metrics["latency"]["median_seconds"],
            "s",
            regression_threshold=20.0,
        )

        # Cost metrics
        _format_table_row(
            "",
            "USD per node",
            base_metrics["cost"]["usd_per_node"],
            curr_metrics["cost"]["usd_per_node"],
            "",
            is_cost=True,
            regression_threshold=10.0,
        )

        # Dispersion metrics
        _format_table_row(
            "",
            "MAD",
            base_metrics["dispersion"]["mad"],
            curr_metrics["dispersion"]["mad"],
            "tokens",
        )

        # Add separator between chunk sizes (except for last one)
        if chunk_size != max(chunk_sizes):
            click.echo("-" * len(header))


def _format_table_row(
    chunk_size_str: str,
    metric: str,
    baseline: float,
    current: float,
    unit: str,
    signed: bool = False,
    higher_is_better: bool = False,
    is_cost: bool = False,
    is_integer: bool = False,
    regression_threshold: float | None = None,
) -> None:
    """Format a single row in the comparison table."""
    base_str = _format_value(baseline, unit, is_cost, is_integer, signed)
    curr_str = _format_value(current, unit, is_cost, is_integer, signed)
    change_str = _calculate_change(
        baseline, current, higher_is_better, regression_threshold
    )

    # Format the row - chunk_size_str is empty for data rows
    click.echo(
        f"{chunk_size_str:<12} | {metric:<20} | {base_str:>12} | {curr_str:>12} | {change_str:>12}"
    )


def _format_value(
    value: float,
    unit: str,
    is_cost: bool = False,
    is_integer: bool = False,
    signed: bool = False,
) -> str:
    """Format a metric value with appropriate precision and units."""
    if is_cost:
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

    if unit and not is_cost:
        formatted += f" {unit}"

    return formatted


def _calculate_change(
    baseline: float,
    current: float,
    higher_is_better: bool = False,
    regression_threshold: float | None = None,
    for_table: bool = False,
) -> str:
    """Calculate and format the change between baseline and current values."""
    if baseline == 0:
        return "—" if for_table else "N/A"

    change_pct = ((current - baseline) / abs(baseline)) * 100

    # Determine if change is good or bad
    if higher_is_better:
        is_improvement = current > baseline
        is_regression = regression_threshold and change_pct < -regression_threshold
    else:
        is_improvement = current < baseline
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


def _format_markdown_comparison(
    baseline: Any, current: Any, chunk_sizes: set[int]
) -> None:
    """Format comparison as markdown table with all chunk sizes."""

    click.echo("# Performance Comparison Report\n")

    # Create unified table
    click.echo("| Chunk Size | Metric | Baseline | Current | Change |")
    click.echo("|------------|--------|----------|---------|--------|")

    for chunk_size in sorted(chunk_sizes):
        base_metrics = baseline.metrics_by_chunk_size[chunk_size]
        curr_metrics = current.metrics_by_chunk_size[chunk_size]

        # Add chunk size header row
        _add_table_row(
            f"**{chunk_size} tokens**",
            "",
            0,
            0,
            "",
            skip_values=True,
        )

        # Target-fit metrics
        _add_table_row(
            "",
            "Median error",
            base_metrics["target_fit"]["median_error"],
            curr_metrics["target_fit"]["median_error"],
            "tokens",
            signed=True,
        )
        _add_table_row(
            "",
            "p95 error",
            base_metrics["target_fit"]["p95_error"],
            curr_metrics["target_fit"]["p95_error"],
            "tokens",
            signed=True,
        )
        _add_table_row(
            "",
            "Within ±10 tokens",
            base_metrics["target_fit"]["percent_within_10"],
            curr_metrics["target_fit"]["percent_within_10"],
            "%",
            higher_is_better=True,
        )

        # Retry metrics
        _add_table_row(
            "",
            "Retry rate",
            base_metrics["retries"]["retry_rate"],
            curr_metrics["retries"]["retry_rate"],
            "",
        )

        # Latency metrics
        _add_table_row(
            "",
            "Median time/node",
            base_metrics["latency"]["median_seconds"],
            curr_metrics["latency"]["median_seconds"],
            "s",
        )

        # Cost metrics
        _add_table_row(
            "",
            "USD per node",
            base_metrics["cost"]["usd_per_node"],
            curr_metrics["cost"]["usd_per_node"],
            "",
            is_cost=True,
        )

        # Dispersion metrics
        _add_table_row(
            "",
            "MAD",
            base_metrics["dispersion"]["mad"],
            curr_metrics["dispersion"]["mad"],
            "tokens",
        )

    click.echo("")


def _add_table_row(
    category: str,
    metric: str,
    baseline: float,
    current: float,
    unit: str,
    signed: bool = False,
    higher_is_better: bool = False,
    is_cost: bool = False,
    is_integer: bool = False,
    skip_values: bool = False,
) -> None:
    """Add a row to the markdown table."""
    if skip_values:
        # This is a header row for a chunk size
        click.echo(f"| {category} | | | | |")
    else:
        base_str = _format_value(baseline, unit, is_cost, is_integer, signed)
        curr_str = _format_value(current, unit, is_cost, is_integer, signed)
        change_str = _calculate_change(
            baseline, current, higher_is_better, for_table=True
        )

        click.echo(
            f"| {category} | {metric} | {base_str} | {curr_str} | {change_str} |"
        )


if __name__ == "__main__":
    cli()
