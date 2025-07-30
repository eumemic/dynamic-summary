"""Telemetry analysis command."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry import (
    analyze_retry_patterns,
    compute_amplification_metrics,
    compute_batch_efficiency,
)


@click.command("analyze")
@click.argument("telemetry_file", type=click.Path(exists=True))
@click.option(
    "--output",
    type=click.Path(),
    help="Output file for analysis report (defaults to stdout)",
)
def analyze(telemetry_file: str, output: Optional[str]) -> None:
    """Analyze telemetry data from a benchmark file."""
    try:
        # Load telemetry data
        with open(telemetry_file) as f:
            data = json.load(f)

        if "telemetry" not in data:
            click.echo("❌ No telemetry data found in file", err=True)
            sys.exit(1)

        telemetry = data["telemetry"]

        # Create config for analysis
        config = RagZoomConfig()

        # Compute all metrics
        try:
            amplification = compute_amplification_metrics(telemetry, config)
            batch_efficiency = compute_batch_efficiency(telemetry)
            retry_patterns = analyze_retry_patterns(telemetry)
        except Exception as e:
            click.echo(f"❌ Error analyzing telemetry: {e}", err=True)
            sys.exit(1)

        # Format report
        report = []
        report.append("TELEMETRY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # Amplification metrics
        report.append("📈 Amplification Metrics:")
        report.append(
            f"  Median cost amplification: {amplification['median_cost']:.2f}x"
        )
        report.append(f"  90th percentile cost: {amplification['cost_p90']:.2f}x")
        report.append(f"  95th percentile cost: {amplification['cost_p95']:.2f}x")
        report.append(
            f"  Median input amplification: {amplification['median_input']:.2f}x"
        )
        report.append(
            f"  Median output amplification: {amplification['median_output']:.2f}x"
        )
        report.append("")

        # Batch efficiency
        report.append("📦 Batch Efficiency:")
        report.append(f"  Total batches: {batch_efficiency['total_batches']}")
        report.append(f"  Total embeddings: {batch_efficiency['total_embeddings']}")
        report.append(f"  Average batch size: {batch_efficiency['avg_batch_size']:.1f}")
        report.append(
            f"  Batch utilization: {batch_efficiency['batch_utilization']:.1f}%"
        )
        report.append("")

        # Retry patterns
        report.append("🔄 Retry Patterns:")
        report.append(f"  Total attempts: {retry_patterns['total_attempts']}")
        report.append(f"  Successful attempts: {retry_patterns['successful_attempts']}")
        report.append(f"  Retry rate: {retry_patterns['retry_rate']:.1f}%")
        report.append(
            f"  Retry success rate: {retry_patterns['retry_success_rate']:.1f}%"
        )

        if retry_patterns["rejection_reasons"]:
            report.append("  Rejection reasons:")
            for reason, count in sorted(
                retry_patterns["rejection_reasons"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                report.append(f"    - {reason}: {count}")

        report.append("")

        # Output report
        report_text = "\n".join(report)
        if output:
            Path(output).write_text(report_text)
            click.echo(f"✅ Analysis report saved to {output}")
        else:
            click.echo(report_text)

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)