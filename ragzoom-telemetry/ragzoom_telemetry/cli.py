"""CLI interface for RagZoom Telemetry."""

import click

from ragzoom_telemetry.analyze import analyze
from ragzoom_telemetry.compare import compare
from ragzoom_telemetry.visualize import visualize


@click.group()
def cli() -> None:
    """RagZoom Telemetry: Developer tools for analyzing telemetry data."""
    pass


cli.add_command(analyze)
cli.add_command(compare)
cli.add_command(visualize)