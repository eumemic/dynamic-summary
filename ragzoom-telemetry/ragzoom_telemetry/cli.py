"""CLI interface for RagZoom Telemetry."""

import click


@click.group()
def cli() -> None:
    """RagZoom Telemetry: Developer tools for analyzing telemetry data."""
    pass


# Commands will be imported from their respective modules
from ragzoom_telemetry.analyze import analyze
from ragzoom_telemetry.compare import compare
from ragzoom_telemetry.visualize import visualize

cli.add_command(analyze)
cli.add_command(compare)
cli.add_command(visualize)