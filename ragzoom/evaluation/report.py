"""Report generation for summary evaluation results."""

import click

from ragzoom.evaluation.types import DIMENSIONS, EvaluationReport, NodeEvaluation


def _format_score_line(dim: str, mean: float, std: float) -> str:
    """Format a single dimension's score line."""
    dim_display = dim.capitalize().ljust(12)
    return f"  {dim_display} {mean:.2f} +/- {std:.2f}"


def _format_outlier(evaluation: NodeEvaluation) -> list[str]:
    """Format an outlier evaluation for display."""
    lines = []

    # Header line with node info
    header = (
        f"  Node {evaluation.node_id[:8]} "
        f"(height {evaluation.height}, pos {evaluation.position_fraction:.2f})"
    )

    # Find which dimensions have low scores
    low_scores = []
    for dim in DIMENSIONS:
        score = getattr(evaluation, dim)
        if score.score <= 2:
            low_scores.append(f"{dim.capitalize()}={score.score}")

    header += f": {', '.join(low_scores)}"
    lines.append(header)

    # Add explanations for low-scoring dimensions
    for dim in DIMENSIONS:
        score = getattr(evaluation, dim)
        if score.score <= 2:
            # Truncate long explanations
            explanation = score.explanation
            if len(explanation) > 100:
                explanation = explanation[:97] + "..."
            lines.append(f'    {dim}: "{explanation}"')

    return lines


def print_report(report: EvaluationReport, threshold: float) -> None:
    """Print formatted evaluation report to stdout.

    Args:
        report: The evaluation report to display
        threshold: Minimum mean score threshold for pass/fail
    """
    # Header
    click.echo()
    click.echo("=" * 55)
    click.echo("SUMMARY QUALITY REPORT")
    click.echo(f"Document: {report.document_id}")
    percentage = (
        (report.nodes_evaluated / report.total_inner_nodes * 100)
        if report.total_inner_nodes > 0
        else 0
    )
    click.echo(
        f"Nodes evaluated: {report.nodes_evaluated} of "
        f"{report.total_inner_nodes} ({percentage:.0f}%)"
    )
    click.echo("=" * 55)

    if not report.evaluations:
        click.echo()
        click.echo("No evaluations to report.")
        return

    # Aggregate scores
    click.echo()
    click.echo("AGGREGATE SCORES (mean +/- std)")
    means = report.mean_scores()
    stds = report.std_scores()
    for dim in DIMENSIONS:
        click.echo(_format_score_line(dim, means[dim], stds[dim]))

    # Outliers
    outliers = report.outliers(threshold=2)
    click.echo()
    if outliers:
        click.echo(f"OUTLIERS (score <= 2): {len(outliers)} found")
        for outlier in outliers[:5]:  # Show at most 5
            for line in _format_outlier(outlier):
                click.echo(line)
            click.echo()
        if len(outliers) > 5:
            click.echo(f"  ... and {len(outliers) - 5} more")
    else:
        click.echo("OUTLIERS (score <= 2): None")

    # Result
    click.echo()
    overall = report.overall_mean()
    passed = report.passed(threshold)
    status = "PASSED" if passed else "FAILED"
    comparison = ">=" if passed else "<"
    click.echo(
        f"RESULT: {status} (mean {overall:.2f} {comparison} threshold {threshold})"
    )
    click.echo()
