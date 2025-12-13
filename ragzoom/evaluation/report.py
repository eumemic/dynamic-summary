"""Report generation for summary evaluation results."""

import click

from ragzoom.evaluation.types import DIMENSIONS, EvaluationReport, NodeEvaluation


def _format_score_line(dim: str, mean: float, std: float) -> str:
    """Format a single dimension's score line."""
    dim_display = dim.capitalize().ljust(12)
    return f"  {dim_display} {mean:.2f} +/- {std:.2f}"


def _format_evaluation(evaluation: NodeEvaluation) -> list[str]:
    """Format an evaluation for display."""
    lines = []

    # Header line with node info and all scores
    scores_str = ", ".join(
        f"{dim[0].upper()}={getattr(evaluation, dim).score}" for dim in DIMENSIONS
    )
    coord = f"({evaluation.height}, {evaluation.level_index})"
    header = f"  Node {coord} @ {evaluation.span_start}: {scores_str}"
    lines.append(header)

    # Show explanation for lowest-scoring dimension
    min_dim = min(DIMENSIONS, key=lambda d: getattr(evaluation, d).score)
    score = getattr(evaluation, min_dim)
    explanation = score.explanation
    if len(explanation) > 100:
        explanation = explanation[:97] + "..."
    lines.append(f'    {min_dim}: "{explanation}"')

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

    # Lowest-scoring nodes (show bottom 5 by min score)
    sorted_evals = sorted(report.evaluations, key=lambda e: (e.min_score, e.mean_score))
    bottom_5 = sorted_evals[:5]
    click.echo()
    click.echo("LOWEST SCORES")
    for evaluation in bottom_5:
        for line in _format_evaluation(evaluation):
            click.echo(line)
        click.echo()

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
