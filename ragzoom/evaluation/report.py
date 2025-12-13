"""Report generation for summary evaluation results."""

import click

from ragzoom.evaluation.issue_summary import RecurringIssue
from ragzoom.evaluation.types import DIMENSIONS, EvaluationReport


def _format_histograms_side_by_side(
    all_scores: dict[str, list[int]], height: int = 8
) -> list[str]:
    """Format vertical ASCII histograms for all dimensions side by side."""
    dims = list(all_scores.keys())
    all_counts = {
        dim: [scores.count(i) for i in range(1, 6)]
        for dim, scores in all_scores.items()
    }
    max_counts = {
        dim: max(counts) if counts else 1 for dim, counts in all_counts.items()
    }

    lines = []

    # Dimension labels centered over each histogram (width 20 each, spacing 4)
    labels = [f"{dim.capitalize():^20}" for dim in dims]
    lines.append("    ".join(labels))
    lines.append("")

    # Build histogram rows
    for row in range(height, 0, -1):
        row_parts = []
        for dim in dims:
            counts = all_counts[dim]
            max_count = max_counts[dim]
            threshold = (row / height) * max_count
            cells = []
            for count in counts:
                if count >= threshold and count > 0:
                    cells.append(" ██ ")
                else:
                    cells.append("    ")
            row_parts.append("".join(cells))
        lines.append("    ".join(row_parts))

    # X-axis separators
    separator = "─" * 20
    lines.append("    ".join([separator] * len(dims)))

    # X-axis labels (1-5)
    axis_labels = "  1   2   3   4   5 "
    lines.append("    ".join([axis_labels] * len(dims)))

    # Counts
    count_parts = []
    for dim in dims:
        count_strs = [f"{c:^4}" for c in all_counts[dim]]
        count_parts.append("".join(count_strs))
    lines.append("    ".join(count_parts))

    return lines


def _format_score_line(dim: str, mean: float, std: float, p5: float, p10: float) -> str:
    """Format a single dimension's score line with percentiles."""
    dim_display = dim.capitalize().ljust(12)
    return f"  {dim_display} {mean:.2f} +/- {std:.2f}    p5={p5:.1f}  p10={p10:.1f}"


def _format_issue(issue: RecurringIssue) -> list[str]:
    """Format a recurring issue for display."""
    lines = []
    # Header with name, score, and count
    lines.append(
        f"  {issue.name} (score: {issue.mean_score:.1f}, {issue.node_count} nodes)"
    )
    # Description
    if issue.description:
        lines.append(f"    {issue.description}")
    # Node IDs (truncate if too many)
    if issue.node_count <= 10:
        node_list = ", ".join(issue.node_ids)
    else:
        node_list = ", ".join(issue.node_ids[:10]) + f", ... (+{issue.node_count - 10})"
    lines.append(f"    Nodes: {node_list}")
    return lines


def print_report(
    report: EvaluationReport,
    threshold: float,
    issues: list[RecurringIssue] | None = None,
) -> None:
    """Print formatted evaluation report to stdout.

    Args:
        report: The evaluation report to display
        threshold: Minimum mean score threshold for pass/fail
        issues: Optional list of recurring issues with node mappings
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

    # Aggregate scores with percentiles
    click.echo()
    click.echo("AGGREGATE SCORES (mean +/- std, percentiles)")
    means = report.mean_scores()
    stds = report.std_scores()
    p5s = report.percentile_scores(5)
    p10s = report.percentile_scores(10)
    for dim in DIMENSIONS:
        click.echo(_format_score_line(dim, means[dim], stds[dim], p5s[dim], p10s[dim]))

    # Failure count
    failure_count = report.failure_count(threshold=2.5)
    failure_pct = (
        (failure_count / len(report.evaluations) * 100) if report.evaluations else 0
    )
    click.echo()
    click.echo(
        f"FAILURES: {failure_count} nodes ({failure_pct:.1f}%) have any dimension < 2.5"
    )

    # Histograms for all dimensions side by side
    click.echo()
    click.echo("SCORE DISTRIBUTIONS")
    all_scores = {
        dim: [getattr(e, dim).score for e in report.evaluations] for dim in DIMENSIONS
    }
    for line in _format_histograms_side_by_side(all_scores):
        click.echo(f"  {line}")

    # Recurring issues (if provided)
    if issues:
        click.echo()
        click.echo("RECURRING ISSUES")
        for issue in issues:
            for line in _format_issue(issue):
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
