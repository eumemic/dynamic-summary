"""JSON and Markdown report generation for Oolong benchmark results.

Mirrors the LongMemEval report, but the unit is a *continuous* partial-credit
score in [0, 1] (Oolong's deterministic metric), not a binary judge verdict —
so the tables report mean scores per type plus the paper's task-averaged score,
and every per-question row records the ``parsed_answer`` and ``served_tilings``
that make a low score attributable (parse failure vs synthesis vs summary-loss).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from ragzoom.evaluation.benchmark_common import assemble_report_json, result_tail
from ragzoom.evaluation.oolong.types import (
    QUESTION_TYPE_NAMES,
    AggregateScores,
    AnswerResult,
    BenchmarkReport,
    QuestionType,
    WindowMetrics,
)


def _scores_to_dict(scores: AggregateScores) -> dict[str, object]:
    """Serialize AggregateScores, converting QuestionType keys to their values."""
    result: dict[str, object] = {
        "by_type": {
            qtype.value: {
                "score": (round(cs.score, 4) if cs.score is not None else None),
                "count": cs.count,
            }
            for qtype, cs in scores.by_type.items()
        },
    }
    if scores.overall_score is not None:
        result["overall_score"] = round(scores.overall_score, 4)
    if scores.task_averaged_score is not None:
        result["task_averaged_score"] = round(scores.task_averaged_score, 4)
    return result


def _window_metrics_to_dict(m: WindowMetrics) -> dict[str, object]:
    """Serialize WindowMetrics for JSON output."""
    return {
        "context_window_id": m.context_window_id,
        "num_episodes": m.num_episodes,
        "indexing_duration_seconds": round(m.indexing_duration_seconds, 3),
    }


def _result_to_dict(r: AnswerResult) -> dict[str, object]:
    """Serialize a single AnswerResult for JSON output."""
    return {
        "question_id": r.question_id,
        "question": r.question,
        "gold_answer": r.gold_answer,
        "question_type": r.question_type.value,
        "generated_answer": r.generated_answer,
        # What was actually extracted from \boxed{} — lets a low score be
        # attributed to a parse failure vs a genuinely wrong answer.
        "parsed_answer": r.parsed_answer,
        "score": round(r.score, 4),
        **result_tail(r.served_tilings, r.cost, r.retrospective),
    }


def save_json(report: BenchmarkReport, path: Path) -> None:
    """Save the full benchmark report as JSON."""
    metadata: dict[str, object] = {
        "answer_model": report.answer_model,
        "config_name": report.config_name,
        "split": report.split,
        "num_questions": report.num_questions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    data = assemble_report_json(
        metadata,
        report.config,
        _scores_to_dict(report.scores),
        [_result_to_dict(r) for r in report.per_question],
    )
    if report.window_metrics:
        data["window_metrics"] = [
            _window_metrics_to_dict(m) for m in report.window_metrics
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _format_score(value: float) -> str:
    """Format a 0-1 score as a percentage string."""
    return f"{value * 100:.1f}%"


def _score_table(report: BenchmarkReport) -> str:
    """Render per-type mean score as a markdown table."""
    scores = report.scores
    qtypes: list[QuestionType] = sorted(scores.by_type, key=lambda q: q.value)

    headers = [QUESTION_TYPE_NAMES[q] for q in qtypes]
    header = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(qtypes)) + "|"
    cells = []
    for q in qtypes:
        cs = scores.by_type[q]
        cells.append(_format_score(cs.score) if cs.score is not None else "—")
    row = "| " + " | ".join(cells) + " |"
    return "\n".join([header, sep, row])


def save_markdown(report: BenchmarkReport, path: Path) -> None:
    """Save a human-readable markdown summary of the benchmark results."""
    scores = report.scores
    lines: list[str] = []
    lines.append("# Oolong Benchmark Results")
    lines.append("")
    lines.append(f"- **Answer model**: {report.answer_model}")
    lines.append(f"- **Config**: {report.config_name}")
    lines.append(f"- **Split**: {report.split}")
    if report.config:
        lines.append(f"- **Budget (B)**: {report.config.get('budget', '?')}")
        lines.append(f"- **Summary model**: {report.config.get('summary_model', '?')}")
        lines.append(
            f"- **Max iterations**: {report.config.get('max_iterations', '?')}"
        )
        sample = report.config.get("sample_size")
        if sample is not None:
            lines.append(f"- **Sample size**: {sample}")
    lines.append(f"- **Questions**: {report.num_questions}")
    lines.append(
        f"- **Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append("")

    if scores.overall_score is not None:
        lines.append("## Score")
        lines.append("")
        lines.append(f"- **Overall**: {_format_score(scores.overall_score)}")
        if scores.task_averaged_score is not None:
            lines.append(
                f"- **Task-averaged**: {_format_score(scores.task_averaged_score)}"
            )
        lines.append("")
        lines.append("### By Question Type")
        lines.append("")
        lines.append(_score_table(report))
        lines.append("")
    else:
        lines.append("_No questions scored._")
        lines.append("")

    # Agent cost summary (only when cost data is present)
    costs = [r.cost for r in report.per_question]
    if costs:
        lines.append("## Agent Cost Summary")
        lines.append("")
        lines.append(
            f"- **Avg recall calls**: {mean(c.retrieval_call_count for c in costs):.1f}"
        )
        lines.append(
            f"- **Avg input tokens**: {mean(c.total_input_tokens for c in costs):,.0f}"
        )
        lines.append(
            f"- **Avg output tokens**: {mean(c.total_output_tokens for c in costs):,.0f}"
        )
        cost_values = [c.total_cost_usd for c in costs if c.total_cost_usd is not None]
        if cost_values:
            lines.append(f"- **Avg cost per question**: ${mean(cost_values):.4f}")
            lines.append(f"- **Total query cost**: ${sum(cost_values):.4f}")
        lines.append("")

    if report.window_metrics:
        idx_durations = [m.indexing_duration_seconds for m in report.window_metrics]
        episodes = [m.num_episodes for m in report.window_metrics]
        lines.append("## Indexing")
        lines.append("")
        lines.append(f"- **Total ingest time**: {sum(idx_durations):.1f}s")
        lines.append(f"- **Avg episodes/window**: {mean(episodes):.0f}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
