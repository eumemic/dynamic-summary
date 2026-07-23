"""JSON and Markdown report generation for LongMemEval benchmark results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from ragzoom.evaluation.benchmark_common import assemble_report_json, result_tail
from ragzoom.evaluation.longmemeval.types import (
    QUESTION_TYPE_NAMES,
    AggregateScores,
    AnswerResult,
    BenchmarkReport,
    HaystackMetrics,
    QuestionType,
)


def _scores_to_dict(scores: AggregateScores) -> dict[str, object]:
    """Serialize AggregateScores, converting QuestionType keys to their values."""
    result: dict[str, object] = {
        "by_type": {
            qtype.value: {
                "accuracy": (
                    round(cs.accuracy, 4) if cs.accuracy is not None else None
                ),
                "count": cs.count,
            }
            for qtype, cs in scores.by_type.items()
        },
    }
    if scores.overall_accuracy is not None:
        result["overall_accuracy"] = round(scores.overall_accuracy, 4)
    if scores.task_averaged_accuracy is not None:
        result["task_averaged_accuracy"] = round(scores.task_averaged_accuracy, 4)
    if scores.abstention_accuracy is not None:
        result["abstention_accuracy"] = round(scores.abstention_accuracy, 4)
    return result


def _haystack_metrics_to_dict(m: HaystackMetrics) -> dict[str, object]:
    """Serialize HaystackMetrics for JSON output."""
    return {
        "question_id": m.question_id,
        "num_sessions": m.num_sessions,
        "num_turns": m.num_turns,
        "indexing_duration_seconds": round(m.indexing_duration_seconds, 3),
    }


def _result_to_dict(r: AnswerResult) -> dict[str, object]:
    """Serialize a single AnswerResult for JSON output."""
    return {
        "question_id": r.question_id,
        "question": r.question,
        "gold_answer": r.gold_answer,
        "question_type": r.question_type.value,
        "is_abstention": r.is_abstention,
        "generated_answer": r.generated_answer,
        "verdict": r.judge_verdict,  # "yes" / "no" / None
        **result_tail(r.served_tilings, r.cost, r.retrospective),
    }


def save_json(report: BenchmarkReport, path: Path) -> None:
    """Save the full benchmark report as JSON."""
    metadata: dict[str, object] = {
        "answer_model": report.answer_model,
        "judge_model": report.judge_model,
        "dataset_variant": report.dataset_variant,
        "num_questions": report.num_questions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    data = assemble_report_json(
        metadata,
        report.config,
        _scores_to_dict(report.scores),
        [_result_to_dict(r) for r in report.per_question],
    )
    if report.haystack_metrics:
        data["haystack_metrics"] = [
            _haystack_metrics_to_dict(m) for m in report.haystack_metrics
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _format_pct(value: float) -> str:
    """Format a 0-1 float as a percentage string."""
    return f"{value * 100:.1f}%"


def _accuracy_table(report: BenchmarkReport) -> str:
    """Render per-type accuracy as a markdown table."""
    scores = report.scores
    qtypes: list[QuestionType] = sorted(scores.by_type, key=lambda q: q.value)

    headers = [QUESTION_TYPE_NAMES[q] for q in qtypes]
    header = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(qtypes)) + "|"
    cells = []
    for q in qtypes:
        cs = scores.by_type[q]
        cells.append(_format_pct(cs.accuracy) if cs.accuracy is not None else "—")
    row = "| " + " | ".join(cells) + " |"
    return "\n".join([header, sep, row])


def save_markdown(report: BenchmarkReport, path: Path) -> None:
    """Save a human-readable markdown summary of the benchmark results."""
    scores = report.scores
    lines: list[str] = []
    lines.append("# LongMemEval Benchmark Results")
    lines.append("")
    lines.append(f"- **Search model**: {report.answer_model}")
    lines.append(f"- **Judge model**: {report.judge_model}")
    lines.append(f"- **Dataset variant**: {report.dataset_variant}")
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

    if scores.overall_accuracy is not None:
        lines.append("## Accuracy")
        lines.append("")
        lines.append(f"- **Overall**: {_format_pct(scores.overall_accuracy)}")
        if scores.task_averaged_accuracy is not None:
            lines.append(
                f"- **Task-averaged**: {_format_pct(scores.task_averaged_accuracy)}"
            )
        if scores.abstention_accuracy is not None:
            lines.append(f"- **Abstention**: {_format_pct(scores.abstention_accuracy)}")
        lines.append("")
        lines.append("### By Question Type")
        lines.append("")
        lines.append(_accuracy_table(report))
        lines.append("")
    else:
        lines.append("_Run in --no-judge mode: no accuracy scores._")
        lines.append("")

    # Agent cost summary (only when cost data is present)
    costs = [r.cost for r in report.per_question]
    if costs:
        lines.append("## Agent Cost Summary")
        lines.append("")
        lines.append(
            f"- **Avg retrieval calls**: {mean(c.retrieval_call_count for c in costs):.1f}"
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

    if report.haystack_metrics:
        idx_durations = [m.indexing_duration_seconds for m in report.haystack_metrics]
        sessions = [m.num_sessions for m in report.haystack_metrics]
        lines.append("## Indexing")
        lines.append("")
        lines.append(f"- **Total ingest time**: {sum(idx_durations):.1f}s")
        lines.append(f"- **Avg sessions/haystack**: {mean(sessions):.0f}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
