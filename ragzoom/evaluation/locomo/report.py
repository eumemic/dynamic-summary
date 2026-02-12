"""JSON and Markdown report generation for LoCoMo benchmark results."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from ragzoom.evaluation.locomo.types import (
    AggregateScores,
    AnswerResult,
    BenchmarkReport,
    ConversationMetrics,
    CostMetrics,
    QACategory,
)

# Category display names for reports
_CATEGORY_NAMES: dict[QACategory, str] = {
    QACategory.SINGLE_HOP: "Single-hop",
    QACategory.MULTI_HOP: "Multi-hop",
    QACategory.TEMPORAL: "Temporal",
    QACategory.OPEN_DOMAIN: "Open-domain",
    QACategory.ADVERSARIAL: "Adversarial",
}


def _scores_to_dict(scores: AggregateScores) -> dict[str, object]:
    """Serialize AggregateScores, converting QACategory keys to strings."""
    result: dict[str, object] = {
        "overall_f1": round(scores.overall_f1, 4),
        "by_category": {
            cat.name.lower(): asdict(score) for cat, score in scores.by_category.items()
        },
    }
    if scores.overall_accuracy is not None:
        result["overall_accuracy"] = round(scores.overall_accuracy, 4)
    return result


def _cost_to_dict(cost: CostMetrics) -> dict[str, object]:
    """Serialize CostMetrics for JSON output."""
    d: dict[str, object] = {
        "total_input_tokens": cost.total_input_tokens,
        "total_output_tokens": cost.total_output_tokens,
        "retrieval_call_count": cost.retrieval_call_count,
        "reasoning_turn_count": cost.reasoning_turn_count,
        "retrieved_tokens_per_call": list(cost.retrieved_tokens_per_call),
    }
    if cost.query_duration_seconds is not None:
        d["query_duration_seconds"] = round(cost.query_duration_seconds, 3)
    if cost.total_cost_usd is not None:
        d["total_cost_usd"] = round(cost.total_cost_usd, 6)
    return d


def _conv_metrics_to_dict(m: ConversationMetrics) -> dict[str, object]:
    """Serialize ConversationMetrics for JSON output."""
    return {
        "sample_id": m.sample_id,
        "num_turns": m.num_turns,
        "num_sessions": m.num_sessions,
        "indexing_duration_seconds": round(m.indexing_duration_seconds, 3),
    }


def _result_to_dict(r: AnswerResult) -> dict[str, object]:
    """Serialize a single AnswerResult for JSON output."""
    d: dict[str, object] = {
        "sample_id": r.sample_id,
        "question": r.question,
        "gold_answer": r.gold_answer,
        "category": r.category.name.lower(),
        "generated_answer": r.generated_answer,
        "verdict": r.judge_verdict,  # A/B/C
        "f1": round(r.token_f1, 4),
    }
    d["cost"] = _cost_to_dict(r.cost)
    if r.retrospective is not None:
        d["retrospective"] = r.retrospective
    return d


def save_json(report: BenchmarkReport, path: Path) -> None:
    """Save the full benchmark report as JSON."""
    metadata: dict[str, object] = {
        "answer_model": report.answer_model,
        "judge_model": report.judge_model,
        "num_conversations": report.num_conversations,
        "num_questions": report.num_questions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if report.config is not None:
        metadata["config"] = report.config

    data: dict[str, object] = {
        "metadata": metadata,
        "scores": _scores_to_dict(report.scores),
        "per_question": [_result_to_dict(r) for r in report.per_question],
    }
    if report.conversation_metrics:
        data["conversation_metrics"] = [
            _conv_metrics_to_dict(m) for m in report.conversation_metrics
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _format_pct(value: float) -> str:
    """Format a 0-1 float as a percentage string."""
    return f"{value * 100:.1f}%"


def _has_accuracy(report: BenchmarkReport) -> bool:
    """Check if the report has accuracy data (i.e., not f1-only mode)."""
    return report.scores.overall_accuracy is not None


def _accuracy_table(report: BenchmarkReport) -> str:
    """Render category accuracy scores as a markdown table."""
    scores = report.scores
    all_cats: list[QACategory] = sorted(scores.by_category)

    cat_headers = [_CATEGORY_NAMES.get(c, c.name) for c in all_cats]
    header = "| Overall |" + " | ".join(cat_headers) + " |"
    sep = "|" + "|".join(["---"] * (1 + len(all_cats))) + "|"

    overall = (
        _format_pct(scores.overall_accuracy)
        if scores.overall_accuracy is not None
        else "—"
    )
    cat_cells = []
    for cat in all_cats:
        cs = scores.by_category[cat]
        cat_cells.append(_format_pct(cs.accuracy) if cs.accuracy is not None else "—")

    row = f"| {overall} |" + " | ".join(cat_cells) + " |"
    return "\n".join([header, sep, row])


def save_markdown(report: BenchmarkReport, path: Path) -> None:
    """Save a human-readable markdown summary of the benchmark results."""
    lines: list[str] = []
    lines.append("# LoCoMo Benchmark Results")
    lines.append("")
    lines.append(f"- **Search model**: {report.answer_model}")
    lines.append(f"- **Judge model**: {report.judge_model}")
    if report.config:
        lines.append(
            f"- **Max iterations**: {report.config.get('max_iterations', '?')}"
        )
        lines.append(f"- **Max budget**: {report.config.get('max_budget', '?')}")
        sample = report.config.get("sample_size")
        if sample is not None:
            lines.append(f"- **Sample size**: {sample}")
    lines.append(f"- **Conversations**: {report.num_conversations}")
    lines.append(f"- **Questions**: {report.num_questions} (excl. adversarial)")
    lines.append(
        f"- **Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append("")

    if _has_accuracy(report):
        lines.append("## Accuracy by Category (Judge)")
        lines.append("")
        lines.append(_accuracy_table(report))
        lines.append("")

    lines.append(f"**Overall F1**: {report.scores.overall_f1:.3f}")
    lines.append("")

    # Agent cost summary (only when cost data is present)
    costs = [r.cost for r in report.per_question if r.cost is not None]
    if costs:
        lines.append("## Agent Cost Summary")
        lines.append("")
        lines.append(
            f"- **Avg retrieval calls**: {mean(c.retrieval_call_count for c in costs):.1f}"
        )
        lines.append(
            f"- **Avg reasoning turns**: {mean(c.reasoning_turn_count for c in costs):.1f}"
        )
        lines.append(
            f"- **Avg input tokens**: {mean(c.total_input_tokens for c in costs):,.0f}"
        )
        lines.append(
            f"- **Avg output tokens**: {mean(c.total_output_tokens for c in costs):,.0f}"
        )
        total_retrieved = [sum(c.retrieved_tokens_per_call) for c in costs]
        lines.append(f"- **Avg retrieved tokens**: {mean(total_retrieved):,.0f}")
        cost_values = [c.total_cost_usd for c in costs if c.total_cost_usd is not None]
        if cost_values:
            lines.append(f"- **Avg cost per question**: ${mean(cost_values):.4f}")
            lines.append(f"- **Total cost**: ${sum(cost_values):.4f}")
        durations = [
            c.query_duration_seconds
            for c in costs
            if c.query_duration_seconds is not None
        ]
        if durations:
            lines.append(f"- **Avg query duration**: {mean(durations):.2f}s")
            lines.append(f"- **Min query duration**: {min(durations):.2f}s")
            lines.append(f"- **Max query duration**: {max(durations):.2f}s")
        lines.append("")

    if report.conversation_metrics:
        idx_durations = [
            m.indexing_duration_seconds for m in report.conversation_metrics
        ]
        lines.append("## Indexing Duration")
        lines.append("")
        lines.append(f"- **Total**: {sum(idx_durations):.1f}s")
        lines.append(f"- **Avg per conversation**: {mean(idx_durations):.1f}s")
        lines.append("")

    # Per-question retrospectives (only present with --profiling)
    retros = [
        (r.question, r.retrospective)
        for r in report.per_question
        if r.retrospective is not None
    ]
    if retros:
        lines.append("## Agent Retrospectives")
        lines.append("")
        for question, retro in retros:
            lines.append(f"**Q: {question}**")
            lines.append(f"> {retro}")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
