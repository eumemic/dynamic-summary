"""JSON and Markdown report generation for LoCoMo benchmark results."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from ragzoom.evaluation.locomo.types import (
    AnswerResult,
    BenchmarkReport,
    BudgetPoint,
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


def _budget_point_to_dict(bp: BudgetPoint) -> dict[str, object]:
    """Serialize a BudgetPoint, converting QACategory keys to strings."""
    result: dict[str, object] = {
        "budget_tokens": bp.budget_tokens,
        "overall_f1": round(bp.overall_f1, 4),
        "by_category": {
            cat.name.lower(): asdict(score) for cat, score in bp.by_category.items()
        },
    }
    if bp.overall_accuracy is not None:
        result["overall_accuracy"] = round(bp.overall_accuracy, 4)
    return result


def _cost_to_dict(cost: CostMetrics) -> dict[str, object]:
    """Serialize CostMetrics for JSON output."""
    return {
        "total_input_tokens": cost.total_input_tokens,
        "total_output_tokens": cost.total_output_tokens,
        "retrieval_call_count": cost.retrieval_call_count,
        "reasoning_turn_count": cost.reasoning_turn_count,
        "retrieved_tokens_per_call": list(cost.retrieved_tokens_per_call),
    }


def _result_to_dict(r: AnswerResult) -> dict[str, object]:
    """Serialize a single AnswerResult for JSON output."""
    d: dict[str, object] = {
        "sample_id": r.sample_id,
        "question": r.question,
        "gold_answer": r.gold_answer,
        "category": r.category.name.lower(),
        "budget_tokens": r.budget_tokens,
        "retrieved_token_count": r.retrieved_token_count,
        "generated_answer": r.generated_answer,
        "verdict": r.judge_verdict,  # A/B/C
        "f1": round(r.token_f1, 4),
    }
    if r.cost is not None:
        d["cost"] = _cost_to_dict(r.cost)
    return d


def save_json(report: BenchmarkReport, path: Path) -> None:
    """Save the full benchmark report as JSON."""
    # Detect max_iterations from cost data
    costs = [r.cost for r in report.per_question if r.cost is not None]
    max_iterations = max((c.retrieval_call_count for c in costs), default=1)

    data: dict[str, object] = {
        "metadata": {
            "answer_model": report.answer_model,
            "judge_model": report.judge_model,
            "num_conversations": report.num_conversations,
            "num_questions": report.num_questions,
            "max_iterations": max_iterations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "budget_accuracy_curve": [
            _budget_point_to_dict(bp) for bp in report.budget_curve
        ],
        "per_question": [_result_to_dict(r) for r in report.per_question],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _format_pct(value: float) -> str:
    """Format a 0-1 float as a percentage string."""
    return f"{value * 100:.1f}%"


def _has_accuracy(report: BenchmarkReport) -> bool:
    """Check if any budget point has accuracy data (i.e., not f1-only mode)."""
    return any(bp.overall_accuracy is not None for bp in report.budget_curve)


def _budget_accuracy_table(report: BenchmarkReport) -> str:
    """Render the budget-accuracy curve as a markdown table."""
    # Collect all categories that appear
    all_cats: list[QACategory] = sorted(
        {cat for bp in report.budget_curve for cat in bp.by_category}
    )

    # Header
    cat_headers = [_CATEGORY_NAMES.get(c, c.name) for c in all_cats]
    header = "| Budget | Overall |" + " | ".join(cat_headers) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(all_cats))) + "|"

    rows = [header, sep]
    for bp in report.budget_curve:
        overall = (
            _format_pct(bp.overall_accuracy) if bp.overall_accuracy is not None else "—"
        )
        cat_cells = []
        for cat in all_cats:
            if cat in bp.by_category:
                cs = bp.by_category[cat]
                cat_cells.append(
                    _format_pct(cs.accuracy) if cs.accuracy is not None else "—"
                )
            else:
                cat_cells.append("—")

        row = f"| {bp.budget_tokens:,} | {overall} |" + " | ".join(cat_cells) + " |"
        rows.append(row)

    return "\n".join(rows)


def save_markdown(report: BenchmarkReport, path: Path) -> None:
    """Save a human-readable markdown summary of the benchmark results."""
    lines: list[str] = []
    lines.append("# LoCoMo Benchmark Results")
    lines.append("")
    lines.append(f"- **Answer model**: {report.answer_model}")
    lines.append(f"- **Judge model**: {report.judge_model}")
    lines.append(f"- **Conversations**: {report.num_conversations}")
    lines.append(f"- **Questions**: {report.num_questions} (excl. adversarial)")
    lines.append(
        f"- **Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append("")

    if _has_accuracy(report):
        lines.append("## Budget-Accuracy Curve (Judge Accuracy)")
        lines.append("")
        lines.append(_budget_accuracy_table(report))
        lines.append("")

    # F1 table
    lines.append("## Budget-F1 Curve (Token F1)")
    lines.append("")
    lines.append("| Budget | Overall F1 |")
    lines.append("|---|---|")
    for bp in report.budget_curve:
        lines.append(f"| {bp.budget_tokens:,} | {bp.overall_f1:.3f} |")
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
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
