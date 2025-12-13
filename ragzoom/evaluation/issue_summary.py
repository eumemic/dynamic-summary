"""LLM-based synthesis of recurring issues across evaluations."""

from ragzoom.contracts.chat_model import ChatModel, Message
from ragzoom.evaluation.types import DIMENSIONS, EvaluationReport

SYSTEM_PROMPT = """You are an expert at analyzing software quality metrics and identifying patterns.

You will be given evaluation results for multiple text summaries. Each evaluation scores the summary on four dimensions (retention, isolation, faithfulness, continuity) from 1-5, with explanations for any defects found.

Your task is to identify the top 2-3 recurring systemic issues across all evaluations. Focus on patterns that appear multiple times, not one-off problems.

Output format - a concise bulleted list:
- **Issue name**: Brief description of the pattern (1-2 sentences max)

Be specific about what's going wrong. Don't just say "summaries have problems" - say exactly what the pattern is (e.g., "preceding context bleeding into summaries as if events occurred in current section")."""


def _build_user_prompt(report: EvaluationReport) -> str:
    """Build the user prompt with all defect explanations."""
    # Collect all non-empty explanations grouped by dimension
    defects_by_dim: dict[str, list[str]] = {dim: [] for dim in DIMENSIONS}

    for evaluation in report.evaluations:
        for dim in DIMENSIONS:
            score = getattr(evaluation, dim)
            if score.score < 5 and score.explanation:
                defects_by_dim[dim].append(f"[score={score.score}] {score.explanation}")

    parts = ["Here are the defects found across all evaluated summaries:\n"]

    for dim in DIMENSIONS:
        defects = defects_by_dim[dim]
        if defects:
            parts.append(f"## {dim.upper()} ({len(defects)} defects)")
            for defect in defects[
                :20
            ]:  # Limit to 20 per dimension to avoid token overflow
                parts.append(f"- {defect}")
            if len(defects) > 20:
                parts.append(f"  ... and {len(defects) - 20} more")
            parts.append("")

    parts.append(
        "Identify the top 2-3 recurring systemic issues across these evaluations."
    )

    return "\n".join(parts)


async def generate_issue_summary(
    report: EvaluationReport,
    chat_model: ChatModel,
) -> str:
    """Generate a synthesis of recurring issues from evaluation results.

    Args:
        report: The evaluation report containing all node evaluations
        chat_model: ChatModel instance for LLM calls

    Returns:
        A bulleted summary of the top recurring issues
    """
    if not report.evaluations:
        return "No evaluations to analyze."

    # Check if there are any defects to summarize
    has_defects = any(
        getattr(e, dim).score < 5 and getattr(e, dim).explanation
        for e in report.evaluations
        for dim in DIMENSIONS
    )

    if not has_defects:
        return "No defects found - all summaries scored perfectly."

    user_prompt = _build_user_prompt(report)

    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = await chat_model.complete(messages, temperature=0.3)
    return result["content"]
