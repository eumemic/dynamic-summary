"""LLM-based synthesis of recurring issues across evaluations."""

import json
from dataclasses import dataclass
from statistics import mean

from ragzoom.contracts.chat_model import ChatModel, Message
from ragzoom.evaluation.types import DIMENSIONS, EvaluationReport, NodeEvaluation

SYSTEM_PROMPT = """You are an expert at analyzing software quality metrics and identifying patterns.

You will be given evaluation results for multiple text summaries. Each evaluation includes a node_id and scores on four dimensions (retention, isolation, faithfulness, continuity) from 1-5, with explanations for any defects found.

Your task is to identify the top 2-3 recurring systemic issues across all evaluations. Focus on patterns that appear multiple times, not one-off problems.

You MUST respond with valid JSON in this exact format:
{
  "issues": [
    {
      "name": "Short issue name",
      "description": "Brief description of the pattern (1-2 sentences max)",
      "node_ids": ["node-id-1", "node-id-2", ...]
    }
  ]
}

Rules:
- Only include issues that affect 2+ nodes
- Each node_id should appear in at most one issue (pick the most relevant)
- Be specific about what's going wrong - not "summaries have problems" but "preceding context bleeding into summaries"
- Use the exact node_ids from the input"""


@dataclass(frozen=True)
class RecurringIssue:
    """A recurring issue identified across multiple nodes."""

    name: str
    description: str
    node_ids: tuple[str, ...]
    mean_score: float

    @property
    def node_count(self) -> int:
        return len(self.node_ids)


def _build_user_prompt(report: EvaluationReport) -> str:
    """Build the user prompt with all defect explanations including node IDs."""
    parts = ["Here are the defects found across all evaluated summaries:\n"]

    # Collect defects by node, including node_id
    for evaluation in report.evaluations:
        node_defects: list[str] = []
        for dim in DIMENSIONS:
            score = getattr(evaluation, dim)
            if score.score < 5 and score.explanation:
                node_defects.append(f"  {dim}[{score.score}]: {score.explanation}")

        if node_defects:
            parts.append(f"Node: {evaluation.node_id}")
            parts.extend(node_defects)
            parts.append("")

    parts.append(
        "Identify the top 2-3 recurring systemic issues. "
        "For each issue, list ALL node_ids that exhibit it."
    )

    return "\n".join(parts)


def _compute_mean_score(
    node_ids: list[str], evaluations: list[NodeEvaluation]
) -> float:
    """Compute mean score across all dimensions for the given nodes."""
    node_id_set = set(node_ids)
    scores: list[float] = []
    for e in evaluations:
        if e.node_id in node_id_set:
            scores.append(e.mean_score)
    return mean(scores) if scores else 0.0


async def generate_issue_summary(
    report: EvaluationReport,
    chat_model: ChatModel,
) -> list[RecurringIssue]:
    """Generate a synthesis of recurring issues from evaluation results.

    Args:
        report: The evaluation report containing all node evaluations
        chat_model: ChatModel instance for LLM calls

    Returns:
        List of RecurringIssue objects with node mappings and scores
    """
    if not report.evaluations:
        return []

    # Check if there are any defects to summarize
    has_defects = any(
        getattr(e, dim).score < 5 and getattr(e, dim).explanation
        for e in report.evaluations
        for dim in DIMENSIONS
    )

    if not has_defects:
        return []

    user_prompt = _build_user_prompt(report)

    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = await chat_model.complete(messages, temperature=0.3)
    content = result["content"]

    # Parse JSON response
    # Handle potential markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: return empty if parsing fails
        return []

    # Build valid node_id set for validation
    valid_node_ids = {e.node_id for e in report.evaluations}

    issues: list[RecurringIssue] = []
    for issue_data in data.get("issues", []):
        # Filter to only valid node_ids
        node_ids = [
            nid for nid in issue_data.get("node_ids", []) if nid in valid_node_ids
        ]
        if len(node_ids) < 2:
            continue  # Skip issues with fewer than 2 nodes

        mean_score = _compute_mean_score(node_ids, report.evaluations)
        issues.append(
            RecurringIssue(
                name=issue_data.get("name", "Unknown"),
                description=issue_data.get("description", ""),
                node_ids=tuple(node_ids),
                mean_score=mean_score,
            )
        )

    # Sort by mean_score ascending (worst issues first)
    issues.sort(key=lambda i: i.mean_score)

    return issues
