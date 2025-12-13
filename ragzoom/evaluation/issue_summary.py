"""LLM-based synthesis of recurring issues across evaluations.

Uses ensemble approach: 10x parallel theme identification + synthesis for consistency.
"""

import asyncio
import json
from dataclasses import dataclass
from statistics import mean
from typing import cast

from ragzoom.contracts.chat_model import ChatModel, Message
from ragzoom.evaluation.types import DIMENSIONS, EvaluationReport, NodeEvaluation

# Type alias for parsed JSON structure
ParsedJSON = dict[str, list[dict[str, str | list[str]]]]

IDENTIFY_THEMES_PROMPT = """You are an expert at analyzing software quality metrics and identifying patterns.

You will be given evaluation results for multiple text summaries. Each evaluation includes scores on four dimensions (retention, isolation, faithfulness, continuity) from 1-5, with explanations for any defects found.

Your task is to identify the top 2-3 recurring systemic issue TYPES across all evaluations. Focus on patterns that appear multiple times, not one-off problems.

You MUST respond with valid JSON in this exact format:
{
  "themes": [
    {
      "name": "Short issue name",
      "description": "Brief description of the pattern (1-2 sentences max)"
    }
  ]
}

Rules:
- Focus on identifying categories of issues, not specific instances
- Be specific about what's going wrong - avoid vague labels like "summaries have problems"
- Output only the theme names and descriptions - do NOT include node IDs"""

SYNTHESIZE_THEMES_PROMPT = """You are an expert at analyzing software quality metrics and consolidating analysis results.

You will be given:
1. Multiple independent analyses of recurring issue themes (from 10 parallel runs)
2. The original defect data with node IDs

Your task is to:
1. Identify themes that appear consistently across multiple analyses (mentioned in 3+ analyses)
2. Merge similar themes with different names into a single canonical name
3. For each consolidated theme, assign the specific node IDs that exhibit that issue

You MUST respond with valid JSON in this exact format:
{
  "issues": [
    {
      "name": "Canonical issue name",
      "description": "Brief description (1-2 sentences max)",
      "node_ids": ["node-id-1", "node-id-2", ...]
    }
  ]
}

Rules:
- Only include themes that appeared in 3+ of the 10 analyses
- CRITICAL: Each node_id must appear in EXACTLY ONE issue. If a node has multiple problems, assign it to its PRIMARY issue category only. Do not duplicate nodes across issues.
- Use the exact node_ids from the defect data
- Only include issues that affect 2+ nodes"""


@dataclass(frozen=True)
class RecurringIssue:
    """A recurring issue identified across multiple nodes."""

    name: str
    description: str
    node_ids: tuple[str, ...]  # Sorted by score ascending (worst first)
    node_scores: tuple[float, ...]  # Parallel array of scores
    mean_score: float

    @property
    def node_count(self) -> int:
        return len(self.node_ids)


@dataclass(frozen=True)
class Theme:
    """A theme identified by one analysis run (no node assignments)."""

    name: str
    description: str


def _build_defects_prompt(report: EvaluationReport, include_node_ids: bool) -> str:
    """Build prompt with defect explanations.

    Args:
        report: The evaluation report
        include_node_ids: If True, include node IDs (for synthesis).
                         If False, omit them (for theme identification).
    """
    parts = ["Here are the defects found across all evaluated summaries:\n"]

    for evaluation in report.evaluations:
        node_defects: list[str] = []
        for dim in DIMENSIONS:
            score = getattr(evaluation, dim)
            if score.score < 5 and score.explanation:
                node_defects.append(f"  {dim}[{score.score}]: {score.explanation}")

        if node_defects:
            if include_node_ids:
                parts.append(f"Node: {evaluation.node_id}")
            parts.extend(node_defects)
            parts.append("")

    return "\n".join(parts)


def _get_node_scores(
    node_ids: list[str], evaluations: list[NodeEvaluation]
) -> dict[str, float]:
    """Get mean score for each node."""
    eval_by_id = {e.node_id: e.mean_score for e in evaluations}
    return {nid: eval_by_id.get(nid, 0.0) for nid in node_ids}


def _parse_json_response(content: str) -> ParsedJSON:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    return cast(ParsedJSON, json.loads(content))


async def _identify_themes(
    defects_prompt: str,
    chat_model: ChatModel,
) -> list[Theme]:
    """Identify recurring themes from defects (single run).

    Uses higher temperature for diversity across parallel runs.
    """
    messages: list[Message] = [
        {"role": "system", "content": IDENTIFY_THEMES_PROMPT},
        {"role": "user", "content": defects_prompt},
    ]

    result = await chat_model.complete(messages, temperature=0.7)
    content = result["content"]

    try:
        data = _parse_json_response(content)
    except json.JSONDecodeError:
        return []

    themes: list[Theme] = []
    for theme_data in data.get("themes", []):
        name = theme_data.get("name")
        description = theme_data.get("description")
        if isinstance(name, str) and isinstance(description, str):
            themes.append(Theme(name=name, description=description))

    return themes


async def _synthesize_themes(
    theme_lists: list[list[Theme]],
    defects_with_ids: str,
    valid_node_ids: set[str],
    evaluations: list[NodeEvaluation],
    chat_model: ChatModel,
) -> list[RecurringIssue]:
    """Synthesize multiple theme analyses into consolidated issues with node assignments."""
    # Format the theme analyses for the synthesis prompt
    analyses_text = []
    for i, themes in enumerate(theme_lists, 1):
        if themes:
            theme_strs = [f"  - {t.name}: {t.description}" for t in themes]
            analyses_text.append(f"Analysis {i}:\n" + "\n".join(theme_strs))
        else:
            analyses_text.append(f"Analysis {i}: (no themes identified)")

    user_prompt = f"""## Theme Analyses from 10 Independent Runs

{chr(10).join(analyses_text)}

## Original Defect Data with Node IDs

{defects_with_ids}

Identify recurring themes that appeared in 3+ analyses and assign node IDs to each."""

    messages: list[Message] = [
        {"role": "system", "content": SYNTHESIZE_THEMES_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = await chat_model.complete(messages, temperature=0.2)
    content = result["content"]

    try:
        data = _parse_json_response(content)
    except json.JSONDecodeError:
        return []

    # Track which nodes have been assigned to enforce one-issue-per-node
    assigned_nodes: set[str] = set()

    issues: list[RecurringIssue] = []
    for issue_data in data.get("issues", []):
        raw_node_ids = issue_data.get("node_ids", [])
        if not isinstance(raw_node_ids, list):
            continue
        # Filter to valid nodes that haven't been assigned yet
        node_ids = [
            nid
            for nid in raw_node_ids
            if nid in valid_node_ids and nid not in assigned_nodes
        ]
        if len(node_ids) < 2:
            continue

        name = issue_data.get("name", "Unknown")
        description = issue_data.get("description", "")
        if not isinstance(name, str):
            name = "Unknown"
        if not isinstance(description, str):
            description = ""

        # Get per-node scores and sort by score (worst first)
        scores_by_id = _get_node_scores(node_ids, evaluations)
        sorted_nodes = sorted(node_ids, key=lambda nid: scores_by_id[nid])
        sorted_scores = tuple(scores_by_id[nid] for nid in sorted_nodes)
        overall_mean = mean(sorted_scores) if sorted_scores else 0.0

        # Mark these nodes as assigned
        assigned_nodes.update(sorted_nodes)

        issues.append(
            RecurringIssue(
                name=name,
                description=description,
                node_ids=tuple(sorted_nodes),
                node_scores=sorted_scores,
                mean_score=overall_mean,
            )
        )

    # Sort by mean_score ascending (worst issues first)
    issues.sort(key=lambda i: i.mean_score)

    return issues


async def generate_issue_summary(
    report: EvaluationReport,
    chat_model: ChatModel,
    num_parallel: int = 10,
) -> list[RecurringIssue]:
    """Generate a synthesis of recurring issues from evaluation results.

    Uses ensemble approach:
    1. Run theme identification 10x in parallel (no node IDs, high temperature)
    2. Synthesize results to consolidate themes and assign nodes

    Args:
        report: The evaluation report containing all node evaluations
        chat_model: ChatModel instance for LLM calls
        num_parallel: Number of parallel theme identification runs (default: 10)

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

    # Build prompts
    defects_without_ids = _build_defects_prompt(report, include_node_ids=False)
    defects_with_ids = _build_defects_prompt(report, include_node_ids=True)

    # Stage 1: Run theme identification in parallel
    theme_tasks = [
        _identify_themes(defects_without_ids, chat_model) for _ in range(num_parallel)
    ]
    theme_lists = await asyncio.gather(*theme_tasks)

    # Stage 2: Synthesize themes and assign nodes
    valid_node_ids = {e.node_id for e in report.evaluations}
    issues = await _synthesize_themes(
        theme_lists,
        defects_with_ids,
        valid_node_ids,
        report.evaluations,
        chat_model,
    )

    return issues
