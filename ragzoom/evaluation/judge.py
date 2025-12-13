"""LLM-based judge for evaluating summary quality."""

import asyncio
import json
import logging

from ragzoom.contracts.chat_model import ChatModel, Message
from ragzoom.evaluation.types import DimensionScore, NodeEvaluation
from ragzoom.exceptions import LLMError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of text summaries.

You will be given:
1. A SUMMARY to evaluate
2. The LEFT CHILD and RIGHT CHILD texts that were combined and summarized
3. Optionally, PRECEDING CONTEXT (the summary of content that came before)

The summary was created by compressing the children's text to roughly half its length. Information loss is expected and acceptable - your job is to assess whether the RIGHT information was kept.

Evaluate the summary on these four dimensions:

## RETENTION (1-5)
Does the summary retain the most important information from the children?
- Consider: key events, important details, essential context
- Remember: ~50% compression means things MUST be cut - focus on whether the RIGHT things were kept
- 5: Excellent prioritization of important content
- 3: Acceptable - captures main points but misses some important details
- 1: Critical information is missing

## ISOLATION (1-5)
Does the summary avoid importing facts from the preceding context as if they occurred in this section?
- The summary MAY use pronouns or references that rely on context (e.g., "he" instead of "Bilbo")
- The summary MAY allude to previous events to aid flow
- The summary MUST NOT present events/facts from the preceding context as if they happened in the current section
- 5: Clean separation - uses context appropriately for flow without importing facts
- 3: Minor bleed - some context details appear but don't seriously mislead
- 1: Major bleed - facts from context presented as current events

## FAITHFULNESS (1-5)
Does the summary avoid hallucination and knowledge contamination?
- All facts in the summary should be traceable to the children or context
- The summary MUST NOT include information the model knows from training data but isn't in the inputs
- Example: If summarizing The Hobbit, don't hint at plot points not yet revealed in the children
- 5: Completely faithful to inputs
- 3: Minor inaccuracies or slight embellishments
- 1: Contains fabricated facts or spoilers from external knowledge

## CONTINUITY (1-5)
Does the summary flow smoothly from the preceding context?
- If no preceding context: Does it read as a coherent opening?
- If preceding context exists: Does it continue naturally without jarring transitions?
- 5: Seamless flow
- 3: Readable but with some awkward transitions
- 1: Disjointed or incoherent

Respond with a JSON object containing your evaluation:
{
  "retention": {"score": <1-5>, "explanation": "<defects only>"},
  "isolation": {"score": <1-5>, "explanation": "<defects only>"},
  "faithfulness": {"score": <1-5>, "explanation": "<defects only>"},
  "continuity": {"score": <1-5>, "explanation": "<defects only>"}
}

IMPORTANT: The explanation field should ONLY describe defects - reasons points were deducted.
- Do NOT mention what went right or praise good aspects
- If the score is 5 (perfect), the explanation MUST be an empty string ""
- For scores 1-4, briefly explain what specific issues caused the deduction"""


def _build_user_prompt(
    summary: str,
    left_text: str,
    right_text: str,
    preceding_context: str | None,
) -> str:
    """Build the user prompt with the texts to evaluate."""
    parts = []

    if preceding_context:
        parts.append(f"## PRECEDING CONTEXT\n{preceding_context}")

    parts.append(f"## LEFT CHILD\n{left_text}")
    parts.append(f"## RIGHT CHILD\n{right_text}")
    parts.append(f"## SUMMARY TO EVALUATE\n{summary}")

    return "\n\n".join(parts)


def _parse_response(response_text: str) -> dict[str, DimensionScore]:
    """Parse LLM JSON response into DimensionScore objects."""
    data = json.loads(response_text)

    result: dict[str, DimensionScore] = {}
    for dimension in ("retention", "isolation", "faithfulness", "continuity"):
        dim_data = data[dimension]
        result[dimension] = DimensionScore(
            score=int(dim_data["score"]),
            explanation=str(dim_data["explanation"]),
        )

    return result


async def evaluate_node(
    *,
    summary: str,
    left_text: str,
    right_text: str,
    preceding_context: str | None,
    chat_model: ChatModel,
) -> dict[str, DimensionScore]:
    """Evaluate a single node's summary on all four dimensions.

    Args:
        summary: The summary text to evaluate
        left_text: Text of the left child node
        right_text: Text of the right child node
        preceding_context: Text of the preceding neighbor (or None)
        chat_model: ChatModel instance for LLM calls

    Returns:
        Dict mapping dimension names to DimensionScore objects

    Raises:
        LLMError: If the API call fails or response is invalid
    """
    user_prompt = _build_user_prompt(
        summary=summary,
        left_text=left_text,
        right_text=right_text,
        preceding_context=preceding_context,
    )

    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        result = await chat_model.complete(messages, temperature=0.1, json_mode=True)
        return _parse_response(result["content"])

    except json.JSONDecodeError as e:
        raise LLMError(
            operation="evaluate_node",
            model=chat_model.model_id,
            message=f"Failed to parse JSON response: {e}",
        ) from e
    except KeyError as e:
        raise LLMError(
            operation="evaluate_node",
            model=chat_model.model_id,
            message=f"Missing required field in response: {e}",
        ) from e
    except Exception as e:
        if isinstance(e, LLMError):
            raise
        raise LLMError(
            operation="evaluate_node",
            model=chat_model.model_id,
            message=f"Evaluation failed: {e}",
        ) from e


async def evaluate_nodes(
    nodes: list[tuple[str, str, str, str, str | None, int, int, int, float]],
    chat_model: ChatModel,
    max_concurrent: int = 10,
) -> list[NodeEvaluation]:
    """Evaluate multiple nodes in parallel with concurrency control.

    Args:
        nodes: List of tuples (node_id, summary, left_text, right_text,
               preceding_context, height, level_index, span_start, compression_ratio)
        chat_model: ChatModel instance for LLM calls
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of NodeEvaluation objects
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_with_limit(
        node_id: str,
        summary: str,
        left_text: str,
        right_text: str,
        preceding_context: str | None,
        height: int,
        level_index: int,
        span_start: int,
        compression_ratio: float,
    ) -> NodeEvaluation:
        async with semaphore:
            scores = await evaluate_node(
                summary=summary,
                left_text=left_text,
                right_text=right_text,
                preceding_context=preceding_context,
                chat_model=chat_model,
            )
            return NodeEvaluation(
                node_id=node_id,
                height=height,
                level_index=level_index,
                span_start=span_start,
                compression_ratio=compression_ratio,
                retention=scores["retention"],
                isolation=scores["isolation"],
                faithfulness=scores["faithfulness"],
                continuity=scores["continuity"],
            )

    tasks = [
        evaluate_with_limit(
            node_id=n[0],
            summary=n[1],
            left_text=n[2],
            right_text=n[3],
            preceding_context=n[4],
            height=n[5],
            level_index=n[6],
            span_start=n[7],
            compression_ratio=n[8],
        )
        for n in nodes
    ]

    return list(await asyncio.gather(*tasks))
