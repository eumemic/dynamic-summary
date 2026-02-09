"""Retrospective self-critique for search profiling."""

from __future__ import annotations

from ragzoom.agent.protocol import BenchmarkingAgent
from ragzoom.search.prompt import RETROSPECTIVE_PROMPT


async def run_retrospective(
    backend: BenchmarkingAgent,
    *,
    question: str,
    answer: str,
    transcript: str,
) -> str:
    """Run a single-shot LLM call to critique the search process.

    Only called when ``SearchConfig.profiling_enabled`` is True.

    Args:
        backend: Agent backend for the retrospective call.
        question: The original question that was searched.
        answer: The final answer the agent produced.
        transcript: Formatted trace of the search iterations.

    Returns:
        Brief self-critique text (3-5 sentences).
    """
    prompt = RETROSPECTIVE_PROMPT.format(
        question=question,
        answer=answer,
        transcript=transcript,
    )

    result = await backend.generate(system_prompt="", user_prompt=prompt)
    return result.answer
