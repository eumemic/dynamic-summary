"""Retrospective self-critique for search profiling."""

from __future__ import annotations

from openai import AsyncOpenAI

from ragzoom.search.prompt import RETROSPECTIVE_PROMPT


async def run_retrospective(
    client: AsyncOpenAI,
    model: str,
    *,
    question: str,
    answer: str,
    transcript: str,
) -> str:
    """Run a single-shot LLM call to critique the search process.

    Only called when ``SearchConfig.profiling_enabled`` is True.

    Args:
        client: Async OpenAI client.
        model: Model ID to use for the retrospective.
        question: The original question that was searched.
        answer: The final answer the agent produced.
        transcript: Full formatted trace of the search session.

    Returns:
        Brief self-critique text (3-5 sentences).
    """
    prompt = RETROSPECTIVE_PROMPT.format(
        question=question,
        answer=answer,
        transcript=transcript,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content or ""
