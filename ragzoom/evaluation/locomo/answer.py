"""Query RagZoom and generate answers for LoCoMo questions."""

from __future__ import annotations

import asyncio

from ragzoom.contracts.chat_model import ChatModel, Message
from ragzoom.wrapper import RagZoom

_ANSWER_SYSTEM_PROMPT = """\
You are answering questions about a conversation between two people.
You will be given retrieved context from the conversation. Answer the \
question based ONLY on the provided context.
If the context does not contain enough information, say "I don't know."
Give the most information-dense answer possible that fully answers the question.
Avoid filler words, hedging, or restating the question."""


async def generate_answer(
    rz: RagZoom,
    chat_model: ChatModel,
    doc_id: str,
    question: str,
    budget_tokens: int,
) -> tuple[str, int]:
    """Query RagZoom for context, then generate an answer via the eval LLM.

    Returns:
        Tuple of (generated_answer, retrieved_token_count).
    """
    # RagZoom.query() is sync/blocking — run in thread to avoid blocking the loop
    response = await asyncio.to_thread(
        rz.query, doc_id, question, budget_tokens=budget_tokens
    )
    context = response.summary

    messages: list[Message] = [
        {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"## Context\n{context}\n\n## Question\n{question}",
        },
    ]
    result = await chat_model.complete(messages, temperature=0.0)
    return (result["content"], response.token_count)
