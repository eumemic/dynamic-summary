"""Placeholder for future Claude Agent SDK backend."""

from __future__ import annotations

from ragzoom.evaluation.locomo.agent.protocol import AgentResult


class AnthropicAgentBackend:
    """Placeholder for future Claude Agent SDK backend."""

    async def generate(
        self,
        doc_id: str,
        question: str,
        budget_tokens: int,
        max_iterations: int,
    ) -> AgentResult:
        raise NotImplementedError(
            "Anthropic agent backend is not yet implemented. "
            "Use the OpenAI backend for agentic evaluation."
        )
