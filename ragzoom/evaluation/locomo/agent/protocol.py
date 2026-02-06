"""Protocol and types for benchmarking agent backends.

A single ``BenchmarkingAgent`` protocol unifies both agentic answer generation
(multi-turn with tools) and single-shot judge calls (no tools, one turn).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ragzoom.evaluation.locomo.types import CostMetrics


@dataclass(frozen=True)
class ToolResult:
    """Value returned by a tool handler."""

    content: str
    is_error: bool = False
    token_count: int = 0


@dataclass(frozen=True)
class ToolDefinition:
    """Backend-agnostic tool definition with an async handler."""

    name: str
    description: str
    parameters: dict[str, object]
    required: tuple[str, ...]
    handler: Callable[[dict[str, object]], Awaitable[ToolResult]]


@dataclass(frozen=True)
class AgentResult:
    """Result from a benchmarking agent call."""

    answer: str
    cost: CostMetrics


def make_agent_result(
    *,
    answer: str,
    total_input: int,
    total_output: int,
    retrieved_tokens: list[int],
    reasoning_turns: int,
    elapsed: float,
) -> AgentResult:
    """Build an AgentResult with CostMetrics from raw counters."""
    return AgentResult(
        answer=answer,
        cost=CostMetrics(
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            retrieval_call_count=len(retrieved_tokens),
            reasoning_turn_count=reasoning_turns,
            retrieved_tokens_per_call=tuple(retrieved_tokens),
            query_duration_seconds=elapsed,
        ),
    )


class BenchmarkingAgent(Protocol):
    """Unified protocol for both agentic answers and single-shot judge calls.

    A judge call is simply ``generate(system, prompt, tools=(), max_turns=1)``.
    """

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: Sequence[ToolDefinition] = (),
        max_turns: int = 1,
        temperature: float | None = None,
    ) -> AgentResult: ...
