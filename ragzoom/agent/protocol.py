"""Protocol and types for model-agnostic agent backends.

A single ``BenchmarkingAgent`` protocol unifies both agentic answer generation
(multi-turn with tools) and single-shot judge calls (no tools, one turn).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CostMetrics:
    """Token usage and iteration counts for one agent call."""

    total_input_tokens: int
    total_output_tokens: int
    retrieval_call_count: int
    reasoning_turn_count: int
    retrieved_tokens_per_call: tuple[int, ...]
    query_duration_seconds: float | None = None
    total_cost_usd: float | None = None

    @classmethod
    def zero(cls) -> CostMetrics:
        """Return a zero-valued sentinel for failed or cost-free calls."""
        return cls(
            total_input_tokens=0,
            total_output_tokens=0,
            retrieval_call_count=0,
            reasoning_turn_count=0,
            retrieved_tokens_per_call=(),
        )


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


# ---------------------------------------------------------------------------
# Conversation history types (backend-agnostic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallRecord:
    """A single tool call made by the assistant."""

    call_id: str
    tool_name: str
    arguments_json: str


@dataclass(frozen=True)
class ToolResultRecord:
    """Result returned for a tool call."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class AssistantTurn:
    """An assistant message, optionally with tool calls."""

    text: str | None
    tool_calls: tuple[ToolCallRecord, ...] = ()


# Flat sequence: str = user msg, AssistantTurn = assistant, ToolResultRecord = tool result
MessageHistory = tuple[str | AssistantTurn | ToolResultRecord, ...]


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentResult:
    """Result from an agent call."""

    answer: str
    cost: CostMetrics
    history: MessageHistory | None = None


def make_agent_result(
    *,
    answer: str,
    total_input: int,
    total_output: int,
    retrieved_tokens: list[int],
    reasoning_turns: int,
    elapsed: float,
    total_cost_usd: float | None = None,
    history: MessageHistory | None = None,
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
            total_cost_usd=total_cost_usd,
        ),
        history=history,
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
        capture_history: bool = False,
        prior_history: MessageHistory | None = None,
    ) -> AgentResult: ...
