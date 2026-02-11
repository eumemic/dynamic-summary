"""Result types for the agentic search system."""

from __future__ import annotations

from dataclasses import dataclass

from ragzoom.agent.protocol import MessageHistory


@dataclass(frozen=True)
class SearchIteration:
    """Record of a single recall tool invocation during search."""

    query: str
    budget_tokens: int
    time_start: str | None
    time_end: str | None
    result_text: str
    result_token_count: int
    agent_reasoning: str


@dataclass(frozen=True)
class SearchProfile:
    """Full profiling trace of a search execution.

    Only populated when ``SearchConfig.profiling_enabled`` is True.
    """

    iterations: tuple[SearchIteration, ...]
    history: MessageHistory
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float | None
    duration_seconds: float
    retrospective: str
    transcript: str


@dataclass(frozen=True)
class SearchResult:
    """The answer returned by a search invocation.

    ``cost`` is always populated.  ``profile`` is None unless profiling is
    enabled via ``SearchConfig.profiling_enabled``.  ``session_id`` is set
    when a ``SessionRegistry`` is configured, enabling follow-up questions.
    """

    answer: str
    cost: SearchCost
    profile: SearchProfile | None
    session_id: str | None = None


@dataclass(frozen=True)
class SearchCost:
    """Token usage and timing for one search invocation."""

    total_input_tokens: int
    total_output_tokens: int
    retrieval_call_count: int
    reasoning_turn_count: int
    retrieved_tokens_per_call: tuple[int, ...]
    duration_seconds: float
    total_cost_usd: float | None
