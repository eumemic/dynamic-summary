"""Result types for the agentic search system."""

from __future__ import annotations

from dataclasses import dataclass


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
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float | None
    duration_seconds: float
    retrospective: str
    transcript: str


@dataclass(frozen=True)
class SearchResult:
    """The answer returned by a search invocation.

    ``profile`` is None unless profiling is enabled on the server.
    """

    answer: str
    profile: SearchProfile | None
