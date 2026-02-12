"""Server-side search configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for the agentic search agent.

    Set at server startup via CLI flags or environment variables.
    Not configurable per-request — the server owns these parameters.
    """

    agent_model: str = "gpt-5-mini"
    max_iterations: int = 5
    max_token_budget: int = 4000
    profiling_enabled: bool = False
