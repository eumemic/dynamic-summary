"""Protocol and result types for agentic answer backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ragzoom.evaluation.locomo.types import CostMetrics


@dataclass(frozen=True)
class AgentResult:
    """Result from an agentic answer generation."""

    answer: str
    cost: CostMetrics


class AgentBackend(Protocol):
    """Protocol for agentic answer backends."""

    async def generate(
        self,
        doc_id: str,
        question: str,
        budget_tokens: int,
        max_iterations: int,
    ) -> AgentResult: ...
