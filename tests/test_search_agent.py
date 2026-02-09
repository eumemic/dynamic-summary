"""Tests for the model-agnostic SearchAgent."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import AsyncMock, patch

import pytest

from ragzoom.agent.protocol import (
    AgentResult,
    CostMetrics,
    ToolDefinition,
    make_agent_result,
)
from ragzoom.client.grpc_client import ExecuteQueryOutput, RetrievalView
from ragzoom.search.config import SearchConfig
from ragzoom.services.query_service import QueryResult


def _make_cost(**overrides: object) -> CostMetrics:
    defaults: dict[str, object] = {
        "total_input_tokens": 100,
        "total_output_tokens": 50,
        "retrieval_call_count": 0,
        "reasoning_turn_count": 1,
        "retrieved_tokens_per_call": (),
        "query_duration_seconds": 0.5,
        "total_cost_usd": 0.001,
    }
    defaults.update(overrides)
    return CostMetrics(**defaults)  # type: ignore[arg-type]


def _make_query_output(token_count: int = 42) -> ExecuteQueryOutput:
    """Build a minimal ExecuteQueryOutput for testing."""
    return ExecuteQueryOutput(
        query_result=QueryResult(
            summary="spans",
            token_count=token_count,
            nodes_retrieved=1,
            tiling_size=1,
            query_id="",
            seed_count=1,
            verbatim_count=0,
            actual_start=0,
            actual_end=0,
        ),
        retrieval=RetrievalView(
            selected_ids=[],
            tiling_ids=[],
            scores={},
            coverage_map={},
            nodes={},
        ),
        visualization="",
        validation_warning="",
    )


class _MockQueryExecutor:
    """Test double satisfying the QueryExecutor protocol."""

    def __init__(self, token_count: int = 42) -> None:
        self._output = _make_query_output(token_count)
        self.call_count = 0

    async def __call__(
        self,
        *,
        document_id: str,
        query: str,
        budget_tokens: int,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> ExecuteQueryOutput:
        self.call_count += 1
        return self._output


class _NoToolBackend:
    """Backend that returns a fixed answer without calling any tools."""

    def __init__(self, answer: str = "Paris", cost: CostMetrics | None = None) -> None:
        self._answer = answer
        self._cost = cost or _make_cost()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: Sequence[ToolDefinition] = (),
        max_turns: int = 1,
        temperature: float | None = None,
    ) -> AgentResult:
        return AgentResult(answer=self._answer, cost=self._cost)


class _ToolCallingBackend:
    """Backend that calls the first tool once, then returns an answer."""

    def __init__(
        self,
        answer: str = "Paris",
        tool_args: dict[str, object] | None = None,
        cost: CostMetrics | None = None,
    ) -> None:
        self._answer = answer
        self._tool_args = tool_args or {
            "query": "capital of France",
            "budget_tokens": 2000,
        }
        self._cost = cost or _make_cost(
            retrieval_call_count=1,
            retrieved_tokens_per_call=(42,),
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: Sequence[ToolDefinition] = (),
        max_turns: int = 1,
        temperature: float | None = None,
    ) -> AgentResult:
        if tools:
            tool = tools[0]
            await tool.handler(self._tool_args)
        return AgentResult(answer=self._answer, cost=self._cost)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSearchReturnsAnswer:
    @pytest.mark.asyncio
    async def test_search_returns_answer(self) -> None:
        from ragzoom.search.agent import SearchAgent

        config = SearchConfig(profiling_enabled=False)
        agent = SearchAgent(config, _NoToolBackend(answer="42"))
        executor = _MockQueryExecutor()

        result = await agent.search("What is the meaning of life?", "doc-1", executor)

        assert result.answer == "42"
        assert result.profile is None


class TestSearchInvokesToolHandler:
    @pytest.mark.asyncio
    async def test_tool_handler_calls_query_executor(self) -> None:
        from ragzoom.search.agent import SearchAgent

        config = SearchConfig(profiling_enabled=False)
        agent = SearchAgent(config, _ToolCallingBackend())
        executor = _MockQueryExecutor()

        result = await agent.search("question", "doc-1", executor)

        assert executor.call_count == 1
        assert result.answer == "Paris"


class TestSearchCapturesIterationsWhenProfiling:
    @pytest.mark.asyncio
    async def test_iterations_populated(self) -> None:
        from ragzoom.search.agent import SearchAgent

        config = SearchConfig(profiling_enabled=True)
        backend = _ToolCallingBackend()
        agent = SearchAgent(config, backend)
        executor = _MockQueryExecutor()

        with patch(
            "ragzoom.search.agent.run_retrospective",
            new_callable=AsyncMock,
            return_value="Looks good.",
        ):
            result = await agent.search("question", "doc-1", executor)

        assert result.profile is not None
        assert len(result.profile.iterations) == 1
        iteration = result.profile.iterations[0]
        assert iteration.query == "capital of France"
        assert iteration.budget_tokens == 2000
        assert iteration.result_token_count == 42


class TestSearchCostFromBackend:
    @pytest.mark.asyncio
    async def test_cost_fields_from_backend(self) -> None:
        from ragzoom.search.agent import SearchAgent

        cost = _make_cost(
            total_input_tokens=500, total_output_tokens=200, total_cost_usd=0.05
        )
        config = SearchConfig(profiling_enabled=True)
        agent = SearchAgent(config, _NoToolBackend(cost=cost))
        executor = _MockQueryExecutor()

        with patch(
            "ragzoom.search.agent.run_retrospective",
            new_callable=AsyncMock,
            return_value="Fine.",
        ):
            result = await agent.search("question", "doc-1", executor)

        assert result.profile is not None
        assert result.profile.total_input_tokens == 500
        assert result.profile.total_output_tokens == 200
        assert result.profile.total_cost_usd == pytest.approx(0.05)


class TestSearchNoProfileWhenDisabled:
    @pytest.mark.asyncio
    async def test_no_profile(self) -> None:
        from ragzoom.search.agent import SearchAgent

        config = SearchConfig(profiling_enabled=False)
        agent = SearchAgent(config, _NoToolBackend())
        executor = _MockQueryExecutor()

        result = await agent.search("question", "doc-1", executor)

        assert result.profile is None


class TestRetrospectiveUsesBackend:
    @pytest.mark.asyncio
    async def test_retrospective_calls_backend(self) -> None:
        from ragzoom.search.agent import SearchAgent

        retro_result = make_agent_result(
            answer="The search was efficient.",
            total_input=50,
            total_output=20,
            retrieved_tokens=[],
            reasoning_turns=1,
            elapsed=0.1,
        )

        class _RetroBackend:
            """Backend that tracks calls to distinguish search vs retrospective."""

            def __init__(self) -> None:
                self.call_count = 0

            async def generate(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                tools: Sequence[ToolDefinition] = (),
                max_turns: int = 1,
                temperature: float | None = None,
            ) -> AgentResult:
                self.call_count += 1
                if not tools:
                    return retro_result
                return AgentResult(answer="Paris", cost=_make_cost())

        backend = _RetroBackend()
        config = SearchConfig(profiling_enabled=True)
        agent = SearchAgent(config, backend)
        executor = _MockQueryExecutor()

        result = await agent.search("question", "doc-1", executor)

        assert result.profile is not None
        assert result.profile.retrospective == "The search was efficient."
        # At least 2 calls: one for search, one for retrospective
        assert backend.call_count >= 2
