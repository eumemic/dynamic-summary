"""Tests for the model-agnostic SearchAgent."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ragzoom.agent.protocol import (
    AgentResult,
    AssistantTurn,
    CostMetrics,
    MessageHistory,
    ToolDefinition,
    make_agent_result,
)
from ragzoom.client.grpc_client import ExecuteQueryOutput, RetrievalView
from ragzoom.search.config import SearchConfig
from ragzoom.search.session import SessionRegistry
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
        resume_session_id: str | None = None,
    ) -> AgentResult:
        return AgentResult(answer=self._answer, cost=self._cost, history=())


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
        resume_session_id: str | None = None,
    ) -> AgentResult:
        if tools:
            tool = tools[0]
            await tool.handler(self._tool_args)
        return AgentResult(answer=self._answer, cost=self._cost, history=())


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
                resume_session_id: str | None = None,
            ) -> AgentResult:
                self.call_count += 1
                if not tools:
                    return retro_result
                return AgentResult(
                    answer="Paris",
                    cost=_make_cost(),
                    history=(),
                    session_id="retro-session",
                )

        backend = _RetroBackend()
        config = SearchConfig(profiling_enabled=True)
        agent = SearchAgent(config, backend)
        executor = _MockQueryExecutor()

        result = await agent.search("question", "doc-1", executor)

        assert result.profile is not None
        assert result.profile.retrospective == "The search was efficient."
        # At least 2 calls: one for search, one for retrospective
        assert backend.call_count >= 2


# ---------------------------------------------------------------------------
# Session-aware backend for session tests
# ---------------------------------------------------------------------------

_SAMPLE_HISTORY: MessageHistory = ("question", AssistantTurn(text="answer"))


class _SessionAwareBackend:
    """Backend that returns session_id and tracks resume_session_id."""

    def __init__(self, answer: str = "Paris") -> None:
        self._answer = answer
        self.call_count = 0
        self.last_resume_session_id: str | None = None

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: Sequence[ToolDefinition] = (),
        max_turns: int = 1,
        temperature: float | None = None,
        resume_session_id: str | None = None,
    ) -> AgentResult:
        self.call_count += 1
        self.last_resume_session_id = resume_session_id
        session_id = resume_session_id or "new-session-id"
        return AgentResult(
            answer=self._answer,
            cost=_make_cost(),
            history=_SAMPLE_HISTORY,
            session_id=session_id,
        )


class TestSessionCreation:
    @pytest.mark.asyncio
    async def test_search_returns_session_id(self) -> None:
        from ragzoom.search.agent import SearchAgent

        config = SearchConfig(profiling_enabled=False)
        registry = SessionRegistry()
        agent = SearchAgent(config, _SessionAwareBackend(), session_registry=registry)
        executor = _MockQueryExecutor()

        result = await agent.search("question", "doc-1", executor)

        assert result.session_id == "new-session-id"

    @pytest.mark.asyncio
    async def test_session_registered_in_registry(self) -> None:
        from ragzoom.search.agent import SearchAgent

        config = SearchConfig(profiling_enabled=False)
        registry = SessionRegistry()
        agent = SearchAgent(config, _SessionAwareBackend(), session_registry=registry)
        executor = _MockQueryExecutor()

        result = await agent.search("question", "doc-1", executor)

        assert result.session_id is not None
        session = registry.get(result.session_id)
        assert session is not None
        assert session.document_id == "doc-1"

    @pytest.mark.asyncio
    async def test_session_id_without_registry(self) -> None:
        """Backend-generated session_id is still returned even without a registry."""
        from ragzoom.search.agent import SearchAgent

        config = SearchConfig(profiling_enabled=False)
        agent = SearchAgent(config, _SessionAwareBackend())
        executor = _MockQueryExecutor()

        result = await agent.search("question", "doc-1", executor)

        assert result.session_id == "new-session-id"


class TestSessionContinuation:
    @pytest.mark.asyncio
    async def test_search_continue_returns_answer(self) -> None:
        from ragzoom.search.agent import SearchAgent

        backend = _SessionAwareBackend(answer="Follow-up answer")
        config = SearchConfig(profiling_enabled=False)
        registry = SessionRegistry()
        agent = SearchAgent(config, backend, session_registry=registry)
        executor = _MockQueryExecutor()

        # Initial search
        initial = await agent.search("first question", "doc-1", executor)
        assert initial.session_id is not None

        # Follow-up
        followup = await agent.search_continue(
            initial.session_id, "second question", executor
        )

        assert followup.answer == "Follow-up answer"
        assert followup.session_id == initial.session_id
        assert backend.last_resume_session_id == initial.session_id

    @pytest.mark.asyncio
    async def test_search_continue_missing_session_raises(self) -> None:
        from ragzoom.search.agent import SearchAgent

        config = SearchConfig(profiling_enabled=False)
        registry = SessionRegistry()
        agent = SearchAgent(config, _SessionAwareBackend(), session_registry=registry)
        executor = _MockQueryExecutor()

        with pytest.raises(KeyError, match="not found"):
            await agent.search_continue("nonexistent", "question", executor)

    @pytest.mark.asyncio
    async def test_search_continue_without_registry_raises(self) -> None:
        from ragzoom.search.agent import SearchAgent

        config = SearchConfig(profiling_enabled=False)
        agent = SearchAgent(config, _SessionAwareBackend())
        executor = _MockQueryExecutor()

        with pytest.raises(RuntimeError, match="requires a SessionRegistry"):
            await agent.search_continue("any-id", "question", executor)


# ---------------------------------------------------------------------------
# Prompt-capturing backend for search_guidance tests
# ---------------------------------------------------------------------------


class _PromptCapturingBackend:
    """Backend that captures the system prompt passed to generate()."""

    def __init__(self, answer: str = "answer") -> None:
        self._answer = answer
        self.captured_system_prompts: list[str] = []

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: Sequence[ToolDefinition] = (),
        max_turns: int = 1,
        temperature: float | None = None,
        resume_session_id: str | None = None,
    ) -> AgentResult:
        self.captured_system_prompts.append(system_prompt)
        session_id = resume_session_id or "sess-1"
        return AgentResult(
            answer=self._answer,
            cost=_make_cost(),
            history=(),
            session_id=session_id,
        )


class TestSearchGuidance:
    @pytest.mark.asyncio
    async def test_guidance_appended_to_system_prompt(self) -> None:
        """search_guidance is appended under a # Search Guidance heading."""
        from ragzoom.search.agent import SearchAgent
        from ragzoom.search.prompt import SEARCH_SYSTEM_PROMPT

        backend = _PromptCapturingBackend()
        config = SearchConfig(profiling_enabled=False)
        agent = SearchAgent(config, backend)
        executor = _MockQueryExecutor()

        await agent.search(
            "question",
            "doc-1",
            executor,
            search_guidance="Use second-person voice.",
        )

        assert len(backend.captured_system_prompts) == 1
        prompt = backend.captured_system_prompts[0]
        assert prompt.startswith(SEARCH_SYSTEM_PROMPT)
        assert "# Search Guidance" in prompt
        assert "Use second-person voice." in prompt

    @pytest.mark.asyncio
    async def test_no_guidance_preserves_default_prompt(self) -> None:
        """When search_guidance is None, the default prompt is used unchanged."""
        from ragzoom.search.agent import SearchAgent
        from ragzoom.search.prompt import SEARCH_SYSTEM_PROMPT

        backend = _PromptCapturingBackend()
        config = SearchConfig(profiling_enabled=False)
        agent = SearchAgent(config, backend)
        executor = _MockQueryExecutor()

        await agent.search("question", "doc-1", executor)

        assert len(backend.captured_system_prompts) == 1
        assert backend.captured_system_prompts[0] == SEARCH_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_empty_guidance_preserves_default_prompt(self) -> None:
        """Empty or whitespace-only guidance does not modify the prompt."""
        from ragzoom.search.agent import SearchAgent
        from ragzoom.search.prompt import SEARCH_SYSTEM_PROMPT

        backend = _PromptCapturingBackend()
        config = SearchConfig(profiling_enabled=False)
        agent = SearchAgent(config, backend)
        executor = _MockQueryExecutor()

        await agent.search("question", "doc-1", executor, search_guidance="   ")

        assert backend.captured_system_prompts[0] == SEARCH_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_search_continue_applies_guidance(self) -> None:
        """search_continue also appends search_guidance to the system prompt."""
        from ragzoom.search.agent import SearchAgent

        backend = _PromptCapturingBackend()
        config = SearchConfig(profiling_enabled=False)
        registry = SessionRegistry()
        agent = SearchAgent(config, backend, session_registry=registry)
        executor = _MockQueryExecutor()

        # Initial search to create session
        result = await agent.search("first", "doc-1", executor)
        assert result.session_id is not None

        # Follow-up with guidance
        await agent.search_continue(
            result.session_id,
            "follow-up",
            executor,
            search_guidance="Answer in second person.",
        )

        # The second call (search_continue) should have the guidance
        assert len(backend.captured_system_prompts) == 2
        prompt = backend.captured_system_prompts[1]
        assert "# Search Guidance" in prompt
        assert "Answer in second person." in prompt
