"""Agentic search: iterative zoom loop driven by an LLM."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Protocol

from ragzoom.agent.protocol import (
    AgentResult,
    BenchmarkingAgent,
    ToolDefinition,
    ToolResult,
)
from ragzoom.client.grpc_client import ExecuteQueryOutput
from ragzoom.output_formatters import format_tiling_spans
from ragzoom.search.config import SearchConfig
from ragzoom.search.prompt import RETROSPECTIVE_FOLLOW_UP, SEARCH_SYSTEM_PROMPT
from ragzoom.search.session import SessionStore
from ragzoom.search.types import (
    SearchCost,
    SearchIteration,
    SearchProfile,
    SearchResult,
)

logger = logging.getLogger(__name__)


class QueryExecutor(Protocol):
    """Async callable that executes a retrieval query.

    Implementations:
    - Server-side: wraps ``execute_query_internal`` (in-process).
    - Client-side: wraps ``RagZoom.query()`` over gRPC.
    """

    async def __call__(
        self,
        *,
        document_id: str,
        query: str,
        budget_tokens: int,
        time_start: str | None = ...,
        time_end: str | None = ...,
    ) -> ExecuteQueryOutput: ...


def _build_recall_tool(
    config: SearchConfig,
    document_id: str,
    query_executor: QueryExecutor,
    iterations: list[SearchIteration],
    profiling: bool,
) -> ToolDefinition:
    """Build the recall ToolDefinition with a handler closure."""

    async def _handle_recall(args: dict[str, object]) -> ToolResult:
        query_text = str(args.get("query", ""))
        raw_budget = args.get("budget_tokens")
        budget = (
            min(int(str(raw_budget)), config.max_token_budget)
            if raw_budget is not None
            else config.max_token_budget
        )
        raw_start = args.get("time_start")
        time_start = str(raw_start) if raw_start and raw_start != "" else None
        raw_end = args.get("time_end")
        time_end = str(raw_end) if raw_end and raw_end != "" else None

        try:
            query_output = await query_executor(
                document_id=document_id,
                query=query_text,
                budget_tokens=budget,
                time_start=time_start,
                time_end=time_end,
            )
            formatted = format_tiling_spans(query_output)
            token_count = query_output.query_result.token_count
            logger.debug(
                "recall(%s, budget=%d) → %d tokens",
                query_text[:50],
                budget,
                token_count,
            )
        except Exception as exc:
            logger.warning("recall(%s) failed: %s", query_text[:50], exc)
            return ToolResult(content=f"Error: {exc}", is_error=True)

        if profiling:
            iterations.append(
                SearchIteration(
                    query=query_text,
                    budget_tokens=budget,
                    time_start=time_start,
                    time_end=time_end,
                    result_text=formatted,
                    result_token_count=token_count,
                    agent_reasoning="",
                )
            )

        return ToolResult(content=formatted, token_count=token_count)

    return ToolDefinition(
        name="recall",
        description=(
            "Retrieve summarized context from the conversation. "
            "Returns variable-resolution text with Span tags showing "
            "time ranges and summarization height."
        ),
        parameters={
            "query": {
                "type": "string",
                "description": "Semantic search keywords",
            },
            "budget_tokens": {
                "type": "integer",
                "description": (
                    f"Max tokens in response (higher = more detail, "
                    f"max: {config.max_token_budget})"
                ),
            },
            "time_start": {
                "type": "string",
                "description": "ISO 8601 lower bound (optional)",
            },
            "time_end": {
                "type": "string",
                "description": "ISO 8601 upper bound (optional)",
            },
        },
        required=("query", "budget_tokens"),
        handler=_handle_recall,
    )


def _make_search_cost(result: AgentResult, elapsed: float) -> SearchCost:
    """Extract SearchCost from an AgentResult."""
    return SearchCost(
        total_input_tokens=result.cost.total_input_tokens,
        total_output_tokens=result.cost.total_output_tokens,
        retrieval_call_count=result.cost.retrieval_call_count,
        reasoning_turn_count=result.cost.reasoning_turn_count,
        retrieved_tokens_per_call=result.cost.retrieved_tokens_per_call,
        duration_seconds=elapsed,
        total_cost_usd=result.cost.total_cost_usd,
    )


class SearchAgent:
    """LLM-driven iterative zoom agent for answering questions from memory."""

    def __init__(
        self,
        config: SearchConfig,
        backend: BenchmarkingAgent,
        session_store: SessionStore | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._session_store = session_store

    async def search(
        self,
        question: str,
        document_id: str,
        query_executor: QueryExecutor,
    ) -> SearchResult:
        """Run the agentic search loop.

        Args:
            question: The user's question to answer.
            document_id: Document to search within.
            query_executor: Callable that executes retrieval queries.

        Returns:
            SearchResult with the answer and optional profiling/session data.
        """
        config = self._config
        profiling = config.profiling_enabled
        needs_history = profiling or self._session_store is not None
        start_time = time.monotonic()

        iterations: list[SearchIteration] = []
        recall_tool = _build_recall_tool(
            config, document_id, query_executor, iterations, profiling
        )

        result = await self._backend.generate(
            SEARCH_SYSTEM_PROMPT,
            question,
            tools=[recall_tool],
            max_turns=config.max_iterations,
            capture_history=needs_history,
        )

        elapsed = time.monotonic() - start_time

        # Create session if a store is configured
        session_id: str | None = None
        history = result.history
        if self._session_store is not None and history is not None:
            session_id = self._session_store.create(document_id, history)

        profile: SearchProfile | None = None
        if profiling:
            transcript = _format_iterations_transcript(
                question, iterations, result.answer
            )
            # Issue a follow-up turn in the same session — the agent has full
            # native context, so the critique is richer than a lossy transcript.
            retro_result = await self._backend.generate(
                SEARCH_SYSTEM_PROMPT,
                RETROSPECTIVE_FOLLOW_UP,
                max_turns=1,
                prior_history=history,
            )
            profile = SearchProfile(
                iterations=tuple(iterations),
                total_input_tokens=result.cost.total_input_tokens,
                total_output_tokens=result.cost.total_output_tokens,
                total_cost_usd=result.cost.total_cost_usd,
                duration_seconds=elapsed,
                retrospective=retro_result.answer,
                transcript=transcript,
            )

        return SearchResult(
            answer=result.answer,
            cost=_make_search_cost(result, elapsed),
            profile=profile,
            session_id=session_id,
        )

    async def search_continue(
        self,
        session_id: str,
        question: str,
        query_executor: QueryExecutor,
    ) -> SearchResult:
        """Continue a search conversation within an existing session.

        Args:
            session_id: Session ID from a previous search result.
            question: The follow-up question.
            query_executor: Callable that executes retrieval queries.

        Returns:
            SearchResult with the answer and the same session_id.

        Raises:
            KeyError: If the session is expired or not found.
        """
        if self._session_store is None:
            raise RuntimeError("search_continue requires a SessionStore")

        session = self._session_store.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found or expired")

        config = self._config
        start_time = time.monotonic()

        iterations: list[SearchIteration] = []
        recall_tool = _build_recall_tool(
            config,
            session.document_id,
            query_executor,
            iterations,
            config.profiling_enabled,
        )

        result = await self._backend.generate(
            SEARCH_SYSTEM_PROMPT,
            question,
            tools=[recall_tool],
            max_turns=config.max_iterations,
            capture_history=True,
            prior_history=session.history,
        )

        elapsed = time.monotonic() - start_time

        # Update session with the new history
        if result.history is not None:
            self._session_store.update(session_id, result.history)

        return SearchResult(
            answer=result.answer,
            cost=_make_search_cost(result, elapsed),
            profile=None,
            session_id=session_id,
        )


def _format_iterations_transcript(
    question: str,
    iterations: Sequence[SearchIteration],
    answer: str,
) -> str:
    """Build a transcript from search iterations for profiling output."""
    lines: list[str] = [f"[QUESTION] {question}", ""]
    for i, it in enumerate(iterations, 1):
        lines.append(f"[RECALL #{i}]")
        lines.append(f"  query: {it.query}")
        lines.append(f"  budget: {it.budget_tokens}")
        if it.time_start:
            lines.append(f"  time_start: {it.time_start}")
        if it.time_end:
            lines.append(f"  time_end: {it.time_end}")
        result_preview = it.result_text
        if len(result_preview) > 500:
            result_preview = result_preview[:500] + "..."
        lines.append(f"  result ({it.result_token_count} tokens): {result_preview}")
        lines.append("")
    lines.append(f"[ANSWER] {answer}")
    return "\n".join(lines)
