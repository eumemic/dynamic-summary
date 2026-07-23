"""Helpers shared by the LoCoMo and LongMemEval benchmark harnesses.

Both harnesses drive the same RagZoom recall path under a fixed token budget and
serialize the same ``CostMetrics``. Those genuinely type-agnostic pieces live
here so the two harnesses share one implementation instead of cloning it.
Anything coupled to a harness's own question/result types stays in that harness.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Protocol, TypeVar

from openai import AsyncOpenAI

from ragzoom.agent.factory import create_backend
from ragzoom.agent.protocol import BenchmarkingAgent, CostMetrics
from ragzoom.client.grpc_client import ExecuteQueryOutput
from ragzoom.exceptions import LLMError
from ragzoom.search import SearchAgent, SearchConfig, SearchCost
from ragzoom.wrapper import RagZoom

if TYPE_CHECKING:
    from ragzoom.evaluation.docker_claude import DockerClaudePool

logger = logging.getLogger(__name__)

_V = TypeVar("_V")


def cost_from_search_cost(sc: SearchCost) -> CostMetrics:
    """Convert a search agent's ``SearchCost`` into the harness ``CostMetrics``.

    The two carry the same token/iteration counts under different field names
    (``duration_seconds`` vs ``query_duration_seconds``); every harness records
    the same translation, so it lives here once.
    """
    return CostMetrics(
        total_input_tokens=sc.total_input_tokens,
        total_output_tokens=sc.total_output_tokens,
        retrieval_call_count=sc.retrieval_call_count,
        reasoning_turn_count=sc.reasoning_turn_count,
        retrieved_tokens_per_call=sc.retrieved_tokens_per_call,
        query_duration_seconds=sc.duration_seconds,
        total_cost_usd=sc.total_cost_usd,
    )


class _PipelineConfig(Protocol):
    """The config fields the shared benchmark pipeline driver reads.

    Covers both the isolated-server bootstrap (``use_isolated_server`` /
    ``skip_ingest`` / settable ``server_address``) and the Docker-pool sizing
    (``use_docker_cli`` / ``search_model`` / ``max_concurrent``).
    """

    use_isolated_server: bool
    skip_ingest: bool
    server_address: str
    use_docker_cli: bool | None
    search_model: str
    max_concurrent: int


_ConfigT = TypeVar("_ConfigT", bound=_PipelineConfig)
_ReportT = TypeVar("_ReportT")


async def run_benchmark_pipeline(
    config: _ConfigT,
    core: Callable[[_ConfigT, DockerClaudePool | None], Awaitable[_ReportT]],
) -> _ReportT:
    """Drive a harness run: bootstrap the server, manage the Docker pool, run core.

    Every harness shares the same outer shell — optionally start a fresh isolated
    server (or verify a running one for ``--skip-ingest``), then run ``core``
    inside a Docker container pool for Claude search models — so it lives here
    once. ``core`` receives the (possibly ``None``) Docker pool and does the
    harness-specific ingest/answer/score work.
    """
    if config.use_isolated_server:
        from ragzoom.evaluation.locomo.server_manager import BenchmarkServerManager

        mgr = BenchmarkServerManager()
        if config.skip_ingest:
            config.server_address = mgr.verify_running()
        else:
            config.server_address = mgr.start_fresh()
    return await _run_with_docker_pool(config, core)


async def _run_with_docker_pool(
    config: _ConfigT,
    core: Callable[[_ConfigT, DockerClaudePool | None], Awaitable[_ReportT]],
) -> _ReportT:
    """Run ``core`` with a Docker container pool for Claude search models.

    Claude search models load plugins and make slow network calls on the host
    CLI (~37s) but start in ~1s inside a container, so they are auto-routed
    through a Docker pool unless ``use_docker_cli`` overrides. The pool is always
    stopped afterward. Non-Claude models pass ``None`` straight through.
    """
    docker_pool: DockerClaudePool | None = None
    use_docker = config.use_docker_cli
    if use_docker is None:
        from ragzoom.agent.factory import is_claude_model

        use_docker = is_claude_model(config.search_model)
        if use_docker:
            logger.info(
                "Auto-enabling Docker CLI for Claude model %s", config.search_model
            )
    if use_docker:
        from ragzoom.daemon import get_daemon_state_dir
        from ragzoom.evaluation.docker_claude import DockerClaudePool

        session_base = get_daemon_state_dir() / "sdk-sessions"
        docker_pool = DockerClaudePool(session_base, config.max_concurrent)
        docker_pool.ensure_running()

    try:
        return await core(config, docker_pool)
    finally:
        if docker_pool is not None:
            docker_pool.stop()


def build_search_agents(
    search_config: SearchConfig,
    search_model: str,
    openai_client: AsyncOpenAI,
    *,
    max_concurrent: int,
    docker_pool: DockerClaudePool | None,
    reasoning_level: str | None,
) -> list[SearchAgent]:
    """Build the pool of ``SearchAgent``s every harness drains into its queue.

    With a Docker pool (Claude search models) each slot gets its own backend
    bound to that slot's CLI path. Otherwise one backend is shared across
    ``max_concurrent`` agents. Returned as a list so each harness owns the queue
    typing; the construction itself — identical across LoCoMo/LongMemEval/Oolong
    — lives here once.
    """
    if docker_pool is not None:
        return [
            SearchAgent(
                search_config,
                create_backend(
                    search_model,
                    openai_client,
                    cli_path=docker_pool.cli_path(i),
                    reasoning_level=reasoning_level,
                ),
            )
            for i in range(docker_pool.size)
        ]

    backend = create_backend(
        search_model, openai_client, reasoning_level=reasoning_level
    )
    agent = SearchAgent(search_config, backend)
    return [agent for _ in range(max_concurrent)]


async def judge_with_retry(
    backend: BenchmarkingAgent,
    system_prompt: str,
    user_prompt: str,
    parse: Callable[[str], _V],
    *,
    operation: str,
    model_id: str,
    max_retries: int,
) -> _V:
    """Run an LLM-as-Judge call and parse its verdict, retrying on parse errors.

    ``parse`` maps the raw answer text to a verdict, raising ``ValueError`` when
    no verdict is present. A parse failure is retried up to ``max_retries`` times;
    once exhausted — or on any non-parse error — this raises ``LLMError``. It
    never silently substitutes a default verdict: a mis-scored answer would
    corrupt the result, so the harness fails hard instead.
    """
    last_error: LLMError | None = None
    for attempt in range(max_retries + 1):
        try:
            result = await backend.generate(system_prompt, user_prompt, temperature=0.0)
            return parse(result.answer)
        except ValueError as e:
            last_error = LLMError(
                operation=operation,
                model=model_id,
                message=f"Judge response parse error: {e}",
            )
            if attempt < max_retries:
                logger.debug(
                    "Retrying judge (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                continue
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(
                operation=operation, model=model_id, message=f"Judge failed: {e}"
            ) from e

    assert last_error is not None
    raise last_error


def wait_for_documents_indexed(
    rz: RagZoom,
    doc_ids: Iterable[str],
    *,
    poll_interval: float = 2.0,
) -> float:
    """Block until every document reaches ``completion_pct >= 100``.

    Returns the wall-clock seconds spent waiting. Callers fold this into their
    own per-haystack metrics. The loop is benchmark-agnostic — it touches only
    document ids and the RagZoom client — so both harnesses share it.
    """
    start = time.monotonic()
    pending = set(doc_ids)

    while pending:
        still_pending: set[str] = set()
        for did in pending:
            status = rz.get_document_status(did)
            if status.completion_pct < 100.0:
                still_pending.add(did)

        if still_pending:
            logger.info(
                "Waiting for indexing: %d/%d documents pending",
                len(still_pending),
                len(pending),
            )
            time.sleep(poll_interval)

        pending = still_pending

    return time.monotonic() - start


class GrpcQueryExecutor:
    """QueryExecutor backed by gRPC ``ExecuteQuery`` via ``RagZoom.query()``.

    Wraps the blocking ``RagZoom.query`` in a thread so many questions can run
    concurrently. The ``budget_tokens`` passed here is the fixed reader budget B
    that the experiment protocol holds constant per arm.
    """

    def __init__(self, rz: RagZoom) -> None:
        self._rz = rz

    async def __call__(
        self,
        *,
        document_id: str,
        query: str,
        budget_tokens: int,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> ExecuteQueryOutput:
        response = await asyncio.to_thread(
            self._rz.query,
            document_id,
            query,
            budget_tokens=budget_tokens,
            time_start=time_start,
            time_end=time_end,
        )
        return response.raw


def assemble_report_json(
    metadata: dict[str, object],
    config: dict[str, object] | None,
    scores: dict[str, object],
    per_question: list[dict[str, object]],
) -> dict[str, object]:
    """Assemble the top-level results-JSON dict shared by every harness.

    Folds ``config`` into ``metadata`` (when present) and bundles the scores and
    per-question rows. Harness-specific blocks (e.g. window/haystack metrics) are
    added by the caller after this returns.
    """
    if config is not None:
        metadata["config"] = config
    return {
        "metadata": metadata,
        "scores": scores,
        "per_question": per_question,
    }


def result_tail(
    served_tilings: tuple[str, ...],
    cost: CostMetrics,
    retrospective: str | None,
) -> dict[str, object]:
    """Build the per-question fields every harness appends to its result row.

    ``served_tilings`` (the text the answerer actually saw, one entry per recall
    call) is always present so failure attribution can rely on it; ``cost`` is
    always serialized; ``retrospective`` only when profiling captured one. Shared
    by every harness's ``_result_to_dict`` tail.
    """
    tail: dict[str, object] = {
        "served_tilings": list(served_tilings),
        "cost": cost_to_dict(cost),
    }
    if retrospective is not None:
        tail["retrospective"] = retrospective
    return tail


def cost_to_dict(cost: CostMetrics) -> dict[str, object]:
    """Serialize ``CostMetrics`` for JSON output (rounding optional float fields)."""
    d: dict[str, object] = {
        "total_input_tokens": cost.total_input_tokens,
        "total_output_tokens": cost.total_output_tokens,
        "retrieval_call_count": cost.retrieval_call_count,
        "reasoning_turn_count": cost.reasoning_turn_count,
        "retrieved_tokens_per_call": list(cost.retrieved_tokens_per_call),
    }
    if cost.query_duration_seconds is not None:
        d["query_duration_seconds"] = round(cost.query_duration_seconds, 3)
    if cost.total_cost_usd is not None:
        d["total_cost_usd"] = round(cost.total_cost_usd, 6)
    return d


def cost_from_dict(cost_raw: object) -> CostMetrics:
    """Reconstruct ``CostMetrics`` from a serialized cost dict (or zero)."""
    if not isinstance(cost_raw, dict):
        return CostMetrics.zero()
    return CostMetrics(
        total_input_tokens=int(cost_raw.get("total_input_tokens", 0)),
        total_output_tokens=int(cost_raw.get("total_output_tokens", 0)),
        retrieval_call_count=int(cost_raw.get("retrieval_call_count", 0)),
        reasoning_turn_count=int(cost_raw.get("reasoning_turn_count", 0)),
        retrieved_tokens_per_call=tuple(
            int(t) for t in cost_raw.get("retrieved_tokens_per_call", ())
        ),
        query_duration_seconds=cost_raw.get("query_duration_seconds"),
        total_cost_usd=cost_raw.get("total_cost_usd"),
    )
