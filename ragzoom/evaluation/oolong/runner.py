"""Orchestrate the Oolong benchmark: ingest, answer under a fixed B, score.

Oolong is RagZoom's home regime. Every question's gold answer is a statistic
over the whole transcript (a count, a per-episode list, a most/least-common
label), so the answerer cannot retrieve a single needle — it must survey the
history and aggregate. A *fixed reader token budget B* (``OolongConfig.budget``)
forces that survey through compression: at small B the verbatim transcript does
not fit, so the agent must lean on the multi-resolution tile. Sweeping B against
a fixed transcript H is how the aggregation experiment plots RagZoom's advantage
as a function of H/B.

Two things distinguish this runner from the LongMemEval one:

1. **No judge.** Scoring is the deterministic Oolong metric (partial credit for
   counts, exact match for labels, set recall for lists) — there is no judge
   model, no API call, no judge noise. ``_evaluate_one`` scores inline.
2. **Per-window ingest dedupe.** Many questions share one context window, so the
   RagZoom document is keyed on the window and each distinct window is ingested
   exactly once (``ingest_unique_windows``).

Each run records the ``summary_model`` that built the trees (the A/B attribution
axis) and, per question, the ``served_tilings`` the answerer saw (so a low score
is attributable to synthesis vs summary-loss vs retrieval-miss).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from openai import AsyncOpenAI

from ragzoom.constants import DEV_GRPC_PORT
from ragzoom.evaluation.benchmark_common import (
    GrpcQueryExecutor,
    build_search_agents,
    cost_from_search_cost,
    run_benchmark_pipeline,
)
from ragzoom.evaluation.oolong.agent.prompt import AGENT_SYSTEM_PROMPT
from ragzoom.evaluation.oolong.ingest import (
    doc_id_for,
    ingest_unique_windows,
    wait_for_indexing,
)
from ragzoom.evaluation.oolong.loader import CONFIG_DND, load_oolong_jsonl
from ragzoom.evaluation.oolong.scoring import extract_boxed_answer, score_answer
from ragzoom.evaluation.oolong.types import (
    AggregateScores,
    AnswerResult,
    BenchmarkReport,
    CategoryScore,
    OolongQuestion,
    QuestionType,
    WindowMetrics,
)
from ragzoom.search import QueryExecutor, SearchConfig, SearchResult
from ragzoom.wrapper import RagZoom

if TYPE_CHECKING:
    from ragzoom.evaluation.docker_claude import DockerClaudePool

logger = logging.getLogger(__name__)


@dataclass
class OolongConfig:
    """Configuration for an Oolong benchmark run."""

    data_path: Path
    config_name: str = CONFIG_DND
    split: str = "test"
    server_address: str = f"127.0.0.1:{DEV_GRPC_PORT}"
    max_concurrent: int = 5
    output_dir: Path = field(default_factory=lambda: Path("oolong_results"))
    skip_ingest: bool = False
    sample_size: int | None = None
    use_isolated_server: bool = True
    search_model: str = SearchConfig.agent_model
    max_iterations: int = SearchConfig.max_iterations
    # The fixed reader budget B — the core experiment knob. Every recall call is
    # capped here, so accuracy is measured at a known H/B ratio. At small B the
    # transcript cannot fit verbatim, forcing aggregation through compression.
    budget: int = SearchConfig.max_token_budget
    profiling: bool = False
    reasoning_level: str | None = None
    use_docker_cli: bool | None = None  # None = auto (Docker for Claude models)
    # The summarizer that built the trees during ingest. Defaults from the same
    # RAGZOOM_SUMMARY_MODEL env the isolated benchmark server inherits, so each
    # run records which summary model is under test (the A/B attribution axis;
    # null here would make a run unattributable).
    summary_model: str | None = field(
        default_factory=lambda: os.environ.get("RAGZOOM_SUMMARY_MODEL")
    )

    def to_dict(self) -> dict[str, object]:
        """Serialize config for embedding in results JSON.

        Converts Path fields to strings and omits transient fields
        (server_address, output_dir) that don't affect results.
        """
        return {
            "data_path": str(self.data_path),
            "config_name": self.config_name,
            "split": self.split,
            "search_model": self.search_model,
            "summary_model": self.summary_model,
            "max_iterations": self.max_iterations,
            "budget": self.budget,
            "max_concurrent": self.max_concurrent,
            "sample_size": self.sample_size,
            "profiling": self.profiling,
            "skip_ingest": self.skip_ingest,
            "use_isolated_server": self.use_isolated_server,
            "use_docker_cli": self.use_docker_cli,
            "reasoning_level": self.reasoning_level,
        }


# ---------------------------------------------------------------------------
# Question framing
# ---------------------------------------------------------------------------

_BOXED_DIRECTIVE = (
    "Answer the question below about the game statistics. Survey the whole "
    "transcript, count exactly, and return ONLY the final answer in \\boxed{}."
)


def boxed_question(question: OolongQuestion) -> str:
    """Frame the question with the mandatory ``\\boxed{}`` answer directive.

    The deterministic scorer parses the answer out of ``\\boxed{}``, so the
    answerer must be told to use it — the upstream oolong-real prompt carries the
    same directive in its preamble, which ingest strips, so it is re-attached to
    the question here rather than buried in the (un-ingested) transcript wrapper.
    """
    return f"{_BOXED_DIRECTIVE}\n\n{question.question}"


# ---------------------------------------------------------------------------
# Single-question evaluation
# ---------------------------------------------------------------------------


@runtime_checkable
class _Answerer(Protocol):
    """The agent surface ``_evaluate_one`` drives: a budget-capped recall search.

    The real ``SearchAgent`` satisfies it, and a test double can too. Declared so
    the agent pulled off the queue can be narrowed from ``object`` to a typed
    ``search`` without importing the concrete agent into the type of the queue.
    """

    async def search(
        self,
        question: str,
        document_id: str,
        query_executor: QueryExecutor,
        *,
        time_start: str | None = None,
        time_end: str | None = None,
        search_guidance: str | None = None,
    ) -> SearchResult: ...


async def _evaluate_one(
    agent_queue: asyncio.Queue[object],
    query_executor: object,
    question: OolongQuestion,
) -> AnswerResult:
    """Evaluate a single question via client-side agentic recall under fixed B.

    Unlike LongMemEval there is no judge: the answer is scored inline against the
    gold with Oolong's deterministic metric, and the parsed ``\\boxed{}`` answer
    is recorded so a low score is attributable to a parse failure vs a wrong one.

    ``agent_queue`` and ``query_executor`` are typed loosely (``object``) so a
    test double can stand in for the concrete ``SearchAgent`` / gRPC executor; the
    agent is narrowed to the ``_Answerer`` protocol before use, and the executor
    is the opaque value the agent threads into its recall calls.
    """
    agent = await agent_queue.get()
    try:
        if not isinstance(agent, _Answerer):
            raise TypeError(f"Queued object is not a searchable agent: {agent!r}")
        doc_id = doc_id_for(question)
        search_result = await agent.search(
            boxed_question(question),
            doc_id,
            cast(QueryExecutor, query_executor),
            search_guidance=AGENT_SYSTEM_PROMPT,
        )
        generated_answer = search_result.answer
        cost = cost_from_search_cost(search_result.cost)
        retrospective: str | None = None
        if search_result.profile is not None:
            retrospective = search_result.profile.retrospective

        return AnswerResult(
            question_id=question.id,
            question=question.question,
            gold_answer=question.answer,
            question_type=question.question_type,
            generated_answer=generated_answer,
            parsed_answer=extract_boxed_answer(generated_answer),
            score=score_answer(gold=question.answer, model_output=generated_answer),
            cost=cost,
            retrospective=retrospective,
            served_tilings=search_result.served_tilings,
        )
    finally:
        agent_queue.put_nowait(agent)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: list[AnswerResult]) -> AggregateScores:
    """Aggregate the deterministic Oolong score by type, overall, and task-averaged.

    ``overall_score`` is the mean over every question (equal-weight); the
    paper's ``task_averaged_score`` is the mean of the per-type means, so a type
    with many questions does not dominate. Empty input yields ``None`` for both
    (no questions, no score) rather than a misleading 0.
    """
    grouped: dict[QuestionType, list[AnswerResult]] = defaultdict(list)
    for r in results:
        grouped[r.question_type].append(r)

    by_type: dict[QuestionType, CategoryScore] = {}
    for qtype, type_results in sorted(grouped.items(), key=lambda kv: kv[0].value):
        by_type[qtype] = CategoryScore(
            score=mean(r.score for r in type_results),
            count=len(type_results),
        )

    if not results:
        return AggregateScores(
            overall_score=None, task_averaged_score=None, by_type=by_type
        )

    per_type = [cs.score for cs in by_type.values() if cs.score is not None]
    return AggregateScores(
        overall_score=mean(r.score for r in results),
        task_averaged_score=mean(per_type) if per_type else None,
        by_type=by_type,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_benchmark(config: OolongConfig) -> BenchmarkReport:
    """Run the full Oolong benchmark pipeline.

    1. Optionally start an isolated server (fresh ingest, or verify for reuse)
    2. Load the parquet, optionally sample
    3. Ingest each distinct context window into RagZoom and wait for indexing
    4. Answer every question via agentic recall under the fixed budget B
    5. Score with the deterministic Oolong metric (no judge)
    6. Aggregate per-type + overall and return
    """
    return await run_benchmark_pipeline(config, _run_benchmark_core)


async def _run_benchmark_core(
    config: OolongConfig, docker_pool: DockerClaudePool | None
) -> BenchmarkReport:
    """Core benchmark implementation: ingest, answer under B, score, aggregate."""
    questions = load_oolong_jsonl(config.data_path)
    logger.info(
        "Loaded %d questions (%s/%s)", len(questions), config.config_name, config.split
    )

    if config.sample_size is not None:
        random.seed(42)
        questions = random.sample(questions, min(config.sample_size, len(questions)))
        logger.info("Sampled %d questions", len(questions))

    rz = RagZoom(server_address=config.server_address)
    window_metrics: tuple[WindowMetrics, ...] = ()
    if config.skip_ingest:
        logger.info("Skipping ingestion (--skip-ingest)")
    else:
        window_metrics = ingest_unique_windows(rz, questions)
        window_metrics = wait_for_indexing(rz, questions, window_metrics)

    # The agentic answerer; every recall call capped at the fixed budget B.
    search_config = SearchConfig(
        agent_model=config.search_model,
        max_iterations=config.max_iterations,
        max_token_budget=config.budget,
        profiling_enabled=config.profiling,
    )
    agents = build_search_agents(
        search_config,
        config.search_model,
        AsyncOpenAI(),
        max_concurrent=config.max_concurrent,
        docker_pool=docker_pool,
        reasoning_level=config.reasoning_level,
    )
    # Typed loosely (Queue[object]) to match _evaluate_one, which narrows each
    # dequeued agent to the _Answerer protocol before driving it.
    agent_queue: asyncio.Queue[object] = asyncio.Queue()
    for agent in agents:
        agent_queue.put_nowait(agent)

    query_executor = GrpcQueryExecutor(rz)

    logger.info(
        "Evaluating %d questions (model=%s, max_iter=%d, budget=%d)",
        len(questions),
        config.search_model,
        config.max_iterations,
        config.budget,
    )

    # Cancel remaining tasks on first failure — don't burn money on a broken
    # run, and never silently substitute fake answers.
    async_tasks = [
        asyncio.create_task(
            _evaluate_one(agent_queue, query_executor, q),
            name=f"eval-{q.id}",
        )
        for q in questions
    ]
    try:
        all_results = list(await asyncio.gather(*async_tasks))
    except BaseException:
        for t in async_tasks:
            t.cancel()
        await asyncio.gather(*async_tasks, return_exceptions=True)
        raise

    scores = aggregate(all_results)
    if scores.overall_score is not None:
        logger.info(
            "Overall score %.1f%% (task-avg %.1f%%)",
            100.0 * scores.overall_score,
            100.0 * (scores.task_averaged_score or 0.0),
        )

    return BenchmarkReport(
        answer_model=config.search_model,
        config_name=config.config_name,
        split=config.split,
        num_questions=len(questions),
        scores=scores,
        per_question=all_results,
        window_metrics=window_metrics,
        config=config.to_dict(),
    )
