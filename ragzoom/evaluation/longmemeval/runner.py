"""Orchestrate the LongMemEval benchmark: ingest, answer under a fixed B, judge.

The experiment protocol from the memory-benchmark-selection decision doc hinges
on one knob: a **fixed reader token budget B** (``LongMemEvalConfig.budget``).
Every recall call the answerer makes is capped at B, so RagZoom's accuracy is
measured at a known H/B ratio. Sweeping B (4K/8K/16K/32K) against a fixed
haystack tier H is exactly how the decisive summarizer experiment plots
RagZoom's advantage as a function of H/B.

Each run records the ``summary_model`` that built the trees (the A/B attribution
axis), the ``served_tilings`` the answerer actually saw (failure attribution),
and per-question-type accuracy (the Q axis, including the abstention tripwire).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from ragzoom.agent.factory import create_backend
from ragzoom.agent.protocol import BenchmarkingAgent
from ragzoom.constants import DEV_GRPC_PORT
from ragzoom.evaluation.benchmark_common import (
    GrpcQueryExecutor,
    build_search_agents,
    cost_from_dict,
    cost_from_search_cost,
    run_benchmark_pipeline,
)
from ragzoom.evaluation.longmemeval.agent.prompt import AGENT_SYSTEM_PROMPT
from ragzoom.evaluation.longmemeval.ingest import (
    doc_id_for,
    ingest_all,
    wait_for_indexing,
)
from ragzoom.evaluation.longmemeval.scoring import judge_answer
from ragzoom.evaluation.longmemeval.types import (
    AggregateScores,
    AnswerResult,
    BenchmarkReport,
    CategoryScore,
    HaystackMetrics,
    JudgeVerdict,
    LongMemEvalQuestion,
    QuestionType,
    detect_variant,
    parse_longmemeval_file,
)
from ragzoom.search import SearchAgent, SearchConfig
from ragzoom.wrapper import RagZoom

if TYPE_CHECKING:
    from ragzoom.evaluation.docker_claude import DockerClaudePool

logger = logging.getLogger(__name__)


@dataclass
class LongMemEvalConfig:
    """Configuration for a LongMemEval benchmark run."""

    data_path: Path
    server_address: str = f"127.0.0.1:{DEV_GRPC_PORT}"
    judge_model: str = "gpt-4.1"
    max_concurrent: int = 5
    output_dir: Path = field(default_factory=lambda: Path("longmemeval_results"))
    skip_ingest: bool = False
    sample_size: int | None = None
    no_judge: bool = False
    rejudge_path: Path | None = None
    use_isolated_server: bool = True
    search_model: str = SearchConfig.agent_model
    max_iterations: int = SearchConfig.max_iterations
    # The fixed reader budget B — the core experiment knob. Every recall call is
    # capped here, so accuracy is measured at a known H/B ratio. Named ``budget``
    # (not ``max_budget``) to make its role in the protocol unmistakable.
    budget: int = SearchConfig.max_token_budget
    profiling: bool = False
    use_docker_cli: bool | None = None  # None = auto (Docker for Claude models)
    reasoning_level: str | None = None
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
            "search_model": self.search_model,
            "judge_model": self.judge_model,
            "summary_model": self.summary_model,
            "max_iterations": self.max_iterations,
            "budget": self.budget,
            "max_concurrent": self.max_concurrent,
            "sample_size": self.sample_size,
            "no_judge": self.no_judge,
            "profiling": self.profiling,
            "skip_ingest": self.skip_ingest,
            "use_isolated_server": self.use_isolated_server,
            "use_docker_cli": self.use_docker_cli,
            "reasoning_level": self.reasoning_level,
        }


# ---------------------------------------------------------------------------
# Query execution (fixed-B recall over gRPC)
# ---------------------------------------------------------------------------


def question_with_date(question: LongMemEvalQuestion) -> str:
    """Prefix the question with the date it is asked on.

    Temporal-reasoning questions are asked "as of" ``question_date``; the
    answerer must reason about durations and recency relative to it.
    """
    return f"[Question asked on {question.question_date}]\n{question.question}"


# ---------------------------------------------------------------------------
# Single-question evaluation
# ---------------------------------------------------------------------------


async def _evaluate_one(
    judge: BenchmarkingAgent | None,
    judge_model_id: str,
    agent_queue: asyncio.Queue[SearchAgent],
    query_executor: GrpcQueryExecutor,
    question: LongMemEvalQuestion,
) -> AnswerResult:
    """Evaluate a single question via client-side agentic recall under fixed B."""
    agent = await agent_queue.get()
    try:
        doc_id = doc_id_for(question)
        search_result = await agent.search(
            question_with_date(question),
            doc_id,
            query_executor,
            search_guidance=AGENT_SYSTEM_PROMPT,
        )
        generated_answer = search_result.answer
        cost = cost_from_search_cost(search_result.cost)
        retrospective: str | None = None
        if search_result.profile is not None:
            retrospective = search_result.profile.retrospective

        verdict: JudgeVerdict | None = None
        if judge is not None:
            verdict = await judge_answer(
                judge,
                question.question_type,
                question.question,
                question.answer,
                generated_answer,
                is_abstention=question.is_abstention,
                model_id=judge_model_id,
            )

        return AnswerResult(
            question_id=question.question_id,
            question=question.question,
            gold_answer=question.answer,
            question_type=question.question_type,
            is_abstention=question.is_abstention,
            generated_answer=generated_answer,
            judge_verdict=verdict,
            cost=cost,
            retrospective=retrospective,
            served_tilings=search_result.served_tilings,
        )
    finally:
        agent_queue.put_nowait(agent)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _accuracy(results: list[AnswerResult]) -> float:
    """Fraction of "yes" verdicts. Caller guarantees results is non-empty."""
    return mean(1.0 if r.judge_verdict == "yes" else 0.0 for r in results)


def aggregate(results: list[AnswerResult]) -> AggregateScores:
    """Aggregate accuracy by question type, task-averaged, and abstention.

    In --no-judge mode (no verdicts) every accuracy field is None.
    """
    has_verdicts = any(r.judge_verdict is not None for r in results)

    by_type: dict[QuestionType, CategoryScore] = {}
    grouped: dict[QuestionType, list[AnswerResult]] = defaultdict(list)
    for r in results:
        grouped[r.question_type].append(r)

    for qtype, type_results in sorted(grouped.items(), key=lambda kv: kv[0].value):
        accuracy: float | None = None
        if has_verdicts:
            accuracy = _accuracy(type_results)
        by_type[qtype] = CategoryScore(accuracy=accuracy, count=len(type_results))

    overall_accuracy: float | None = None
    task_averaged_accuracy: float | None = None
    abstention_accuracy: float | None = None
    if has_verdicts:
        overall_accuracy = _accuracy(results)
        per_type = [cs.accuracy for cs in by_type.values() if cs.accuracy is not None]
        task_averaged_accuracy = mean(per_type) if per_type else None
        abstention_results = [r for r in results if r.is_abstention]
        if abstention_results:
            abstention_accuracy = _accuracy(abstention_results)

    return AggregateScores(
        overall_accuracy=overall_accuracy,
        task_averaged_accuracy=task_averaged_accuracy,
        abstention_accuracy=abstention_accuracy,
        by_type=by_type,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_benchmark(config: LongMemEvalConfig) -> BenchmarkReport:
    """Run the full LongMemEval benchmark pipeline.

    1. Optionally start an isolated server (fresh ingest, or verify for reuse)
    2. Parse the dataset, optionally sample
    3. Ingest every haystack into RagZoom and wait for indexing
    4. Answer every question via agentic recall under the fixed budget B
    5. Judge with type-specific prompts (incl. abstention)
    6. Aggregate per-type + overall and return
    """
    return await run_benchmark_pipeline(config, _run_benchmark_core)


async def _run_benchmark_core(
    config: LongMemEvalConfig, docker_pool: DockerClaudePool | None
) -> BenchmarkReport:
    """Core benchmark implementation (separated for Docker lifecycle)."""
    variant = detect_variant(config.data_path)
    questions = parse_longmemeval_file(config.data_path)
    logger.info("Parsed %d questions (variant=%s)", len(questions), variant)

    if config.sample_size is not None:
        random.seed(42)
        questions = random.sample(questions, min(config.sample_size, len(questions)))
        logger.info("Sampled %d questions", len(questions))

    # Ingest (skip if docs are already indexed)
    rz = RagZoom(server_address=config.server_address)
    haystack_metrics: tuple[HaystackMetrics, ...] = ()
    if config.skip_ingest:
        logger.info("Skipping ingestion (--skip-ingest)")
    else:
        haystack_metrics = ingest_all(rz, questions)
        haystack_metrics = wait_for_indexing(rz, questions, haystack_metrics)

    # Judge (OpenAI-backed; no Docker needed)
    openai_client = AsyncOpenAI()
    judge: BenchmarkingAgent | None = None
    if not config.no_judge:
        judge = create_backend(config.judge_model, openai_client)

    # Build the agent queue — one SearchAgent per concurrency slot, every recall
    # call capped at the fixed budget B.
    search_config = SearchConfig(
        agent_model=config.search_model,
        max_iterations=config.max_iterations,
        max_token_budget=config.budget,
        profiling_enabled=config.profiling,
    )
    agent_queue: asyncio.Queue[SearchAgent] = asyncio.Queue()
    for agent in build_search_agents(
        search_config,
        config.search_model,
        openai_client,
        max_concurrent=config.max_concurrent,
        docker_pool=docker_pool,
        reasoning_level=config.reasoning_level,
    ):
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
            _evaluate_one(judge, config.judge_model, agent_queue, query_executor, q),
            name=f"eval-{q.question_id}",
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

    if judge is not None:
        correct = sum(1 for r in all_results if r.judge_verdict == "yes")
        logger.info(
            "%d/%d correct (%.1f%%)",
            correct,
            len(all_results),
            100.0 * correct / len(all_results),
        )

    scores = aggregate(all_results)

    return BenchmarkReport(
        answer_model=config.search_model,
        judge_model=config.judge_model,
        dataset_variant=variant,
        num_questions=len(questions),
        scores=scores,
        per_question=all_results,
        haystack_metrics=haystack_metrics,
        config=config.to_dict(),
    )


# ---------------------------------------------------------------------------
# Rejudge from cached results
# ---------------------------------------------------------------------------


async def rejudge(config: LongMemEvalConfig) -> BenchmarkReport:
    """Re-judge cached answers from a previous results.json (no server needed).

    Re-runs the type-specific judge on each stored (answer, gold) pair and
    re-aggregates. Preserves ``served_tilings`` so the failure-attribution
    instrumentation survives a re-judge.
    """
    assert config.rejudge_path is not None
    with open(config.rejudge_path) as f:
        data = json.load(f)

    per_question_raw = data["per_question"]
    assert isinstance(per_question_raw, list)

    openai_client = AsyncOpenAI()
    judge_backend = create_backend(config.judge_model, openai_client)
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def _rejudge_one(entry: dict[str, object]) -> AnswerResult:
        question = str(entry["question"])
        gold_answer = str(entry["gold_answer"])
        generated_answer = str(entry["generated_answer"])
        question_type = QuestionType(str(entry["question_type"]))
        is_abstention = bool(entry.get("is_abstention", False))
        async with semaphore:
            verdict = await judge_answer(
                judge_backend,
                question_type,
                question,
                gold_answer,
                generated_answer,
                is_abstention=is_abstention,
                model_id=config.judge_model,
            )

        served_raw = entry.get("served_tilings", ())
        assert isinstance(served_raw, list | tuple)
        served_tilings = tuple(str(t) for t in served_raw)

        return AnswerResult(
            question_id=str(entry["question_id"]),
            question=question,
            gold_answer=gold_answer,
            question_type=question_type,
            is_abstention=is_abstention,
            generated_answer=generated_answer,
            judge_verdict=verdict,
            cost=cost_from_dict(entry.get("cost")),
            served_tilings=served_tilings,
        )

    all_results = list(
        await asyncio.gather(*[_rejudge_one(entry) for entry in per_question_raw])
    )

    metadata = data.get("metadata", {})
    assert isinstance(metadata, dict)

    original_config = metadata.get("config")
    if isinstance(original_config, dict):
        rejudge_config: dict[str, object] = {
            **original_config,
            "judge_model": config.judge_model,
        }
    else:
        rejudge_config = config.to_dict()

    return BenchmarkReport(
        answer_model=str(metadata.get("answer_model", config.search_model)),
        judge_model=config.judge_model,
        dataset_variant=str(metadata.get("dataset_variant", "unknown")),
        num_questions=len(all_results),
        scores=aggregate(all_results),
        per_question=all_results,
        config=rejudge_config,
    )
