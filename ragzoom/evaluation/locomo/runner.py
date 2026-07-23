"""Orchestrate the LoCoMo benchmark: ingest, sweep budgets, aggregate."""

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
from ragzoom.evaluation.locomo.ingest import (
    doc_id_for,
    ingest_all,
    wait_for_indexing,
)
from ragzoom.evaluation.locomo.scoring import compute_token_f1, judge_answer
from ragzoom.evaluation.locomo.types import (
    AggregateScores,
    AnswerResult,
    BenchmarkReport,
    CategoryScore,
    ConversationMetrics,
    JudgeVerdict,
    QACategory,
    QAPair,
    parse_locomo_file,
)
from ragzoom.search import SearchAgent, SearchConfig
from ragzoom.wrapper import RagZoom

if TYPE_CHECKING:
    from ragzoom.evaluation.docker_claude import DockerClaudePool

logger = logging.getLogger(__name__)


@dataclass
class LoCoMoConfig:
    """Configuration for a LoCoMo benchmark run."""

    data_path: Path
    server_address: str = f"127.0.0.1:{DEV_GRPC_PORT}"
    judge_model: str = "gpt-4.1"
    max_concurrent: int = 10
    output_dir: Path = field(default_factory=lambda: Path("locomo_results"))
    skip_ingest: bool = False
    sample_size: int | None = None
    f1_only: bool = False
    rejudge_path: Path | None = None
    use_isolated_server: bool = True
    search_model: str = SearchConfig.agent_model
    max_iterations: int = SearchConfig.max_iterations
    max_budget: int = SearchConfig.max_token_budget
    profiling: bool = False
    use_docker_cli: bool | None = None  # None = auto (Docker for Claude models)
    reasoning_level: str | None = None
    # The summarizer that built the trees during ingest. Defaults from the same
    # RAGZOOM_SUMMARY_MODEL env the isolated benchmark server inherits, so each
    # run records which summary model is under test (required for A/B
    # attribution; prior runs logged null and were unattributable).
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
            "max_budget": self.max_budget,
            "max_concurrent": self.max_concurrent,
            "sample_size": self.sample_size,
            "f1_only": self.f1_only,
            "profiling": self.profiling,
            "skip_ingest": self.skip_ingest,
            "use_isolated_server": self.use_isolated_server,
            "use_docker_cli": self.use_docker_cli,
            "reasoning_level": self.reasoning_level,
        }


# ---------------------------------------------------------------------------
# Single-question evaluation
# ---------------------------------------------------------------------------


async def _evaluate_one(
    judge: BenchmarkingAgent | None,
    judge_model_id: str,
    agent_queue: asyncio.Queue[SearchAgent],
    query_executor: GrpcQueryExecutor,
    doc_id: str,
    qa: QAPair,
) -> AnswerResult:
    """Evaluate a single QA pair using client-side agentic search."""
    agent = await agent_queue.get()
    try:
        search_result = await agent.search(qa.question, doc_id, query_executor)
        generated_answer = search_result.answer
        cost = cost_from_search_cost(search_result.cost)
        retrospective: str | None = None
        if search_result.profile is not None:
            retrospective = search_result.profile.retrospective

        verdict: JudgeVerdict | None = None
        if judge is not None:
            verdict = await judge_answer(
                judge,
                qa.question,
                qa.gold_answer,
                generated_answer,
                model_id=judge_model_id,
            )
        f1 = compute_token_f1(generated_answer, qa.gold_answer)

        return AnswerResult(
            sample_id=qa.sample_id,
            question=qa.question,
            gold_answer=qa.gold_answer,
            category=qa.category,
            generated_answer=generated_answer,
            judge_verdict=verdict,
            token_f1=f1,
            cost=cost,
            retrospective=retrospective,
            served_tilings=search_result.served_tilings,
        )
    finally:
        agent_queue.put_nowait(agent)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(results: list[AnswerResult]) -> AggregateScores:
    """Aggregate accuracy and F1 across all results."""
    has_verdicts = any(r.judge_verdict is not None for r in results)

    by_category: dict[QACategory, CategoryScore] = {}
    grouped: dict[QACategory, list[AnswerResult]] = defaultdict(list)
    for r in results:
        grouped[r.category].append(r)

    for cat, cat_results in sorted(grouped.items()):
        accuracy: float | None = None
        if has_verdicts:
            accuracy = mean(1.0 if r.judge_verdict == "A" else 0.0 for r in cat_results)
        by_category[cat] = CategoryScore(
            accuracy=accuracy,
            f1=mean(r.token_f1 for r in cat_results),
            count=len(cat_results),
        )

    overall_accuracy: float | None = None
    if has_verdicts:
        overall_accuracy = mean(1.0 if r.judge_verdict == "A" else 0.0 for r in results)

    return AggregateScores(
        overall_accuracy=overall_accuracy,
        overall_f1=mean(r.token_f1 for r in results),
        by_category=by_category,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_benchmark(config: LoCoMoConfig) -> BenchmarkReport:
    """Run the full LoCoMo benchmark pipeline.

    1. Optionally start an isolated server
    2. Parse the dataset
    3. Ingest all conversations into RagZoom
    4. Wait for indexing to complete
    5. Evaluate all non-adversarial QA pairs via server-side search
    6. Aggregate and return results
    """
    return await run_benchmark_pipeline(config, _run_benchmark_core)


async def _run_benchmark_core(
    config: LoCoMoConfig, docker_pool: DockerClaudePool | None
) -> BenchmarkReport:
    """Core benchmark implementation (separated for Docker lifecycle)."""
    # 1. Parse
    conversations = parse_locomo_file(config.data_path)
    logger.info("Parsed %d conversations", len(conversations))

    # 2. Ingest (skip if docs are already indexed)
    rz = RagZoom(server_address=config.server_address)
    conv_metrics: tuple[ConversationMetrics, ...] = ()
    if config.skip_ingest:
        logger.info("Skipping ingestion (--skip-ingest)")
    else:
        conv_metrics = ingest_all(rz, conversations)
        conv_metrics = wait_for_indexing(rz, conversations, conv_metrics)

    # 3. Set up judge (no Docker needed — judge is typically OpenAI)
    openai_client = AsyncOpenAI()

    judge: BenchmarkingAgent | None = None
    if not config.f1_only:
        judge = create_backend(config.judge_model, openai_client)

    # 4. Collect non-adversarial QA pairs
    qa_items: list[tuple[str, QAPair]] = []
    for conv in conversations:
        did = doc_id_for(conv)
        for qa in conv.qa_pairs:
            if qa.category != QACategory.ADVERSARIAL:
                qa_items.append((did, qa))

    if config.sample_size is not None:
        random.seed(42)
        qa_items = random.sample(qa_items, min(config.sample_size, len(qa_items)))

    # 5. Build agent queue — one SearchAgent per concurrency slot
    search_config = SearchConfig(
        agent_model=config.search_model,
        max_iterations=config.max_iterations,
        max_token_budget=config.max_budget,
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
        len(qa_items),
        config.search_model,
        config.max_iterations,
        config.max_budget,
    )

    # 6. Evaluate all questions via client-side agentic search.
    #    Cancel remaining tasks on first failure — don't burn money on a
    #    broken run, and never silently substitute fake answers.
    async_tasks = [
        asyncio.create_task(
            _evaluate_one(
                judge,
                config.judge_model,
                agent_queue,
                query_executor,
                doc_id,
                qa,
            ),
            name=f"eval-{qa.sample_id}",
        )
        for doc_id, qa in qa_items
    ]
    try:
        all_results = list(await asyncio.gather(*async_tasks))
    except BaseException:
        for t in async_tasks:
            t.cancel()
        # Wait for cancellations to propagate before re-raising
        await asyncio.gather(*async_tasks, return_exceptions=True)
        raise

    # Log results
    if judge is not None:
        correct = sum(1 for r in all_results if r.judge_verdict == "A")
        logger.info(
            "%d/%d correct (%.1f%%), F1=%.3f",
            correct,
            len(all_results),
            100.0 * correct / len(all_results),
            mean(r.token_f1 for r in all_results),
        )
    else:
        logger.info("F1=%.3f (f1-only)", mean(r.token_f1 for r in all_results))

    # 7. Aggregate
    scores = _aggregate(all_results)

    return BenchmarkReport(
        answer_model=config.search_model,
        judge_model=config.judge_model,
        num_conversations=len(conversations),
        num_questions=len(qa_items),
        scores=scores,
        per_question=all_results,
        conversation_metrics=conv_metrics,
        config=config.to_dict(),
    )


# ---------------------------------------------------------------------------
# Rejudge from cached results
# ---------------------------------------------------------------------------


async def rejudge(config: LoCoMoConfig) -> BenchmarkReport:
    """Re-judge cached answers from a previous benchmark run.

    Loads per-question results from ``config.rejudge_path``, re-runs
    the LLM judge on each stored (generated_answer, gold_answer) pair,
    recomputes F1, and re-aggregates into a fresh report. No RagZoom
    server is needed.
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
        async with semaphore:
            verdict = await judge_answer(
                judge_backend,
                question,
                gold_answer,
                generated_answer,
                model_id=config.judge_model,
            )
        f1 = compute_token_f1(generated_answer, gold_answer)
        category_str = str(entry["category"])
        cost = cost_from_dict(entry.get("cost"))

        served_raw = entry.get("served_tilings", ())
        assert isinstance(served_raw, list | tuple)
        served_tilings = tuple(str(t) for t in served_raw)

        return AnswerResult(
            sample_id=str(entry["sample_id"]),
            question=question,
            gold_answer=gold_answer,
            category=QACategory[category_str.upper()],
            generated_answer=generated_answer,
            judge_verdict=verdict,
            token_f1=f1,
            cost=cost,
            served_tilings=served_tilings,
        )

    all_results = list(
        await asyncio.gather(*[_rejudge_one(entry) for entry in per_question_raw])
    )

    metadata = data.get("metadata", {})
    assert isinstance(metadata, dict)
    num_questions = len({(r.sample_id, r.question) for r in all_results})
    num_conversations_raw = metadata.get("num_conversations", 0)
    assert isinstance(num_conversations_raw, int)

    # Preserve original config if present, override judge_model for rejudge
    original_config = metadata.get("config")
    rejudge_config: dict[str, object] | None = None
    if isinstance(original_config, dict):
        rejudge_config = {**original_config, "judge_model": config.judge_model}
    else:
        rejudge_config = config.to_dict()

    return BenchmarkReport(
        answer_model=str(metadata.get("answer_model", config.search_model)),
        judge_model=config.judge_model,
        num_conversations=num_conversations_raw,
        num_questions=num_questions,
        scores=_aggregate(all_results),
        per_question=all_results,
        config=rejudge_config,
    )
