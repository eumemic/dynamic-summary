"""Orchestrate the LoCoMo benchmark: ingest, sweep budgets, aggregate."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from ragzoom.agent.factory import create_backend
from ragzoom.agent.protocol import BenchmarkingAgent
from ragzoom.client.grpc_client import ExecuteQueryOutput
from ragzoom.constants import DEV_GRPC_PORT
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
    CostMetrics,
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

    def to_dict(self) -> dict[str, object]:
        """Serialize config for embedding in results JSON.

        Converts Path fields to strings and omits transient fields
        (server_address, output_dir) that don't affect results.
        """
        return {
            "data_path": str(self.data_path),
            "search_model": self.search_model,
            "judge_model": self.judge_model,
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


class _GrpcQueryExecutor:
    """QueryExecutor backed by gRPC ``ExecuteQuery`` via ``RagZoom.query()``."""

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


async def _evaluate_one(
    judge: BenchmarkingAgent | None,
    judge_model_id: str,
    agent_queue: asyncio.Queue[SearchAgent],
    query_executor: _GrpcQueryExecutor,
    doc_id: str,
    qa: QAPair,
) -> AnswerResult:
    """Evaluate a single QA pair using client-side agentic search."""
    agent = await agent_queue.get()
    try:
        search_result = await agent.search(qa.question, doc_id, query_executor)
        generated_answer = search_result.answer
        sc = search_result.cost
        cost = CostMetrics(
            total_input_tokens=sc.total_input_tokens,
            total_output_tokens=sc.total_output_tokens,
            retrieval_call_count=sc.retrieval_call_count,
            reasoning_turn_count=sc.reasoning_turn_count,
            retrieved_tokens_per_call=sc.retrieved_tokens_per_call,
            query_duration_seconds=sc.duration_seconds,
            total_cost_usd=sc.total_cost_usd,
        )
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
    if config.use_isolated_server:
        from ragzoom.evaluation.locomo.server_manager import BenchmarkServerManager

        mgr = BenchmarkServerManager()
        if config.skip_ingest:
            config.server_address = mgr.verify_running()
        else:
            config.server_address = mgr.start_fresh()
    return await _run_benchmark_impl(config)


async def _run_benchmark_impl(config: LoCoMoConfig) -> BenchmarkReport:
    """Core benchmark implementation."""
    # 0. Docker container pool for Claude SDK calls.
    #    Auto-enabled for Claude models: the host CLI loads plugins and makes
    #    slow network calls (~37s), while Docker containers start in ~1s.
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
        return await _run_benchmark_core(config, docker_pool)
    finally:
        if docker_pool is not None:
            docker_pool.stop()


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
    if docker_pool is not None:
        for i in range(docker_pool.size):
            backend = create_backend(
                config.search_model,
                openai_client,
                cli_path=docker_pool.cli_path(i),
                reasoning_level=config.reasoning_level,
            )
            agent_queue.put_nowait(SearchAgent(search_config, backend))
    else:
        backend = create_backend(
            config.search_model,
            openai_client,
            reasoning_level=config.reasoning_level,
        )
        agent = SearchAgent(search_config, backend)
        for _ in range(config.max_concurrent):
            agent_queue.put_nowait(agent)

    query_executor = _GrpcQueryExecutor(rz)

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
        cost_raw = entry.get("cost")
        if isinstance(cost_raw, dict):
            cost = CostMetrics(
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
        else:
            cost = CostMetrics.zero()

        return AnswerResult(
            sample_id=str(entry["sample_id"]),
            question=question,
            gold_answer=gold_answer,
            category=QACategory[category_str.upper()],
            generated_answer=generated_answer,
            judge_verdict=verdict,
            token_f1=f1,
            cost=cost,
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
