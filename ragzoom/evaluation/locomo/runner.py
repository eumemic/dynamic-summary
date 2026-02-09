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

from openai import AsyncOpenAI

from ragzoom.constants import DEV_GRPC_PORT
from ragzoom.evaluation.locomo.agent.backends.openai import OpenAIBackend
from ragzoom.evaluation.locomo.agent.protocol import BenchmarkingAgent
from ragzoom.evaluation.locomo.ingest import (
    doc_id_for,
    ingest_all,
    wait_for_indexing,
)
from ragzoom.evaluation.locomo.scoring import compute_token_f1, judge_answer
from ragzoom.evaluation.locomo.types import (
    AnswerResult,
    BenchmarkReport,
    BudgetPoint,
    CategoryScore,
    ConversationMetrics,
    CostMetrics,
    JudgeVerdict,
    QACategory,
    QAPair,
    parse_locomo_file,
)
from ragzoom.wrapper import RagZoom

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
    use_isolated_server: bool = False
    # Legacy fields kept for backward compatibility with saved configs
    answer_model: str = "gpt-4o-mini"
    budgets: list[int] = field(default_factory=lambda: [0])
    max_iterations: int = 1
    agent_model: str | None = None


# ---------------------------------------------------------------------------
# Tool & backend factories
# ---------------------------------------------------------------------------


def _is_anthropic_model(model_id: str) -> bool:
    """Check if a model ID corresponds to an Anthropic model."""
    return model_id.startswith("claude-")


def _create_backend(model_id: str, openai_client: AsyncOpenAI) -> BenchmarkingAgent:
    """Create a BenchmarkingAgent for the given model ID."""
    if _is_anthropic_model(model_id):
        from ragzoom.evaluation.locomo.agent.backends.anthropic import AnthropicBackend

        return AnthropicBackend(model_id)
    return OpenAIBackend(openai_client, model_id)


# ---------------------------------------------------------------------------
# Single-question evaluation
# ---------------------------------------------------------------------------


async def _evaluate_one(
    judge: BenchmarkingAgent | None,
    judge_model_id: str,
    rz: RagZoom,
    doc_id: str,
    qa: QAPair,
    semaphore: asyncio.Semaphore,
) -> AnswerResult:
    """Evaluate a single QA pair using server-side agentic search."""
    async with semaphore:
        search_result = await asyncio.to_thread(rz.search, qa.question, doc_id)
        generated_answer = search_result.answer

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

        cost: CostMetrics | None = None

        return AnswerResult(
            sample_id=qa.sample_id,
            question=qa.question,
            gold_answer=qa.gold_answer,
            category=qa.category,
            budget_tokens=0,
            retrieved_token_count=0,
            generated_answer=generated_answer,
            judge_verdict=verdict,
            token_f1=f1,
            cost=cost,
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_budget(results: list[AnswerResult], budget: int) -> BudgetPoint:
    """Aggregate results for a single budget level."""
    budget_results = [r for r in results if r.budget_tokens == budget]
    has_verdicts = any(r.judge_verdict is not None for r in budget_results)

    by_category: dict[QACategory, CategoryScore] = {}
    grouped: dict[QACategory, list[AnswerResult]] = defaultdict(list)
    for r in budget_results:
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
        overall_accuracy = mean(
            1.0 if r.judge_verdict == "A" else 0.0 for r in budget_results
        )

    return BudgetPoint(
        budget_tokens=budget,
        overall_accuracy=overall_accuracy,
        overall_f1=mean(r.token_f1 for r in budget_results),
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

        async with BenchmarkServerManager() as mgr:
            config.server_address = mgr.address
            return await _run_benchmark_impl(config)
    return await _run_benchmark_impl(config)


async def _run_benchmark_impl(config: LoCoMoConfig) -> BenchmarkReport:
    """Core benchmark implementation."""
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

    # 3. Set up judge
    openai_client = AsyncOpenAI()

    judge: BenchmarkingAgent | None = None
    if not config.f1_only:
        judge = _create_backend(config.judge_model, openai_client)

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

    logger.info(
        "Evaluating %d questions via server-side search",
        len(qa_items),
    )

    # 5. Evaluate all questions via server-side search
    all_results: list[AnswerResult] = []
    semaphore = asyncio.Semaphore(config.max_concurrent)

    tasks = [
        _evaluate_one(judge, config.judge_model, rz, doc_id, qa, semaphore)
        for doc_id, qa in qa_items
    ]
    all_results = list(await asyncio.gather(*tasks))

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

    # 6. Aggregate (single budget=0 point for search mode)
    budget_curve = [_aggregate_budget(all_results, 0)]

    answer_model = config.agent_model or config.answer_model

    return BenchmarkReport(
        answer_model=answer_model,
        judge_model=config.judge_model,
        num_conversations=len(conversations),
        num_questions=len(qa_items),
        budget_curve=budget_curve,
        per_question=all_results,
        conversation_metrics=conv_metrics,
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
    judge_backend = _create_backend(config.judge_model, openai_client)
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
        budget_raw = entry["budget_tokens"]
        assert isinstance(
            budget_raw, int
        ), f"budget_tokens must be int, got {type(budget_raw)}"
        retrieved_raw = entry["retrieved_token_count"]
        assert isinstance(
            retrieved_raw, int
        ), f"retrieved_token_count must be int, got {type(retrieved_raw)}"
        return AnswerResult(
            sample_id=str(entry["sample_id"]),
            question=question,
            gold_answer=gold_answer,
            category=QACategory[category_str.upper()],
            budget_tokens=budget_raw,
            retrieved_token_count=retrieved_raw,
            generated_answer=generated_answer,
            judge_verdict=verdict,
            token_f1=f1,
        )

    all_results = list(
        await asyncio.gather(*[_rejudge_one(entry) for entry in per_question_raw])
    )

    budgets = sorted({r.budget_tokens for r in all_results})
    budget_curve = [_aggregate_budget(all_results, b) for b in budgets]

    metadata = data.get("metadata", {})
    assert isinstance(metadata, dict)
    num_questions = len({(r.sample_id, r.question) for r in all_results})
    num_conversations_raw = metadata.get("num_conversations", 0)
    assert isinstance(num_conversations_raw, int)

    return BenchmarkReport(
        answer_model=str(metadata.get("answer_model", config.answer_model)),
        judge_model=config.judge_model,
        num_conversations=num_conversations_raw,
        num_questions=num_questions,
        budget_curve=budget_curve,
        per_question=all_results,
    )
