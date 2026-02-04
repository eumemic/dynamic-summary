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

from ragzoom.adapters.openai_chat_model import OpenAIChatModel
from ragzoom.constants import DEV_GRPC_PORT
from ragzoom.evaluation.locomo.answer import generate_answer
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
    JudgeVerdict,
    QACategory,
    QAPair,
    parse_locomo_file,
)
from ragzoom.wrapper import RagZoom

logger = logging.getLogger(__name__)

DEFAULT_BUDGETS = [500, 1000, 2000, 4000, 8000]


@dataclass
class LoCoMoConfig:
    """Configuration for a LoCoMo benchmark run."""

    data_path: Path
    server_address: str = f"127.0.0.1:{DEV_GRPC_PORT}"
    answer_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4.1"
    budgets: list[int] = field(default_factory=lambda: list(DEFAULT_BUDGETS))
    max_concurrent: int = 10
    output_dir: Path = field(default_factory=lambda: Path("locomo_results"))
    skip_ingest: bool = False
    sample_size: int | None = None
    f1_only: bool = False
    rejudge_path: Path | None = None


# ---------------------------------------------------------------------------
# Single-question evaluation
# ---------------------------------------------------------------------------


async def _evaluate_one(
    rz: RagZoom,
    answer_model: OpenAIChatModel,
    judge_model: OpenAIChatModel | None,
    doc_id: str,
    qa: QAPair,
    budget: int,
    semaphore: asyncio.Semaphore,
) -> AnswerResult:
    """Evaluate a single QA pair at a given budget."""
    async with semaphore:
        generated, token_count = await generate_answer(
            rz, answer_model, doc_id, qa.question, budget
        )
        verdict: JudgeVerdict | None = None
        if judge_model is not None:
            verdict = await judge_answer(
                judge_model, qa.question, qa.gold_answer, generated
            )
        f1 = compute_token_f1(generated, qa.gold_answer)

        return AnswerResult(
            sample_id=qa.sample_id,
            question=qa.question,
            gold_answer=qa.gold_answer,
            category=qa.category,
            budget_tokens=budget,
            retrieved_token_count=token_count,
            generated_answer=generated,
            judge_verdict=verdict,
            token_f1=f1,
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

    1. Parse the dataset
    2. Ingest all conversations into RagZoom
    3. Wait for indexing to complete
    4. For each budget, evaluate all non-adversarial QA pairs
    5. Aggregate and return results
    """
    # 1. Parse
    conversations = parse_locomo_file(config.data_path)
    logger.info("Parsed %d conversations", len(conversations))

    # 2. Ingest (skip if docs are already indexed)
    rz = RagZoom(server_address=config.server_address)
    if config.skip_ingest:
        logger.info("Skipping ingestion (--skip-ingest)")
    else:
        ingest_all(rz, conversations)
        wait_for_indexing(rz, conversations)

    # 4. Set up LLM clients
    openai_client = AsyncOpenAI()
    answer_model = OpenAIChatModel(openai_client, config.answer_model)
    judge: OpenAIChatModel | None = None
    if not config.f1_only:
        judge = OpenAIChatModel(openai_client, config.judge_model)

    # 5. Collect non-adversarial QA pairs
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
        "Evaluating %d questions at %d budgets (%d total evaluations)",
        len(qa_items),
        len(config.budgets),
        len(qa_items) * len(config.budgets),
    )

    # 6. Sweep budgets
    all_results: list[AnswerResult] = []
    semaphore = asyncio.Semaphore(config.max_concurrent)

    for budget in config.budgets:
        logger.info("Starting budget=%d", budget)
        tasks = [
            _evaluate_one(rz, answer_model, judge, doc_id, qa, budget, semaphore)
            for doc_id, qa in qa_items
        ]
        budget_results = await asyncio.gather(*tasks)
        all_results.extend(budget_results)

        # Log intermediate results
        avg_f1 = mean(r.token_f1 for r in budget_results)
        if judge is not None:
            correct = sum(1 for r in budget_results if r.judge_verdict == "A")
            logger.info(
                "Budget=%d: %d/%d correct (%.1f%%), F1=%.3f",
                budget,
                correct,
                len(budget_results),
                100.0 * correct / len(budget_results),
                avg_f1,
            )
        else:
            logger.info("Budget=%d: F1=%.3f (f1-only)", budget, avg_f1)

    # 7. Aggregate
    budget_curve = [_aggregate_budget(all_results, b) for b in config.budgets]

    return BenchmarkReport(
        answer_model=config.answer_model,
        judge_model=config.judge_model,
        num_conversations=len(conversations),
        num_questions=len(qa_items),
        budget_curve=budget_curve,
        per_question=all_results,
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
    judge_model = OpenAIChatModel(openai_client, config.judge_model)
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def _rejudge_one(entry: dict[str, object]) -> AnswerResult:
        question = str(entry["question"])
        gold_answer = str(entry["gold_answer"])
        generated_answer = str(entry["generated_answer"])
        async with semaphore:
            verdict = await judge_answer(
                judge_model, question, gold_answer, generated_answer
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
