"""Naive baselines for Oolong: answer the aggregation benchmark WITHOUT RagZoom.

This is the mirror of the LongMemEval baseline harness, and the decisive test of
RagZoom's thesis. On needle QA, flat top-k RAG already beats RagZoom; RagZoom's
*only* predicted niche is aggregation, where flat top-k structurally cannot
represent a whole-history distribution (it surfaces only the k most-similar
chunks, but an Oolong answer is a statistic over *every* episode). This harness
scores the dumb alternatives on the SAME seed-42 questions, with the SAME
deterministic Oolong metric, as ``run-oolong --sample N`` — so "RagZoom at budget
B vs just stuffing the transcript in" becomes a directly comparable number.

Three strategies build a flat context string from a question's transcript and ask
a held-constant answer model (default the same ``gpt-5-mini`` the RagZoom runs
use), reusing the dataset-agnostic machinery in
:mod:`ragzoom.evaluation.baseline_common`:

  * ``full`` — every episode/turn in chronological (episode) order, no budget;
    the "stuff it all in" upper bound (and the arm most likely to overflow).
  * ``truncate`` — the most-recent lines within token budget B; the honest
    matched-budget comparison (RagZoom gets B, so does this).
  * ``topk`` — chunk the transcript by turn, embed with the same
    ``text-embedding-3-small`` the harness indexes with, retrieve the most-similar
    chunks until B. THE structurally decisive flat-RAG-vs-RagZoom comparison.

Unlike LongMemEval there is NO judge: the answer is scored inline with Oolong's
deterministic metric (``oolong.scoring.score_answer``), reused verbatim from the
runner so the two harnesses score identically. A context that overflows the
answer model is a reportable result — the provider's rejection is recorded as an
explicit ``context_overflow`` outcome and graded as the (unanswerable) loss it
is, never crashed and never silently truncated.

Everything that touches the network — the answer backend, the embedder — is
injected, so the whole pipeline is unit-tested with mocks; the CLI is the only
place that constructs the real clients.
"""

from __future__ import annotations

# jscpd:ignore-start - shared-module import list necessarily overlaps the
# LongMemEval baseline's; the duplication is in *what is imported*, not logic.
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ragzoom.agent.protocol import BenchmarkingAgent
from ragzoom.evaluation.baseline_common import (
    DEFAULT_EMBEDDING_MODEL,
    BaselineStrategy,
    EmbedderProtocol,
    TopKResult,
    full_context,
    sample_seed42,
    serve_and_answer,
    topk_lines,
    truncate_lines_to_budget,
)

# jscpd:ignore-end
from ragzoom.evaluation.oolong.ingest import split_episodes
from ragzoom.evaluation.oolong.loader import CONFIG_DND, load_oolong_jsonl
from ragzoom.evaluation.oolong.runner import aggregate, boxed_question
from ragzoom.evaluation.oolong.scoring import extract_boxed_answer, score_answer
from ragzoom.evaluation.oolong.types import (
    AnswerResult,
    BenchmarkReport,
    OolongQuestion,
)

logger = logging.getLogger(__name__)

# The same answerer the RagZoom runs hold constant, so the only thing that
# changes between "RagZoom at B" and "baseline at B" is the memory strategy.
DEFAULT_ANSWER_MODEL = "gpt-5-mini"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OolongBaselineConfig:
    """Configuration for a naive-baseline Oolong run.

    Mirrors the fields ``run-oolong``'s config records so the two ``results.json``
    files line up for analysis. ``summary_model`` is repurposed as a baseline
    label (``"baseline:topk"`` etc.) because there is no summarizer — that label
    is how an analysis script tells a baseline arm apart from a RagZoom arm.
    """

    data_path: Path
    strategy: BaselineStrategy
    budget: int | None
    config_name: str = CONFIG_DND
    split: str = "test"
    answer_model: str = DEFAULT_ANSWER_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    sample_size: int | None = None
    max_concurrent: int = 5
    output_dir: Path = field(default_factory=lambda: Path("oolong_baseline_results"))

    def label(self) -> str:
        """The ``baseline:<strategy>`` label recorded as ``summary_model``."""
        return f"baseline:{self.strategy.value}"

    def to_dict(self) -> dict[str, object]:
        """Serialize config for embedding in results JSON.

        Keeps every field an analysis script keys on (``strategy``, ``budget``,
        ``answer_model``, ``config_name``/``split``, ``sample_size``, and the
        ``summary_model`` baseline label) and drops transient ones (``output_dir``)
        that do not affect results.
        """
        return {
            "data_path": str(self.data_path),
            "strategy": self.strategy.value,
            "budget": self.budget,
            "answer_model": self.answer_model,
            "embedding_model": self.embedding_model,
            # The baseline label lives where RagZoom records its summarizer, so
            # one analysis script reads both kinds of run.
            "summary_model": self.label(),
            "config_name": self.config_name,
            "split": self.split,
            "sample_size": self.sample_size,
            "max_concurrent": self.max_concurrent,
        }


# ---------------------------------------------------------------------------
# Sampling — the runner's exact seed-42 logic
# ---------------------------------------------------------------------------


def sample_questions(
    questions: list[OolongQuestion], sample_size: int | None
) -> list[OolongQuestion]:
    """Sample exactly as ``runner._run_benchmark_core`` does (seed 42).

    Delegates to the shared seed-42 sampler so the baseline scores the *same*
    question set as ``run-oolong --sample N`` — the whole point of the comparison.
    """
    return sample_seed42(questions, sample_size)


# ---------------------------------------------------------------------------
# Context construction — render the transcript into chronological lines
# ---------------------------------------------------------------------------


def chronological_lines(question: OolongQuestion) -> list[str]:
    """Render the transcript as one line per turn, oldest episode first.

    The instruction preamble is stripped (it is task framing, not transcript —
    exactly what ingest excludes from the memory tree), each episode is prefixed
    with an ``=== Episode N ===`` marker so the answerer can reason per-episode
    (the temporal signal multi-doc questions need), and each turn becomes one
    line so a turn is the unit of top-k retrieval.
    """
    lines: list[str] = []
    for index, episode in enumerate(split_episodes(question.context_window_text), 1):
        lines.append(f"=== Episode {index} ===")
        for turn in episode.splitlines():
            stripped = turn.strip()
            if stripped:
                lines.append(stripped)
    return lines


def build_context(
    question: OolongQuestion,
    strategy: BaselineStrategy,
    *,
    budget: int | None,
) -> str:
    """Build the answerer's context for the non-retrieval strategies.

    ``full`` ignores ``budget`` entirely (it is the unbounded upper bound).
    ``truncate`` requires a budget — calling it without one is a configuration
    error, not something to paper over with an arbitrary default. ``topk`` is
    asynchronous (it embeds) and lives in :func:`build_context_topk`.
    """
    lines = chronological_lines(question)
    if strategy is BaselineStrategy.FULL:
        return full_context(lines)
    if strategy is BaselineStrategy.TRUNCATE:
        if budget is None:
            raise ValueError("truncate strategy requires a token budget")
        return truncate_lines_to_budget(lines, budget)
    raise ValueError(
        f"build_context does not handle {strategy.value!r}; "
        "use build_context_topk for top-k retrieval"
    )


async def build_context_topk(
    question: OolongQuestion,
    *,
    embedder: EmbedderProtocol,
    budget: int,
) -> TopKResult:
    """Flat top-k retrieval: the most query-similar turns until ``budget``.

    The retrieval query is the bare question. On an aggregation question this is
    exactly where flat top-k is expected to fail: it can only surface the k
    turns most similar to the question, but the answer counts events across the
    *whole* transcript — so the admitted chunks are a biased sample, not the
    distribution the gold statistic is computed over.
    """
    return await topk_lines(
        query=question.question,
        lines=chronological_lines(question),
        embedder=embedder,
        budget=budget,
    )


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = (
    "You are analyzing a transcript of a Dungeons & Dragons game to answer a "
    "question about its statistics. The answer is an aggregate over the whole "
    "transcript — a count, a per-episode list, or a most/least-common label. "
    "Use only the transcript shown below, survey it exhaustively, count exactly, "
    "and never estimate."
)


def _build_user_prompt(question: OolongQuestion, context: str) -> str:
    """Assemble the answerer's user message: transcript, then the boxed question.

    The question is framed with :func:`boxed_question` — the same mandatory
    ``\\boxed{}`` directive the RagZoom Oolong agent uses — so the deterministic
    scorer can parse the answer out identically for both kinds of run.
    """
    return "Game transcript:\n" f"{context}\n\n" f"{boxed_question(question)}"


async def answer_one(
    *,
    answer_backend: BenchmarkingAgent,
    question: OolongQuestion,
    strategy: BaselineStrategy,
    budget: int | None,
    embedder: EmbedderProtocol | None,
    answer_model: str,
) -> AnswerResult:
    """Evaluate one question with the chosen baseline strategy.

    The shared :func:`serve_and_answer` builds the strategy's context, calls the
    answerer (no temperature override), and records a context-length rejection as
    an explicit ``context_overflow`` outcome. This harness then scores the result
    with the reused deterministic Oolong metric — there is no judge, and the
    overflow text parses to no boxed answer, so an overflow is graded as the loss
    it is (never crashed, never silently truncated).
    """
    served = await serve_and_answer(
        answer_backend=answer_backend,
        lines=chronological_lines(question),
        query=question.question,
        strategy=strategy,
        budget=budget,
        embedder=embedder,
        answer_model=answer_model,
        answer_system_prompt=ANSWER_SYSTEM_PROMPT,
        build_user_prompt=lambda context: _build_user_prompt(question, context),
        question_label=question.id,
    )

    return AnswerResult(
        question_id=question.id,
        question=question.question,
        gold_answer=question.answer,
        question_type=question.question_type,
        generated_answer=served.generated_answer,
        parsed_answer=extract_boxed_answer(served.generated_answer),
        score=score_answer(gold=question.answer, model_output=served.generated_answer),
        cost=served.cost,
        served_tilings=(served.served_tiling,),
    )


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


async def run_baseline(
    config: OolongBaselineConfig,
    *,
    answer_backend: BenchmarkingAgent,
    embedder: EmbedderProtocol | None,
) -> BenchmarkReport:
    """Run a baseline over the sampled questions and return a report.

    Loads the parquet, samples with the runner's exact seed-42 logic, answers and
    scores every question concurrently (bounded by ``max_concurrent``), and
    aggregates with the runner's own :func:`aggregate` so the scores are computed
    identically to a RagZoom run. The backend/embedder are injected so this is
    fully unit-testable; the CLI builds the real ones. There is no ingest and no
    server — the baseline never touches RagZoom.

    On the first failure every outstanding task is cancelled — a broken run must
    not keep spending, and must never substitute a fabricated answer.
    """
    questions = load_oolong_jsonl(config.data_path)
    logger.info(
        "Loaded %d questions (%s/%s)",
        len(questions),
        config.config_name,
        config.split,
    )

    questions = sample_questions(questions, config.sample_size)
    logger.info(
        "Evaluating %d questions (strategy=%s)", len(questions), config.strategy
    )

    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def _one(q: OolongQuestion) -> AnswerResult:
        async with semaphore:
            return await answer_one(
                answer_backend=answer_backend,
                question=q,
                strategy=config.strategy,
                budget=config.budget,
                embedder=embedder,
                answer_model=config.answer_model,
            )

    tasks = [
        asyncio.create_task(_one(q), name=f"oolong-baseline-{q.id}") for q in questions
    ]
    try:
        results = list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    scores = aggregate(results)
    if scores.overall_score is not None:
        logger.info(
            "Overall score %.1f%% (task-avg %.1f%%)",
            100.0 * scores.overall_score,
            100.0 * (scores.task_averaged_score or 0.0),
        )

    return BenchmarkReport(
        answer_model=config.answer_model,
        config_name=config.config_name,
        split=config.split,
        num_questions=len(questions),
        scores=scores,
        per_question=results,
        window_metrics=(),
        config=config.to_dict(),
    )
