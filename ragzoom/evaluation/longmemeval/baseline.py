"""Naive baselines for LongMemEval: answer WITHOUT RagZoom.

These baselines exist to give RagZoom an *external* yardstick. The nano-vs-Sonnet
numbers are an internal RagZoom ablation; they say nothing about whether the
hierarchy beats the dumb alternatives. This harness scores those alternatives on
the *same* sampled questions with the *same* judge, so "RagZoom at budget B vs
just stuffing the history in" becomes a directly comparable number.

Three strategies, each building a context string from a question's haystack and
then asking a held-constant answer model (default the same ``gpt-5-mini`` the
RagZoom runs used):

  * ``full`` — concatenate every session/turn in chronological order, each line
    prefixed with its timestamp and role. No budget; this is the "stuff it all
    in" upper bound (and the arm most likely to overflow the context window).
  * ``truncate`` — keep the most-RECENT turns until a token budget B is reached.
    The honest matched-budget comparison: RagZoom gets budget B, so does this.
  * ``topk`` — chunk turns, embed with the same ``text-embedding-3-small`` the
    harness uses, and retrieve the most-similar chunks to the question until the
    budget B is filled. The flat-RAG comparison: is the hierarchy worth it
    against vanilla top-k retrieval?

A context that overflows the answer model is itself a reportable result ("you
literally cannot stuff variant-M in"): the provider's context-length rejection
is recorded as an explicit ``context_overflow`` outcome (verdict graded as
usual on an answer that says so), never crashed and never silently truncated.

Everything that touches the network — the answer backend, the judge backend,
the embedder — is injected, so the whole pipeline is unit-tested with mocks and
the CLI is the only place that constructs the real clients.
"""

from __future__ import annotations

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
from ragzoom.evaluation.longmemeval.ingest import parse_longmemeval_timestamp
from ragzoom.evaluation.longmemeval.runner import aggregate
from ragzoom.evaluation.longmemeval.scoring import judge_answer
from ragzoom.evaluation.longmemeval.types import (
    AnswerResult,
    BenchmarkReport,
    HaystackMetrics,
    LongMemEvalQuestion,
    Session,
    detect_variant,
    parse_longmemeval_file,
)

logger = logging.getLogger(__name__)

# The same answerer the RagZoom runs held constant, so the only thing that
# changes between "RagZoom at B" and "baseline at B" is the memory strategy.
DEFAULT_ANSWER_MODEL = "gpt-5-mini"
DEFAULT_JUDGE_MODEL = "gpt-4.1"

# A concise, neutral answerer prompt. It must not coach the model toward
# abstention or verbosity — the baseline's job is to expose what a plain
# "read the transcript and answer" call achieves.
ANSWER_SYSTEM_PROMPT = (
    "You are answering a question about a user based on the conversation history "
    "between the user and an AI assistant shown below. Use only the information in "
    "the history. Answer concisely. If the history does not contain enough "
    "information to answer, say that the question cannot be answered from the "
    "history."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BaselineConfig:
    """Configuration for a naive-baseline LongMemEval run.

    Mirrors the fields ``run-longmemeval``'s config records so the two
    ``results.json`` files line up for analysis. ``summary_model`` is repurposed
    as a baseline label (``"baseline:full"`` etc.) because there is no
    summarizer — that label is how an analysis script tells a baseline arm apart
    from a RagZoom arm.
    """

    data_path: Path
    strategy: BaselineStrategy
    budget: int | None
    answer_model: str = DEFAULT_ANSWER_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    sample_size: int | None = None
    max_concurrent: int = 5
    no_judge: bool = False
    output_dir: Path = field(default_factory=lambda: Path("baseline_results"))

    def label(self) -> str:
        """The ``baseline:<strategy>`` label recorded as ``summary_model``."""
        return f"baseline:{self.strategy.value}"

    def to_dict(self) -> dict[str, object]:
        """Serialize config for embedding in results JSON.

        Keeps every field an analysis script keys on (``strategy``, ``budget``,
        ``answer_model``, ``judge_model``, ``dataset_variant``, ``sample_size``,
        and the ``summary_model`` baseline label) and drops transient ones
        (``output_dir``) that do not affect results.
        """
        return {
            "data_path": str(self.data_path),
            "strategy": self.strategy.value,
            "budget": self.budget,
            "answer_model": self.answer_model,
            "judge_model": self.judge_model,
            "embedding_model": self.embedding_model,
            # The baseline label lives where RagZoom records its summarizer, so
            # one analysis script reads both kinds of run.
            "summary_model": self.label(),
            "dataset_variant": detect_variant(self.data_path),
            "sample_size": self.sample_size,
            "max_concurrent": self.max_concurrent,
            "no_judge": self.no_judge,
        }


# ---------------------------------------------------------------------------
# Sampling — byte-for-byte the runner's seed-42 logic
# ---------------------------------------------------------------------------


def sample_questions(
    questions: list[LongMemEvalQuestion], sample_size: int | None
) -> list[LongMemEvalQuestion]:
    """Sample exactly as ``runner._run_benchmark_core`` does (seed 42).

    Delegates to the shared seed-42 sampler so the baseline scores the *same*
    question set as ``run-longmemeval --sample N`` — the whole point of the
    comparison.
    """
    return sample_seed42(questions, sample_size)


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------


def _ordered_sessions(question: LongMemEvalQuestion) -> list[Session]:
    """Return the haystack sessions sorted oldest-first by their timestamp.

    Sessions are stored in file order (sometimes newest-first); the baselines
    need a stable chronological order so "concatenate everything in order" and
    "keep the most-recent turns" are well-defined.
    """
    return sorted(
        question.haystack_sessions,
        key=lambda s: parse_longmemeval_timestamp(s.date),
    )


def chronological_turns(question: LongMemEvalQuestion) -> list[str]:
    """Render every turn as one timestamped, role-prefixed line, oldest-first.

    Each line is ``[<iso-timestamp>] <role>: <content>`` so the answerer sees
    both *when* something was said and *who* said it — the temporal signal the
    temporal-reasoning questions need.
    """
    lines: list[str] = []
    for session in _ordered_sessions(question):
        ts = parse_longmemeval_timestamp(session.date)
        for turn in session.turns:
            lines.append(f"[{ts}] {turn.role}: {turn.content.strip()}")
    return lines


def build_context(
    question: LongMemEvalQuestion,
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
    lines = chronological_turns(question)
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


# ---------------------------------------------------------------------------
# Top-k flat retrieval
# ---------------------------------------------------------------------------


async def build_context_topk(
    question: LongMemEvalQuestion,
    *,
    embedder: EmbedderProtocol,
    budget: int,
) -> TopKResult:
    """Flat top-k retrieval: the most query-similar turns until ``budget``.

    Each turn is one chunk (matching the harness's session-leaf granularity at
    turn level); the bare question is the retrieval query. The shared admission
    ranks chunks by cosine similarity and greedily fills the budget, emitting the
    admitted chunks in chronological order so the answerer reads a coherent (if
    sparse) transcript rather than a relevance-sorted jumble.
    """
    return await topk_lines(
        query=question_text_for_embedding(question),
        lines=chronological_turns(question),
        embedder=embedder,
        budget=budget,
    )


def question_text_for_embedding(question: LongMemEvalQuestion) -> str:
    """The text embedded as the retrieval query (the bare question)."""
    return question.question


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------


async def answer_one(
    *,
    answer_backend: BenchmarkingAgent,
    judge: BenchmarkingAgent | None,
    judge_model: str,
    question: LongMemEvalQuestion,
    strategy: BaselineStrategy,
    budget: int | None,
    embedder: EmbedderProtocol | None,
    answer_model: str,
) -> AnswerResult:
    """Evaluate one question with the chosen baseline strategy.

    The shared :func:`serve_and_answer` builds the strategy's context, asks the
    answer model (no temperature override), and records a context-length
    rejection as an explicit ``context_overflow`` outcome. This harness then
    judges the result — the question is still judged (on an answer that states
    the overflow) so the verdict reflects reality and the abstention tripwire is
    never silently scored.
    """
    served = await serve_and_answer(
        answer_backend=answer_backend,
        lines=chronological_turns(question),
        query=question_text_for_embedding(question),
        strategy=strategy,
        budget=budget,
        embedder=embedder,
        answer_model=answer_model,
        answer_system_prompt=ANSWER_SYSTEM_PROMPT,
        build_user_prompt=lambda context: _build_user_prompt(question, context),
        question_label=question.question_id,
    )

    verdict = None
    if judge is not None:
        verdict = await judge_answer(
            judge,
            question.question_type,
            question.question,
            question.answer,
            served.generated_answer,
            is_abstention=question.is_abstention,
            model_id=judge_model,
        )

    return AnswerResult(
        question_id=question.question_id,
        question=question.question,
        gold_answer=question.answer,
        question_type=question.question_type,
        is_abstention=question.is_abstention,
        generated_answer=served.generated_answer,
        judge_verdict=verdict,
        cost=served.cost,
        served_tilings=(served.served_tiling,),
    )


def _build_user_prompt(question: LongMemEvalQuestion, context: str) -> str:
    """Assemble the answerer's user message: history, then the dated question.

    The question carries its ``question_date`` so temporal-reasoning questions
    can be answered "as of" that date, matching the RagZoom runner's framing.
    """
    return (
        "Conversation history:\n"
        f"{context}\n\n"
        f"[Question asked on {question.question_date}]\n"
        f"{question.question}"
    )


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


def _haystack_metrics(
    questions: list[LongMemEvalQuestion],
) -> tuple[HaystackMetrics, ...]:
    """Per-question session/turn counts. Indexing duration is 0 (no ingest)."""
    metrics: list[HaystackMetrics] = []
    for q in questions:
        num_turns = sum(len(s.turns) for s in q.haystack_sessions)
        metrics.append(
            HaystackMetrics(
                question_id=q.question_id,
                num_sessions=len(q.haystack_sessions),
                num_turns=num_turns,
                indexing_duration_seconds=0.0,
            )
        )
    return tuple(metrics)


async def run_baseline(
    config: BaselineConfig,
    *,
    answer_backend: BenchmarkingAgent,
    judge: BenchmarkingAgent | None,
    embedder: EmbedderProtocol | None,
) -> BenchmarkReport:
    """Run a baseline over the sampled questions and return a report.

    Parses the dataset, samples with the runner's exact seed-42 logic, answers
    and judges every question concurrently (bounded by ``max_concurrent``), and
    aggregates with the runner's own :func:`aggregate` so the scores are
    computed identically to a RagZoom run. The backends/judge/embedder are
    injected so this is fully unit-testable; the CLI builds the real ones.

    On the first failure every outstanding task is cancelled — a broken run
    must not keep spending, and must never substitute a fabricated answer.
    """
    variant = detect_variant(config.data_path)
    questions = parse_longmemeval_file(config.data_path)
    logger.info("Parsed %d questions (variant=%s)", len(questions), variant)

    questions = sample_questions(questions, config.sample_size)
    logger.info(
        "Evaluating %d questions (strategy=%s)", len(questions), config.strategy
    )

    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def _one(q: LongMemEvalQuestion) -> AnswerResult:
        async with semaphore:
            return await answer_one(
                answer_backend=answer_backend,
                judge=judge,
                judge_model=config.judge_model,
                question=q,
                strategy=config.strategy,
                budget=config.budget,
                embedder=embedder,
                answer_model=config.answer_model,
            )

    tasks = [
        asyncio.create_task(_one(q), name=f"baseline-{q.question_id}")
        for q in questions
    ]
    try:
        results = list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    if judge is not None:
        correct = sum(1 for r in results if r.judge_verdict == "yes")
        logger.info(
            "%d/%d correct (%.1f%%)",
            correct,
            len(results),
            100.0 * correct / len(results) if results else 0.0,
        )

    return BenchmarkReport(
        answer_model=config.answer_model,
        judge_model=config.judge_model,
        dataset_variant=variant,
        num_questions=len(questions),
        scores=aggregate(results),
        per_question=results,
        haystack_metrics=_haystack_metrics(questions),
        config=config.to_dict(),
    )
