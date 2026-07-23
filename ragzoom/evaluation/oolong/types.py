"""Data types and record parsing for the Oolong (oolong-real) benchmark.

Each oolong-real record is one (question, context-window) pair drawn from the
``oolongbench/oolong-real`` dataset. The context window is a block of one to
many D&D episode transcripts; the question asks for an aggregate statistic over
it (a count, an ordered list, a most/least-common label). The gold ``answer`` is
a single string whose *form* (integer / label / comma-list) determines how it is
scored — see ``scoring.py``.

Reference: arXiv:2511.02817 / github.com/abertsch72/oolong (oolong-real `dnd`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# CostMetrics is canonical in ragzoom.agent.protocol; re-exported here so the
# rest of the harness imports it from one place (mirrors the LongMemEval types).
from ragzoom.agent.protocol import CostMetrics as CostMetrics  # noqa: F401


class QuestionType(str, Enum):
    """Oolong-real question categories (the ``question_type`` field).

    The four values are the cross product of {single-episode, multi-episode} ×
    {dice rolls, spells}. ``singledoc_*`` questions aggregate over one episode;
    ``multidoc_*`` questions aggregate across the whole multi-episode window and
    must also reason about *episode order* (e.g. "the last spell cast in each
    episode") — which is why ingest encodes episode order as temporal metadata.
    """

    SINGLEDOC_ROLLS = "singledoc_rolls"
    SINGLEDOC_SPELLS = "singledoc_spells"
    MULTIDOC_ROLLS = "multidoc_rolls"
    MULTIDOC_SPELLS = "multidoc_spells"


# Human-readable names for report tables.
QUESTION_TYPE_NAMES: dict[QuestionType, str] = {
    QuestionType.SINGLEDOC_ROLLS: "Counting (single-doc, rolls)",
    QuestionType.SINGLEDOC_SPELLS: "Counting (single-doc, spells)",
    QuestionType.MULTIDOC_ROLLS: "Aggregation (multi-doc, rolls)",
    QuestionType.MULTIDOC_SPELLS: "Aggregation (multi-doc, spells)",
}


@dataclass(frozen=True)
class OolongQuestion:
    """A single oolong-real evaluation instance.

    ``context_window_text`` carries the upstream instruction preamble, the
    player→character mapping, and the ``[START OF EPISODE]``-delimited
    transcript(s). The preamble is stripped before ingest (it is task framing,
    not conversation); see ``ingest.strip_instruction_preamble``.
    """

    id: str
    context_window_id: str
    context_window_text: str
    question: str
    answer: str  # gold answer string; its form drives scoring
    question_type: QuestionType
    episodes: tuple[int, ...]
    campaign: str


@dataclass(frozen=True)
class AnswerResult:
    """Result of evaluating one question.

    Unlike LongMemEval's binary ``judge_verdict``, Oolong records a continuous
    ``score`` in [0, 1] (partial credit for near-miss counts) and the
    ``parsed_answer`` actually extracted from the model's ``\\boxed{}`` output,
    so a low score can be attributed to a parse failure vs a wrong answer.
    """

    question_id: str
    question: str
    gold_answer: str
    question_type: QuestionType
    generated_answer: str  # the raw agent output
    parsed_answer: str  # what was extracted from \boxed{} (or raw fallback)
    score: float  # Oolong metric in [0, 1]
    cost: CostMetrics
    retrospective: str | None = None  # populated only with --profiling
    # The formatted tiling text returned by each recall call the answerer made,
    # in call order — exactly what the answerer saw. Persisted so failure
    # attribution (synthesis vs summary-loss vs retrieval) is possible from
    # results.json alone. Always present (possibly empty).
    served_tilings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategoryScore:
    """Aggregated score for one question type."""

    score: float | None  # None when there are no questions of this type
    count: int


@dataclass(frozen=True)
class AggregateScores:
    """Aggregated scores across all evaluated questions.

    ``overall_score`` weights every question equally. ``task_averaged_score``
    averages the per-type scores (the metric the paper reports as a "task
    average"), so types with few questions are not drowned out.
    """

    overall_score: float | None
    task_averaged_score: float | None
    by_type: dict[QuestionType, CategoryScore]


@dataclass(frozen=True)
class WindowMetrics:
    """Timing and size metadata for one ingested context window."""

    context_window_id: str
    num_episodes: int
    indexing_duration_seconds: float


@dataclass
class BenchmarkReport:
    """Full benchmark output."""

    answer_model: str
    config_name: str  # "dnd" or "toy_dnd"
    split: str  # "validation" or "test"
    num_questions: int
    scores: AggregateScores
    per_question: list[AnswerResult]
    window_metrics: tuple[WindowMetrics, ...] = ()
    config: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_oolong_records(records: list[dict[str, object]]) -> list[OolongQuestion]:
    """Parse raw oolong-real records (dicts) into typed question objects.

    A record with an unrecognized ``question_type`` fails hard rather than being
    silently dropped — a missing category would corrupt the per-type breakdown
    the aggregation analysis depends on.
    """
    return [_parse_record(r) for r in records]


def _parse_record(raw: dict[str, object]) -> OolongQuestion:
    episodes_raw = raw["episodes"]
    assert isinstance(episodes_raw, list | tuple)

    return OolongQuestion(
        id=str(raw["id"]),
        context_window_id=str(raw["context_window_id"]),
        context_window_text=str(raw["context_window_text"]),
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        question_type=QuestionType(str(raw["question_type"])),
        episodes=tuple(int(e) for e in episodes_raw),
        campaign=str(raw["campaign"]),
    )
