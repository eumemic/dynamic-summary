"""Data types and JSON parsing for the LongMemEval benchmark.

LongMemEval evaluates chat assistants on long-term interactive memory.
Each instance bundles a question with its full multi-session haystack — the
conversation sessions the system must search through, ranging from ~3 (oracle)
to ~500 (the ``_M`` variant) sessions per question. The ``_S`` (~115K tokens)
and ``_M`` (~1.5M tokens) variants are structurally identical JSON; they differ
only in haystack size, so the same parser handles both.

Reference: arXiv:2410.10813 / github.com/xiaowu0162/LongMemEval
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

# Judge verdicts: yes=correct, no=incorrect (LongMemEval is binary, unlike
# LoCoMo's three-way A/B/C). Abstention questions reuse the same yes/no axis:
# yes = the model correctly recognized the question as unanswerable.
JudgeVerdict = Literal["yes", "no"]


class QuestionType(str, Enum):
    """LongMemEval question categories.

    Abstention is *not* a question type — it is a flag (``is_abstention``)
    orthogonal to type, signalled by an ``_abs`` suffix on the question id.
    An abstention question still carries one of these types but its gold
    "answer" is an explanation of why the question is unanswerable.
    """

    SINGLE_SESSION_USER = "single-session-user"
    SINGLE_SESSION_ASSISTANT = "single-session-assistant"
    SINGLE_SESSION_PREFERENCE = "single-session-preference"
    MULTI_SESSION = "multi-session"
    TEMPORAL_REASONING = "temporal-reasoning"
    KNOWLEDGE_UPDATE = "knowledge-update"


# Human-readable names for report tables.
QUESTION_TYPE_NAMES: dict[QuestionType, str] = {
    QuestionType.SINGLE_SESSION_USER: "Single-session (user)",
    QuestionType.SINGLE_SESSION_ASSISTANT: "Single-session (assistant)",
    QuestionType.SINGLE_SESSION_PREFERENCE: "Single-session (preference)",
    QuestionType.MULTI_SESSION: "Multi-session",
    QuestionType.TEMPORAL_REASONING: "Temporal reasoning",
    QuestionType.KNOWLEDGE_UPDATE: "Knowledge update",
}


# CostMetrics is canonical in ragzoom.agent.protocol; re-exported for symmetry
# with the LoCoMo harness and so report.py can import it from one place.
from ragzoom.agent.protocol import CostMetrics as CostMetrics  # noqa: E402


@dataclass(frozen=True)
class Turn:
    """A single conversation turn within a session."""

    role: str  # "user" or "assistant"
    content: str
    has_answer: bool = False  # True if this turn contains gold evidence


@dataclass(frozen=True)
class Session:
    """A conversation session with its turns and timestamp."""

    session_id: str
    date: str  # raw LongMemEval format, e.g. "2023/04/10 (Mon) 17:50"
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class LongMemEvalQuestion:
    """A single LongMemEval evaluation instance.

    Each instance bundles a question with its full haystack context — every
    conversation session the system must search through.
    """

    question_id: str
    question_type: QuestionType
    question: str
    answer: str
    question_date: str  # raw format, e.g. "2023/04/10 (Mon) 23:07"
    haystack_sessions: tuple[Session, ...]
    answer_session_ids: tuple[str, ...]  # ground-truth evidence session ids
    is_abstention: bool  # True for unanswerable questions (_abs id suffix)


@dataclass(frozen=True)
class AnswerResult:
    """Result of evaluating one question."""

    question_id: str
    question: str
    gold_answer: str
    question_type: QuestionType
    is_abstention: bool
    generated_answer: str
    judge_verdict: JudgeVerdict | None  # None in --no-judge mode
    cost: CostMetrics
    retrospective: str | None = None  # populated only with --profiling
    # The formatted tiling text returned by each recall call the answerer
    # made, in call order — i.e. exactly what the answerer saw. Persisted so
    # failure attribution (synthesis vs summary-loss vs retrieval) is possible
    # from results.json alone. Always present (possibly empty).
    served_tilings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategoryScore:
    """Aggregated scores for one question type."""

    accuracy: float | None  # None when running in --no-judge mode
    count: int


@dataclass(frozen=True)
class AggregateScores:
    """Aggregated scores across all evaluated questions.

    ``overall_accuracy`` weights every question equally. ``task_averaged_accuracy``
    averages the per-type accuracies (the metric reported in the paper, so types
    with few questions are not drowned out). ``abstention_accuracy`` is computed
    over only the unanswerable questions — a safety tripwire for compaction-induced
    hallucination, kept separate so it never inflates the headline number.
    """

    overall_accuracy: float | None
    task_averaged_accuracy: float | None
    abstention_accuracy: float | None
    by_type: dict[QuestionType, CategoryScore]


@dataclass(frozen=True)
class HaystackMetrics:
    """Timing and size metadata for one ingested haystack."""

    question_id: str
    num_sessions: int
    num_turns: int
    indexing_duration_seconds: float


@dataclass
class BenchmarkReport:
    """Full benchmark output."""

    answer_model: str
    judge_model: str
    dataset_variant: str  # "oracle", "s", "m", or "unknown"
    num_questions: int
    scores: AggregateScores
    per_question: list[AnswerResult]
    haystack_metrics: tuple[HaystackMetrics, ...] = ()
    config: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_session(
    session_data: list[dict[str, object]],
    session_id: str,
    date: str,
) -> Session:
    """Parse a single session from raw JSON turn data."""
    turns: list[Turn] = []
    for turn_data in session_data:
        assert isinstance(turn_data, dict)
        turns.append(
            Turn(
                role=str(turn_data.get("role", "unknown")),
                content=str(turn_data.get("content", "")),
                has_answer=bool(turn_data.get("has_answer", False)),
            )
        )
    return Session(session_id=session_id, date=date, turns=tuple(turns))


def _parse_question(raw: dict[str, object]) -> LongMemEvalQuestion:
    """Parse a single LongMemEval question from raw JSON data.

    The three parallel arrays ``haystack_sessions``, ``haystack_session_ids``,
    and ``haystack_dates`` are zipped positionally — a length mismatch is a
    corrupt instance and fails hard rather than silently truncating.
    """
    question_id = str(raw["question_id"])
    question_type = QuestionType(str(raw["question_type"]))
    question = str(raw["question"])
    answer = str(raw["answer"])
    question_date = str(raw["question_date"])

    haystack_dates = raw["haystack_dates"]
    haystack_session_ids = raw["haystack_session_ids"]
    haystack_sessions_raw = raw["haystack_sessions"]
    answer_session_ids = raw.get("answer_session_ids", [])

    assert isinstance(haystack_dates, list)
    assert isinstance(haystack_session_ids, list)
    assert isinstance(haystack_sessions_raw, list)
    assert isinstance(answer_session_ids, list)

    if not (
        len(haystack_dates) == len(haystack_session_ids) == len(haystack_sessions_raw)
    ):
        raise ValueError(
            f"Haystack arrays misaligned for {question_id}: "
            f"{len(haystack_sessions_raw)} sessions, "
            f"{len(haystack_session_ids)} ids, {len(haystack_dates)} dates"
        )

    sessions: list[Session] = []
    for i, session_data in enumerate(haystack_sessions_raw):
        assert isinstance(session_data, list)
        session_id = str(haystack_session_ids[i])
        date = str(haystack_dates[i])
        sessions.append(_parse_session(session_data, session_id, date))

    # Abstention is flagged by an "_abs" suffix on the question id — matching
    # the official evaluation's convention.
    is_abstention = question_id.endswith("_abs")

    return LongMemEvalQuestion(
        question_id=question_id,
        question_type=question_type,
        question=question,
        answer=answer,
        question_date=question_date,
        haystack_sessions=tuple(sessions),
        answer_session_ids=tuple(str(s) for s in answer_session_ids),
        is_abstention=is_abstention,
    )


def parse_longmemeval_file(path: Path) -> list[LongMemEvalQuestion]:
    """Parse a LongMemEval JSON file into typed question objects.

    Supports both the list layout (the released ``_S``/``_M``/oracle files) and
    a ``{question_id: question}`` dict layout. The same parser handles every
    variant — they differ only in haystack size, not structure.
    """
    with open(path) as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list):
        questions = raw_data
    elif isinstance(raw_data, dict):
        questions = list(raw_data.values())
    else:
        raise ValueError(f"Unexpected top-level JSON type: {type(raw_data)}")

    return [_parse_question(q) for q in questions]


def detect_variant(path: Path) -> str:
    """Detect the dataset variant from the filename.

    Returns one of ``"oracle"``, ``"s"``, ``"m"``, or ``"unknown"``. The
    variant is recorded in results so each run is self-describing about which
    H tier (and therefore which H/B regime) it measured.
    """
    name = path.stem.lower()
    if "oracle" in name:
        return "oracle"
    if "_m" in name:
        return "m"
    if "_s" in name:
        return "s"
    return "unknown"
