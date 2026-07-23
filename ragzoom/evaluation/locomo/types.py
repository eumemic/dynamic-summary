"""Data types and JSON parsing for the LoCoMo benchmark."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal

# Judge verdicts: A=correct, B=incorrect, C=not attempted
JudgeVerdict = Literal["A", "B", "C"]


class QACategory(IntEnum):
    """LoCoMo question categories (1-indexed in the actual dataset)."""

    SINGLE_HOP = 1
    MULTI_HOP = 2
    TEMPORAL = 3
    OPEN_DOMAIN = 4
    ADVERSARIAL = 5


@dataclass(frozen=True)
class Turn:
    """A single dialogue turn."""

    dia_id: str
    content: str  # e.g. "[CAROLINE]: Hey Mel!"


@dataclass(frozen=True)
class Session:
    """A conversation session with its turns and timestamp."""

    index: int
    turns: tuple[Turn, ...]
    timestamp: str  # ISO 8601 from session_N_date_time


@dataclass(frozen=True)
class QAPair:
    """A question-answer pair with category and evidence."""

    question: str
    gold_answer: str
    category: QACategory
    evidence_ids: tuple[str, ...]
    sample_id: str  # back-reference to conversation


@dataclass(frozen=True)
class LoCoMoConversation:
    """A parsed LoCoMo conversation with sessions and QA pairs."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[Session, ...]
    qa_pairs: tuple[QAPair, ...]


# CostMetrics is canonical in ragzoom.agent.protocol; re-exported for
# backward compatibility with code that imports from this module.
from ragzoom.agent.protocol import CostMetrics as CostMetrics  # noqa: E402


@dataclass(frozen=True)
class AnswerResult:
    """Result of evaluating one QA pair."""

    sample_id: str
    question: str
    gold_answer: str
    category: QACategory
    generated_answer: str
    judge_verdict: JudgeVerdict | None  # A=correct, B=incorrect, C=not attempted
    token_f1: float
    cost: CostMetrics
    retrospective: str | None = None  # Populated when --profiling is enabled
    # The formatted tiling text returned by each recall call the answerer
    # made, in call order — i.e. exactly what the answerer saw. Persisted so
    # failure attribution (synthesis vs summary-loss vs retrieval) is possible
    # from results.json alone.
    served_tilings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategoryScore:
    """Aggregated scores for one QA category."""

    accuracy: float | None  # None when running in f1-only mode
    f1: float
    count: int


@dataclass(frozen=True)
class AggregateScores:
    """Aggregated scores across all evaluated questions."""

    overall_accuracy: float | None  # None when running in f1-only mode
    overall_f1: float
    by_category: dict[QACategory, CategoryScore]


@dataclass(frozen=True)
class ConversationMetrics:
    """Timing and size metadata for one ingested conversation."""

    sample_id: str
    num_turns: int
    num_sessions: int
    indexing_duration_seconds: float


@dataclass
class BenchmarkReport:
    """Full benchmark output."""

    answer_model: str
    judge_model: str
    num_conversations: int
    num_questions: int
    scores: AggregateScores
    per_question: list[AnswerResult]
    conversation_metrics: tuple[ConversationMetrics, ...] = ()
    config: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


def _parse_session(
    conversation_dict: dict[str, object], index: int, timestamp: str
) -> Session:
    """Parse a single session from the conversation dict.

    The actual LoCoMo format nests sessions under a ``conversation`` dict
    with turns using ``speaker``/``text`` fields (not ``role``/``content``).
    Turn content is formatted as ``[SPEAKER]: text`` to preserve identity.
    """
    session_list = conversation_dict[f"session_{index}"]
    assert isinstance(session_list, list), f"session_{index} must be a list"

    turns: list[Turn] = []
    for turn_data in session_list:
        assert isinstance(turn_data, dict)
        speaker = str(turn_data.get("speaker", "Unknown"))
        text = str(turn_data.get("text", ""))
        content = f"[{speaker.upper()}]: {text}"
        dia_id = str(turn_data.get("dia_id", f"D{index}:{len(turns) + 1}"))
        turns.append(Turn(dia_id=dia_id, content=content))

    return Session(index=index, turns=tuple(turns), timestamp=timestamp)


def _parse_qa_pairs(
    raw_qa: list[dict[str, object]], sample_id: str
) -> tuple[QAPair, ...]:
    """Parse QA pairs from the raw JSON data."""
    pairs: list[QAPair] = []
    for qa in raw_qa:
        category_val = qa.get("category")
        if category_val is None:
            continue  # skip QA pairs without a category
        assert isinstance(
            category_val, int
        ), f"category must be int, got {type(category_val)}"

        evidence = qa.get("evidence", [])
        assert isinstance(evidence, list)

        # Category 5 (adversarial) uses "adversarial_answer" instead of "answer"
        gold = qa.get("answer") or qa.get("adversarial_answer")
        if gold is None:
            continue  # skip QA pairs with no answer at all

        pairs.append(
            QAPair(
                question=str(qa["question"]),
                gold_answer=str(gold),
                category=QACategory(category_val),
                evidence_ids=tuple(str(e) for e in evidence),
                sample_id=sample_id,
            )
        )
    return tuple(pairs)


def _parse_conversation(raw: dict[str, object]) -> LoCoMoConversation:
    """Parse a single conversation from the raw JSON data.

    The actual LoCoMo format has sessions nested under a ``conversation``
    dict rather than at the top level.  Speaker names are inferred from
    the first session's turns when ``speaker_a``/``speaker_b`` keys are
    absent (which is the norm in locomo10.json).
    """
    sample_id = str(raw["sample_id"])

    # Sessions live under the "conversation" sub-dict
    conv_dict = raw.get("conversation", raw)
    assert isinstance(conv_dict, dict), "conversation field must be a dict"

    # Discover session indices by scanning keys
    session_indices: list[int] = sorted(
        int(m.group(1))
        for key in conv_dict
        if (m := _SESSION_KEY_RE.match(key)) is not None
    )

    sessions: list[Session] = []
    for idx in session_indices:
        ts_key = f"session_{idx}_date_time"
        timestamp = str(conv_dict.get(ts_key, ""))
        if not timestamp:
            raise ValueError(f"Missing {ts_key} for {sample_id}")
        sessions.append(_parse_session(conv_dict, idx, timestamp))

    # Infer speaker names from the first session's turns if not at top level
    speaker_a = str(raw.get("speaker_a", ""))
    speaker_b = str(raw.get("speaker_b", ""))
    if not speaker_a and sessions:
        speakers_seen: list[str] = []
        for turn in sessions[0].turns:
            # Extract speaker from "[SPEAKER]: text" format
            if turn.content.startswith("["):
                name = turn.content.split("]")[0][1:]
                if name not in speakers_seen:
                    speakers_seen.append(name)
        if len(speakers_seen) >= 1:
            speaker_a = speakers_seen[0]
        if len(speakers_seen) >= 2:
            speaker_b = speakers_seen[1]

    raw_qa = raw.get("qa", [])
    assert isinstance(raw_qa, list)
    qa_pairs = _parse_qa_pairs(raw_qa, sample_id)

    return LoCoMoConversation(
        sample_id=sample_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=tuple(sessions),
        qa_pairs=qa_pairs,
    )


def parse_locomo_file(path: Path) -> list[LoCoMoConversation]:
    """Parse a LoCoMo JSON file into typed conversation objects.

    The file should contain a list of conversation objects, each with
    session_N, session_N_date_time, and qa fields.
    """
    with open(path) as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        # Some formats use {sample_id: conversation} layout
        conversations = list(raw_data.values())
    elif isinstance(raw_data, list):
        conversations = raw_data
    else:
        raise ValueError(f"Unexpected top-level JSON type: {type(raw_data)}")

    return [_parse_conversation(conv) for conv in conversations]
