"""Ingest LongMemEval haystacks into RagZoom as temporal documents.

Each question carries its own multi-session haystack (sessions are *per
question*, not a shared corpus). One RagZoom document is built per question;
each session becomes one ``AppendUnit`` whose text is the session's turns and
whose timestamp is the session date — mirroring the LoCoMo ingest, but at
session rather than turn granularity (a LongMemEval haystack can hold ~500
sessions of many turns each; one unit per session keeps the leaf count and the
ingest cost tractable while still letting the recall agent zoom to verbatim
turns within a session).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

from ragzoom.evaluation.benchmark_common import wait_for_documents_indexed
from ragzoom.evaluation.longmemeval.types import HaystackMetrics, LongMemEvalQuestion
from ragzoom.wrapper import AppendUnit, RagZoom

logger = logging.getLogger(__name__)

# LongMemEval dates look like "2023/04/10 (Mon) 17:50" — date, an optional
# parenthesised weekday, then a 24-hour time. The weekday is informational and
# ignored. A few instances omit the time, so it is optional and defaults to
# midnight.
_LME_TS_RE = re.compile(
    r"(\d{4})/(\d{1,2})/(\d{1,2})"  # YYYY/MM/DD
    r"(?:\s*\([^)]*\))?"  # optional "(Mon)"
    r"(?:\s+(\d{1,2}):(\d{2}))?"  # optional HH:MM
)


def parse_longmemeval_timestamp(ts: str) -> str:
    """Convert a LongMemEval date string to ISO 8601 (UTC).

    Input:  "2023/04/10 (Mon) 17:50"   Output: "2023-04-10T17:50:00+00:00"
    Input:  "2023/04/10"               Output: "2023-04-10T00:00:00+00:00"

    Raises ValueError on an unparseable string — we never silently substitute a
    placeholder date, because a wrong timestamp would corrupt the temporal
    ordering the hierarchy depends on.
    """
    m = _LME_TS_RE.search(ts)
    if not m:
        raise ValueError(f"Cannot parse LongMemEval timestamp: {ts!r}")

    year = int(m.group(1))
    month = int(m.group(2))
    day = int(m.group(3))
    hour = int(m.group(4)) if m.group(4) is not None else 0
    minute = int(m.group(5)) if m.group(5) is not None else 0

    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return dt.isoformat()


def doc_id_for(question: LongMemEvalQuestion) -> str:
    """Return the RagZoom document ID for a question's haystack."""
    return f"lme-{question.question_id}"


def render_session_text(question: LongMemEvalQuestion, session_index: int) -> str:
    """Render one haystack session as the text for a single AppendUnit.

    Each turn is one line, prefixed with its role, so the recall agent can see
    speaker identity within the verbatim leaf.
    """
    session = question.haystack_sessions[session_index]
    lines = [f"{turn.role}: {turn.content.strip()}" for turn in session.turns]
    return "\n".join(lines)


def ingest_haystack(rz: RagZoom, question: LongMemEvalQuestion) -> HaystackMetrics:
    """Ingest one question's haystack into RagZoom as a temporal document.

    Clears any existing document first (idempotent), then batch-appends one
    ``AppendUnit`` per session with the session's date as its timestamp.

    Returns metrics including the wall-clock duration of the append call.
    """
    start = time.monotonic()
    did = doc_id_for(question)
    rz.clear(did)

    units: list[AppendUnit] = []
    num_turns = 0
    for i, session in enumerate(question.haystack_sessions):
        text = render_session_text(question, i)
        if not text:
            # An empty session carries no information and cannot be summarized;
            # skip it rather than appending a blank leaf.
            continue
        iso_ts = parse_longmemeval_timestamp(session.date)
        units.append(AppendUnit(text=text, time_start=iso_ts, time_end=iso_ts))
        num_turns += len(session.turns)

    if not units:
        raise ValueError(
            f"Haystack for {question.question_id} has no non-empty sessions"
        )

    rz.batch_append(did, units)
    elapsed = time.monotonic() - start
    logger.info(
        "Ingested %s: %d sessions (%d turns) (%.1fs)",
        did,
        len(units),
        num_turns,
        elapsed,
    )
    return HaystackMetrics(
        question_id=question.question_id,
        num_sessions=len(units),
        num_turns=num_turns,
        indexing_duration_seconds=elapsed,
    )


def ingest_all(
    rz: RagZoom, questions: list[LongMemEvalQuestion]
) -> tuple[HaystackMetrics, ...]:
    """Ingest every question's haystack. Idempotent via clear-then-append."""
    return tuple(ingest_haystack(rz, q) for q in questions)


def wait_for_indexing(
    rz: RagZoom,
    questions: list[LongMemEvalQuestion],
    metrics: tuple[HaystackMetrics, ...],
    poll_interval: float = 2.0,
) -> tuple[HaystackMetrics, ...]:
    """Block until every haystack is fully indexed (completion_pct >= 100).

    Returns updated metrics whose ``indexing_duration_seconds`` includes the
    summarization/embedding wait time.
    """
    wait_elapsed = wait_for_documents_indexed(
        rz, [doc_id_for(q) for q in questions], poll_interval=poll_interval
    )
    logger.info(
        "All %d haystacks fully indexed (waited %.1fs)", len(questions), wait_elapsed
    )

    return tuple(
        HaystackMetrics(
            question_id=m.question_id,
            num_sessions=m.num_sessions,
            num_turns=m.num_turns,
            indexing_duration_seconds=m.indexing_duration_seconds + wait_elapsed,
        )
        for m in metrics
    )
