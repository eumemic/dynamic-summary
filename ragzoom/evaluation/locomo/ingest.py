"""Ingest LoCoMo conversations into RagZoom as temporal documents."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

from ragzoom.evaluation.benchmark_common import wait_for_documents_indexed
from ragzoom.evaluation.locomo.types import ConversationMetrics, LoCoMoConversation
from ragzoom.wrapper import AppendUnit, RagZoom

logger = logging.getLogger(__name__)

# Pattern: "1:56 pm on 8 May, 2023" or "12:24 am on 7 April, 2023"
_LOCOMO_TS_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _parse_locomo_timestamp(ts: str) -> str:
    """Convert LoCoMo's human-readable timestamps to ISO 8601.

    Input:  "1:56 pm on 8 May, 2023"
    Output: "2023-05-08T13:56:00Z"
    """
    m = _LOCOMO_TS_RE.search(ts)
    if not m:
        raise ValueError(f"Cannot parse LoCoMo timestamp: {ts!r}")

    hour = int(m.group(1))
    minute = int(m.group(2))
    ampm = m.group(3).lower()
    day = int(m.group(4))
    month_name = m.group(5).lower()
    year = int(m.group(6))

    month = _MONTH_MAP.get(month_name)
    if month is None:
        raise ValueError(f"Unknown month: {m.group(5)!r} in timestamp {ts!r}")

    # Convert 12-hour to 24-hour
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return dt.isoformat()


def doc_id_for(conv: LoCoMoConversation) -> str:
    """Return the RagZoom document ID for a LoCoMo conversation."""
    return f"locomo-{conv.sample_id}"


def ingest_conversation(rz: RagZoom, conv: LoCoMoConversation) -> ConversationMetrics:
    """Ingest one conversation into RagZoom as a temporal document.

    Clears any existing document first, then batch-appends all turns
    with one AppendUnit per turn. Each turn gets its session's timestamp
    converted from LoCoMo's human-readable format to ISO 8601.

    Returns metrics including wall-clock duration of the API calls.
    """
    start = time.monotonic()
    did = doc_id_for(conv)
    rz.clear(did)

    units: list[AppendUnit] = []
    for session in conv.sessions:
        iso_ts = _parse_locomo_timestamp(session.timestamp)
        for turn in session.turns:
            units.append(
                AppendUnit(
                    text=turn.content,
                    time_start=iso_ts,
                    time_end=iso_ts,
                )
            )

    if not units:
        raise ValueError(f"Conversation {conv.sample_id} has no turns")

    rz.batch_append(did, units)
    elapsed = time.monotonic() - start
    logger.info(
        "Ingested %s: %d turns across %d sessions (%.1fs)",
        did,
        len(units),
        len(conv.sessions),
        elapsed,
    )
    return ConversationMetrics(
        sample_id=conv.sample_id,
        num_turns=len(units),
        num_sessions=len(conv.sessions),
        indexing_duration_seconds=elapsed,
    )


def ingest_all(
    rz: RagZoom, conversations: list[LoCoMoConversation]
) -> tuple[ConversationMetrics, ...]:
    """Ingest all conversations. Idempotent via clear-then-append."""
    return tuple(ingest_conversation(rz, conv) for conv in conversations)


def wait_for_indexing(
    rz: RagZoom,
    conversations: list[LoCoMoConversation],
    metrics: tuple[ConversationMetrics, ...],
    poll_interval: float = 2.0,
) -> tuple[ConversationMetrics, ...]:
    """Block until all conversations are fully indexed.

    Polls document status until completion_pct >= 100 for every document.
    Returns updated metrics with indexing duration that includes the
    summarization/embedding wait time.
    """
    wait_elapsed = wait_for_documents_indexed(
        rz, [doc_id_for(conv) for conv in conversations], poll_interval=poll_interval
    )
    logger.info(
        "All %d documents fully indexed (waited %.1fs)",
        len(conversations),
        wait_elapsed,
    )

    # Update each conversation's duration to include the wait time
    return tuple(
        ConversationMetrics(
            sample_id=m.sample_id,
            num_turns=m.num_turns,
            num_sessions=m.num_sessions,
            indexing_duration_seconds=m.indexing_duration_seconds + wait_elapsed,
        )
        for m in metrics
    )
