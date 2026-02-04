"""Ingest LoCoMo conversations into RagZoom as temporal documents."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

from ragzoom.evaluation.locomo.types import LoCoMoConversation
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


def ingest_conversation(rz: RagZoom, conv: LoCoMoConversation) -> None:
    """Ingest one conversation into RagZoom as a temporal document.

    Clears any existing document first, then batch-appends all turns
    with one AppendUnit per turn. Each turn gets its session's timestamp
    converted from LoCoMo's human-readable format to ISO 8601.
    """
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
    logger.info(
        "Ingested %s: %d turns across %d sessions",
        did,
        len(units),
        len(conv.sessions),
    )


def ingest_all(rz: RagZoom, conversations: list[LoCoMoConversation]) -> None:
    """Ingest all conversations. Idempotent via clear-then-append."""
    for conv in conversations:
        ingest_conversation(rz, conv)


def wait_for_indexing(
    rz: RagZoom,
    conversations: list[LoCoMoConversation],
    poll_interval: float = 2.0,
) -> None:
    """Block until all conversations are fully indexed.

    Polls document status until completion_pct >= 100 for every document.
    """
    pending = {doc_id_for(conv) for conv in conversations}

    while pending:
        still_pending: set[str] = set()
        for did in pending:
            status = rz.get_document_status(did)
            if status.completion_pct < 100.0:
                still_pending.add(did)

        if still_pending:
            logger.info(
                "Waiting for indexing: %d/%d documents pending",
                len(still_pending),
                len(pending),
            )
            time.sleep(poll_interval)

        pending = still_pending

    logger.info("All %d documents fully indexed", len(conversations))
