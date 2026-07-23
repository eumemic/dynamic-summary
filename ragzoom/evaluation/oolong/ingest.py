"""Ingest Oolong (oolong-real) context windows into RagZoom as temporal docs.

A context window is one block of one-to-many D&D episode transcripts. Each
episode becomes one ``AppendUnit`` whose text is the episode's transcript and
whose timestamp encodes the episode's *order* in the window — mirroring the
LongMemEval ingest (one unit per session, session date as timestamp), but here
the temporal axis is synthesized from episode order because the raw transcripts
carry no per-turn timestamps.

Two things distinguish this from the LongMemEval ingest:

1. **The instruction preamble is stripped.** ``context_window_text`` wraps the
   transcript in task framing (a description of the format, a player→character
   map, and the ``\\boxed{}`` answer directive). That framing is *not*
   conversation and must not be summarized into the memory tree — only the
   episode bodies are ingested.

2. **The document id keys on the context window, not the question.** Many
   questions share one window, so keying the RagZoom document on
   ``context_window_id`` lets a window be ingested once and reused across all
   its questions — the dedupe lever that keeps ingest cost tractable.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone

from ragzoom.evaluation.benchmark_common import wait_for_documents_indexed
from ragzoom.evaluation.oolong.types import OolongQuestion, WindowMetrics
from ragzoom.wrapper import AppendUnit, RagZoom

logger = logging.getLogger(__name__)

_START_MARKER = "[START OF EPISODE]"
_END_MARKER = "[END OF EPISODE]"

# The instruction preamble describes the format *in prose* — "delimited by
# [START OF EPISODE] and [END OF EPISODE]" — so the marker strings also appear
# mid-sentence there. A bare substring search would mistake that mention for the
# real transcript boundary. The genuine markers occupy a whole line on their own;
# these line-anchored patterns match only those, never the inline prose mention.
_START_LINE_RE = re.compile(rf"^[ \t]*{re.escape(_START_MARKER)}[ \t]*$", re.MULTILINE)
_END_LINE_RE = re.compile(rf"^[ \t]*{re.escape(_END_MARKER)}[ \t]*$", re.MULTILINE)

# Episode order is the only temporal signal in oolong-real, so we synthesize a
# monotonic per-episode timestamp from a fixed epoch (one day apart per episode).
# The absolute date is meaningless; only the ordering matters — it lets the
# recall agent reason about "the last spell in each episode" via time windows.
_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)


def episode_timestamp(episode_index: int) -> str:
    """Return a synthetic monotonic ISO-8601 timestamp for an episode index.

    Episode ``i`` maps to ``epoch + i days``. Strictly increasing in ``i`` so the
    hierarchy preserves episode order; the absolute date is not meaningful.
    """
    return (_EPOCH + timedelta(days=episode_index)).isoformat()


# Each ingested leaf must stay under BOTH the server's per-unit cap
# (MAX_UNIT_CHARS = 50000, beyond which it silently truncates in client-managed
# chunking mode) AND the embedding token limit (~8000 tokens). 8000 chars
# (~2000 tokens) gives fine retrieval granularity with comfortable margin on
# both — Oolong episodes run 150K-200K chars, so a whole episode is far too big
# to be one leaf.
_MAX_LEAF_CHARS = 8000


def chunk_episode_text(text: str, max_chars: int = _MAX_LEAF_CHARS) -> list[str]:
    """Split one episode's transcript into <=max_chars leaf chunks on line boundaries.

    Accumulates whole lines until the next would exceed ``max_chars``; a single
    line longer than ``max_chars`` is hard-split so no chunk ever exceeds the cap
    (the server would otherwise silently truncate it, losing transcript content).
    Every non-blank chunk is preserved, in order.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        while len(line) > max_chars:
            if current:
                chunks.append("".join(current))
                current, size = [], 0
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if size + len(line) > max_chars and current:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def leaf_timestamp(episode_index: int, chunk_index: int) -> str:
    """Strictly-increasing ISO timestamp for chunk ``chunk_index`` of an episode.

    Episode ``i`` maps to ``epoch + i days``; chunk ``j`` adds ``j`` seconds, so
    the hierarchy preserves both episode order and within-episode chunk order
    (an episode would need >86400 chunks to collide with the next, far beyond any
    real transcript).
    """
    return (_EPOCH + timedelta(days=episode_index, seconds=chunk_index)).isoformat()


def strip_instruction_preamble(context_window_text: str) -> str:
    """Drop the task-framing preamble, keeping the transcript body onward.

    The body starts at the first ``[START OF EPISODE]`` marker. Everything before
    it (the format description, the player→character mapping, the ``\\boxed{}``
    directive) is task framing, not conversation, and is excluded from ingest.

    Raises ValueError if no episode marker is present — we never ingest the raw
    instruction text as if it were memory.
    """
    m = _START_LINE_RE.search(context_window_text)
    if m is None:
        raise ValueError(
            "Context window has no [START OF EPISODE] marker; cannot locate the "
            "transcript body to ingest."
        )
    return context_window_text[m.start() :]


def split_episodes(context_window_text: str) -> list[str]:
    """Split a context window into per-episode transcript bodies (markers removed).

    Multi-episode windows concatenate episodes, each wrapped in
    ``[START OF EPISODE]`` / ``[END OF EPISODE]``. This returns the inner text of
    each episode, in order, with the markers stripped and whitespace trimmed.

    Raises ValueError if no episode is found.
    """
    body = strip_instruction_preamble(context_window_text)
    episodes: list[str] = []
    for start_match in _START_LINE_RE.finditer(body):
        inner_start = start_match.end()
        end_match = _END_LINE_RE.search(body, inner_start)
        # A final episode may be unterminated; take the remainder in that case.
        inner_end = end_match.start() if end_match is not None else len(body)
        episode = body[inner_start:inner_end].strip()
        if episode:
            episodes.append(episode)

    if not episodes:
        raise ValueError("Context window contains no non-empty episodes.")
    return episodes


def render_episode_text(episode_block: str) -> str:
    """Render one episode (already marker-delimited or inner) as ingest text.

    Idempotent: strips episode markers if present and trims, so it can be called
    on either a raw ``[START OF EPISODE]...`` block or an already-inner body.
    """
    text = episode_block.replace(_START_MARKER, "").replace(_END_MARKER, "")
    return text.strip()


def doc_id_for(question: OolongQuestion) -> str:
    """Return the RagZoom document id for a question's context window.

    Keyed on ``context_window_id`` (not the question id) so the many questions
    that share a window ingest the tree exactly once.
    """
    return f"oolong-{question.context_window_id}"


def ingest_window(rz: RagZoom, question: OolongQuestion) -> WindowMetrics:
    """Ingest one question's context window into RagZoom as a temporal document.

    Clears any existing document first (idempotent), then batch-appends one
    ``AppendUnit`` per episode with a monotonic per-episode timestamp. Returns
    metrics including the wall-clock duration of the append call.
    """
    start = time.monotonic()
    did = doc_id_for(question)
    rz.clear(did)

    episodes = split_episodes(question.context_window_text)
    units: list[AppendUnit] = []
    for i, episode in enumerate(episodes):
        text = render_episode_text(episode)
        if not text:
            continue
        # Oolong episodes (150K-200K chars) far exceed the per-leaf cap, so each
        # episode is chunked into leaf-sized units. Without this the server
        # silently truncates each episode to 50K chars, ingesting only a third
        # of the history RagZoom is meant to compress.
        for j, chunk in enumerate(chunk_episode_text(text)):
            ts = leaf_timestamp(i, j)
            units.append(AppendUnit(text=chunk, time_start=ts, time_end=ts))

    if not units:
        raise ValueError(
            f"Context window {question.context_window_id} has no non-empty episodes"
        )

    rz.batch_append(did, units)
    elapsed = time.monotonic() - start
    logger.info("Ingested %s: %d episodes (%.1fs)", did, len(units), elapsed)
    return WindowMetrics(
        context_window_id=question.context_window_id,
        num_episodes=len(units),
        indexing_duration_seconds=elapsed,
    )


def ingest_unique_windows(
    rz: RagZoom, questions: list[OolongQuestion]
) -> tuple[WindowMetrics, ...]:
    """Ingest each *distinct* context window exactly once.

    Questions sharing a ``context_window_id`` map to one RagZoom document, so we
    ingest the first occurrence of each window and skip the rest — the dedupe
    lever that keeps Oolong ingest affordable (a single 24-episode window can
    back dozens of questions).
    """
    seen: set[str] = set()
    metrics: list[WindowMetrics] = []
    for q in questions:
        if q.context_window_id in seen:
            continue
        seen.add(q.context_window_id)
        metrics.append(ingest_window(rz, q))
    return tuple(metrics)


def wait_for_indexing(
    rz: RagZoom,
    questions: list[OolongQuestion],
    metrics: tuple[WindowMetrics, ...],
    poll_interval: float = 2.0,
) -> tuple[WindowMetrics, ...]:
    """Block until every distinct context window is fully indexed.

    Returns updated metrics whose ``indexing_duration_seconds`` includes the
    summarization/embedding wait time.
    """
    doc_ids = list({doc_id_for(q) for q in questions})
    wait_elapsed = wait_for_documents_indexed(rz, doc_ids, poll_interval=poll_interval)
    logger.info(
        "All %d context windows fully indexed (waited %.1fs)",
        len(doc_ids),
        wait_elapsed,
    )

    return tuple(
        WindowMetrics(
            context_window_id=m.context_window_id,
            num_episodes=m.num_episodes,
            indexing_duration_seconds=m.indexing_duration_seconds + wait_elapsed,
        )
        for m in metrics
    )
