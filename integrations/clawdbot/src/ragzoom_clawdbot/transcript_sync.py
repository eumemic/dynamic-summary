"""Clawdbot transcript sync adapter for ragzoom.

Reads Clawdbot session JSONL files, groups messages into conversation turns,
transcribes them to readable text, and ingests them into ragzoom as temporal
leaf nodes.

Clawdbot sessions are linear (no branching/revert), so this adapter uses
a simpler sync model than the Claude Code adapter: track the last synced
message ID and append new turns.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from claude_transcriber import Transcriber

from ragzoom.wrapper import AppendUnit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clawdbot JSONL → Claude Code format normalization
# ---------------------------------------------------------------------------


def normalize_clawdbot_entry(entry: dict[str, object]) -> dict[str, object] | None:
    """Convert a Clawdbot JSONL entry to Claude Code format.

    Clawdbot wraps messages as:
        {"type": "message", "id": "...", "parentId": "...", "message": {"role": "user|assistant|toolResult", ...}}

    Claude Code uses:
        {"type": "user|assistant", "uuid": "...", "parentUuid": "...", "message": {...}}

    Returns None for non-message entries (session, model_change, custom, etc.)
    """
    if entry.get("type") != "message":
        return None

    msg = entry.get("message", {})
    if not isinstance(msg, dict):
        return None
    role = msg.get("role", "")
    entry_id = entry.get("id", "")
    parent_id = entry.get("parentId")
    timestamp = entry.get("timestamp", "")

    if role == "user":
        # User messages pass through with type normalization
        return {
            "type": "user",
            "uuid": entry_id,
            "parentUuid": parent_id,
            "timestamp": timestamp,
            "message": _normalize_message_content(msg),
        }

    elif role == "assistant":
        # Assistant messages: normalize toolCall → tool_use in content blocks
        normalized_msg = _normalize_assistant_message(msg)
        return {
            "type": "assistant",
            "uuid": entry_id,
            "parentUuid": parent_id,
            "timestamp": timestamp,
            "message": normalized_msg,
        }

    elif role == "toolResult":
        # Tool results become user messages with toolUseResult
        tool_call_id = msg.get("toolCallId", "")
        content = msg.get("content", [])

        # Extract text from content blocks
        text_content = _extract_text_from_content(content)

        return {
            "type": "user",
            "uuid": entry_id,
            "parentUuid": parent_id,
            "timestamp": timestamp,
            "toolUseResult": {
                "tool_use_id": tool_call_id,
                "content": text_content,
            },
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": text_content,
                    }
                ],
            },
        }

    return None


def _normalize_message_content(msg: dict[str, object]) -> dict[str, object]:
    """Normalize a message dict, keeping role and content."""
    return {
        "role": msg.get("role", "user"),
        "content": msg.get("content", ""),
    }


def _normalize_assistant_message(msg: dict[str, object]) -> dict[str, object]:
    """Normalize assistant message content blocks.

    Converts Clawdbot toolCall blocks to Claude Code tool_use format.
    """
    content = msg.get("content", [])
    if not isinstance(content, list):
        return {"role": "assistant", "content": content}

    normalized_blocks: list[object] = []
    for block in content:
        if not isinstance(block, dict):
            normalized_blocks.append(block)
            continue

        block_type = block.get("type", "")

        if block_type == "toolCall":
            # Convert toolCall → tool_use
            normalized_blocks.append(
                {
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("arguments", {}),
                }
            )
        elif block_type == "thinking":
            # Skip thinking blocks for cleaner transcripts
            continue
        else:
            normalized_blocks.append(block)

    return {"role": "assistant", "content": normalized_blocks}


def _extract_text_from_content(content: object) -> str:
    """Extract text from Clawdbot content blocks."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    texts.append(str(text))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts) if texts else ""

    return str(content)


# ---------------------------------------------------------------------------
# JSONL reading
# ---------------------------------------------------------------------------


def iter_clawdbot_jsonl(
    path: Path,
) -> Iterator[tuple[dict[str, object], int]]:
    """Iterate over entries in a Clawdbot session JSONL file.

    Yields (entry_dict, line_number) tuples.
    """
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), i
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON at line {i}")
                continue


def read_clawdbot_messages(path: Path) -> list[dict[str, object]]:
    """Read all message entries from a Clawdbot JSONL file.

    Returns normalized Claude Code format records with original Clawdbot
    entry IDs preserved.
    """
    messages: list[dict[str, object]] = []
    for entry, _ in iter_clawdbot_jsonl(path):
        normalized = normalize_clawdbot_entry(entry)
        if normalized is not None:
            messages.append(normalized)
    return messages


# ---------------------------------------------------------------------------
# Turn grouping
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A conversation turn: user prompt → assistant response (+ tool calls)."""

    records: list[dict[str, object]]
    """Normalized records in this turn, in order."""

    time_start: str
    """ISO 8601 timestamp of the first message."""

    time_end: str
    """ISO 8601 timestamp of the last message."""

    @property
    def first_id(self) -> str:
        return str(self.records[0].get("uuid", "")) if self.records else ""

    @property
    def last_id(self) -> str:
        return str(self.records[-1].get("uuid", "")) if self.records else ""


def group_into_turns(records: list[dict[str, object]]) -> list[Turn]:
    """Group normalized records into conversation turns.

    A turn starts with a user message that is NOT a tool result, and includes
    all subsequent messages until the next user prompt.
    """
    if not records:
        return []

    turns: list[Turn] = []
    current_turn: list[dict[str, object]] = []

    for record in records:
        is_tool_result = "toolUseResult" in record
        is_user = record.get("type") == "user" and not is_tool_result

        if is_user:
            # Start a new turn
            if current_turn:
                turns.append(_build_turn(current_turn))
            current_turn = [record]
        elif current_turn:
            # Add to current turn
            current_turn.append(record)
        # Skip orphan messages before the first user prompt

    if current_turn:
        turns.append(_build_turn(current_turn))

    return turns


def _build_turn(records: list[dict[str, object]]) -> Turn:
    """Build a Turn from a list of normalized records."""
    first_ts = str(records[0].get("timestamp", ""))
    last_ts = str(records[-1].get("timestamp", ""))
    return Turn(records=records, time_start=first_ts, time_end=last_ts)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe_turn(turn: Turn) -> str:
    """Transcribe a turn to human-readable text using claude_transcriber."""
    transcriber = Transcriber()
    chunks = []

    for record in turn.records:
        result = transcriber.transcribe(record)
        if result:
            chunks.append(result)

    return "\n\n".join(chunks)


def turns_to_append_units(turns: list[Turn]) -> list[AppendUnit]:
    """Convert turns to AppendUnits for ragzoom ingestion."""
    units = []
    for turn in turns:
        text = transcribe_turn(turn)
        if text.strip():
            units.append(
                AppendUnit(
                    text=text,
                    time_start=turn.time_start,
                    time_end=turn.time_end,
                )
            )
    return units


# ---------------------------------------------------------------------------
# Sync state
# ---------------------------------------------------------------------------


@dataclass
class SyncState:
    """Tracks what has been synced to ragzoom."""

    document_id: str
    last_message_id: str | None = None
    """ID of the last synced message."""

    span_end: int = 0
    """Document span position after last sync."""

    turns_synced: int = 0
    """Number of turns synced so far."""

    entries: list[dict[str, object]] = field(default_factory=list)
    """Per-turn tracking entries."""

    @classmethod
    def load(cls, path: Path) -> SyncState | None:
        """Load state from JSONL file. Returns None if not found."""
        if not path.exists():
            return None
        lines = path.read_text().strip().split("\n")
        if not lines or not lines[0]:
            return None

        header = json.loads(lines[0])
        entries = [json.loads(line) for line in lines[1:] if line.strip()]

        return cls(
            document_id=header["document_id"],
            last_message_id=header.get("last_message_id"),
            span_end=header.get("span_end", 0),
            turns_synced=header.get("turns_synced", 0),
            entries=entries,
        )

    def save(self, path: Path) -> None:
        """Save state to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "document_id": self.document_id,
            "last_message_id": self.last_message_id,
            "span_end": self.span_end,
            "turns_synced": self.turns_synced,
        }
        lines = [json.dumps(header)]
        lines.extend(json.dumps(e) for e in self.entries)
        path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Full sync pipeline
# ---------------------------------------------------------------------------


@dataclass
class SyncResult:
    """Result of a sync operation."""

    document_id: str
    new_turns: int
    total_turns: int
    new_span_end: int


def sync_transcript(
    transcript_path: Path,
    state_path: Path,
    client: object,
    document_id: str | None = None,
) -> SyncResult:
    """Sync a Clawdbot transcript to ragzoom.

    Args:
        transcript_path: Path to the Clawdbot session JSONL file
        state_path: Path to store sync state
        client: RagZoom wrapper instance (with batch_append, clear, etc.)
        document_id: Override document ID (defaults to transcript filename stem)

    Returns:
        SyncResult describing what was done
    """
    # Load or create state
    state = SyncState.load(state_path)
    if state is None:
        doc_id = document_id or f"clawdbot-{transcript_path.stem}"
        state = SyncState(document_id=doc_id)
    else:
        if document_id:
            state.document_id = document_id

    # Read all messages
    all_records = read_clawdbot_messages(transcript_path)

    # Group into turns
    all_turns = group_into_turns(all_records)

    # Find new turns (skip already synced ones)
    new_turns = all_turns[state.turns_synced :]

    if not new_turns:
        return SyncResult(
            document_id=state.document_id,
            new_turns=0,
            total_turns=len(all_turns),
            new_span_end=state.span_end,
        )

    # Convert to append units
    units = turns_to_append_units(new_turns)

    if units:
        # Batch append to ragzoom
        batch_append_method = getattr(client, "batch_append")
        batch_append_method(state.document_id, units)

        # Update state
        cumulative_span = state.span_end
        for unit, turn in zip(units, new_turns):
            cumulative_span += len(unit.text)
            state.entries.append(
                {
                    "last_id": turn.last_id,
                    "first_id": turn.first_id,
                    "span_end": cumulative_span,
                }
            )

        state.span_end = cumulative_span
        state.last_message_id = new_turns[-1].last_id

    state.turns_synced = len(all_turns)
    state.save(state_path)

    return SyncResult(
        document_id=state.document_id,
        new_turns=len(new_turns),
        total_turns=len(all_turns),
        new_span_end=state.span_end,
    )
