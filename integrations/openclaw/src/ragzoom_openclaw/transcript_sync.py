"""OpenClaw transcript sync adapter for RagZoom.

Reads OpenClaw session JSONL files, transcribes each message step to readable
text, and ingests them into RagZoom as temporal leaf nodes.

Key differences from Claude Code adapter:
- OpenClaw uses id/parentId (not uuid/parentUuid)
- Uses type="message" with message.role (not type="user"/"assistant")
- Has type="compaction" (not isCompactSummary flag)
- Per-step granularity (not turn grouping)
- Preserves thinking blocks with 💭 marker
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ragzoom.wrapper import AppendUnit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL reading
# ---------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[tuple[dict, int]]:
    """Iterate over JSONL records with line numbers."""
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), lineno
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON at line {lineno}")
                continue


def iter_jsonl_reversed(path: Path) -> Iterator[dict]:
    """Iterate over JSONL records in reverse order."""
    records = []
    for record, _ in iter_jsonl(path):
        records.append(record)
    for record in reversed(records):
        yield record


# ---------------------------------------------------------------------------
# OpenClaw format helpers
# ---------------------------------------------------------------------------


def get_uuid(record: dict) -> str | None:
    """Get UUID from OpenClaw record (uses 'id' field)."""
    id_val = record.get("id")
    return id_val if isinstance(id_val, str) else None


def get_parent_uuid(record: dict) -> str | None:
    """Get parent UUID from OpenClaw record (uses 'parentId' field)."""
    parent = record.get("parentId")
    return parent if isinstance(parent, str) else None


def is_compaction_summary(record: dict) -> bool:
    """Check if record is a compaction summary."""
    return record.get("type") == "compaction"


def get_message_role(record: dict) -> str | None:
    """Get role from OpenClaw message record."""
    if record.get("type") != "message":
        return None
    message = record.get("message", {})
    if isinstance(message, dict):
        return message.get("role")
    return None


def should_include_record(record: dict) -> bool:
    """Include only user and assistant messages, excluding meta/compaction."""
    role = get_message_role(record)
    if role not in ("user", "assistant"):
        return False
    if is_compaction_summary(record):
        return False
    return True


# ---------------------------------------------------------------------------
# Transcription (per-step, with thinking preserved)
# ---------------------------------------------------------------------------


def _summarize_tool_call(block: dict) -> str | None:
    """Summarize a tool call for memory-friendly output.
    
    Returns a brief description or None to skip entirely.
    """
    name = block.get("name", "unknown")
    args = block.get("arguments", {})
    
    # File operations — show what was touched
    if name in ("write", "Write"):
        path = args.get("path", args.get("file_path", ""))
        if path:
            filename = path.split("/")[-1]
            return f"[Wrote: {filename}]"
        return "[Wrote file]"
    
    if name in ("read", "Read"):
        path = args.get("path", args.get("file_path", ""))
        if path:
            filename = path.split("/")[-1]
            return f"[Read: {filename}]"
        return None  # Skip reads entirely, usually just context gathering
    
    if name in ("edit", "Edit"):
        path = args.get("path", args.get("file_path", ""))
        if path:
            filename = path.split("/")[-1]
            return f"[Edited: {filename}]"
        return "[Edited file]"
    
    if name == "exec":
        cmd = args.get("command", "")
        if cmd:
            short_cmd = cmd[:50] + "..." if len(cmd) > 50 else cmd
            return f"[Ran: {short_cmd}]"
        return "[Ran command]"
    
    if name == "web_search":
        query = args.get("query", "")
        return f"[Searched: {query}]" if query else "[Web search]"
    
    if name == "web_fetch":
        url = args.get("url", "")
        return f"[Fetched: {url[:50]}...]" if len(url) > 50 else f"[Fetched: {url}]"
    
    if name == "message":
        action = args.get("action", "")
        return f"[Message: {action}]"
    
    if name == "gateway":
        action = args.get("action", "")
        return f"[Gateway: {action}]"
    
    # Internal operations — skip
    if name in ("memory_search", "memory_get"):
        return None
    if name in ("sessions_list", "sessions_history", "sessions_spawn", "session_status"):
        return None
    if name == "process":
        return None
    
    # Default: show tool name
    return f"[Tool: {name}]"


def extract_text_content(content: list | str) -> str:
    """Extract text from message content, preserving thinking blocks."""
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                
                if block_type == "text":
                    text = block.get("text", "")
                    if text.strip():
                        texts.append(text)
                
                elif block_type == "thinking":
                    # Preserve thinking with marker
                    thinking_text = block.get("thinking", "")
                    if thinking_text.strip():
                        texts.append(f"💭 {thinking_text}")
                
                elif block_type == "toolCall":
                    summary = _summarize_tool_call(block)
                    if summary:
                        texts.append(summary)
                
                elif block_type == "toolResult":
                    pass  # Skip tool results entirely
                
            elif isinstance(block, str):
                if block.strip():
                    texts.append(block)
        
        return "\n".join(texts)
    
    return str(content) if content else ""


def transcribe_record(record: dict) -> str:
    """Transcribe a single OpenClaw record to text."""
    role = get_message_role(record)
    if role is None:
        return ""
    
    message = record.get("message", {})
    content = message.get("content", [])
    
    text = extract_text_content(content)
    if not text.strip():
        return ""
    
    role_label = "User" if role == "user" else "Assistant"
    return f"{role_label}: {text}"


# ---------------------------------------------------------------------------
# Step dataclass
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """A single conversation step with metadata."""
    uuid: str
    timestamp: str
    record: dict


# ---------------------------------------------------------------------------
# Ancestry / parent map
# ---------------------------------------------------------------------------


def build_parent_map(transcript_path: Path) -> dict[str, str | None]:
    """Build uuid -> parentUuid map with compaction bridging."""
    parent_map: dict[str, str | None] = {}
    entries: list[tuple[str, str | None, bool]] = []
    
    for record, _ in iter_jsonl(transcript_path):
        uuid = get_uuid(record)
        if uuid is None:
            continue
        parent_uuid = get_parent_uuid(record)
        is_compact = is_compaction_summary(record)
        entries.append((uuid, parent_uuid, is_compact))
    
    last_regular_uuid: str | None = None
    
    for i, (uuid, parent_uuid, is_compact) in enumerate(entries):
        if parent_uuid is None and not is_compact:
            # Check if followed by compaction
            is_followed_by_compact = False
            for j in range(i + 1, len(entries)):
                _, _, next_is_compact = entries[j]
                if next_is_compact:
                    is_followed_by_compact = True
                    break
                _, next_parent, _ = entries[j]
                if next_parent != uuid:
                    break
            
            if is_followed_by_compact and last_regular_uuid is not None:
                parent_map[uuid] = last_regular_uuid
            else:
                parent_map[uuid] = None
        else:
            parent_map[uuid] = parent_uuid
        
        if not is_compact:
            last_regular_uuid = uuid
    
    return parent_map


def build_records_map(transcript_path: Path) -> dict[str, dict]:
    """Build UUID -> record mapping."""
    records = {}
    for record, _ in iter_jsonl(transcript_path):
        uuid = get_uuid(record)
        if uuid is not None:
            records[uuid] = record
    return records


def get_current_head(transcript_path: Path) -> str | None:
    """Get the UUID of the most recent message."""
    for record in iter_jsonl_reversed(transcript_path):
        if is_compaction_summary(record):
            continue
        uuid = get_uuid(record)
        if uuid is not None:
            return uuid
    return None


def build_ancestry_chain(
    head_uuid: str,
    stop_uuid: str | None,
    records: dict[str, dict],
    parent_map: dict[str, str | None],
) -> list[str]:
    """Collect UUIDs from stop_uuid to head_uuid in chronological order."""
    if head_uuid == stop_uuid:
        return []
    
    chain = []
    current: str | None = head_uuid
    
    while current is not None and current != stop_uuid:
        if current not in records:
            break
        chain.append(current)
        current = parent_map.get(current)
    
    chain.reverse()
    return chain


# ---------------------------------------------------------------------------
# Convert to steps and append units
# ---------------------------------------------------------------------------


def filter_to_steps(
    uuids: list[str],
    records_by_uuid: dict[str, dict],
) -> list[Step]:
    """Filter UUIDs to steps (user/assistant messages only)."""
    steps = []
    for uuid in uuids:
        record = records_by_uuid.get(uuid)
        if record is None:
            continue
        if not should_include_record(record):
            continue
        timestamp = record.get("timestamp")
        if not isinstance(timestamp, str):
            continue
        steps.append(Step(uuid=uuid, timestamp=timestamp, record=record))
    return steps


def steps_to_append_units(steps: list[Step]) -> list[tuple[AppendUnit, Step]]:
    """Convert steps to AppendUnits for RagZoom ingestion.
    
    Returns pairs of (unit, step) to maintain alignment when some steps
    are filtered out (e.g., steps with empty transcriptions).
    """
    results: list[tuple[AppendUnit, Step]] = []
    for step in steps:
        text = transcribe_record(step.record)
        if text.strip():
            results.append((
                AppendUnit(
                    text=text,
                    time_start=step.timestamp,
                    time_end=step.timestamp,
                ),
                step,
            ))
    return results


# ---------------------------------------------------------------------------
# Sync state
# ---------------------------------------------------------------------------


@dataclass
class SyncState:
    """Tracks what has been synced to RagZoom."""

    document_id: str
    last_message_id: str | None = None
    """ID of the last synced message."""

    span_end: int = 0
    """Document span position after last sync."""

    steps_synced: int = 0
    """Number of steps synced so far."""

    entries: list[dict[str, object]] = field(default_factory=list)
    """Per-step tracking entries."""

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
            steps_synced=header.get("steps_synced", 0),
            entries=entries,
        )

    def save(self, path: Path) -> None:
        """Save state to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "document_id": self.document_id,
            "last_message_id": self.last_message_id,
            "span_end": self.span_end,
            "steps_synced": self.steps_synced,
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
    new_steps: int
    total_steps: int
    new_span_end: int


def sync_transcript(
    transcript_path: Path,
    state_path: Path,
    client: object,
    document_id: str | None = None,
) -> SyncResult:
    """Sync an OpenClaw transcript to RagZoom.

    Args:
        transcript_path: Path to the OpenClaw session JSONL file
        state_path: Path to store sync state
        client: RagZoom wrapper instance (with batch_append, clear, etc.)
        document_id: Override document ID (defaults to transcript filename stem)

    Returns:
        SyncResult describing what was done
    """
    # Load or create state
    state = SyncState.load(state_path)
    if state is None:
        doc_id = document_id or f"openclaw-{transcript_path.stem}"
        state = SyncState(document_id=doc_id)
    else:
        if document_id:
            state.document_id = document_id

    # Get current head
    head = get_current_head(transcript_path)
    if head is None:
        return SyncResult(
            document_id=state.document_id,
            new_steps=0,
            total_steps=0,
            new_span_end=state.span_end,
        )

    # Build maps
    records = build_records_map(transcript_path)
    parent_map = build_parent_map(transcript_path)

    # Get all UUIDs from root to head
    all_uuids = build_ancestry_chain(head, None, records, parent_map)

    # Filter to message steps
    all_steps = filter_to_steps(all_uuids, records)

    # Find new steps (skip already synced ones)
    new_steps = all_steps[state.steps_synced:]

    if not new_steps:
        return SyncResult(
            document_id=state.document_id,
            new_steps=0,
            total_steps=len(all_steps),
            new_span_end=state.span_end,
        )

    # Convert to append units (paired with their source steps)
    units_with_steps = steps_to_append_units(new_steps)

    if units_with_steps:
        # Extract just the units for batch append
        units = [unit for unit, _ in units_with_steps]

        # Batch append to RagZoom
        batch_append_method = getattr(client, "batch_append")
        batch_append_method(state.document_id, units)

        # Update state
        cumulative_span = state.span_end
        for unit, step in units_with_steps:
            cumulative_span += len(unit.text)
            state.entries.append({
                "uuid": step.uuid,
                "span_end": cumulative_span,
            })

        state.span_end = cumulative_span
        state.last_message_id = units_with_steps[-1][1].uuid

    state.steps_synced = len(all_steps)
    state.save(state_path)

    return SyncResult(
        document_id=state.document_id,
        new_steps=len(units_with_steps) if units_with_steps else 0,
        total_steps=len(all_steps),
        new_span_end=state.span_end,
    )
