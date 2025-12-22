"""Transcript sync with revert detection."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ragzoom.jsonl_reader import iter_jsonl

STATE_VERSION = 2


@dataclass
class SessionStateHeader:
    """Header line of session state file."""

    document_id: str
    version: int = STATE_VERSION

    def to_json(self) -> dict[str, object]:
        return {"document_id": self.document_id, "version": self.version}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> SessionStateHeader:
        doc_id = data.get("document_id")
        if not isinstance(doc_id, str):
            raise TypeError(f"document_id must be str, got {type(doc_id)}")
        version = data.get("version", 1)
        if not isinstance(version, int):
            raise TypeError(f"version must be int, got {type(version)}")
        return cls(document_id=doc_id, version=version)


@dataclass
class AppendEntry:
    """A single entry in the append log."""

    last_uuid: str
    """UUID of the last message in this append."""

    span_end: int
    """Document span position after this append."""

    def to_json(self) -> dict[str, object]:
        """Serialize to JSON-compatible dict."""
        return {"last_uuid": self.last_uuid, "span_end": self.span_end}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> AppendEntry:
        """Deserialize from JSON dict."""
        last_uuid = data["last_uuid"]
        span_end = data["span_end"]
        if not isinstance(last_uuid, str):
            raise TypeError(f"last_uuid must be str, got {type(last_uuid)}")
        if not isinstance(span_end, int):
            raise TypeError(f"span_end must be int, got {type(span_end)}")
        return cls(last_uuid=last_uuid, span_end=span_end)


class AppendLog:
    """JSONL-based log of append operations."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, entry: AppendEntry) -> None:
        """Append an entry to the log."""
        with open(self._path, "a") as f:
            f.write(json.dumps(entry.to_json()) + "\n")

    def last_entry(self) -> AppendEntry | None:
        """Get the last entry, or None if empty."""
        if not self._path.exists():
            return None

        # Read last line efficiently
        content = self._path.read_text().rstrip("\n")
        if not content:
            return None

        last_line = content.rsplit("\n", 1)[-1]
        return AppendEntry.from_json(json.loads(last_line))

    def truncate_to(self, last_uuid: str) -> None:
        """Remove all entries after the one with the given uuid."""
        entries = list(self)
        found_idx = None
        for i, entry in enumerate(entries):
            if entry.last_uuid == last_uuid:
                found_idx = i
                break

        if found_idx is None:
            raise ValueError(f"uuid {last_uuid!r} not found in append log")

        # Rewrite file with only entries up to and including found_idx
        with open(self._path, "w") as f:
            for entry in entries[: found_idx + 1]:
                f.write(json.dumps(entry.to_json()) + "\n")

    def find_valid_prefix(
        self, current_head: str, parent_map: dict[str, str | None]
    ) -> AppendEntry | None:
        """Find the last entry whose uuid is an ancestor of current_head.

        Walks backwards through the append log, checking each entry's last_uuid
        against the ancestor chain of current_head.

        Returns None if the log is empty, signaling the caller should transcribe
        the entire ancestor chain from root to current_head.
        """
        entries = list(self)
        if not entries:
            return None

        # Get the last indexed uuid
        last_indexed = entries[-1].last_uuid

        # Find common ancestor between last_indexed and current_head
        common = find_common_ancestor(last_indexed, current_head, parent_map)

        # No common ancestor means completely disjoint branches - user reverted
        # to before anything we indexed, so start fresh
        if common is None:
            return None

        # Walk backwards through entries to find the one containing the common ancestor
        # An entry is valid if its last_uuid is the common ancestor or an ancestor of it
        common_ancestors = _get_ancestors(common, parent_map)
        common_ancestors.add(common)

        for entry in reversed(entries):
            if entry.last_uuid in common_ancestors:
                return entry

        return None

    def __iter__(self) -> Iterator[AppendEntry]:
        """Iterate over all entries."""
        if not self._path.exists():
            return

        for record, _ in iter_jsonl(self._path):
            yield AppendEntry.from_json(record)


@dataclass
class SessionState:
    """Session state with header and append log."""

    header: SessionStateHeader
    entries: list[AppendEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> SessionState | None:
        """Load session state from JSONL file.

        Returns None if file doesn't exist.
        """
        if not path.exists():
            return None

        lines = path.read_text().strip().split("\n")
        if not lines or not lines[0]:
            return None

        header = SessionStateHeader.from_json(json.loads(lines[0]))
        entries = [
            AppendEntry.from_json(json.loads(line)) for line in lines[1:] if line
        ]
        return cls(header=header, entries=entries)

    def save(self, path: Path) -> None:
        """Save session state to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(self.header.to_json())]
        lines.extend(json.dumps(entry.to_json()) for entry in self.entries)
        path.write_text("\n".join(lines) + "\n")

    def append_log(self) -> AppendLog:
        """Get an AppendLog view backed by this state's entries."""
        return _SessionAppendLog(self)


class _SessionAppendLog(AppendLog):
    """AppendLog backed by SessionState entries (in-memory)."""

    def __init__(self, state: SessionState) -> None:
        # Dummy path - not used for in-memory operations
        super().__init__(Path("/dev/null"))
        self._state = state

    def append(self, entry: AppendEntry) -> None:
        self._state.entries.append(entry)

    def last_entry(self) -> AppendEntry | None:
        if not self._state.entries:
            return None
        return self._state.entries[-1]

    def truncate_to(self, last_uuid: str) -> None:
        for i, entry in enumerate(self._state.entries):
            if entry.last_uuid == last_uuid:
                self._state.entries = self._state.entries[: i + 1]
                return
        raise ValueError(f"uuid {last_uuid!r} not found in append log")

    def __iter__(self) -> Iterator[AppendEntry]:
        return iter(self._state.entries)


def build_parent_map(transcript_path: Path) -> dict[str, str | None]:
    """Build a uuid -> parentUuid map from a transcript file."""
    parent_map: dict[str, str | None] = {}

    for record, _ in iter_jsonl(transcript_path):
        uuid = record.get("uuid")
        if isinstance(uuid, str):
            parent_uuid = record.get("parentUuid")
            if parent_uuid is None or isinstance(parent_uuid, str):
                parent_map[uuid] = parent_uuid
            else:
                # parentUuid exists but is not a string - skip this record
                continue

    return parent_map


def find_common_ancestor(
    x: str, y: str, parent_map: dict[str, str | None]
) -> str | None:
    """Find the most recent common ancestor of x and y.

    Traces both ancestor chains until they intersect.
    Returns None if x and y have no common ancestor (completely disjoint branches).
    Raises KeyError if either x or y is not in the parent map.
    """
    if x not in parent_map:
        raise KeyError(x)
    if y not in parent_map:
        raise KeyError(y)

    # Get all ancestors of x (including x itself)
    x_ancestors = _get_ancestors(x, parent_map)
    x_ancestors.add(x)

    # Walk up y's chain until we find a common ancestor
    current: str | None = y
    while current is not None:
        if current in x_ancestors:
            return current
        current = parent_map.get(current)

    # No intersection - completely disjoint branches (e.g., user reverted to
    # before the first message and started a new conversation)
    return None


def _get_ancestors(uuid: str, parent_map: dict[str, str | None]) -> set[str]:
    """Get all ancestors of a uuid (not including the uuid itself)."""
    ancestors: set[str] = set()
    current = parent_map.get(uuid)
    while current is not None:
        ancestors.add(current)
        current = parent_map.get(current)
    return ancestors


def get_ancestor_chain(
    target: str, ancestor: str | None, parent_map: dict[str, str | None]
) -> list[str]:
    """Get ordered chain from ancestor to target (exclusive of ancestor).

    Args:
        target: The endpoint uuid
        ancestor: The starting point uuid (exclusive), or None for root
        parent_map: uuid -> parentUuid mapping

    Returns:
        List of uuids from ancestor's child to target, in forward order.
        Empty list if target == ancestor.

    Raises:
        ValueError: If ancestor is not actually an ancestor of target
    """
    if target == ancestor:
        return []

    # Walk backwards from target to ancestor, collecting the chain
    chain: list[str] = []
    current: str | None = target

    while current is not None and current != ancestor:
        chain.append(current)
        current = parent_map.get(current)

    # If ancestor is not None but we hit None without finding it, it's not an ancestor
    if ancestor is not None and current != ancestor:
        raise ValueError(f"{ancestor!r} is not an ancestor of {target!r}")

    # Reverse to get forward order (ancestor's child first, target last)
    chain.reverse()
    return chain


@dataclass
class SyncPlan:
    """Plan for syncing a transcript to the indexed document."""

    uuids_to_transcribe: list[str]
    """UUIDs to transcribe and append, in order."""

    truncate_to_span: int | None
    """If set, truncate document to this span before appending."""

    truncate_to_uuid: str | None
    """UUID of the valid prefix entry (for truncating append log)."""


def compute_sync_plan(
    current_head: str,
    append_log: AppendLog,
    parent_map: dict[str, str | None],
) -> SyncPlan:
    """Compute what operations are needed to sync transcript to document.

    Args:
        current_head: UUID of the current transcript head
        append_log: The append log tracking what we've indexed
        parent_map: uuid -> parentUuid mapping from transcript

    Returns:
        SyncPlan describing truncation and transcription needed
    """
    last_entry = append_log.last_entry()

    # Empty append log: transcribe entire ancestor chain from root
    if last_entry is None:
        chain = get_ancestor_chain(current_head, None, parent_map)
        return SyncPlan(
            uuids_to_transcribe=chain,
            truncate_to_span=None,
            truncate_to_uuid=None,
        )

    # Already synced: nothing to do
    if last_entry.last_uuid == current_head:
        return SyncPlan(
            uuids_to_transcribe=[],
            truncate_to_span=None,
            truncate_to_uuid=None,
        )

    # Find valid prefix (handles revert detection)
    valid_prefix = append_log.find_valid_prefix(current_head, parent_map)

    if valid_prefix is None:
        # Disjoint branches: truncate everything and start fresh
        chain = get_ancestor_chain(current_head, None, parent_map)
        return SyncPlan(
            uuids_to_transcribe=chain,
            truncate_to_span=0,
            truncate_to_uuid=None,
        )

    # Get chain from valid prefix to current head
    chain = get_ancestor_chain(current_head, valid_prefix.last_uuid, parent_map)

    # Determine if truncation is needed
    if valid_prefix.last_uuid == last_entry.last_uuid:
        # No revert, just append new messages
        return SyncPlan(
            uuids_to_transcribe=chain,
            truncate_to_span=None,
            truncate_to_uuid=None,
        )
    else:
        # Revert happened: truncate to valid prefix
        return SyncPlan(
            uuids_to_transcribe=chain,
            truncate_to_span=valid_prefix.span_end,
            truncate_to_uuid=valid_prefix.last_uuid,
        )


def get_current_head(transcript_path: Path) -> str | None:
    """Get the UUID of the most recent message in the transcript.

    Reads the transcript forward, returning the last uuid seen.
    Returns None if no messages with uuid found.
    """
    last_uuid: str | None = None

    for record, _ in iter_jsonl(transcript_path):
        uuid = record.get("uuid")
        if isinstance(uuid, str):
            last_uuid = uuid

    return last_uuid


def transcribe_uuids(
    transcript_path: Path,
    uuids: list[str],
) -> str:
    """Transcribe specific messages by UUID into readable text.

    Args:
        transcript_path: Path to the JSONL transcript
        uuids: UUIDs to transcribe, in order

    Returns:
        Concatenated transcript text for the specified messages
    """
    if not uuids:
        return ""

    # Build uuid -> record lookup
    uuid_set = set(uuids)
    records_by_uuid: dict[str, dict[str, object]] = {}

    for record, _ in iter_jsonl(transcript_path):
        uuid = record.get("uuid")
        if isinstance(uuid, str) and uuid in uuid_set:
            records_by_uuid[uuid] = record

    # Transcribe in order
    chunks: list[str] = []
    for uuid in uuids:
        if uuid not in records_by_uuid:
            continue
        record = records_by_uuid[uuid]

        chunk = _transcribe_record(record)
        if chunk:
            chunks.append(chunk)

    return "\n\n".join(chunks)


def _transcribe_record(record: dict[str, object]) -> str | None:
    """Transcribe a single record to readable text."""
    record_type = record.get("type")

    if record_type == "user":
        # Skip tool results
        if "toolUseResult" in record:
            return None
        text = _extract_user_text(record)
        if text.strip():
            return f"[USER]\n{text}"

    elif record_type == "assistant":
        text, tool_count = _extract_assistant_content(record)
        parts: list[str] = []
        if text.strip():
            parts.append(f"[ASSISTANT]\n{text}")
        if tool_count > 0:
            parts.append(f"[Used {tool_count} tool{'s' if tool_count > 1 else ''}]")
        if parts:
            return "\n\n".join(parts)

    return None


# jscpd:ignore-start - Similar to claude_transcript.py; will consolidate after migration
def _extract_user_text(record: dict[str, object]) -> str:
    """Extract text content from a user message."""
    message = record.get("message", {})
    if not isinstance(message, dict):
        return str(message)

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                texts.append(block)
        return "".join(texts).strip()

    return str(content).strip()


def _extract_assistant_content(record: dict[str, object]) -> tuple[str, int]:
    """Extract text and tool count from an assistant message."""
    message = record.get("message", {})
    if not isinstance(message, dict):
        return str(message), 0

    content = message.get("content", [])
    if not isinstance(content, list):
        return str(content), 0

    texts: list[str] = []
    tool_count = 0

    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if isinstance(text, str) and text.strip():
                texts.append(text)
        elif block_type == "tool_use":
            tool_count += 1

    return "\n\n".join(texts), tool_count


# jscpd:ignore-end


@dataclass
class SyncResult:
    """Result from executing a sync operation."""

    document_id: str
    truncated: bool
    truncate_span: int | None
    appended_uuids: list[str]
    new_span_end: int


def execute_sync(
    transcript_path: Path,
    state_path: Path,
    client: object,
) -> SyncResult:
    """Execute a complete sync operation.

    Args:
        transcript_path: Path to the JSONL transcript
        state_path: Path to the session state file
        client: RagZoom client with append() and truncate() methods

    Returns:
        SyncResult describing what was done
    """
    # Load or create session state
    state = SessionState.load(state_path)
    if state is None:
        # Generate document_id from transcript filename
        document_id = transcript_path.stem
        state = SessionState(header=SessionStateHeader(document_id=document_id))

    document_id = state.header.document_id

    # Get current transcript head
    current_head = get_current_head(transcript_path)
    if current_head is None:
        # Empty transcript, nothing to sync
        return SyncResult(
            document_id=document_id,
            truncated=False,
            truncate_span=None,
            appended_uuids=[],
            new_span_end=0,
        )

    # Build parent map and compute sync plan
    parent_map = build_parent_map(transcript_path)
    append_log = state.append_log()
    plan = compute_sync_plan(current_head, append_log, parent_map)

    # Nothing to do
    if not plan.uuids_to_transcribe and plan.truncate_to_span is None:
        last_entry = append_log.last_entry()
        span_end = last_entry.span_end if last_entry else 0
        return SyncResult(
            document_id=document_id,
            truncated=False,
            truncate_span=None,
            appended_uuids=[],
            new_span_end=span_end,
        )

    truncated = False
    truncate_span: int | None = None

    # Execute truncation if needed
    if plan.truncate_to_span is not None:
        truncate_method = getattr(client, "truncate")
        truncate_method(document_id, plan.truncate_to_span)
        truncated = True
        truncate_span = plan.truncate_to_span

        # Truncate the append log
        if plan.truncate_to_uuid is not None:
            append_log.truncate_to(plan.truncate_to_uuid)
        else:
            # Disjoint branches - clear the state entries
            state.entries = []

    # Transcribe and append
    if plan.uuids_to_transcribe:
        text = transcribe_uuids(transcript_path, plan.uuids_to_transcribe)
        if text:
            append_method = getattr(client, "append")
            result = append_method(document_id, text)
            # Get span_end from result
            # The append result should contain document stats that tell us the new position
            # For now, we estimate based on token count (rough approximation)
            # TODO: Add span_end to IndexingResult when available
            last_entry = append_log.last_entry()
            prev_span_end = last_entry.span_end if last_entry else 0
            if truncate_span is not None:
                prev_span_end = truncate_span
            # Use chunks_created as a proxy for new span extent
            new_span_end = prev_span_end + getattr(result, "chunks_created", 0)

            # Record in append log
            append_log.append(
                AppendEntry(
                    last_uuid=plan.uuids_to_transcribe[-1],
                    span_end=new_span_end,
                )
            )
    else:
        last_entry = append_log.last_entry()
        new_span_end = last_entry.span_end if last_entry else 0

    # Save state
    state.save(state_path)

    return SyncResult(
        document_id=document_id,
        truncated=truncated,
        truncate_span=truncate_span,
        appended_uuids=plan.uuids_to_transcribe,
        new_span_end=new_span_end,
    )
