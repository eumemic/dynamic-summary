"""Transcript sync with revert detection."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from claude_transcriber import Transcriber

from ragzoom.wrapper import AppendUnit
from ragzoom_claude_code.jsonl_reader import iter_jsonl, iter_jsonl_reversed

# Pattern to extract command name from invocation message
_COMMAND_NAME_PATTERN = re.compile(r"<command-name>(/[\w-]+)</command-name>")


@dataclass
class Turn:
    """A conversation turn with timestamp range.

    A turn groups messages from a user prompt through the assistant's complete
    response cycle. Used for timestamped transcript sync where each turn becomes
    one leaf node with temporal metadata.
    """

    uuids: list[str]
    """UUIDs of messages in this turn, in chronological order."""

    time_start: str
    """ISO 8601 timestamp of the first message in the turn."""

    time_end: str
    """ISO 8601 timestamp of the last message in the turn."""


def _is_command_output_or_expansion(
    record: dict[str, object], records_by_uuid: dict[str, dict[str, object]]
) -> bool:
    """Check if record is command output or expansion (part of a command's turn).

    Command output: User message with <local-command-stdout>
    Command expansion: User message whose parent is a command invocation
    """
    if record.get("type") != "user":
        return False

    message = record.get("message")
    if not isinstance(message, dict):
        return False

    content = message.get("content")
    if not content:
        return False

    # Check string content for command output markers
    if isinstance(content, str):
        if "<local-command-stdout>" in content:
            return True

    # Check if parent is a command invocation (command expansion case)
    parent_uuid = record.get("parentUuid")
    if isinstance(parent_uuid, str):
        parent_record = records_by_uuid.get(parent_uuid)
        if parent_record is not None:
            if _is_command_invocation(parent_record) is not None:
                return True

    return False


def _is_user_prompt(
    record: dict[str, object], records_by_uuid: dict[str, dict[str, object]]
) -> bool:
    """Check if record is a user prompt (not a tool result or command output)."""
    if record.get("type") != "user":
        return False
    if "toolUseResult" in record:
        return False
    # Command output/expansion messages should not start a new turn
    if _is_command_output_or_expansion(record, records_by_uuid):
        return False
    return True


def is_user_message(record: dict[str, object]) -> bool:
    """Check if record is a real user message (starts a turn).

    A user message starts a new turn if it's a user type message that is NOT:
    - A tool result (has toolUseResult field)
    - A command output (contains <local-command-stdout>)

    Unlike _is_user_prompt(), this doesn't check for command expansion since
    that requires access to the parent record. For the stateless truncation
    point algorithm, we only need to identify turn boundaries, and command
    expansions still mark turn boundaries from the user's perspective.

    Args:
        record: The JSONL record to check

    Returns:
        True if the record is a user message that starts a turn
    """
    if record.get("type") != "user":
        return False
    if "toolUseResult" in record:
        return False

    # Check for command output marker
    message = record.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and "<local-command-stdout>" in content:
            return False

    return True


def _parse_timestamp(timestamp_str: str) -> datetime:
    """Parse an ISO 8601 timestamp string to datetime.

    Handles both 'Z' suffix and '+00:00' timezone formats.

    Args:
        timestamp_str: ISO 8601 timestamp string

    Returns:
        datetime object (timezone-aware if input has timezone)

    Raises:
        ValueError: If timestamp format is invalid
    """
    # Replace 'Z' with '+00:00' for Python's fromisoformat
    normalized = timestamp_str.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _get_parent_uuid(rec: dict[str, object]) -> str | None:
    """Extract parentUuid as string or None from a record."""
    parent = rec.get("parentUuid")
    return parent if isinstance(parent, str) else None


def find_truncation_point(
    head_uuid: str,
    records: dict[str, dict[str, object]],
    indexed_time_end: datetime | None,
) -> tuple[str | None, str | None]:
    """Find the connection point between current transcript and indexed content.

    Uses a sliding window algorithm to walk backward from head_uuid, finding
    where the current transcript connects to previously indexed content. The
    algorithm ensures we stop at a turn boundary (user message) to maintain
    atomic turn semantics.

    The function returns (R, S) where:
    - R is the connection point UUID (last record in indexed content)
    - S is the first record to append (successor of R in the chain)

    Cases:
    - First sync (indexed_time_end is None): Returns (None, head_uuid)
      meaning "no indexed content, append from head back to root"
    - Normal append: R.timestamp == indexed_time_end (approximately)
      meaning "continue from where we left off"
    - Revert detected: R.timestamp < indexed_time_end
      meaning "truncate orphaned content, then append from S"

    The sliding window ensures we always stop at a turn boundary:
    1. R.timestamp <= indexed_time_end (R is within indexed range)
    2. S is a user message OR S is None (turn boundary)

    Args:
        head_uuid: UUID of the current transcript head
        records: UUID -> record mapping (must include parentUuid and timestamp)
        indexed_time_end: Last indexed timestamp, or None if empty document.
            Must be timezone-aware if provided, as record timestamps are parsed
            as timezone-aware datetimes.

    Returns:
        (R, S) tuple where:
        - R: UUID of connection point, or None for first sync / complete reindex
        - S: UUID of first record to append, or None if nothing to append
    """
    if indexed_time_end is None:
        # First sync: no indexed content, append everything from head
        return (None, head_uuid)

    # Sliding window: S is the successor, R is the current node
    s_uuid: str | None = None  # Starts null at end of chain
    r_uuid: str | None = head_uuid

    while r_uuid is not None:
        record = records.get(r_uuid)
        if record is None:
            # Record not found - treat as end of chain
            break

        # Get timestamp from record
        timestamp_str = record.get("timestamp")
        if not isinstance(timestamp_str, str):
            # No timestamp - can't compare, continue walking
            s_uuid = r_uuid
            r_uuid = _get_parent_uuid(record)
            continue

        try:
            record_time = _parse_timestamp(timestamp_str)
        except ValueError:
            # Invalid timestamp - continue walking
            s_uuid = r_uuid
            r_uuid = _get_parent_uuid(record)
            continue

        # Check if R is within indexed range
        if record_time <= indexed_time_end:
            # R is in indexed content - check if S is a turn boundary
            if s_uuid is None:
                # S is None means we're at the end of chain - valid boundary
                return (r_uuid, s_uuid)

            s_record = records.get(s_uuid)
            if s_record is not None and is_user_message(s_record):
                # S is a user message - valid turn boundary
                return (r_uuid, s_uuid)

            # S is not a turn boundary - continue sliding to find one

        # Slide window backward
        s_uuid = r_uuid
        r_uuid = _get_parent_uuid(record)

    # Walked entire chain without finding indexed content
    # This means complete reindex is needed
    return (None, head_uuid)


def build_ancestry_chain(
    head_uuid: str,
    stop_uuid: str | None,
    records: dict[str, dict[str, object]],
) -> list[str]:
    """Collect UUIDs from stop_uuid to head_uuid in chronological order.

    Walks backward from head_uuid through parent links until reaching stop_uuid
    (exclusive) or the root (parentUuid=None). Returns the collected UUIDs in
    chronological order (oldest first).

    Only UUIDs that exist in the records dict are included. If a parent UUID
    is missing from records, the chain stops at that point.

    This function is designed to work with the same records dict used by
    find_truncation_point(), making it easy to build the ancestry chain
    for records that need to be appended.

    Args:
        head_uuid: UUID of the endpoint (most recent record)
        stop_uuid: UUID to stop at (exclusive), or None to include entire chain
        records: UUID -> record mapping (must include parentUuid field)

    Returns:
        List of UUIDs from stop_uuid's successor to head_uuid, in chronological
        order. Empty list if head_uuid == stop_uuid or head_uuid not in records.
    """
    if head_uuid == stop_uuid:
        return []

    # Walk backward from head, collecting UUIDs that exist in records
    chain: list[str] = []
    current: str | None = head_uuid

    while current is not None and current != stop_uuid:
        record = records.get(current)
        if record is None:
            # Current UUID not in records - can't include or trace further
            break
        chain.append(current)
        current = _get_parent_uuid(record)

    # Reverse to get chronological order (oldest first)
    chain.reverse()
    return chain


def _should_skip_record(record: dict[str, object]) -> bool:
    """Check if record should be filtered out of turn grouping.

    Filters:
    - Compaction summaries (isCompactSummary=True) - already LLM-generated
    - Queue operations (type="queue-operation") - internal Claude Code state
    - Meta records (isMeta=True) - injected content like skill expansions,
      embedded PDFs, command templates (static documentation that repeats)
    """
    if record.get("isCompactSummary"):
        return True
    if record.get("type") == "queue-operation":
        return True
    if record.get("isMeta"):
        return True
    return False


def group_into_turns(
    uuids: list[str],
    records_by_uuid: dict[str, dict[str, object]],
) -> list[Turn]:
    """Group message UUIDs into conversation turns.

    A turn starts with a user prompt (user message without toolUseResult) and
    includes all subsequent messages until the next user prompt. This includes
    assistant messages and tool results.

    Filters out:
    - Compaction summaries (isCompactSummary=True)
    - Queue operations (type="queue-operation")
    - Meta records (isMeta=True) - skill expansions, PDFs, command templates
    - UUIDs not found in records_by_uuid

    Args:
        uuids: Message UUIDs in chronological order
        records_by_uuid: UUID -> record mapping

    Returns:
        List of Turn objects, each containing UUIDs and timestamps

    Raises:
        ValueError: If a record in a turn is missing a timestamp field
    """
    if not uuids:
        return []

    turns: list[Turn] = []
    current_turn_uuids: list[str] = []

    for uuid in uuids:
        record = records_by_uuid.get(uuid)
        if record is None:
            continue

        if _should_skip_record(record):
            continue

        if _is_user_prompt(record, records_by_uuid):
            # Finish current turn if it has content
            if current_turn_uuids:
                turns.append(_build_turn(current_turn_uuids, records_by_uuid))
            # Start new turn
            current_turn_uuids = [uuid]
        elif current_turn_uuids:
            # Add to current turn (assistant message or tool result)
            current_turn_uuids.append(uuid)
        # Orphan messages without a current turn are skipped

    # Finalize the last turn
    if current_turn_uuids:
        turns.append(_build_turn(current_turn_uuids, records_by_uuid))

    return turns


def _get_record_timestamp(uuid: str, record: dict[str, object]) -> str:
    """Extract and validate timestamp from a record.

    Raises:
        ValueError: If record is missing timestamp field
    """
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str):
        raise ValueError(f"Record {uuid} missing timestamp field")
    return timestamp


def _extract_segment_timestamps(
    uuids: list[str],
    records_by_uuid: dict[str, dict[str, object]],
) -> tuple[str, str] | None:
    """Extract time_start and time_end from a segment's first/last messages.

    Returns (time_start, time_end) tuple if timestamps are available,
    or None if no timestamps found.
    """
    if not uuids:
        return None

    # Find first UUID with a valid timestamp
    time_start: str | None = None
    for uuid in uuids:
        record = records_by_uuid.get(uuid)
        if record is not None:
            ts = record.get("timestamp")
            if isinstance(ts, str):
                time_start = ts
                break

    # Find last UUID with a valid timestamp
    time_end: str | None = None
    for uuid in reversed(uuids):
        record = records_by_uuid.get(uuid)
        if record is not None:
            ts = record.get("timestamp")
            if isinstance(ts, str):
                time_end = ts
                break

    if time_start is not None and time_end is not None:
        return (time_start, time_end)
    return None


def _build_turn(
    uuids: list[str],
    records_by_uuid: dict[str, dict[str, object]],
) -> Turn:
    """Build a Turn from a list of UUIDs.

    Args:
        uuids: Message UUIDs in this turn (must be non-empty)
        records_by_uuid: UUID -> record mapping

    Returns:
        Turn with timestamps from first and last messages

    Raises:
        ValueError: If uuids is empty, UUID not in records, or missing timestamp
    """
    if not uuids:
        raise ValueError("Cannot build turn from empty UUID list")

    first_uuid, last_uuid = uuids[0], uuids[-1]

    if first_uuid not in records_by_uuid:
        raise ValueError(f"UUID {first_uuid} not in records_by_uuid")
    if last_uuid not in records_by_uuid:
        raise ValueError(f"UUID {last_uuid} not in records_by_uuid")

    first_record = records_by_uuid[first_uuid]
    last_record = records_by_uuid[last_uuid]

    return Turn(
        uuids=uuids,
        time_start=_get_record_timestamp(first_uuid, first_record),
        time_end=_get_record_timestamp(last_uuid, last_record),
    )


def turns_to_append_units(
    turns: list[Turn],
    records_by_uuid: dict[str, dict[str, object]],
) -> list[AppendUnit]:
    """Convert turns to AppendUnits for batch indexing.

    Each turn is transcribed to text and bundled with its timestamps
    into an AppendUnit for temporal indexing.

    Args:
        turns: List of conversation turns
        records_by_uuid: UUID -> record mapping for transcription

    Returns:
        List of AppendUnits with text and timestamps from each turn
    """
    if not turns:
        return []

    result: list[AppendUnit] = []
    for turn in turns:
        text = transcribe_uuids_from_map(turn.uuids, records_by_uuid)
        result.append(
            AppendUnit(
                text=text,
                time_start=turn.time_start,
                time_end=turn.time_end,
            )
        )
    return result


@dataclass
class SessionStateHeader:
    """Header line of session state file."""

    document_id: str

    last_pid: int | None = None
    """PID of the Claude Code process for this session."""

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {"document_id": self.document_id}
        if self.last_pid is not None:
            result["last_pid"] = self.last_pid
        return result

    @classmethod
    def from_json(cls, data: dict[str, object]) -> SessionStateHeader:
        doc_id = data.get("document_id")
        if not isinstance(doc_id, str):
            raise TypeError(f"document_id must be str, got {type(doc_id)}")
        last_pid = data.get("last_pid")
        if last_pid is not None and not isinstance(last_pid, int):
            raise TypeError(f"last_pid must be int or None, got {type(last_pid)}")
        return cls(document_id=doc_id, last_pid=last_pid)


@dataclass
class AppendEntry:
    """A single entry in the append log.

    For turn-level tracking, each entry represents one conversation turn.
    The first_uuid marks the turn's start (for revert detection) and
    last_uuid marks its end (for continuation detection).
    """

    last_uuid: str
    """UUID of the last message in this append (turn end)."""

    span_end: int
    """Document span position after this append."""

    first_uuid: str | None = None
    """UUID of the first message in this append (turn start).

    Used for turn-granularity revert detection: if a common ancestor
    falls between first_uuid and last_uuid (within the turn), we truncate
    to before this turn rather than keeping it.

    None for backward compatibility with pre-turn-tracking entries.
    """

    def to_json(self) -> dict[str, object]:
        """Serialize to JSON-compatible dict."""
        result: dict[str, object] = {
            "last_uuid": self.last_uuid,
            "span_end": self.span_end,
        }
        if self.first_uuid is not None:
            result["first_uuid"] = self.first_uuid
        return result

    @classmethod
    def from_json(cls, data: dict[str, object]) -> AppendEntry:
        """Deserialize from JSON dict."""
        last_uuid = data["last_uuid"]
        span_end = data["span_end"]
        first_uuid = data.get("first_uuid")
        if not isinstance(last_uuid, str):
            raise TypeError(f"last_uuid must be str, got {type(last_uuid)}")
        if not isinstance(span_end, int):
            raise TypeError(f"span_end must be int, got {type(span_end)}")
        if first_uuid is not None and not isinstance(first_uuid, str):
            raise TypeError(f"first_uuid must be str or None, got {type(first_uuid)}")
        return cls(last_uuid=last_uuid, span_end=span_end, first_uuid=first_uuid)


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

        For turn-level tracking: if the common ancestor falls WITHIN a turn
        (between first_uuid and last_uuid) rather than at a turn boundary
        (at last_uuid), we return the PREVIOUS turn's entry. This ensures
        we truncate to before the partially-matching turn.

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

        # Walk backwards through entries to find one whose last_uuid is an ancestor
        # of the common ancestor (meaning the turn ended at or before the common point)
        common_ancestors = _get_ancestors(common, parent_map)
        common_ancestors.add(common)

        for i, entry in enumerate(reversed(entries)):
            if entry.last_uuid in common_ancestors:
                # Check if common ancestor is WITHIN this turn (not at its boundary)
                # This only applies to turn-level entries that have first_uuid set
                if entry.first_uuid is not None and entry.last_uuid != common:
                    # Common ancestor is within this turn, not at its end.
                    # Check if common is actually within this turn's UUID range.
                    # Get all UUIDs from first_uuid to last_uuid (the turn's range)
                    turn_ancestors = _get_ancestors(entry.last_uuid, parent_map)
                    turn_ancestors.add(entry.last_uuid)

                    # If the common ancestor is in the turn's range but not at the end,
                    # we need to return the previous entry (before this turn)
                    if common in turn_ancestors and common != entry.last_uuid:
                        # Find the first_uuid's ancestors to determine turn boundary
                        first_ancestors = _get_ancestors(entry.first_uuid, parent_map)
                        # If common is an ancestor of first_uuid (before this turn),
                        # this entry is still valid
                        if common in first_ancestors:
                            return entry
                        # Common is within this turn - return previous entry
                        actual_idx = len(entries) - 1 - i
                        if actual_idx > 0:
                            return entries[actual_idx - 1]
                        return None
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


def _get_state_dir() -> Path:
    """Get the transcript state directory from environment or default."""
    import os

    state_dir_str = os.environ.get("RAGZOOM_STATE_DIR", "data/transcript-state")
    return Path(state_dir_str)


def get_state_path(document_id: str) -> Path:
    """Get the path to the state file for a document."""
    state_dir = _get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{document_id}.jsonl"


def set_session_pid(document_id: str, pid: int) -> None:
    """Set the PID for a session, creating state file if needed.

    Called by SessionStart hook to register the Claude Code PID before
    any tool calls. Preserves existing state fields if the file exists.
    """
    state_path = get_state_path(document_id)
    state = SessionState.load(state_path)
    if state is None:
        state = SessionState(header=SessionStateHeader(document_id=document_id))
    state.header.last_pid = pid
    state.save(state_path)


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
    """Build a uuid -> parentUuid map from a transcript file.

    Handles compaction boundaries by bridging "segments". When a message has
    parentUuid=None but is immediately followed by a compaction summary, this
    indicates a session resume. We bridge the gap by setting that message's
    parent to the last message before the system/compaction pair.

    This allows ancestor chain traversal to span multiple compaction events.
    """
    parent_map: dict[str, str | None] = {}

    # First pass: collect all records with their order and identify compaction points
    records: list[tuple[str, str | None, bool]] = []  # (uuid, parentUuid, is_compact)
    for record, _ in iter_jsonl(transcript_path):
        uuid = record.get("uuid")
        if isinstance(uuid, str):
            parent_uuid = record.get("parentUuid")
            if parent_uuid is not None and not isinstance(parent_uuid, str):
                continue
            is_compact = bool(record.get("isCompactSummary"))
            records.append((uuid, parent_uuid, is_compact))

    # Second pass: build parent map with compaction bridging
    # Track the last "regular" uuid before each compaction boundary
    last_regular_uuid: str | None = None

    for i, (uuid, parent_uuid, is_compact) in enumerate(records):
        if parent_uuid is None and not is_compact:
            # This message has no parent. Check if it's followed by a compaction.
            # If so, bridge it to the last regular message before this point.
            is_followed_by_compact = False
            for j in range(i + 1, len(records)):
                _, _, next_is_compact = records[j]
                if next_is_compact:
                    is_followed_by_compact = True
                    break
                # Stop looking if we hit another regular message
                _, next_parent, _ = records[j]
                if next_parent != uuid:
                    break

            if is_followed_by_compact and last_regular_uuid is not None:
                # Bridge to the last regular message before this segment
                parent_map[uuid] = last_regular_uuid
            else:
                parent_map[uuid] = None
        else:
            parent_map[uuid] = parent_uuid

        # Update last_regular_uuid (messages that aren't part of compaction)
        if not is_compact:
            last_regular_uuid = uuid

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


def get_compaction_uuid(transcript_path: Path) -> str | None:
    """Find the UUID just before the most recent compaction.

    Scans backwards from the end of the transcript for efficiency,
    since compaction is typically recent.

    Returns:
        UUID of the message just before compaction, or None if no compaction.
    """
    found_compaction = False
    for record in iter_jsonl_reversed(transcript_path):
        if record.get("isCompactSummary"):
            # Found compaction - the next UUID we see (going backwards)
            # is the one just before the compaction in chronological order
            found_compaction = True
            continue
        uuid = record.get("uuid")
        if isinstance(uuid, str):
            if found_compaction:
                return uuid
    return None


def get_current_head(transcript_path: Path) -> str | None:
    """Get the UUID of the most recent message in the transcript.

    Scans backwards from the end for efficiency, since the head is at the end.
    Returns None if no messages with uuid found.
    """
    for record in iter_jsonl_reversed(transcript_path):
        if record.get("isCompactSummary"):
            continue
        uuid = record.get("uuid")
        if isinstance(uuid, str):
            return uuid
    return None


def build_records_map(
    transcript_path: Path,
    uuids: set[str],
) -> dict[str, dict[str, object]]:
    """Build a UUID -> record lookup for the specified UUIDs.

    Args:
        transcript_path: Path to the JSONL transcript
        uuids: Set of UUIDs to extract

    Returns:
        UUID -> record mapping
    """
    records_by_uuid: dict[str, dict[str, object]] = {}

    for record, _ in iter_jsonl(transcript_path):
        if record.get("isCompactSummary"):
            continue

        uuid = record.get("uuid")
        if isinstance(uuid, str) and uuid in uuids:
            records_by_uuid[uuid] = record

    return records_by_uuid


def _is_command_invocation(record: dict[str, object]) -> str | None:
    """Check if record is a command invocation, return command name if so."""
    if record.get("type") != "user":
        return None
    text = _extract_user_text_raw(record)
    match = _COMMAND_NAME_PATTERN.search(text)
    return match.group(1) if match else None


def _extract_user_text_raw(record: dict[str, object]) -> str:
    """Extract raw text from a user message without cleaning."""
    message = record.get("message", {})
    if not isinstance(message, dict):
        return str(message)

    content = message.get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                texts.append(block)
        return "".join(texts)

    return str(content)


def transcribe_uuids_from_map(
    uuids: list[str],
    records_by_uuid: dict[str, dict[str, object]],
) -> str:
    """Transcribe UUIDs using claude-transcriber library.

    Uses the Transcriber class from claude-transcriber to convert JSONL records
    to human-readable text matching Claude Code's /export format.

    The Transcriber is stateful - it tracks pending tool uses to properly format
    tool results. A single instance is used per batch to maintain this state.

    Args:
        uuids: UUIDs to transcribe, in order
        records_by_uuid: Pre-built UUID -> record lookup

    Returns:
        Concatenated transcript text
    """
    if not uuids:
        return ""

    transcriber = Transcriber()
    chunks: list[str] = []

    for uuid in uuids:
        record = records_by_uuid.get(uuid)
        if record is None:
            continue

        result = transcriber.transcribe(record)
        if result:
            chunks.append(result)

    return "\n\n".join(chunks)


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

    # Get current head
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
        state.save(state_path)
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

    # Group UUIDs into turns and batch append with temporal metadata
    appended_uuids: list[str] = []
    new_span_end = 0
    if plan.uuids_to_transcribe:
        # Build records map once for all UUIDs
        records_by_uuid = build_records_map(
            transcript_path, set(plan.uuids_to_transcribe)
        )

        # Group UUIDs into conversation turns
        turns = group_into_turns(plan.uuids_to_transcribe, records_by_uuid)

        if turns:
            # Convert turns to AppendUnits with timestamps
            append_units = turns_to_append_units(turns, records_by_uuid)

            # Pair units with turns, keeping only non-empty units
            units_with_turns = [
                (unit, turn)
                for unit, turn in zip(append_units, turns)
                if unit.text.strip()
            ]

            if units_with_turns:
                non_empty_units = [unit for unit, _ in units_with_turns]

                # Call batch_append with all units at once
                batch_append_method = getattr(client, "batch_append")
                batch_append_method(document_id, non_empty_units)

                # Determine starting span position for this batch
                last_entry = append_log.last_entry()
                span_start = last_entry.span_end if last_entry else 0

                # Record one AppendEntry per turn with cumulative span_end
                cumulative_span = span_start
                for unit, turn in units_with_turns:
                    cumulative_span += len(unit.text)

                    # Collect UUIDs excluding tool results
                    turn_uuids = [
                        uuid
                        for uuid in turn.uuids
                        if uuid in records_by_uuid
                        and not (
                            records_by_uuid[uuid].get("type") == "user"
                            and "toolUseResult" in records_by_uuid[uuid]
                        )
                    ]
                    appended_uuids.extend(turn_uuids)

                    if turn_uuids:
                        append_log.append(
                            AppendEntry(
                                last_uuid=turn_uuids[-1],
                                span_end=cumulative_span,
                                first_uuid=turn_uuids[0],
                            )
                        )

                # Use cumulative span for consistency with stored entries
                # (matches what we recorded in each turn's AppendEntry)
                new_span_end = cumulative_span

    # Fall back to existing span_end if nothing was appended
    if not appended_uuids:
        last_entry = append_log.last_entry()
        new_span_end = last_entry.span_end if last_entry else 0

    # Save state
    state.save(state_path)

    return SyncResult(
        document_id=document_id,
        truncated=truncated,
        truncate_span=truncate_span,
        appended_uuids=appended_uuids,
        new_span_end=new_span_end,
    )
