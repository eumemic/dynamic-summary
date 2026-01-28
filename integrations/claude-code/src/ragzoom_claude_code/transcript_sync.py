"""Transcript sync with revert detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from claude_transcriber import Transcriber

from ragzoom.wrapper import AppendUnit
from ragzoom_claude_code.jsonl_reader import iter_jsonl, iter_jsonl_reversed

# Pattern to extract command name from invocation message
_COMMAND_NAME_PATTERN = re.compile(r"<command-name>(/[\w-]+)</command-name>")

# Summarization guidance for conversation transcripts
# Instructs the LLM to preserve narrative structure, identity, and decision outcomes
CONVERSATION_SUMMARIZATION_GUIDANCE = """
This is a conversation transcript between a human and an AI assistant.

When summarizing, preserve:
- **Identity and agency**: Who said what, who performed which actions
- **Decisions and outcomes**: What was decided, what actions were taken
- **Cause and effect**: Why things happened, the reasoning behind decisions
- **Chronological flow**: The temporal sequence of events

Focus on the narrative of what happened and why, not just the facts.
Preserve exact technical terms, file paths, function names, and code references.
"""


def _get_temp_dir() -> Path:
    """Get the temp directory for session files.

    Uses /tmp by default. Separated for testability.
    """
    return Path("/tmp")


def get_session_document_id(pid: int) -> str | None:
    """Read document_id from PID-keyed temp file.

    Used for discovered identity (Claude Code model) where the SessionStart
    hook writes the session ID to /tmp/ragzoom-session-{pid}.

    Args:
        pid: Process ID of the Claude Code process

    Returns:
        The document_id if temp file exists and has content, None otherwise
    """
    temp_dir = _get_temp_dir()
    temp_path = temp_dir / f"ragzoom-session-{pid}"

    if not temp_path.exists():
        return None

    content = temp_path.read_text().strip()
    if not content:
        return None

    return content


@dataclass
class Step:
    """A single conversation step with timestamp.

    Each step is one JSONL record that passes filtering. Each Step represents
    a single point-in-time message with time_start = time_end = timestamp.
    """

    uuid: str
    """UUID of the message for this step."""

    timestamp: str
    """ISO 8601 timestamp of this step."""


def _should_include_record(record: dict[str, object]) -> bool:
    """Include only user and assistant messages, excluding meta/compaction.

    Determines if a JSONL record should become a Step for indexing. This is
    the positive inclusion filter for step-level chunking.

    Includes:
    - User messages (type="user")
    - Assistant messages (type="assistant")
    - Tool results (type="user" with toolUseResult) - as their own steps

    Excludes:
    - Queue operations (type="queue-operation")
    - Compaction summaries (isCompactSummary=True)
    - Meta records (isMeta=True) - skill expansions, PDFs, templates

    Args:
        record: The JSONL record to check

    Returns:
        True if the record should become a Step, False otherwise
    """
    record_type = record.get("type")
    if record_type not in ("user", "assistant"):
        return False
    if record.get("isCompactSummary"):
        return False
    if record.get("isMeta"):
        return False
    return True


def filter_to_steps(
    uuids: list[str],
    records_by_uuid: dict[str, dict[str, object]],
) -> list[Step]:
    """Filter UUIDs to steps (user/assistant messages only).

    Each JSONL record that passes _should_include_record() becomes its own Step
    with a point-in-time timestamp. This enables fine-grained temporal retrieval.

    Args:
        uuids: Message UUIDs in chronological order
        records_by_uuid: UUID -> record mapping

    Returns:
        List of Step objects for records that pass filtering
    """
    steps: list[Step] = []
    for uuid in uuids:
        record = records_by_uuid.get(uuid)
        if record is None:
            continue
        if not _should_include_record(record):
            continue
        timestamp = record.get("timestamp")
        if not isinstance(timestamp, str):
            continue
        steps.append(Step(uuid=uuid, timestamp=timestamp))
    return steps


def steps_to_append_units(
    steps: list[Step],
    records_by_uuid: dict[str, dict[str, object]],
) -> list[AppendUnit]:
    """Convert steps to AppendUnits for batch indexing.

    Each step is transcribed individually with time_start = time_end (point-in-time).
    This enables fine-grained temporal queries at the message level rather than
    turn level.

    Steps whose UUID is not in records_by_uuid or that transcribe to empty/whitespace
    are skipped.

    Args:
        steps: List of conversation steps
        records_by_uuid: UUID -> record mapping for transcription

    Returns:
        List of AppendUnits with text and point-in-time timestamps
    """
    if not steps:
        return []

    result: list[AppendUnit] = []
    for step in steps:
        text = transcribe_uuids_from_map([step.uuid], records_by_uuid)
        if text.strip():
            result.append(
                AppendUnit(
                    text=text,
                    time_start=step.timestamp,
                    time_end=step.timestamp,
                )
            )
    return result


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

    Walks backward from head_uuid to find where the current transcript connects
    to previously indexed content. With step-level chunking, every record is a
    valid truncation point (no turn boundary constraints).

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

    # Walk backward: S is the successor, R is the current node
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

        # Check if R is within indexed range - stop immediately if so
        if record_time <= indexed_time_end:
            return (r_uuid, s_uuid)

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
    parent_map: dict[str, str | None],
) -> list[str]:
    """Collect UUIDs from stop_uuid to head_uuid in chronological order.

    Walks backward from head_uuid through parent links until reaching stop_uuid
    (exclusive) or the root (parentUuid=None). Returns the collected UUIDs in
    chronological order (oldest first).

    Only UUIDs that exist in the records dict are included. Uses parent_map
    for traversal to bridge compaction boundaries.

    Args:
        head_uuid: UUID of the endpoint (most recent record)
        stop_uuid: UUID to stop at (exclusive), or None to include entire chain
        records: UUID -> record mapping
        parent_map: UUID -> parentUuid mapping that bridges compaction
            boundaries. Use build_parent_map() to create this.

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
        if current not in records:
            # Current UUID not in records - can't include or trace further
            break
        chain.append(current)
        current = parent_map.get(current)

    # Reverse to get chronological order (oldest first)
    chain.reverse()
    return chain


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
    """Result from executing a sync operation.

    The stateless sync algorithm derives state from the document status API,
    eliminating the need for UUID-based tracking or span positions.
    """

    document_id: str
    """Document ID that was synced."""

    truncated: bool
    """True if a revert was detected and orphaned content was removed."""

    truncate_cutoff_time: str | None
    """ISO 8601 timestamp used for truncation, if truncated is True."""

    steps_appended: int
    """Number of conversation steps appended to the document."""


def _handle_revert_detection(
    r_uuid: str | None,
    indexed_time_end: datetime | None,
    records: dict[str, dict[str, object]],
    client: object,
    document_id: str,
) -> str | None:
    """Detect and handle revert by comparing connection point to indexed content.

    Returns the cutoff timestamp string if a revert was detected and truncation
    was performed, or None if no revert occurred.
    """
    if r_uuid is None or indexed_time_end is None:
        return None

    r_record = records.get(r_uuid)
    if r_record is None:
        return None

    r_timestamp_str = r_record.get("timestamp")
    if not isinstance(r_timestamp_str, str):
        return None

    r_timestamp = _parse_timestamp(r_timestamp_str)
    if r_timestamp >= indexed_time_end:
        return None

    # Revert detected: truncate orphaned content
    truncate_from_time = getattr(client, "truncate_from_time")
    truncate_from_time(document_id, r_timestamp_str)
    return r_timestamp_str


def _build_records_map(transcript_path: Path) -> dict[str, dict[str, object]]:
    """Build UUID -> record map from transcript.

    Includes all records with UUIDs, including compaction summaries.
    Compaction summaries are needed for parent-child chain traversal
    but are filtered out during transcription.
    """
    records: dict[str, dict[str, object]] = {}
    for record, _ in iter_jsonl(transcript_path):
        uuid = record.get("uuid")
        if isinstance(uuid, str):
            records[uuid] = record
    return records


def execute_sync(
    transcript_path: Path,
    document_id: str,
    client: object,
) -> SyncResult:
    """Execute a complete sync operation using stateless algorithm.

    The stateless algorithm derives sync state from two sources:
    1. The JSONL transcript (source of truth for content)
    2. RagZoom document status (source of truth for what's indexed)

    This makes sync idempotent and crash-safe, eliminating external state files.

    Args:
        transcript_path: Path to the JSONL transcript
        document_id: Document ID to sync to (typically the transcript filename stem)
        client: RagZoom client with get_document_status(), batch_append(),
                and truncate_from_time() methods

    Returns:
        SyncResult describing what was done
    """

    # Get current head
    current_head = get_current_head(transcript_path)
    if current_head is None:
        # Empty transcript, nothing to sync
        return SyncResult(
            document_id=document_id,
            truncated=False,
            truncate_cutoff_time=None,
            steps_appended=0,
        )

    # Get indexed state from RagZoom document status
    get_document_status = getattr(client, "get_document_status")
    doc_status = get_document_status(document_id)

    # Parse indexed_time_end if document has temporal content
    indexed_time_end: datetime | None = None
    if doc_status.exists and doc_status.time_end is not None:
        indexed_time_end = _parse_timestamp(doc_status.time_end)

    # Build records map for stateless algorithm
    records = _build_records_map(transcript_path)

    # Build parent map with compaction bridging for ancestry traversal
    parent_map = build_parent_map(transcript_path)

    # Find truncation point using sliding window algorithm
    r_uuid, s_uuid = find_truncation_point(current_head, records, indexed_time_end)

    # Build list of UUIDs to append (using parent_map to bridge compactions)
    uuids_to_append = (
        []
        if s_uuid is None
        else build_ancestry_chain(current_head, r_uuid, records, parent_map)
    )

    # Detect and handle revert (returns cutoff time if truncation occurred)
    truncate_cutoff_time = _handle_revert_detection(
        r_uuid, indexed_time_end, records, client, document_id
    )

    if not uuids_to_append:
        # Nothing to append
        return SyncResult(
            document_id=document_id,
            truncated=truncate_cutoff_time is not None,
            truncate_cutoff_time=truncate_cutoff_time,
            steps_appended=0,
        )

    # Filter UUIDs to conversation steps (user/assistant messages only)
    steps = filter_to_steps(uuids_to_append, records)
    if not steps:
        return SyncResult(
            document_id=document_id,
            truncated=truncate_cutoff_time is not None,
            truncate_cutoff_time=truncate_cutoff_time,
            steps_appended=0,
        )

    # Convert steps to AppendUnits (steps_to_append_units already filters empty)
    non_empty = steps_to_append_units(steps, records)

    if not non_empty:
        return SyncResult(
            document_id=document_id,
            truncated=truncate_cutoff_time is not None,
            truncate_cutoff_time=truncate_cutoff_time,
            steps_appended=0,
        )

    # Batch append in chunks to avoid gRPC timeout on large syncs.
    # Each chunk completes within the default 30s timeout.
    chunk_size = 200
    batch_append = getattr(client, "batch_append")
    for i in range(0, len(non_empty), chunk_size):
        chunk = non_empty[i : i + chunk_size]
        batch_append(
            document_id,
            chunk,
            summarization_guidance=CONVERSATION_SUMMARIZATION_GUIDANCE,
        )

    return SyncResult(
        document_id=document_id,
        truncated=truncate_cutoff_time is not None,
        truncate_cutoff_time=truncate_cutoff_time,
        steps_appended=len(non_empty),
    )
