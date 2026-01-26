---
status: READY
---

# Stateless Transcript Sync

## Overview

Refactor the Claude Code transcript sync to eliminate external state tracking (the "append log"). The sync algorithm derives all necessary state from two sources: the JSONL transcript (source of truth) and the RagZoom document status (indexed state). This makes sync idempotent, crash-safe, and eliminates an entire class of state desynchronization bugs.

## Problem Statement

The current transcript sync maintains an "append log" that tracks:
- Which message UUIDs have been synced
- Span positions in the RagZoom document
- Turn boundaries for revert detection

This external state can diverge from actual document state:

| Failure Mode | Cause | Impact |
|--------------|-------|--------|
| State desync | Crash between append and state save | Duplicates or missing content |
| Corrupted state | Partial write, disk full | Sync completely broken |
| Concurrent access | No locking | Lost entries, wrong state |
| Stale references | UUIDs deleted from transcript | Crashes on revert detection |

**Root cause:** The append log duplicates information that already exists in the JSONL (message order, timestamps, ancestry) and RagZoom (what's indexed, time range).

## Requirements

### 1. Eliminate the Append Log

Remove all append log machinery:
- `AppendLog` class
- `AppendEntry` dataclass
- State file at `$RAGZOOM_STATE_DIR/<document_id>.jsonl`
- All UUID tracking for "what's been synced"

**Rationale:** The JSONL is the source of truth for content. RagZoom is the source of truth for what's indexed. We don't need a third source of truth.

### 2. Use Document Status for Indexed State

Query RagZoom for what's already indexed:

```python
status = client.get_document_status(document_id)
indexed_time_end = status.time_end  # None if document doesn't exist or is empty
```

This single value tells us: "RagZoom has content up to this timestamp."

### 3. Connection Point Algorithm

Find where the current transcript connects to indexed content using a sliding window walk:

```python
def find_truncation_point(
    head_uuid: str,
    records: dict[str, Record],
    indexed_time_end: datetime | None,
) -> tuple[str | None, str | None]:
    """Walk backward to find valid truncation point.

    Returns:
        (R, S) where:
        - R: UUID of the connection point (last indexed record), or None for first sync
        - S: UUID of first record to append (successor of R), or head_uuid for first sync
    """
    if indexed_time_end is None:
        # First sync: no indexed content, append everything
        return (None, head_uuid)

    S = None  # Successor (starts null at end of log)
    R = head_uuid

    while R is not None:
        record = records[R]

        # Stop when: R is in indexed range AND S is a turn boundary
        if record.timestamp <= indexed_time_end:
            if S is None or is_user_message(records[S]):
                return (R, S)

        # Slide window backward
        S = R
        R = record.parent_uuid

    # Walked entire chain without finding indexed content
    # This means complete reindex needed
    return (None, head_uuid)


def is_user_message(record: Record) -> bool:
    """True if record is a real user message (starts a turn)."""
    return record.type == "user" and not record.tool_use_result
```

**Key insight:** The sliding window `(R, S)` ensures we stop at a turn boundary. We only stop when:
1. `R.timestamp <= indexed_time_end` (R is within indexed content)
2. `S` is a UserMessage OR `S` is None (we're at a turn boundary)

This handles mid-turn reverts automatically by continuing to walk back until we hit a clean turn boundary.

### 4. Revert Detection and Truncation

Reverts are detected implicitly by comparing timestamps:

```python
R, S = find_truncation_point(head, records, indexed_time_end)

if R is not None and records[R].timestamp < indexed_time_end:
    # Gap between connection point and indexed end = revert happened
    # Content between R.timestamp and indexed_time_end is orphaned
    client.truncate_from_time(document_id, cutoff_time=records[R].timestamp)
```

**Normal append:** `R.timestamp == indexed_time_end` (approximately), no truncation needed.

**Revert case:** `R.timestamp < indexed_time_end`, truncate orphaned content.

### 5. Build Ancestry Chain

Collect records from S to head (the records we need to append):

```python
def build_ancestry_chain(head_uuid: str, stop_uuid: str | None, records: dict) -> list[str]:
    """Collect UUIDs from head back to (but not including) stop_uuid."""
    chain = []
    current = head_uuid

    while current is not None and current != stop_uuid:
        chain.append(current)
        current = records[current].parent_uuid

    chain.reverse()  # Chronological order
    return chain
```

**Important:** This only includes records in the current ancestry chain, automatically excluding orphaned records from past reverts (which remain in the append-only JSONL but are not ancestors of the current head).

### 6. Group into Turns and Append

Use existing turn grouping logic:

```python
uuids_to_append = build_ancestry_chain(head, R, records)
turns = group_into_turns(uuids_to_append, records)

units = []
for turn in turns:
    text = transcribe_turn(turn.uuids, records)
    units.append(AppendUnit(
        text=text,
        time_start=turn.time_start,
        time_end=turn.time_end,
    ))

client.batch_append(document_id, units)
```

### 7. First Sync Case

When `indexed_time_end` is None (document doesn't exist or is empty):
- `find_truncation_point` returns `(None, head_uuid)`
- No truncation needed
- Append entire ancestry chain from head back to root

### 8. Idempotency

Running sync twice produces the same result:
1. Second run queries document status, gets `indexed_time_end = T`
2. Walks backward, finds connection at T
3. No gap detected (T == T), no truncation
4. Ancestry chain from T to head is empty (or only contains already-indexed records)
5. No-op or minimal duplicate detection

**Note:** Exact idempotency depends on whether batch_append is idempotent for duplicate content. For safety, sync should check if ancestry chain yields any new turns before calling batch_append.

## API Changes

### transcript_sync.py

**Remove:**
- `AppendLog` class
- `AppendEntry` dataclass
- `_SessionAppendLog` class
- State file read/write logic
- UUID-based sync tracking

**Add:**
```python
def find_truncation_point(
    head_uuid: str,
    records: dict[str, Record],
    indexed_time_end: datetime | None,
) -> tuple[str | None, str | None]:
    """Find connection point using sliding window algorithm."""

def build_ancestry_chain(
    head_uuid: str,
    stop_uuid: str | None,
    records: dict[str, Record],
) -> list[str]:
    """Collect UUIDs from head to stop point in chronological order."""
```

**Modify:**
```python
def execute_sync(transcript_path: Path, document_id: str, client: RagzoomClient) -> SyncResult:
    # New flow:
    # 1. Get document status
    # 2. Find truncation point
    # 3. Truncate if revert detected
    # 4. Build ancestry chain
    # 5. Group into turns
    # 6. Batch append
```

### MCP Server

No changes needed. The MCP server calls `execute_sync` which now uses the stateless algorithm internally.

## Implementation Outline

### Phase 1: Add Core Functions

1. Implement `find_truncation_point` with sliding window
2. Implement `build_ancestry_chain`
3. Unit tests for both functions with various scenarios:
   - Normal append
   - Revert to turn boundary
   - Revert mid-turn (should round down)
   - First sync (empty document)
   - Complete reindex (no common ancestor)

### Phase 2: Refactor execute_sync

1. Replace append log logic with new algorithm
2. Use `client.get_document_status()` for indexed state
3. Use `client.truncate_from_time()` for reverts
4. Remove state file creation/reading
5. Integration tests with real transcripts

### Phase 3: Cleanup

1. Remove `AppendLog`, `AppendEntry`, `_SessionAppendLog`
2. Remove state file handling code
3. Update documentation
4. Remove `RAGZOOM_STATE_DIR` environment variable (if no longer needed)

## Acceptance Criteria

1. Sync works without any local state files
2. Normal append case: new turns appended correctly
3. Revert case: orphaned content removed, new content appended
4. Mid-turn revert: correctly rounds down to turn boundary
5. First sync: entire transcript indexed
6. Idempotent: running sync twice is safe (no duplicates, no errors)
7. Crash-safe: sync can be interrupted and resumed correctly
8. Concurrent-safe: multiple syncs on same document don't corrupt state (handled by RagZoom's document locking)

## Migration

No migration needed. The new algorithm works with existing RagZoom documents:
- Documents with content: status query returns time_end, sync continues from there
- Empty documents: treated as first sync

Old state files (if any exist) are simply ignored. They can be cleaned up manually or left in place.

## Dependencies

- `temporal-document-apis.md`: Document status and truncate_from_time APIs (READY)
- `timestamped-transcript-sync.md`: Turn grouping and temporal metadata (READY)

## Non-Goals

- Backward compatibility with append log state files (clean break)
- Span-based tracking (fully replaced by temporal tracking)
- Sub-turn granularity (turns remain atomic)
