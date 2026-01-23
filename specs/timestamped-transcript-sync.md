---
status: READY
---

# Timestamped Transcript Sync

## Overview

Extend the Claude Code transcript sync to index conversation turns with temporal metadata, enabling time-windowed queries like "what was discussed around 2pm?"

## Problem Statement

The current transcript sync (`transcript_sync.py`) converts Claude Code JSONL transcripts to text and indexes them in RagZoom. However:

1. **No timestamps**: Indexed content lacks temporal metadata, so time-windowed queries aren't possible
2. **Coarse chunking**: Uses compaction boundaries to define AppendUnits, which then get further split by fixed-length chunking. We want: 1 Turn = 1 AppendUnit = 1 Chunk.
3. **Ad-hoc transcription**: Custom JSONL→text code duplicates functionality available in the `claude-transcriber` library

Claude Code transcripts contain ISO 8601 timestamps on every message. This spec bridges that data to RagZoom's temporal metadata support, using `claude-transcriber` for clean transcription.

## Requirements

### 1. Per-Turn Chunking

Each conversation turn becomes exactly one leaf node:

```
Turn = UserMessage (AssistantMessage | ToolResult)*
```

Where:
- **UserMessage**: A `type: "user"` record WITHOUT `toolUseResult` field (real user input)
- **AssistantMessage**: A `type: "assistant"` record (may contain text, tool_use, or both)
- **ToolResult**: A `type: "user"` record WITH `toolUseResult` field (automatic tool response)

A turn contains the full interaction cycle: user asks → assistant responds → tools called → results returned → assistant continues → until next real user input.

**Excluded from turns:**
- Compaction summaries (`isCompactSummary: true`) - already LLM-generated
- Queue operations (`type: "queue-operation"`) - internal Claude Code state

**Edge cases:**
- Standalone user message with no assistant response (e.g., `/command`) → valid single-message turn

### 2. Turn Timestamp Assignment

Each turn gets a time range from its constituent messages:

| Field | Value |
|-------|-------|
| `time_start` | Timestamp of the first message in the turn (the user message) |
| `time_end` | Timestamp of the last message in the turn (last assistant message, or user message if standalone) |

Timestamps are extracted from the `timestamp` field in JSONL records, which are already ISO 8601 with timezone.

### 3. Batch Append with AppendUnit

Use `batch_append()` with `AppendUnit` objects that bundle text and timestamps:

```python
from ragzoom import AppendUnit

client.batch_append(
    document_id=doc_id,
    units=[
        AppendUnit(text=turn1_text, time_start=turn1_start, time_end=turn1_end),
        AppendUnit(text=turn2_text, time_start=turn2_start, time_end=turn2_end),
        AppendUnit(text=turn3_text, time_start=turn3_start, time_end=turn3_end),
    ],
)
```

This requires client-controlled chunking (`target_chunk_tokens=None`), enforced by the temporal metadata constraint (`temporal-metadata.md` §5).

### 4. Turn Grouping Algorithm

```python
def group_into_turns(
    uuids: list[str],
    records_by_uuid: dict[str, dict[str, object]],
) -> list[Turn]:
    """Group message UUIDs into turns.

    Returns:
        List of Turn objects, each containing:
        - uuids: list[str] - UUIDs in this turn
        - time_start: str - ISO 8601 timestamp of first message
        - time_end: str - ISO 8601 timestamp of last message
    """
```

**Algorithm:**
1. Filter to user/assistant messages (exclude compaction summaries, queue ops)
2. Walk through in order
3. Start new turn on each UserMessage (user without `toolUseResult`)
4. Accumulate AssistantMessages and ToolResults into current turn
5. Extract `time_start` from first UUID's timestamp, `time_end` from last

### 5. AppendEntry Tracking

Each turn's `AppendEntry.last_uuid` = the last message UUID in that turn.

On revert detection:
- Common ancestor found in middle of a turn → truncate to before that turn
- Re-index from the turn boundary

This maintains atomic turn semantics for sync state.

### 6. Transcription with claude-transcriber

Use the `claude-transcriber` library instead of custom JSONL→text code:

```python
from claude_transcriber import Transcriber

def transcribe_turn(uuids: list[str], records_by_uuid: dict) -> str:
    """Transcribe a single turn using claude-transcriber."""
    transcriber = Transcriber()
    parts = []
    for uuid in uuids:
        record = records_by_uuid.get(uuid)
        if record:
            result = transcriber.transcribe(record)
            if result:
                parts.append(result)
    return "\n\n".join(parts)
```

**Benefits:**
- Standard output format matching Claude Code `/export`
- Already handles tool use formatting, user text cleaning, compaction skipping
- Maintained separately, cleaner separation of concerns

## API Changes

### transcript_sync.py

**New dataclass:**
```python
@dataclass
class Turn:
    """A conversation turn with timestamp range."""
    uuids: list[str]
    time_start: str  # ISO 8601
    time_end: str    # ISO 8601
```

**New function:**
```python
def group_into_turns(
    uuids: list[str],
    records_by_uuid: dict[str, dict[str, object]],
) -> list[Turn]:
    """Group message UUIDs into conversation turns."""
```

**Modified function:**
```python
def execute_sync(...) -> SyncResult:
    # Changes:
    # 1. Group uuids_to_transcribe into turns
    # 2. Transcribe each turn separately
    # 3. Call batch_append with turn texts and timestamps
    # 4. Track last_uuid per turn in append log
```

### CLI

No CLI changes - sync command behavior unchanged, just adds timestamps internally.

## Implementation Outline

### Phase 1: Turn Grouping

1. Add `Turn` dataclass
2. Implement `group_into_turns()` with filtering and accumulation logic
3. Unit tests for turn boundary detection

### Phase 2: Timestamp Extraction

1. Extract `timestamp` field from JSONL records
2. Compute `time_start`/`time_end` per turn
3. Unit tests for timestamp extraction

### Phase 3: Sync Integration

1. Modify `execute_sync()` to use turn-level processing
2. Build `AppendUnit` list for `batch_append()`
3. Update `AppendEntry` tracking to use turn boundaries
4. Integration tests with real transcript files

### Phase 4: Validation

1. Verify temporal documents are created (check `is_temporal` flag)
2. Test time-windowed queries on synced transcripts
3. Test revert detection with turn-level granularity

## Acceptance Criteria

1. ⬚ Each conversation turn becomes exactly one leaf node
2. ⬚ Turn `time_start` = first message timestamp, `time_end` = last message timestamp
3. ⬚ Synced documents are temporal (`is_temporal=True`)
4. ⬚ Time-windowed queries work on synced transcripts
5. ⬚ Tool-only assistant messages are batched within their turn
6. ⬚ Standalone user messages (no response) create valid single-message turns
7. ⬚ Revert detection works at turn granularity
8. ⬚ Compaction summaries are not indexed (skipped)
9. ⬚ Uses `claude-transcriber` library for JSONL→text conversion

## Non-Goals

- **Subagent transcripts**: This spec covers main session transcripts only
- **Retroactive timestamping**: Existing non-temporal indexes require re-sync
- **Sub-turn granularity**: Individual messages within a turn share the turn's time range

## Dependencies

- `temporal-metadata.md` §5: Client-controlled chunking requirement
- `temporal-metadata.md` §12-13: AppendUnit API for batch_append()
- `claude-transcriber` pip package for JSONL→text transcription
