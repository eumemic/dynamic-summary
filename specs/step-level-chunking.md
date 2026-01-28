---
status: READY
---

# Step-Level Chunking for Transcript Sync

## Problem

The current transcript sync groups messages into "turns" (user prompt through assistant response cycle). Each turn becomes one leaf node. This is problematic because:

1. **Turns can be arbitrarily long** - With capable agents, a single turn may contain hours of work with dozens of tool calls
2. **Time-based retrieval is imprecise** - The `recall` tool can only zoom to turn-level granularity, defeating fine-grained temporal queries
3. **Wasted context** - Retrieving a turn to find one specific tool call pulls in the entire turn's content

## Solution

Replace turn-level chunking with step-level chunking where each meaningful JSONL record becomes its own leaf node with `time_start = time_end = record.timestamp`.

### What is a "Step"?

A step is any JSONL record that represents actual conversation content:

```python
def _should_include_record(record: dict[str, object]) -> bool:
    """Include only user and assistant messages, excluding meta/compaction."""
    record_type = record.get("type")
    if record_type not in ("user", "assistant"):
        return False
    if record.get("isCompactSummary"):
        return False
    if record.get("isMeta"):
        return False
    return True
```

This includes:
- **User messages** - prompts from the human
- **Tool results** - `type="user"` with `toolUseResult` field
- **Assistant messages** - responses with text and/or tool calls
- **Command outputs/expansions** - now their own steps (previously bundled with command invocation)

This excludes:
- `type="queue-operation"` - internal Claude Code state
- `isCompactSummary=True` - LLM-generated summaries (not actual conversation)
- `isMeta=True` - injected content (skill expansions, PDFs, templates)

### Timestamp Handling

Each step gets:
- `time_start = record.timestamp`
- `time_end = record.timestamp`

This is a point-in-time rather than a range, enabling precise temporal queries.

## Changes Required

### 1. Replace `Turn` with `Step` dataclass

```python
@dataclass
class Step:
    """A single conversation step with timestamp.

    Each step is one JSONL record that passes filtering.
    """
    uuid: str
    timestamp: str  # ISO 8601
```

### 2. Replace `group_into_turns()` with `filter_to_steps()`

```python
def filter_to_steps(
    uuids: list[str],
    records_by_uuid: dict[str, dict[str, object]],
) -> list[Step]:
    """Filter UUIDs to steps (user/assistant messages only).

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
```

### 3. Replace `turns_to_append_units()` with `steps_to_append_units()`

```python
def steps_to_append_units(
    steps: list[Step],
    records_by_uuid: dict[str, dict[str, object]],
) -> list[AppendUnit]:
    """Convert steps to AppendUnits for indexing.

    Each step is transcribed individually with time_start = time_end.
    """
    result: list[AppendUnit] = []
    for step in steps:
        text = transcribe_uuids_from_map([step.uuid], records_by_uuid)
        if text.strip():
            result.append(AppendUnit(
                text=text,
                time_start=step.timestamp,
                time_end=step.timestamp,
            ))
    return result
```

### 4. Simplify `find_truncation_point()`

The current algorithm uses a sliding window to find turn boundaries. With step-level granularity, every record is a valid truncation point.

Remove:
- `_is_user_prompt()` - no longer needed for boundary detection
- `is_user_message()` - no longer needed for boundary detection
- `_is_command_output_or_expansion()` - command outputs are now their own steps

Simplify to find the first record where `timestamp <= indexed_time_end`.

### 5. Update `SyncResult`

```python
@dataclass
class SyncResult:
    document_id: str
    truncated: bool
    truncate_cutoff_time: str | None
    steps_appended: int  # was: turns_appended
```

### 6. Update `execute_sync()`

Replace:
```python
turns = group_into_turns(uuids_to_append, records)
append_units = turns_to_append_units(turns, records)
```

With:
```python
steps = filter_to_steps(uuids_to_append, records)
append_units = steps_to_append_units(steps, records)
```

## Code to Remove

These functions become dead code:
- `Turn` dataclass
- `group_into_turns()`
- `turns_to_append_units()`
- `_build_turn()`
- `_is_user_prompt()`
- `is_user_message()` (if not used elsewhere)
- `_is_command_output_or_expansion()`

## Test Updates

Tests in `test_group_into_turns.py` should be replaced with tests for `filter_to_steps()`.

Key test cases:
1. User and assistant messages are included
2. Tool results (`toolUseResult`) are included as their own steps
3. Command outputs are their own steps (not grouped)
4. Meta records are filtered out
5. Compaction summaries are filtered out
6. Queue operations are filtered out
7. Records without timestamps are skipped
8. Each step gets `time_start = time_end = timestamp`

## Migration

No data migration needed - the RagZoom document format is unchanged. Only the granularity of leaves changes. Existing indexed documents will work but with coarser granularity until re-synced.

## Risks

1. **More leaves = more embeddings** - Each step generates an embedding call. A turn with 20 tool calls becomes 20+ leaves instead of 1. Monitor for performance impact.

2. **More summarization work** - The hierarchical summarizer will have more leaf pairs to summarize. This should be manageable since summarization is async.

3. **Smaller context per leaf** - Individual steps may lack context that was previously provided by the full turn. The summarization hierarchy should capture this context at higher levels.
