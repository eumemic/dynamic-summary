# Temporal Documents

Temporal documents store time-series data with timestamp metadata, enabling time-windowed queries like "what happened between 2pm and 3pm" in addition to content-based search.

## When to Use Temporal Documents

Use temporal documents for:
- **Conversation transcripts** - Chat logs, meeting notes, support tickets
- **Event streams** - Log files, audit trails, system events
- **Time-series content** - News feeds, sensor data with text annotations
- **Any content where "when" matters** - If you need to query by time range, use temporal documents

## Quick Start

### 1. Create a Config File

Temporal documents **require** `target_chunk_tokens: null` (client-controlled chunking):

```json
{
  "target_chunk_tokens": null
}
```

Save this as `temporal-config.json`.

### 2. Index with Timestamps

```bash
# Start the server (if not already running)
ragzoom server start

# Index temporal content via Python API
python -c "
from ragzoom.client import GrpcRagzoomClient

with GrpcRagzoomClient() as client:
    # First append determines temporality - timestamps make it temporal
    client.batch_append(
        document_id='meeting-notes',
        units=[
            {'content': b'Meeting started, Alice joined', 'time_start': '2024-01-21T14:00:00Z', 'time_end': '2024-01-21T14:00:05Z'},
            {'content': b'Bob presented Q4 results', 'time_start': '2024-01-21T14:05:00Z', 'time_end': '2024-01-21T14:15:00Z'},
            {'content': b'Discussion about budget allocation', 'time_start': '2024-01-21T14:15:00Z', 'time_end': '2024-01-21T14:30:00Z'},
        ],
        target_chunk_tokens=None,  # Required for temporal documents
    )
"
```

### 3. Query by Time Window

```bash
# Query content within a time range
ragzoom query "What was discussed?" -d meeting-notes \
  --time-start "2024-01-21T14:10:00Z" \
  --time-end "2024-01-21T14:20:00Z"
```

## Configuration

### Required Setting: `target_chunk_tokens: null`

Temporal documents require client-controlled chunking. This is mandatory because:

1. **Timestamps are semantic metadata** - You provide one timestamp per logical unit (e.g., one conversation turn)
2. **Server chunking breaks semantics** - If the server splits your content, it can't assign meaningful timestamps to the fragments
3. **One-to-one mapping** - Each unit you append becomes exactly one leaf node with your timestamps

### Config File Example

```json
{
  "target_chunk_tokens": null,
  "summarization_guidance": "These are meeting notes. Preserve participant names, action items, and decisions."
}
```

### CLI Usage

```bash
# Using config file
ragzoom index meeting.txt --config temporal-config.json --document-id meeting

# Or specify directly
ragzoom index meeting.txt --target-chunk-tokens none --document-id meeting
```

### Error: Config Mismatch

If you try to index temporal content without `target_chunk_tokens: null`, you'll see:

```
Error: Temporal documents require target_chunk_tokens=null in config.

Temporal documents preserve a one-to-one mapping between your input units and
leaf nodes, which is required for accurate timestamp-based queries.

To fix:
  - Config file: Set "target_chunk_tokens": null
  - CLI flag: Use --target-chunk-tokens none
```

## Timestamp Format

RagZoom accepts ISO 8601 timestamps with timezone information:

```
2024-01-21T14:30:00Z           # UTC with Z suffix
2024-01-21T14:30:00+00:00      # UTC with offset
2024-01-21T14:30:00-05:00      # EST timezone
2024-01-21T14:30:00.123456Z    # With microseconds
```

**Important**: Timezone is required. Timestamps without timezone (e.g., `2024-01-21T14:30:00`) are rejected.

## How It Works

### Document Temporality

Documents are **all-or-nothing temporal**:
- First append **with** timestamps → document becomes temporal (requires timestamps on all future appends)
- First append **without** timestamps → document becomes non-temporal (rejects timestamps on future appends)

This prevents ambiguity from mixing timestamped and non-timestamped content.

### Timestamp Propagation

Timestamps propagate through the tree hierarchy:
- **Leaf nodes**: Store exactly what you provide (`time_start`, `time_end`)
- **Inner nodes**: Computed from children (`time_start = left_child.time_start`, `time_end = right_child.time_end`)

### Time-Windowed Queries

When you query with `--time-start` and `--time-end`:
1. RagZoom finds leaves whose time ranges overlap your query window
2. Maps those leaves to document spans
3. Retrieves content from those spans using existing query logic

Any leaf whose `[time_start, time_end]` overlaps your query window is included.

## Python API

### Single Append

```python
from ragzoom.client import GrpcRagzoomClient

with GrpcRagzoomClient() as client:
    client.append_text(
        document_id="events",
        content=b"User logged in",
        timestamp=("2024-01-21T14:30:00Z", "2024-01-21T14:30:01Z"),
        target_chunk_tokens=None,
    )
```

### Batch Append

```python
from ragzoom.client import GrpcRagzoomClient

with GrpcRagzoomClient() as client:
    client.batch_append(
        document_id="chat-log",
        units=[
            {"content": b"Hello!", "time_start": "2024-01-21T14:00:00Z", "time_end": "2024-01-21T14:00:00Z"},
            {"content": b"Hi there!", "time_start": "2024-01-21T14:00:05Z", "time_end": "2024-01-21T14:00:05Z"},
        ],
        target_chunk_tokens=None,
    )
```

### Time-Windowed Query

```python
from ragzoom.client import GrpcRagzoomClient

with GrpcRagzoomClient() as client:
    result = client.execute_query(
        query="What happened?",
        document_id="chat-log",
        time_start="2024-01-21T14:00:00Z",
        time_end="2024-01-21T14:01:00Z",
    )
    print(result.summary)
```

## JSON Output

When using `--json` with temporal documents, the output includes time information:

```bash
ragzoom query "summary" -d meeting --json
```

```json
{
  "summary": "...",
  "time_start": "2024-01-21T14:00:00Z",
  "time_end": "2024-01-21T14:30:00Z",
  "nodes": [...]
}
```

## Common Patterns

### Conversation Indexing

Each conversation turn becomes one leaf with its timestamp:

```python
turns = [
    {"role": "user", "content": "Hello", "timestamp": "2024-01-21T14:00:00Z"},
    {"role": "assistant", "content": "Hi! How can I help?", "timestamp": "2024-01-21T14:00:02Z"},
]

units = [
    {"content": f"{t['role']}: {t['content']}".encode(), "time_start": t["timestamp"], "time_end": t["timestamp"]}
    for t in turns
]

client.batch_append(document_id="session", units=units, target_chunk_tokens=None)
```

### Log File Processing

```python
import re
from datetime import datetime

log_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.+)'
units = []

for line in log_lines:
    match = re.match(log_pattern, line)
    if match:
        ts = datetime.fromisoformat(match.group(1)).isoformat() + "Z"
        units.append({"content": line.encode(), "time_start": ts, "time_end": ts})

client.batch_append(document_id="app-logs", units=units, target_chunk_tokens=None)
```

## Validation Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "Temporal documents require target_chunk_tokens=null" | Timestamps provided but chunking enabled | Set `target_chunk_tokens: null` in config |
| "Temporal document requires timestamps on all chunks" | Appending to temporal doc without timestamps | Add `time_start` and `time_end` to your append |
| "Non-temporal document does not accept timestamps" | Appending timestamps to non-temporal doc | Remove timestamps, or create a new temporal document |
| "Time-windowed query on non-temporal document" | Using `--time-start/--time-end` on regular doc | Query without time parameters, or re-index as temporal |
| "Invalid timestamp format" | Missing timezone or malformed ISO 8601 | Use format like `2024-01-21T14:30:00Z` |

## Best Practices

1. **Plan document type upfront** - Decide if a document needs temporal queries before first index
2. **Use consistent timezones** - Store all timestamps in UTC for consistency
3. **Include meaningful time ranges** - For instant events, use same timestamp for start and end
4. **Combine with content queries** - Time windows work with regular content search for precise results
