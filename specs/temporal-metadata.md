---
status: READY
---

# Temporal Metadata for Client-Controlled Chunking

## Overview

Add optional temporal metadata (`time_start` and `time_end` timestamps) to chunks in client-controlled chunking mode. This enables time-windowed queries analogous to existing span-windowed queries, where time ranges map to document spans for retrieval.

## Problem Statement

RagZoom currently supports spatial navigation (document spans) but not temporal navigation. For time-series data like conversation transcripts, log files, or event streams, users need to query "what happened between T1 and T2" in addition to "what's in this section of the document."

## Requirements

### 1. Temporal Documents

- Documents are **all-or-nothing temporal**: either all nodes have timestamps or none do
- Document metadata includes `is_temporal: bool` flag (stored in database)
- **Temporality is inferred from first append**:
  - First append WITH timestamps → document becomes temporal (requires timestamps on all future appends)
  - First append WITHOUT timestamps → document becomes non-temporal (rejects timestamps on all future appends)
- Temporal documents REQUIRE timestamps on all appended chunks (error if missing)
- Non-temporal documents REJECT timestamps on appended chunks (error if provided)

**Rationale**: Mixing timestamped and non-timestamped content creates ambiguity. A document either represents temporally-ordered data or it doesn't. Inferring temporality from first append avoids redundant API parameters.

### 2. Timestamp Format

- **ISO 8601 string format with timezone**
  - Examples: `"2024-01-21T14:30:00Z"`, `"2024-01-21T14:30:00+00:00"`, `"2024-01-21T14:30:00.123456-05:00"`
  - MUST include timezone information (Z suffix or offset)
  - Parsed using Python's `datetime.fromisoformat()`
- Clients can provide:
  - Both `time_start` and `time_end` (different timestamps for chunk boundaries)
  - Single timestamp (used for both start and end)
- Internally converted to Unix timestamp (float seconds) for storage and queries

**Rationale**: ISO 8601 with timezone is human-readable, universally supported, and unambiguous. Internal conversion to Unix timestamp enables efficient numeric comparisons.

### 3. Tree Propagation

Timestamps propagate up the tree like spans:
- **Leaves**: Get exactly what the client sends (`time_start`, `time_end`)
- **Inner nodes**:
  - `time_start = left_child.time_start`
  - `time_end = right_child.time_end`

**Rationale**: Inner nodes represent time ranges spanning their children, maintaining the tree's hierarchical time coverage.

### 4. Time-Windowed Queries

Time windows map to span windows:

1. **Query Input**: `time_start` and `time_end` parameters
2. **Leaf Lookup**:
   - `leaf_start`: Earliest leaf L where `query.time_start <= L.time_end`
   - `leaf_end`: Latest leaf L where `L.time_start <= query.time_end`
3. **Span Mapping**: Use `leaf_start.span_start` and `leaf_end.span_end` as span window parameters
4. **Execute**: Run existing span-windowed query logic

**Overlap Semantics**: Any leaf whose time range `[time_start, time_end]` overlaps the query window is included.

**Rationale**: This design reuses ALL existing span query logic (coverage builder, edge-max calculations, vector filtering). Time is metadata that guides span selection, not a parallel query system.

### 5. Validation Rules

**At index time:**
- Temporal document + missing timestamps → Error
- Non-temporal document + provided timestamps → Error
- `time_end < time_start` on any leaf → Error
- Inner node timestamps MUST match children (enforced by tree builder)

**At query time:**
- Time-windowed query on non-temporal document → Error (clear message)
- `time_end < time_start` in query → Error

**Rationale**: Fail fast with clear errors. No silent fallbacks or dummy values.

## API Changes

### Python Client API

```python
# Single append with timestamp
ragzoom.append(
    document_id="session_abc",
    text="User said hello",
    timestamp="2024-01-21T14:30:00Z",  # Optional, single timestamp or (start, end) tuple
)

# Batch append with timestamps (ISO 8601 - any valid format with timezone)
ragzoom.batch_append(
    document_id="session_abc",
    units=[
        "User said hello",
        "Assistant responded",
    ],
    timestamps=[
        "2024-01-21T14:30:00Z",           # Single timestamp (start = end)
        ("2024-01-21T14:30:05Z", "2024-01-21T14:30:12Z"),  # Range (start, end)
    ],
)

# Document temporality is INFERRED from first append:
# - First append WITH timestamps → document becomes temporal (requires timestamps on all future appends)
# - First append WITHOUT timestamps → document becomes non-temporal (rejects timestamps on all future appends)

# Query with time window
ragzoom.query(
    document_id="session_abc",
    query_text="authentication bug",
    time_start="2024-01-21T14:00:00Z",
    time_end="2024-01-21T15:00:00Z",
)
```

### gRPC Protocol

Extend `AppendTextRequest`:
```protobuf
message AppendTextRequest {
  string document_id = 1;
  bytes content = 2;
  bool collect_telemetry = 3;
  bool replace_existing = 4;
  optional Timestamp timestamp = 5;  // Optional timestamp for this chunk
}
```

Extend `BatchAppendTextRequest`:
```protobuf
message BatchAppendTextRequest {
  string document_id = 1;
  repeated bytes units = 2;
  bool collect_telemetry = 3;
  repeated Timestamp timestamps = 4;  // Optional, parallel to units (must match length if provided)
}

message Timestamp {
  string time_start = 1;  // ISO 8601 with timezone (e.g., "2024-01-21T14:30:00Z" or "2024-01-21T14:30:00+00:00")
  optional string time_end = 2;  // If omitted, time_end = time_start
}
```

Extend `ExecuteQueryRequest`:
```protobuf
message ExecuteQueryRequest {
  // ... existing fields ...
  optional string time_start = 20;  // ISO 8601 with timezone
  optional string time_end = 21;    // ISO 8601 with timezone
}
```

Extend `DocumentStatus`:
```protobuf
message DocumentStatus {
  // ... existing fields ...
  bool is_temporal = 5;
}
```

**ISO 8601 Parsing**: Accept any valid ISO 8601 format with timezone information. Examples:
- `2024-01-21T14:30:00Z` (UTC with Z suffix)
- `2024-01-21T14:30:00+00:00` (UTC with offset)
- `2024-01-21T14:30:00.123456+00:00` (with microseconds)
- Use Python's `datetime.fromisoformat()` for parsing
- Reject timestamps without timezone info (e.g., `2024-01-21T14:30:00`) with clear error

## Data Model Changes

### Database Schema

**PostgresTreeNode** (add columns):
```python
time_start: Mapped[float | None] = mapped_column(Float, nullable=True)
time_end: Mapped[float | None] = mapped_column(Float, nullable=True)
```

**Document** (add column):
```python
is_temporal: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
```

**Migration**: Add columns with defaults, existing documents are non-temporal.

### TypedDict / Protocols

**NodeDataDict**:
```python
class NodeDataDict(TypedDict, total=False):
    # ... existing fields ...
    time_start: float | None
    time_end: float | None
```

**TreeNode Protocol**:
```python
@runtime_checkable
class TreeNode(Protocol):
    # ... existing fields ...
    time_start: float | None
    time_end: float | None
```

### Vector Metadata

Extend `normalize_metadata_from_dict()`:
```python
def normalize_metadata_from_dict(meta: dict[str, object]) -> MetaDict:
    return {
        # ... existing fields ...
        "time_start": coerce_float(meta.get("time_start")),
        "time_end": coerce_float(meta.get("time_end")),
    }
```

**Note**: Vector backends (pgvector, chroma, python) automatically inherit these fields through the normalization layer.

## Implementation Outline

### Phase 1: Storage & Tree Building

1. Add database columns (`time_start`, `time_end` to nodes, `is_temporal` to documents)
2. Extend `NodeDataDict` and `TreeNode` protocol
3. Update `LeafSpec` dataclass to include timestamps
4. Modify `_build_leaf_specs()` to accept and propagate timestamps
5. Modify `build_tree()` to compute inner node timestamps from children
6. Update vector metadata normalization

### Phase 2: Indexing API

1. Extend gRPC proto with timestamp fields (`AppendTextRequest` and `BatchAppendTextRequest`)
2. Update `GrpcRagzoomClient` to send timestamps
3. Extend Python API:
   - `RagZoom.append(text, timestamp=None)` - single timestamp or (start, end) tuple
   - `RagZoom.batch_append(units, timestamps=None)` - list parallel to units
4. Update `AppendExecutor` to:
   - Infer `is_temporal` from presence of timestamps on first append
   - Set `document.is_temporal` flag in database
   - Validate timestamp presence vs existing `is_temporal` flag on subsequent appends
   - Parse ISO 8601 to Unix timestamp (float seconds) using `datetime.fromisoformat()`
   - Pass timestamps to `_build_leaf_specs()`
5. Add validation for `time_end >= time_start`

### Phase 3: Query API & Time→Span Mapping

1. Extend `ExecuteQueryRequest` proto with time window fields
2. Add `time_start` / `time_end` parameters to Python `query()` method
3. Implement `get_leaf_at_time_position()` in node repository:
   - SQL query for earliest/latest leaves overlapping time window
   - Return leaf with its `span_start` / `span_end`
4. Add time→span mapping logic in `retrieve.py`:
   - Detect time-windowed query
   - Look up `leaf_start` and `leaf_end`
   - Map to `span_start` / `span_end`
   - Fall through to existing span query path
5. Add validation: reject time queries on non-temporal documents

### Phase 4: Testing

1. **Unit tests**:
   - Timestamp validation (missing, mismatched, invalid format)
   - Tree propagation (verify inner node timestamps)
   - ISO 8601 parsing edge cases
2. **Integration tests**:
   - Index temporal document, query with time window
   - Verify leaf lookup correctness
   - Verify time→span mapping accuracy
   - Mixed queries (time + content)
3. **Error cases**:
   - Time query on non-temporal document
   - Temporal append without timestamps
   - Non-temporal append with timestamps

## Open Questions

None. All design decisions resolved.

## Acceptance Criteria

1. ✅ Client can index chunks with ISO 8601 timestamps via `append()` and `batch_append()`
2. ✅ Document temporality is inferred from first append (with/without timestamps)
3. ✅ Timestamps propagate correctly through tree (leaves = client values, inner nodes = children boundaries)
4. ✅ Client can query with `time_start` and `time_end` parameters
5. ✅ Time windows map correctly to span windows (leaf lookup + span mapping)
6. ✅ Validation enforces all-or-nothing temporality per document:
   - Temporal doc + missing timestamps → Error
   - Non-temporal doc + provided timestamps → Error
7. ✅ Time-windowed queries produce correct results (retrieve content within time range)
8. ✅ Attempting time query on non-temporal document raises clear error
9. ✅ Vector metadata includes `time_start` and `time_end` fields
10. ✅ Accept any valid ISO 8601 format with timezone, reject timestamps without timezone

## Non-Goals

- **Automatic timestamp inference**: Clients MUST provide timestamps explicitly
- **Time-based indexing strategies**: Timestamps are metadata only, not index keys
- **Cross-document temporal queries**: Time windows remain single-document scoped
- **Timezone conversion**: Server parses ISO 8601 as-is; clients handle timezone conversion if needed
- **Sub-second precision beyond microseconds**: Float seconds (Unix timestamp) provides microsecond precision, which is sufficient for most use cases

## Future Enhancements

- **Relative time queries**: "last hour", "previous 5 messages"
- **Time-based truncation**: Delete all content before/after timestamp
- **Temporal analytics**: Query frequency over time, temporal gaps
- **Multi-document timelines**: Query across multiple temporal documents by time range
