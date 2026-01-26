---
status: READY
---

# Temporal Document APIs

## Overview

Add two new APIs for temporal document management: `document-status` for querying document metadata and completion progress, and `truncate_from_time` for removing content after a given timestamp. These APIs enable stateless synchronization workflows where the document itself is the source of truth for sync state.

## Problem Statement

Currently, syncing temporal documents (like conversation transcripts) requires maintaining external state (an "append log") to track what content has been indexed. This external state can diverge from the actual document state, causing subtle bugs:

- Daemon crashes between append and state save → state desync
- Concurrent syncs → state corruption
- Corrupted state file → manual intervention required

By exposing document metadata and temporal truncation as first-class APIs, sync clients can derive all necessary state from the document itself.

## Requirements

### 1. Document Status API

A new `document-status` command and gRPC method that returns document metadata:

**CLI:**
```bash
ragzoom document-status <document-id> [--json]
```

**Response fields:**
| Field | Type | Description |
|-------|------|-------------|
| `document_id` | string | Document identifier |
| `exists` | bool | Whether the document exists |
| `is_temporal` | bool | Whether document has temporal metadata |
| `leaf_count` | int | Number of leaf nodes |
| `node_count` | int | Total nodes (leaves + inner) |
| `complete_forest_size` | int | Expected nodes when fully indexed: `2N - popcount(N)` |
| `completion_pct` | float | `node_count / complete_forest_size * 100` |
| `time_start` | string? | ISO 8601 timestamp of earliest content (null if non-temporal or empty) |
| `time_end` | string? | ISO 8601 timestamp of latest content (null if non-temporal or empty) |

**Example output (JSON):**
```json
{
  "document_id": "session-abc123",
  "exists": true,
  "is_temporal": true,
  "leaf_count": 100,
  "node_count": 142,
  "complete_forest_size": 197,
  "completion_pct": 72.1,
  "time_start": "2026-01-25T22:47:42Z",
  "time_end": "2026-01-26T07:04:15Z"
}
```

**Example output (human-readable):**
```
Document: session-abc123
Type: temporal
Leaves: 100
Nodes: 142 / 197 (72.1% complete)
Time range: 2026-01-25T22:47:42Z to 2026-01-26T07:04:15Z
```

**Rationale:** This lightweight endpoint provides sync clients with the key information they need (time_end for finding connection points, existence check) without executing a full query that could return large amounts of data.

### 2. Complete Forest Size Formula

The complete forest size uses the binary forest formula:

```python
def complete_forest_size(leaf_count: int) -> int:
    """Expected total nodes when fully indexed.

    RagZoom builds a forest of perfect binary trees.
    N leaves decompose into popcount(N) trees (binary representation).
    Each tree with 2^k leaves has 2^k - 1 inner nodes.
    Total inner nodes = N - popcount(N).
    Total nodes = N + (N - popcount(N)) = 2N - popcount(N).
    """
    if leaf_count <= 0:
        return 0
    popcount = bin(leaf_count).count("1")
    return 2 * leaf_count - popcount
```

**Examples:**
| Leaves | Binary | Popcount | Complete Forest Size |
|--------|--------|----------|---------------------|
| 8 | 0b1000 | 1 | 15 |
| 7 | 0b0111 | 3 | 11 |
| 5 | 0b0101 | 2 | 8 |
| 100 | 0b1100100 | 3 | 197 |

### 3. Truncate from Time API

A new gRPC method to remove all nodes whose time range extends beyond a cutoff:

**Method:** `TruncateFromTime(document_id, cutoff_time)`

**Behavior:**
- Delete all nodes where `time_end > cutoff_time`
- This includes both leaf nodes and inner (summary) nodes
- Orphaned children (kept nodes whose parents were deleted) get `parent_id = NULL`
- Delete corresponding vectors from the vector index

**Rationale:** This is the temporal analog of the existing `truncate_from_span` method. The `time_end > cutoff` condition correctly handles both leaves (content after cutoff) and summaries (any summary covering content after cutoff).

**Validation:**
- Document must exist → error if not found
- Document must be temporal → error if `is_temporal = false`
- Cutoff time must be valid ISO 8601 → error if malformed

**Response:**
```python
TruncateResult(
    document_id: str,
    deleted_node_ids: list[str],
    cutoff_time: str,  # Echo back the cutoff
)
```

### 4. gRPC Service Definition

```protobuf
message DocumentStatusRequest {
  string document_id = 1;
}

message DocumentStatusResponse {
  string document_id = 1;
  bool exists = 2;
  bool is_temporal = 3;
  int32 leaf_count = 4;
  int32 node_count = 5;
  int32 complete_forest_size = 6;
  float completion_pct = 7;
  optional string time_start = 8;
  optional string time_end = 9;
}

message TruncateFromTimeRequest {
  string document_id = 1;
  string cutoff_time = 2;  // ISO 8601
}

message TruncateFromTimeResponse {
  string document_id = 1;
  repeated string deleted_node_ids = 2;
  string cutoff_time = 3;
}

service RagZoom {
  // ... existing methods ...
  rpc GetDocumentStatus(DocumentStatusRequest) returns (DocumentStatusResponse);
  rpc TruncateFromTime(TruncateFromTimeRequest) returns (TruncateFromTimeResponse);
}
```

## API Changes

### CLI

New command:
```bash
ragzoom document-status <document-id> [--json]
```

### Python Client

```python
from ragzoom.client import GrpcRagzoomClient

with GrpcRagzoomClient() as client:
    # Get document status
    status = client.get_document_status(document_id="session-abc")
    print(f"Time range: {status.time_start} to {status.time_end}")
    print(f"Completion: {status.completion_pct}%")

    # Truncate temporal document
    result = client.truncate_from_time(
        document_id="session-abc",
        cutoff_time="2026-01-26T03:00:00Z",
    )
    print(f"Deleted {len(result.deleted_node_ids)} nodes")
```

## Implementation Outline

### Phase 1: Document Status

1. Add `get_document_status` method to storage backend
   - Query document record for `is_temporal`
   - Count leaves and total nodes
   - Get min/max timestamps from leaf nodes
2. Add gRPC service method
3. Add CLI command
4. Unit tests for status calculation

### Phase 2: Truncate from Time

1. Add `delete_nodes_from_time` method to storage backend
   - SQL: `DELETE FROM tree_nodes WHERE document_id = ? AND time_end > ?`
   - First NULL out parent_id on kept children (same pattern as span truncate)
   - Delete vectors for removed nodes
2. Add gRPC service method
3. Unit tests for truncation correctness
4. Integration tests with temporal documents

## Acceptance Criteria

1. `ragzoom document-status <doc>` returns accurate metadata for existing documents
2. `document-status` returns `exists: false` for non-existent documents
3. `document-status` includes correct time range for temporal documents
4. `completion_pct` accurately reflects indexing progress using `2N - popcount(N)` formula
5. `truncate_from_time` removes all nodes where `time_end > cutoff`
6. `truncate_from_time` correctly orphans kept children (NULL parent_id)
7. `truncate_from_time` removes vectors for deleted nodes
8. `truncate_from_time` errors on non-temporal documents
9. Both APIs are accessible via gRPC and Python client

## Dependencies

- `temporal-metadata.md`: Temporal document support (COMPLETE)
- `daemon-lifecycle.md`: gRPC server infrastructure (COMPLETE)
