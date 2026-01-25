---
status: COMPLETE
---

# JSON Output Mode for Query

## Overview

Add `--json` flag to `ragzoom query` that outputs machine-readable JSON instead of human-formatted text. Includes temporal span information for each tiling node to enable iterative zoom workflows.

## Goals

1. **Machine-readable output** - Enable programmatic consumption of query results
2. **Temporal spans** - Include span_start/span_end and time_start/time_end for each node
3. **Plugin integration** - Provide structured data for Claude Code plugin to present and zoom

## Non-Goals

- Changing the default human-readable output
- JSON output for other commands (future work)
- Streaming JSON output

## Usage

```bash
# Human output (default, unchanged)
ragzoom query "what happened?" -d session.txt

# JSON output
ragzoom query --json "what happened?" -d session.txt
```

## JSON Schema

```json
{
  "summary": "The assembled summary text...",
  "token_count": 1234,
  "seed_count": 3,
  "tiling_size": 5,
  "actual_span": {
    "start": 0,
    "end": 45678
  },
  "tiling": [
    {
      "node_id": "abc123",
      "text": "Node summary or leaf text...",
      "span_start": 0,
      "span_end": 1500,
      "time_start": "2024-01-21T10:00:00Z",
      "time_end": "2024-01-21T10:15:00Z",
      "height": 2,
      "is_seed": true,
      "token_count": 250
    },
    {
      "node_id": "def456",
      "text": "Another node...",
      "span_start": 1500,
      "span_end": 3000,
      "time_start": "2024-01-21T10:15:00Z",
      "time_end": "2024-01-21T10:30:00Z",
      "height": 0,
      "is_seed": false,
      "token_count": 180
    }
  ],
  "query": "what happened?",
  "document_id": "session.txt"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `summary` | string | Assembled summary from tiling nodes |
| `token_count` | int | Total tokens in summary |
| `seed_count` | int | Number of seed nodes retrieved |
| `tiling_size` | int | Number of nodes in tiling |
| `actual_span.start` | int | Actual span start (may differ from requested) |
| `actual_span.end` | int | Actual span end |
| `tiling` | array | Ordered list of tiling nodes |
| `tiling[].node_id` | string | Unique node identifier |
| `tiling[].text` | string | Node text content |
| `tiling[].span_start` | int | Character position in source document |
| `tiling[].span_end` | int | Character position in source document |
| `tiling[].time_start` | string? | ISO 8601 timestamp (null if not temporal) |
| `tiling[].time_end` | string? | ISO 8601 timestamp (null if not temporal) |
| `tiling[].height` | int | Tree height (0 = leaf) |
| `tiling[].is_seed` | bool | Whether this node was a seed |
| `tiling[].token_count` | int | Tokens in this node |
| `query` | string | Original query text |
| `document_id` | string | Document that was queried |

### Temporal Fields

- `time_start` and `time_end` are only present for temporal documents
- For non-temporal documents, these fields are `null`
- Times are always ISO 8601 with timezone (e.g., `2024-01-21T10:00:00Z`)

## Implementation

### CLI Changes

Add `--json` flag to query command:

```python
@click.option("--json", "output_json", is_flag=True, help="Output JSON instead of text")
def query(..., output_json: bool):
    ...
    if output_json:
        output = build_json_output(response)
        click.echo(json.dumps(output, indent=2))
    else:
        # existing human output
```

### Response Building

```python
def build_json_output(response: QueryResponse) -> dict:
    tiling = []
    for node_id in response.retrieval.tiling_ids:
        node = response.retrieval.nodes.get(node_id)
        if node:
            tiling.append({
                "node_id": node_id,
                "text": node.text,
                "span_start": node.span_start,
                "span_end": node.span_end,
                "time_start": node.time_start,  # May be None
                "time_end": node.time_end,      # May be None
                "height": node.height,
                "is_seed": node_id in response.retrieval.selected_ids,
                "token_count": node.token_count,
            })

    return {
        "summary": response.query_result.summary,
        "token_count": response.query_result.token_count,
        "seed_count": response.query_result.seed_count,
        "tiling_size": response.query_result.tiling_size,
        "actual_span": {
            "start": response.actual_start,
            "end": response.actual_end,
        },
        "tiling": tiling,
        "query": query_text,
        "document_id": document_id,
    }
```

## Use Cases

### Iterative Zoom

Plugin can present summary, user clicks a time range, plugin re-queries with narrowed window:

```bash
# Initial broad query
ragzoom query --json "summarize" -d session.txt

# User sees tiling spans, wants to zoom into 10:00-10:30
ragzoom query --json "summarize" -d session.txt \
  --time-start 2024-01-21T10:00:00Z \
  --time-end 2024-01-21T10:30:00Z
```

### Integration Testing

Programmatic verification of retrieval behavior:

```bash
result=$(ragzoom query --json "test" -d doc.txt)
span_end=$(echo "$result" | jq '.tiling[0].span_end')
```

## Testing

### Unit Tests

- JSON output matches schema
- Temporal fields present for temporal documents
- Temporal fields null for non-temporal documents
- Tiling order preserved

### Integration Tests

```bash
# Verify JSON is valid
ragzoom query --json "test" -d doc.txt | jq .

# Verify temporal spans
ragzoom query --json "test" -d temporal-doc.txt | jq '.tiling[0].time_start'
```

## Compatibility

- `--json` can combine with existing flags (`--debug`, `--profile`, etc.)
- When `--json` is used, `--debug` visualization is omitted (not JSON-compatible)
- Exit codes unchanged (0 success, non-zero error)
- Errors output as JSON when `--json` flag is present:
  ```json
  {"error": "Document not found", "code": "NOT_FOUND"}
  ```
