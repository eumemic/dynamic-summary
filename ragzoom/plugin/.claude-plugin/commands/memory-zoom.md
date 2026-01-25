---
allowed-tools: Bash, Read
description: Zoom into a specific time range from previous memory results
argument-hint: <range> [query]
---

# /memory-zoom

Zoom into a specific time range from previous `/memory` query results for more detail.

Arguments: "$ARGUMENTS"

## Overview

This command re-queries memory focused on a specific time range, providing more detailed (lower height, more verbatim) results. Use it after `/memory` returns numbered tiling spans.

## Process

1. **Parse Range**: Determine time bounds from span number or explicit times
2. **Query with Time Bounds**: Call `ragzoom query --json` with `--time-start` and `--time-end`
3. **Format Results**: Display with sub-numbering (e.g., 2.1, 2.2) showing zoom hierarchy
4. **Offer Navigation**: Suggest deeper zoom, return to overview, or new query

## Argument Parsing

- **Numeric index**: `/memory-zoom 2` - zoom into span [2] from previous results
- **Dotted notation**: `/memory-zoom 2.1` - zoom deeper into sub-span [2.1]
- **Explicit range**: `/memory-zoom "10:45-10:52"` or ISO 8601 timestamps
- **With query**: `/memory-zoom 2 "token storage"` - zoom with refined query

If no query provided, reuse the original query from the previous `/memory` call.

## Execution

### Step 1: Parse the Range Argument

If numeric (including dotted notation like "2.1"):
1. Find the previous `/memory` JSON output in conversation context
2. Locate the tiling span at that index (1-indexed) in the `tiling` array
3. Extract its `time_start` and `time_end` values (or `span_start`/`span_end` if non-temporal)

The JSON structure from `/memory` contains:
```json
{
  "tiling": [
    {"time_start": "2024-01-15T10:15:00", "time_end": "2024-01-15T10:32:00", "span_start": 0, "span_end": 2000, ...}
  ],
  "query": "original query text",
  "document_id": "session-abc123"
}
```

If the argument looks like a time (contains ":" or ISO 8601 format), parse directly as a time window.

### Step 2: Determine Session Document

```bash
session_id="$(basename "$CLAUDE_SESSION_PATH" .jsonl)"
```

### Step 3: Query with Time Bounds

Use the `document_id` and `query` from the previous `/memory` results.

```bash
# Temporal documents (time_start/time_end are set)
ragzoom query --json -d "$document_id" \
  --time-start "2024-01-15T10:45:00" \
  --time-end "2024-01-15T10:52:00" \
  "original query or refined query"

# Non-temporal documents (time fields are null, use character positions)
ragzoom query --json -d "$document_id" \
  --span-start 4500 \
  --span-end 6200 \
  "original query or refined query"
```

### Step 4: Display Zoomed Results

```
Zooming into [2] 10:45-10:52: Token storage
==========================================

[2.1] 10:45-10:47 (2 min) - height 1
    Discussed secure storage options...

[2.2] 10:47-10:50 (3 min) - height 0 (verbatim)
    Implemented localStorage with AES encryption...

-----------------------------------------
Zoom deeper: `/memory-zoom 2.1` | Back to overview: `/memory`
```

For non-temporal documents (where `time_start`/`time_end` are null in the JSON), format with span positions:

```
Zooming into [2] span 4500-6200
================================

[2.1] span 4500-5100 - height 1
    Content from this range...
```

## Error Handling

- **Invalid range**: Number doesn't match a tiling span. Show available spans.
- **No previous results**: Suggest running `/memory` first.
- **Empty results**: Time window too narrow. Suggest broadening.
- **Non-temporal document**: Explain that zoom uses character spans instead of times.

## Examples

```
/memory-zoom 2                       # Zoom into span [2]
/memory-zoom 2 "encryption"          # Zoom with refined query
/memory-zoom 2.1                     # Deeper zoom into sub-span
/memory-zoom "10:30-11:00"           # Explicit time range
```

## Follow-up Actions

After displaying results, remind the user:
- `/memory-zoom N.M` - Zoom deeper
- `/memory` - Return to overview
- `/memory "query"` - Fresh search
