---
allowed-tools: Bash, Read
description: Query your session memory with optional time range
argument-hint: [query] [--time-range "last 30 minutes"]
---

# /memory

Query the indexed session transcript for relevant context using RagZoom.

Arguments: "$ARGUMENTS"

## Overview

This command queries your session memory (indexed transcript) and returns relevant excerpts with temporal spans. Results are formatted for interactive exploration - you can zoom into specific time ranges for more detail.

## Process

1. **Parse Arguments**: Extract query text and optional time range
2. **Sync if Needed**: Ensure session transcript is current
3. **Query Memory**: Call `ragzoom query --json` with parameters
4. **Format Results**: Present results with numbered spans for zoom selection
5. **Offer Follow-up**: Suggest zooming or refining the query

## Argument Parsing

- **Query only**: `/memory "authentication flow"` - searches for the term
- **Empty query**: `/memory` - returns a general overview of recent activity
- **With time range**: `/memory "auth" --time-range "last 30 minutes"` or `--time-range "10:00-11:00"`

Time ranges support:
- Relative: "last N minutes/hours"
- Absolute: "HH:MM-HH:MM" (today's times)
- ISO 8601: Full datetime strings

## Execution

### Step 1: Determine Session Document

The document ID is the current Claude Code session ID. You can find the session path from:
- `$CLAUDE_SESSION_PATH` environment variable (if set by hook)
- Or locate the JSONL in `~/.claude/projects/<project-hash>/`

Extract session ID: `basename "$session_path" .jsonl`

### Step 2: Sync Session (if recent changes)

```bash
ragzoom sync-claude-code-transcript "$session_path"
```

### Step 3: Query Memory

Build the query command based on arguments:

```bash
# Basic query
ragzoom query --json -d "$session_id" "your query here"

# With time range
ragzoom query --json -d "$session_id" --time-start "2024-01-15T10:00:00" --time-end "2024-01-15T11:00:00" "your query"

# Overview (empty query, minimal seeds)
ragzoom query --json -d "$session_id" --num-seeds 0 ""
```

### Step 4: Parse and Format Results

The JSON response structure:
```json
{
  "summary": "Combined summary text",
  "token_count": 500,
  "seed_count": 3,
  "tiling_size": 5,
  "actual_span": {"start": 0, "end": 10000},
  "tiling": [
    {
      "node_id": "...",
      "text": "Excerpt content...",
      "span_start": 0,
      "span_end": 2000,
      "time_start": "2024-01-15T10:15:00",
      "time_end": "2024-01-15T10:32:00",
      "height": 2,
      "is_seed": true,
      "token_count": 100
    }
  ],
  "query": "authentication flow",
  "document_id": "session-abc123"
}
```

### Step 5: Display Results

Format for user interaction:

```
Memory Results for "authentication flow"
========================================

[1] 10:15-10:32 (17 min) - height 2, seed
    Discussed OAuth vs JWT, decided on JWT with refresh tokens...

[2] 10:45-10:52 (7 min) - height 1
    Implemented secure token storage in localStorage with encryption...

[3] 11:20-11:35 (15 min) - height 0 (verbatim)
    Created Express middleware for JWT validation...

-----------------------------------------
Type a number to zoom in (e.g., `/memory-zoom 2`), or ask a follow-up question.
```

For non-temporal documents (time fields are null), show span positions instead of times.

## Error Handling

- **Document not found**: Session may not be synced yet. Offer to run sync.
- **No results**: The query didn't match any indexed content. Suggest broader terms.
- **Server not running**: RagZoom daemon auto-starts, but report if it fails.

## Examples

**Basic recall**:
```
/memory "what did we discuss about error handling"
```

**Time-bounded search**:
```
/memory "database schema" --time-range "last hour"
```

**Session overview**:
```
/memory
```

## Follow-up Actions

After displaying results, remind the user they can:
- `/memory-zoom N` - Zoom into tiling span N for more detail
- `/memory "refined query"` - Search with different terms
- `/memory-sync` - Force sync if results seem stale
