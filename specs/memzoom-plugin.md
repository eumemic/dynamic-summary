---
status: READY
---

# Memzoom Claude Code Plugin

## Overview

A Claude Code plugin that provides seamless memory augmentation for coding sessions. Syncs Claude Code transcripts to RagZoom and enables interactive memory retrieval with temporal zoom.

## Goals

1. **Plug-and-play** - Install plugin, get memory augmentation with zero config
2. **Full lifecycle** - Handles both syncing and querying
3. **Interactive zoom** - User can drill into specific time ranges
4. **Lives in ragzoom repo** - Single source of truth, versioned with core

## Non-Goals

- Multi-user/team memory sharing
- Cross-session memory (each session is its own document)
- Real-time streaming sync (batch sync is sufficient)

## Plugin Structure

```
ragzoom/
├── plugin/
│   ├── plugin.json           # Plugin manifest
│   ├── commands/
│   │   ├── memory.md         # /memory command
│   │   ├── memory-sync.md    # /memory-sync command
│   │   └── memory-zoom.md    # /memory-zoom command
│   ├── hooks/
│   │   └── auto-sync.json    # Auto-sync on session events
│   ├── skills/
│   │   └── memory-recall.md  # Skill for memory retrieval
│   └── agents/
│       └── memory-agent.md   # Agent for complex recall tasks
```

## Commands

### /memory

Query session memory with optional time window.

```markdown
---
description: Query your session memory
arguments:
  - name: query
    description: What to recall (empty for overview)
    required: false
  - name: time-range
    description: Time window (e.g., "last 30 minutes", "10:00-11:00")
    required: false
---

# Memory Query

Query the indexed session transcript for relevant context.

## Instructions

1. Ensure session is synced (run sync if needed)
2. Call `ragzoom query --json` with the query
3. Present results with temporal spans
4. Offer to zoom into specific time ranges
```

### /memory-sync

Sync current session transcript to RagZoom.

```markdown
---
description: Sync session transcript to memory
---

# Memory Sync

Sync the current Claude Code session transcript to RagZoom for indexing.

## Instructions

1. Find the session transcript path
2. Call `ragzoom sync-claude-code-transcript`
3. Report sync status (new messages indexed, total indexed)
```

### /memory-zoom

Zoom into a specific time range from previous query.

```markdown
---
description: Zoom into a time range from memory results
arguments:
  - name: range
    description: Time range to zoom into (index from previous results or ISO times)
    required: true
---

# Memory Zoom

Re-query memory focused on a specific time range.

## Instructions

1. Parse the time range (could be "1" for first tiling span, or explicit times)
2. Re-run query with --time-start and --time-end
3. Present zoomed results
4. Offer further zoom or return to overview
```

## Hooks

### Auto-Sync Hook

Sync transcript on session pause/resume:

```json
{
  "hooks": [
    {
      "event": "SessionStart",
      "command": "ragzoom sync-claude-code-transcript --session-path ${CLAUDE_SESSION_PATH}"
    }
  ]
}
```

## Skills

### memory-recall

Triggered when user asks about past conversation:

```markdown
---
description: Recall details from earlier in this session
triggers:
  - "what did we discuss"
  - "earlier we talked about"
  - "remember when"
  - "what was the"
---

# Memory Recall Skill

When the user asks about something from earlier in the session, use RagZoom
to retrieve the relevant context.

## Instructions

1. Sync session if not recently synced
2. Query memory with user's question
3. Present relevant excerpts with timestamps
4. Offer to zoom for more detail
```

## Interactive Zoom Workflow

### Query Results Display

When `/memory` returns results, format them for zoom selection:

```
Memory Results for "authentication flow"
========================================

[1] 10:15-10:32 (17 min) - Initial auth discussion
    Discussed OAuth vs JWT, decided on JWT with refresh tokens...

[2] 10:45-10:52 (7 min) - Token storage
    Implemented secure token storage in localStorage with encryption...

[3] 11:20-11:35 (15 min) - Auth middleware
    Created Express middleware for JWT validation...

─────────────────────────────────────────
Type a number to zoom in, or ask a follow-up question.
```

### Zoom Interaction

User types "2" or "/memory-zoom 2":

```
Zooming into 10:45-10:52: Token storage
=======================================

[2.1] 10:45-10:47 - Storage options
    You asked about secure storage options. I suggested:
    - HttpOnly cookies (most secure)
    - localStorage with encryption (simpler)
    - sessionStorage (per-tab only)

[2.2] 10:47-10:50 - Implementation
    We implemented localStorage approach with AES encryption...

[2.3] 10:50-10:52 - Testing
    Added tests for token encryption/decryption...

─────────────────────────────────────────
Type a number to zoom deeper, "back" to return, or ask a question.
```

## Dependencies

The plugin requires:
- RagZoom CLI installed (`pip install ragzoom` or `pip install git+...`)
- Daemon auto-starts on first command (per daemon-lifecycle spec)

## Configuration

### Plugin Settings

```yaml
# .claude/memzoom.local.yaml
auto_sync: true           # Sync on session start
sync_interval_minutes: 5  # Background sync interval (0 = manual only)
server_address: null      # Use default (localhost:50051)
```

### Environment Variables

```bash
RAGZOOM_SERVER_ADDRESS=localhost:50051  # Override server address
MEMZOOM_AUTO_SYNC=false                 # Disable auto-sync
```

## Installation

### From Git (development)

```bash
claude plugins add github:eumemic/dynamic-summary/ragzoom/plugin
```

### From PyPI (production)

```bash
pip install ragzoom
claude plugins add ragzoom
```

## Testing

### Manual Testing

1. Install plugin in Claude Code
2. Start a session, have a conversation
3. Run `/memory-sync` - verify sync completes
4. Run `/memory "what did we discuss"` - verify results
5. Zoom into a result - verify narrowed view
6. Zoom again - verify deeper drill-down

### Automated Testing

- Mock RagZoom CLI responses
- Verify command argument parsing
- Verify zoom state management
- Verify display formatting

## Rollout

1. Create plugin directory structure in ragzoom repo
2. Implement /memory-sync command
3. Implement /memory command with JSON output parsing
4. Implement /memory-zoom with state tracking
5. Add auto-sync hook
6. Add memory-recall skill
7. Documentation and examples
