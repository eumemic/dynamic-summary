---
allowed-tools: Bash, Read
description: Manually sync session transcript to memory
---

# /memory-sync

Sync the current Claude Code session transcript to RagZoom for indexing.

Arguments: "$ARGUMENTS"

## Overview

This command manually triggers a sync of the current session's transcript to RagZoom. While syncing happens automatically on session start via hook, you can use this command to force a sync if:
- Results from `/memory` seem stale
- You want to ensure recent conversation is indexed before querying
- Auto-sync is disabled in your configuration

## Process

1. **Locate Session Transcript**: Find the JSONL file for the current session
2. **Call Sync**: Execute `ragzoom sync-claude-code-transcript`
3. **Report Status**: Show how many messages were synced

## Execution

### Step 1: Determine Session Path

The session transcript path can be found from:
- `$CLAUDE_SESSION_PATH` environment variable (if set by hook)
- Or search `~/.claude/projects/` for recent JSONL files

```bash
# Check if session path is set
if [ -n "$CLAUDE_SESSION_PATH" ]; then
  session_path="$CLAUDE_SESSION_PATH"
else
  # Find the most recently modified JSONL in ~/.claude/projects
  session_path=$(find ~/.claude/projects -name "*.jsonl" -type f -mmin -60 2>/dev/null | head -1)
fi
echo "Session path: $session_path"
```

### Step 2: Run Sync Command

```bash
ragzoom sync-claude-code-transcript "$session_path"
```

The sync command will:
- Parse the JSONL transcript
- Extract message turns with timestamps
- Incrementally index new content (skipping already-indexed messages)
- Return status showing progress

### Step 3: Report Results

After running the sync, report to the user:
- Whether sync was successful
- How many new messages were indexed
- Total messages now in memory

Example output to display:
```
Memory Sync Complete
====================
Session: abc123...
New messages indexed: 15
Total messages: 127
Last sync: just now
```

## Error Handling

- **Session not found**: If `$CLAUDE_SESSION_PATH` is not set and no recent JSONL found, explain that the session transcript couldn't be located.
- **Server not running**: The daemon will auto-start, but if it fails, report the error.
- **Sync failure**: Report the error message from the CLI.

## Examples

**Basic sync**:
```
/memory-sync
```

**Verify sync before querying**:
```
User: Let me sync first and then check what we discussed
/memory-sync
/memory "database schema"
```

## Follow-up Actions

After syncing, suggest:
- `/memory "query"` - Search your session memory
- `/memory` - Get an overview of the session
