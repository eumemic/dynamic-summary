# Claude Code Transcript Architecture

Claude Code stores conversation transcripts locally as JSONL files. Understanding this structure is essential for debugging memory issues and testing the sync pipeline.

## Storage Location

```
~/.claude/projects/<project-path>/<session-id>.jsonl
```

### Path Components

- **Base directory**: `~/.claude/projects/`
- **Project path**: Working directory with `/` replaced by `-`
  - `/Users/tom/code/myproject` → `-Users-tom-code-myproject`
  - `/Users/tom/code/dynamic-summary/worktrees/worktree-1` → `-Users-tom-code-dynamic-summary-worktrees-worktree-1`
- **Session file**: `<uuid>.jsonl` where UUID is the session ID

### Finding Transcripts

```bash
# List all transcripts for current project
ls ~/.claude/projects/-Users-tom-code-dynamic-summary-worktrees-worktree-1/*.jsonl

# Find most recent transcript (likely the active session)
ls -lt ~/.claude/projects/-Users-tom-code-dynamic-summary-worktrees-worktree-1/*.jsonl | head -1

# Check file size (larger = longer conversation)
ls -lh ~/.claude/projects/-Users-tom-code-dynamic-summary-worktrees-worktree-1/*.jsonl
```

## JSONL Structure

Each line is a self-contained JSON object with a `type` field indicating the record type.

### Record Types

| Type | Description | Contains |
|------|-------------|----------|
| `user` | User messages | `message.content` with text |
| `assistant` | Claude responses | `message.content` with text and tool calls |
| `system` | System prompts | Context, reminders, CLAUDE.md content |
| `summary` | Compaction summaries | Compressed history after context limits |
| `file-history-snapshot` | File state | Tracked file backups at points in time |
| `queue-operation` | Task queue | Background task operations |

### Key Fields

Common fields across record types:

```json
{
  "type": "user|assistant|system|...",
  "uuid": "unique-message-id",
  "parentUuid": "parent-message-id (for threading)",
  "timestamp": "ISO-8601 timestamp",
  "message": { "content": "..." }
}
```

### Threading Model

Messages form a tree via `parentUuid`:
- Each message points to its parent
- Reverts create branches (same parent, different children)
- The "current head" is the most recent message in the active branch

## Transcript as Source of Truth

The JSONL file is the **authoritative record** of the conversation. The memory service indexes this into a searchable tree, but the transcript can always be re-indexed if needed.

This architecture enables:
- **Safe recovery**: Reset + re-sync rebuilds from scratch
- **Revert handling**: Service detects branch changes and truncates accordingly
- **Incremental sync**: Only new bytes since last cursor position are sent

## Inspecting Transcripts

### Count record types
```bash
cat session.jsonl | python3 -c "
import sys, json
types = {}
for line in sys.stdin:
    t = json.loads(line).get('type', '?')
    types[t] = types.get(t, 0) + 1
for t, c in sorted(types.items(), key=lambda x: -x[1]):
    print(f'{c:5} {t}')
"
```

### View a specific record
```bash
# First record
head -1 session.jsonl | python3 -m json.tool

# Record at line N
sed -n 'Np' session.jsonl | python3 -m json.tool
```

### Find message by UUID
```bash
grep "uuid-prefix" session.jsonl | python3 -m json.tool
```

## Memory Service Integration

The memory service:
1. **Syncs** transcript deltas via gRPC (`IngestSession`)
2. **Stores** raw JSONL in `session_raw_data.jsonl_content`
3. **Indexes** into a hierarchical tree with embeddings
4. **Tracks** cursor position for incremental updates

The `test-sync` command pushes a transcript to the PR environment for testing this pipeline without affecting production.
