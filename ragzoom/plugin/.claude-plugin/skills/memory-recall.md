---
allowed-tools: Bash, Read
description: Recall details from earlier in this session
triggers:
  - "what did we discuss"
  - "earlier we talked"
  - "remember when"
  - "what was"
  - "earlier in our conversation"
  - "mentioned before"
  - "we discussed"
  - "previously talked"
  - "what happened with"
  - "recall"
---

# Memory Recall Skill

When the user asks about earlier conversation content, query RagZoom to retrieve relevant context from the indexed session transcript.

## When to Use

Trigger when the user references past conversation:
- "What did we discuss about authentication?"
- "Earlier we talked about the database schema"
- "Remember when we fixed that bug?"
- "What was the error message we saw?"

## Process

1. **Extract Query**: Identify what the user is trying to recall
2. **Sync if Needed**: Ensure recent conversation is indexed
3. **Query Memory**: Search the session transcript
4. **Present Results**: Show relevant excerpts conversationally

## Execution

### Step 1: Parse Query

Extract search terms from the user's question:
- "What did we discuss about **authentication**?" → query: "authentication"
- "Remember when we **fixed the bug in the login**?" → query: "fixed bug login"
- "Earlier we talked about **database migrations**" → query: "database migrations"

### Step 2: Locate Session

```bash
if [ -n "$CLAUDE_SESSION_PATH" ]; then
  session_path="$CLAUDE_SESSION_PATH"
else
  # Find the most recently modified JSONL in ~/.claude/projects
  session_path=$(find ~/.claude/projects -name "*.jsonl" -type f 2>/dev/null -exec ls -t {} + | head -1)
fi
session_id=$(basename "$session_path" .jsonl)
```

### Step 3: Sync Session

```bash
ragzoom sync-claude-code-transcript "$session_path"
```

### Step 4: Query Memory

```bash
ragzoom query --json -d "$session_id" "extracted query terms"
```

### Step 5: Present Results

Format results conversationally:

```
Based on our earlier conversation about "authentication":

At 10:15-10:32, we discussed OAuth vs JWT. You asked about security
tradeoffs and we decided on JWT with refresh tokens...

At 10:45-10:52, we implemented token storage using localStorage
with AES encryption...

Would you like more detail on any part?
```

For non-temporal documents (time fields are null), use span references:

```
Earlier in our conversation about "authentication":

In the first part of the session (span 0-2000), we discussed OAuth vs JWT...

Would you like more detail?
```

## Response Guidelines

1. **Be Natural**: Present results conversationally, not as raw search output
2. **Include Context**: Mention timestamps or session phases when available
3. **Summarize First**: Lead with key points, then offer details
4. **Suggest Navigation**: Mention `/memory-zoom N` for drilling deeper

Example:
```
We discussed [topic] around [time]. The key points were:
- Point 1
- Point 2

I can provide more detail if you'd like.
```

## Error Handling

- **Session not synced**: Run sync first, then query
- **No results**: "I don't see discussion about that topic. Could you rephrase or be more specific?"
- **Server unavailable**: Auto-start handles this

## Examples

**User**: "What did we discuss about error handling?"
**Action**: Query "error handling", present relevant excerpts with timestamps

**User**: "Earlier we talked about the database schema, what was it?"
**Action**: Query "database schema", provide schema details discussed

**User**: "Remember when we fixed that authentication bug?"
**Action**: Query "authentication bug fix", show debugging session

## Follow-up

After recalling information:
- Suggest `/memory-zoom N` for more detail on specific excerpts
- Offer `/memory "refined query"` for different search terms
- Mention `/memory-sync` if results seem stale
