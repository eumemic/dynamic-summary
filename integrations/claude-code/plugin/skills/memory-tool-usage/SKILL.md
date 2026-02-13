---
name: memory-tool-usage
description: This skill should be used when needing to recall earlier conversation details, after compaction occurs, when resuming a session, when details feel fuzzy, or when the user references something from earlier in the conversation. Provides guidance on effective use of the `recall` memory tool.
---

# Memory Tool Usage

Guidance for effective use of the `recall` memory tool to recall information from earlier in the conversation after compaction.

## How the Tool Works

The `recall` tool uses server-side agentic search: you ask a natural language question, and the server searches through the conversation at multiple levels of detail to find the best answer. No manual zooming, keyword construction, or budget tuning needed — just ask what you want to know.

```python
recall(query="What was I working on before the refactor?")
# Returns: A concise answer synthesized from conversation history

recall(query="What error did we see in the JWT validation?")
# Returns: The specific error details from that discussion
```

### Session Resumption

Each recall response includes a `Session: <id>` footer. Pass this back as `session_id` on follow-up queries to resume the search agent's session — it remembers what it already found and can refine without starting over.

```python
# Initial broad query
recall(query="When did we discuss the authentication bug?")
# Returns answer + "Session: abc123"

# Follow-up within the same search session
recall(query="What was the root cause?", session_id="abc123")
# Agent already has context, can go deeper immediately
```

### Time Constraints

Use `time_start`/`time_end` to constrain the search to a specific period. This is useful for zooming into a known time range after a broad initial query reveals the relevant window.

```python
# Broad question first
recall(query="When did we discuss the authentication bug?")
# Answer mentions it happened around 2024-01-15T14:00:00

# Zoom into that time window for more detail
recall(query="What was the root cause of the auth bug?", time_start="2024-01-15T14:00:00", time_end="2024-01-15T16:00:00")
```

The server enforces these bounds — all internal recall calls are clamped to the specified window.

## When to Use Memory Proactively

- **After compaction**: "What was I working on? What were the specific details?"
- **When resuming a session**: "What decisions did I make recently? What's still open?"
- **Before making changes to code discussed earlier**: "What did we decide about the API design?"
- **When user references something from earlier**: Don't guess, look it up
- **When details feel fuzzy**: If uncertain about specifics, query rather than assume

## Tips for Effective Queries

1. **Ask natural questions**: "What error did we see in the auth flow?" works better than keyword lists like "authentication error JWT"

2. **Be specific about what you want to know**: "What was the root cause of the login bug?" is better than "login bug"

3. **Multiple focused questions beat one broad question**: Ask targeted questions rather than one vague catch-all

4. **Recent content is often verbatim**: Content near the compaction point hasn't been summarized yet, so recall is highly accurate

## Managing Your Memory Document

The session ID and transcript path are injected on every session start.

### Check Document Status

To see what's indexed (node count, time range, completion):

```bash
ragzoom document-status <session-id>
```

Example output:
```
Document: 31d97397-5ee6-4189-8cd6-f9e2b0f7ea42
Type: temporal
Leaves: 208
Nodes: 229 / 413 (55.4% complete)
Time range: 2026-01-25T22:47:42Z to 2026-01-26T17:42:56Z
```

### Reset Memory

To clear and re-sync your memory document:

```bash
# Wipe and re-sync from transcript
ragzoom-claude-code reset <transcript-path>

# Wipe only (no re-sync)
ragzoom-claude-code reset <transcript-path> --no-resync
```

## Anti-patterns to Avoid

- Proceeding with vague recollection when specifics matter
- Assuming the compaction summary captured everything important
- Not verifying details before acting on half-remembered context
- Guessing at implementation details instead of looking them up

## Continuous Improvement

These guidelines are a living document. After each retrieval:

- **Introspect on effectiveness**: Did you get what you needed? Was there a more efficient approach?
- **Experiment with new patterns**: Try different query strategies
- **Propose improvements**: When discovering a better practice or anti-pattern, suggest an update to these guidelines
