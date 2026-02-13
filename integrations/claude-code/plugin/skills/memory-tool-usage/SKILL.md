---
name: memory-tool-usage
description: This skill should be used when needing to recall earlier conversation details, after compaction occurs, when resuming a session, when details feel fuzzy, or when the user references something from earlier in the conversation. Provides guidance on effective use of the `recall` memory tool.
---

# Memory Tool Usage

The `recall` tool is a conversation with your memory. Ask natural questions about what happened earlier — what you worked on, what you decided, what went wrong — and follow up to dig deeper.

```python
recall(query="What was I working on before the refactor?")
recall(query="What error did we see in the JWT validation?")
```

## Follow-up Questions

Each response includes a `Session: <id>` footer. Pass it back on follow-up queries so the conversation continues where it left off.

```python
# Initial question
recall(query="When did we discuss the authentication bug?")
# Returns answer + "Session: abc123"

# Follow-up — picks up where the last answer left off
recall(query="What was the root cause?", session_id="abc123")
```

## Time Constraints

Use `time_start`/`time_end` to focus on a specific period. Useful when a broad query reveals the relevant time window.

```python
recall(query="When did we discuss the authentication bug?")
# Answer mentions it was around 2024-01-15T14:00:00

recall(query="What was the root cause of the auth bug?",
       time_start="2024-01-15T14:00:00", time_end="2024-01-15T16:00:00")
```

## When to Use Memory Proactively

- **After compaction**: "What was I working on? What were the specific details?"
- **When resuming a session**: "What decisions did I make recently? What's still open?"
- **Before making changes to code discussed earlier**: "What did we decide about the API design?"
- **When user references something from earlier**: Don't guess, look it up
- **When details feel fuzzy**: If uncertain about specifics, query rather than assume

## Tips

1. **Ask natural questions**: "What error did we see in the auth flow?" works better than keyword lists like "authentication error JWT"
2. **Be specific about what you want to know**: "What was the root cause of the login bug?" is better than "login bug"
3. **Multiple focused questions beat one broad question**: Ask targeted questions rather than one vague catch-all

## Anti-patterns to Avoid

- Constructing keyword soup instead of asking a real question
- Proceeding with vague recollection when specifics matter
- Assuming the compaction summary captured everything important
- Guessing at implementation details instead of looking them up
