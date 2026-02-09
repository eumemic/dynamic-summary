---
name: memory-tool-usage
description: This skill should be used when needing to recall earlier conversation details, after compaction occurs, when resuming a session, when details feel fuzzy, or when the user references something from earlier in the conversation. Provides guidance on effective use of the `recall` memory tool.
---

# Memory Tool Usage

Guidance for effective use of the `recall` memory tool to recall information from earlier in the conversation after compaction.

## How the Tool Works

The `recall` tool uses server-side agentic search: you ask a question, and the server automatically searches through the conversation at multiple levels of detail to find the best answer. No manual zooming or budget tuning needed.

```python
recall(query="What was the authentication bug we discussed?")
# Returns: A concise answer synthesized from conversation history
```

## When to Use Memory Proactively

- **After compaction**: Query "What was I working on? What were the specific details?"
- **When resuming a session**: Refresh on recent decisions, open questions, implementation details
- **Before making changes to code discussed earlier**: "What did we decide about X?"
- **When user references something from earlier**: Don't guess, look it up
- **When details feel fuzzy**: If uncertain about specifics, query rather than assume

## Tips for Effective Queries

1. **Be specific**: "What error message did we see in the JWT validation?" is better than "authentication"

2. **Use natural language**: The tool understands semantic queries, so phrase questions naturally

3. **Multiple focused queries beat one broad query**: Ask targeted questions rather than one vague catch-all

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
