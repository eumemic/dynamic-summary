---
name: memory-tool-usage
description: This skill should be used when needing to recall earlier conversation details, after compaction occurs, when resuming a session, when details feel fuzzy, or when the user references something from earlier in the conversation. Provides guidance on effective use of the `recall` memory tool.
---

# Memory Tool Usage

Guidance for effective use of the `recall` memory tool to recall information from earlier in the conversation after compaction.

## When to Use Memory Proactively

- **After compaction**: Query "What was I working on? What were the specific details?"
- **When resuming a session**: Refresh on recent decisions, open questions, implementation details
- **Before making changes to code discussed earlier**: "What did we decide about X?"
- **When user references something from earlier**: Don't guess, look it up
- **When details feel fuzzy**: If uncertain about specifics, query rather than assume

## The Iterative Zoom Workflow

The memory tool is designed for iterative exploration, not single-shot queries.

### Step 1: Survey

Start broad to get the time range layout:

```python
recall(query="authentication bug", token_budget=2000)

# Returns summaries + time ranges like:
# [2024-01-10T09:00:00Z to 2024-01-10T12:00:00Z] height=3
# [2024-01-10T14:00:00Z to 2024-01-10T16:30:00Z] height=2  <-- mentions auth bug
# [2024-01-10T16:30:00Z to 2024-01-10T18:00:00Z] height=1
```

### Step 2: Zoom

Drill into the relevant time range for more detail:

```python
recall(query="authentication bug", token_budget=2000,
         time_start="2024-01-10T14:00:00Z", time_end="2024-01-10T16:30:00Z")

# Same budget, smaller time range = more verbatim content
```

### Step 3: Zoom Aggressively

For specific details (commands run, exact decisions, code snippets), use **sub-hour windows**:

```python
recall(query="daemon restart commands", token_budget=1500,
         time_start="2024-01-10T15:22:00Z", time_end="2024-01-10T15:26:00Z")

# Tight window = height=0 verbatim content, no repetitive summaries
```

## Understanding Node Heights

- **height=0**: Verbatim content from the original conversation
- **height=1+**: Increasingly compressed summaries
- **Recent time ranges**: Often still verbatim (not yet summarized)
- **Old time ranges**: Usually summarized into higher-level nodes

If results contain repetitive context you already know:
1. Your time window is too broad - zoom in tighter
2. Use sub-hour windows when you need specifics

## Key Insight: Window Size Determines Content Type

| Window Size | What You Get | Use For |
|-------------|--------------|---------|
| Full session (no time params) | High-level summaries | Orientation, "what did we work on?" |
| Multi-hour window | Mix of summaries and some detail | Finding relevant periods |
| Sub-hour window | Mostly verbatim content | Specific commands, decisions, code |
| Minutes-only window | Pure verbatim (height=0) | Exact recall of what was said/done |

## Anti-patterns to Avoid

### Not Zooming Tight Enough

**Problem**: Querying broad time windows and getting the same problem/context description repeated:
```python
# BAD: 3-hour window returns summaries with repeated context
recall(query="verification", time_start="2024-01-10T00:38:00Z", time_end="2024-01-10T03:31:00Z")
```

**Solution**: Zoom to the specific moment:
```python
# GOOD: 4-minute window returns exactly what happened
recall(query="verification", time_start="2024-01-10T03:22:00Z", time_end="2024-01-10T03:26:00Z")
```

### Parallel Broad Queries

**Don't do this:**
```python
# BAD: Multiple broad queries all hitting the same summaries
recall(query="summarization hints", token_budget=3000)
recall(query="structured node data", token_budget=3000)
recall(query="cost per node", token_budget=3000)
```

**Do this instead:**
```python
# GOOD: Survey once, then zoom into specific time ranges
recall(query="brainstorm session", token_budget=2000)
# Notice discussion is in time range 14:00-15:30

recall(query="summarization hints", time_start="2024-01-10T14:00:00Z", time_end="2024-01-10T14:30:00Z")
recall(query="cost per node", time_start="2024-01-10T15:00:00Z", time_end="2024-01-10T15:30:00Z")
```

### Other Anti-patterns

- Proceeding with vague recollection when specifics matter
- Assuming the compaction summary captured everything important
- Not verifying details before acting on half-remembered context
- Guessing at implementation details instead of looking them up

## Tips for Effective Retrieval

1. **Recent content is often verbatim**: Time ranges near the compaction point haven't been summarized yet

2. **Use semantic queries**: The tool finds relevant content by meaning, not just keywords

3. **Budget vs. precision tradeoff**: Higher `token_budget` gives more content but may include less relevant nodes; constraining time ranges gives more precision

4. **Multiple focused queries beat one huge query**: After surveying, targeted queries into specific time ranges are more effective than one massive token budget

5. **Time ranges compound with query terms**: The query seeds expansion toward matching content; tight time windows ensure you get verbatim nodes

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

## Continuous Improvement

These guidelines are a living document. After each retrieval:

- **Introspect on effectiveness**: Did you get what you needed? Was there a more efficient approach?
- **Experiment with new patterns**: Try different query strategies, token budgets, and zoom sequences
- **Propose improvements**: When discovering a better practice or anti-pattern, suggest an update to these guidelines
