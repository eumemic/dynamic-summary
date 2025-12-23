# Memory Usage Guide

This guide covers effective use of the `remember` memory tool for recalling information from earlier in the conversation after compaction.

## Continuously Improve These Guidelines

These guidelines are a living document. As you use the memory tool:

- **Introspect on effectiveness**: After each retrieval, ask yourself: Did I get what I needed? Was there a more efficient approach?
- **Experiment with new patterns**: Try different query strategies, token budgets, and zoom sequences to discover what works best
- **Propose improvements**: When you discover a better practice or a new anti-pattern, pause your current work and suggest an update to these guidelines to the user

The goal is for each agent session to leave these guidelines better than it found them.

## When to Use Memory Proactively

- **After compaction**: Query "What was I working on? What were the specific details?"
- **When resuming a session**: Refresh on recent decisions, open questions, implementation details
- **Before making changes to code discussed earlier**: "What did we decide about X?"
- **When user references something from earlier**: Don't guess, look it up
- **When details feel fuzzy**: If you're uncertain about specifics, query rather than assume

## The Iterative Zoom Workflow

The memory tool is designed for iterative exploration, not single-shot queries.

### Step 1: Survey

Start broad to get the span layout:

```python
remember(query="authentication bug", token_budget=2000)

# Returns summaries + node spans like:
# [0-349459] height=9
# [349459-522506] height=8
# [522506-564383] height=6  <-- mentions auth bug
# ...
```

### Step 2: Zoom

Drill into the relevant span for more detail:

```python
remember(query="authentication bug", token_budget=2000,
         span_start=522506, span_end=564383)

# Same budget, smaller region = more verbatim content
```

### Step 3: Repeat

Continue zooming until you hit height=0 (verbatim) nodes.

## Understanding Node Heights

- **height=0**: Verbatim content from the original conversation
- **height=1+**: Increasingly compressed summaries
- **High spans** (near the end): Recent content, often still verbatim
- **Low spans** (near 0): Old content, usually summarized

If results are all high-level summaries (high heights), either:
1. Increase `token_budget` to fit more detail
2. Constrain `span_start`/`span_end` to a smaller region

## Anti-patterns to Avoid

### Parallel Full-Scope Queries

**Don't do this:**
```python
# BAD: Multiple queries all hitting the same high-level summaries
remember(query="summarization hints", token_budget=3000)
remember(query="structured node data", token_budget=3000)
remember(query="cost per node", token_budget=3000)
# ... all return redundant old summaries
```

This wastes tokens on the same high-level summaries repeated across queries.

**Do this instead:**
```python
# GOOD: Survey once, then zoom into the relevant region
remember(query="brainstorm session", token_budget=2000)
# Notice discussion is in spans 580000-650000

remember(query="summarization hints", span_start=580000, span_end=650000)
remember(query="cost per node", span_start=580000, span_end=650000)
# Now these hit verbatim content from the actual discussion
```

### Other Anti-patterns

- Proceeding with vague recollection when specifics matter
- Assuming the compaction summary captured everything important
- Not verifying details before acting on half-remembered context
- Guessing at implementation details instead of looking them up

## Tips for Effective Retrieval

1. **Recent content is often verbatim**: Spans near the compaction boundary haven't been summarized yet

2. **Use semantic queries**: The tool finds relevant content by meaning, not just keywords

3. **Budget vs. precision tradeoff**: Higher `token_budget` gives more content but may include less relevant nodes; constraining spans gives more precision

4. **Multiple focused queries beat one huge query**: After surveying, targeted queries into specific spans are more effective than one massive token budget
