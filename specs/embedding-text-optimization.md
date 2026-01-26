---
status: READY
---

# Embedding Text Optimization

## Job to Be Done

Produce high-quality embedding vectors for leaves of any size by generating retrieval-optimized text that fits within embedding model limits. Large conversation turns (which can exceed 8000 tokens under client-managed chunking) must be compressed intelligently while preserving searchability.

## Background

The current embedding flow has a gap:

1. **Preceding context** gets summarized to ~200 tokens (`target_embedding_context_tokens`)
2. **Leaf text** is used verbatim
3. These are concatenated and sent to the embedding API

This breaks when leaf text exceeds the 8000 token embedding limit (text-embedding-3-small). Client-managed chunking allows leaves up to 50k characters (~12.5k tokens), causing embedding jobs to fail with `ValueError: Item 0 exceeds embedding token limit`.

Additionally, the current design misses an opportunity: embeddings work best when text is focused (100-500 tokens) rather than exhaustive. Longer text "dilutes" the vector across many concepts, weakening retrieval quality.

## Solution

Replace the two-step process (summarize context, concatenate with leaf) with a single-step process that produces **retrieval-optimized text** from the combined input.

New behavior:
- If `(preceding_context + leaf_text) <= target_embedding_tokens`: passthrough unchanged
- If exceeds target: LLM generates retrieval-optimized text within budget

The LLM prompt explicitly optimizes for cosine similarity matching - preserving key terms, named entities, and searchable concepts rather than just compressing.

## Configuration

### New Parameter

```python
target_embedding_tokens: int  # Default: 500
```

Target token count for text sent to embedding. When combined input exceeds this, LLM summarizes to fit.

### Removed Parameter

```python
target_embedding_context_tokens: int  # REMOVED - no longer needed
```

This parameter is obsolete. The new unified approach handles both context and leaf together.

### IndexConfig Changes

```python
@dataclass
class IndexConfig:
    target_embedding_tokens: int = 500  # New
    # target_embedding_context_tokens removed
    # ... rest unchanged
```

### default_config.json

```json
{
  "target_embedding_tokens": 500,
  ...
}
```

## Behavior

### Embedding Text Generation

```python
def prepare_embedding_text(preceding_context: str, leaf_text: str, target_tokens: int) -> str:
    combined = f"{preceding_context}\n{leaf_text}" if preceding_context else leaf_text
    combined_tokens = count_tokens(combined)

    if combined_tokens <= target_tokens:
        return combined  # Passthrough

    # LLM generates retrieval-optimized summary
    return llm_optimize_for_retrieval(preceding_context, leaf_text, target_tokens)
```

### LLM Prompt Design

The prompt should:
1. Take both preceding context and leaf text as input
2. Produce text optimized for semantic search matching
3. Preserve key terms, named entities, and concepts that users might query
4. Stay within the token budget
5. Prioritize leaf content over context when space is limited

Example prompt structure:
```
System: You produce text optimized for semantic search. Your output will be
embedded and matched against user queries via cosine similarity. Preserve
key terms, named entities, and searchable concepts.

User: Summarize the following into a retrieval-optimized text of at most
{target_words} words. Prioritize content from TARGET over CONTEXT.

<CONTEXT>
{preceding_context}
</CONTEXT>

<TARGET>
{leaf_text}
</TARGET>
```

### Storage

- **Leaf text in database**: Unchanged (original verbatim text preserved)
- **Embedding vector**: Based on retrieval-optimized text
- **preceding_context_summary field**: Repurposed to store the retrieval-optimized text (for debugging/inspection)

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| No preceding context | Optimize leaf text alone |
| Combined < target | Passthrough, no LLM call |
| Leaf alone > target | Compress leaf text to fit |
| Empty leaf | Embed empty string (existing behavior) |
| LLM output > target | Accept if within 8000 limit; retry logic applies |

## Migration

1. Remove `target_embedding_context_tokens` from IndexConfig
2. Add `target_embedding_tokens` with default 500
3. Update config validation to reject old parameter name (clear error message)
4. Existing embeddings remain valid; new embeddings use new logic

## Acceptance Criteria

1. **Oversized leaves embed successfully**: Leaves with 9000+ tokens no longer fail
2. **Passthrough for small content**: No LLM call when combined tokens <= target
3. **Retrieval quality**: Embeddings match relevant queries (manual verification)
4. **Config migration**: Old configs with `target_embedding_context_tokens` get clear deprecation error
5. **Original text preserved**: Leaf text in database unchanged; only embedding uses optimized version

## Non-Goals

- Changing how preceding context is retrieved (tiling algorithm unchanged)
- Modifying summary job behavior (internal node summarization unchanged)
- Adjusting the 8000 token embedding API limit
- Re-embedding existing leaves (they continue working)
