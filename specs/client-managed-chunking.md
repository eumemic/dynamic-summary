---
status: COMPLETE
---

# Client-Managed Chunking with Dynamic Summary Targets

## Job to Be Done

Enable transcript indexing where each conversation turn becomes exactly one leaf node, preserving semantic coherence. The memory service layer needs to append whole turns without RagZoom fragmenting them, while summary compression targets scale appropriately with source span size.

## Background

Currently, RagZoom's text splitter uses fixed-size chunks (~200 tokens by default). This works well for prose documents but fragments conversation turns, causing summarization to operate on incoherent pieces. The memory service layer already identifies turn boundaries during transcript sync—it just needs RagZoom to respect them.

## Solution

Two changes to RagZoom core:

1. **Client-managed chunking mode**: When `target_chunk_tokens` is `None`, each unit passed to `append()` or `append_batch()` becomes exactly one leaf. No further splitting occurs.

2. **Dynamic summary targets**: Instead of a fixed target at all tree heights, targets shrink by 2x per level: `target = span_tokens / 2^height`, with a 50-token floor below which text passes through unsummarized.

## Activation

- `target_chunk_tokens: int` becomes `target_chunk_tokens: int | None`
- When `None`: client-managed chunking + dynamic summary targets
- When set to an integer: current behavior (fixed chunking + fixed summary targets)
- Fully backward compatible—existing configs work unchanged

## Temporal Documents Always Use Client-Managed Chunking

**Behavior:** When appending to a temporal document (timestamps provided), `target_chunk_tokens` is **ignored**. The system automatically uses client-managed chunking regardless of config.

### Rationale

If the server were to chunk temporal data, it would need to assign timestamps to sub-chunks. There's no correct way to do this:
- **Same range for all chunks** → falsifies data (makes sequential events appear simultaneous)
- **Evenly divide the time range** → fabricates timestamps that don't reflect reality
- **Null timestamps on sub-chunks** → loses temporal information

Rather than requiring users to configure this correctly, the system enforces it automatically: any append with timestamps uses client-managed chunking.

### Implementation

In `append_executor.py`, when timestamps are present:
1. Truncation behavior uses client-managed mode (truncate > 50k chars, preserve empty units)
2. Splitter is bypassed entirely—each input unit becomes exactly one leaf
3. A debug log message notes when `target_chunk_tokens` is being ignored

```python
# Determine if we're in temporal mode (forces client-controlled chunking)
is_temporal_mode = has_timestamps or is_temporal

if is_temporal_mode:
    chunks = [unit_text]  # Bypass splitter
else:
    chunks = self._splitter.split_text(unit_text)
```

This ensures temporal documents work correctly regardless of server config.

## Behavior Changes

### Append Operations

When `target_chunk_tokens is None`:

| Operation | Current Behavior | New Behavior |
|-----------|-----------------|--------------|
| `append(text)` | Split into multiple chunks | Create exactly one leaf |
| `append_batch(units)` | Split each unit, forced boundaries between | Each unit = one leaf, no splitting |
| Empty/whitespace unit | Filtered out | Creates leaf (embed the empty string) |
| Unit > 50k characters | N/A (would be split) | Warn and truncate to 50k chars |

### Summary Target Calculation

When `target_chunk_tokens is None`:

```python
def get_summary_target(node_span_chars: int, height: int, chars_per_token: float) -> int:
    span_tokens = node_span_chars / chars_per_token
    target = span_tokens / (2 ** height)

    # Floor: below 50 tokens, pass through without summarization
    if target < 50:
        return 0  # Signal passthrough

    return int(target)
```

Where:
- `node_span_chars = span_end - span_start`
- `chars_per_token` = document-level ratio (see below)
- Height 0 leaves: no summarization (target equals source)
- Height 1+: progressive 2x compression per level

### chars_per_token Tracking

Maintain `chars_per_token` ratio in memory (not persisted):

1. **On server startup**: Query all leaves for the document, sum `span_end - span_start` and `token_count`, compute ratio
2. **After each append**: Re-run the same query to update the ratio
3. **Empty document**: Ratio is undefined until first append; use 4.0 as fallback if needed before first data

This avoids adding columns to the database while keeping the ratio accurate.

### Embedding Context Target

New config field: `target_embedding_context_tokens: int` (default: 200)

Used for `contextualize_text()` when preparing embedding content. This replaces the use of `target_chunk_tokens` for embedding context summarization, ensuring embedding context has a sensible target even when `target_chunk_tokens` is `None`.

## Configuration

### IndexConfig Changes

```python
@dataclass
class IndexConfig:
    target_chunk_tokens: int | None  # None = client-managed chunking
    target_embedding_context_tokens: int  # New field, default 200
    # ... rest unchanged
```

### default_config.json

```json
{
  "target_chunk_tokens": 200,
  "target_embedding_context_tokens": 200,
  ...
}
```

Memory service will use a config with `target_chunk_tokens: null` to enable client-managed mode.

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Unit is empty string | Create leaf with empty text, embed empty string |
| Unit is whitespace only | Create leaf with whitespace, embed it |
| Unit > 50k characters | Log warning, truncate to 50k chars, create leaf |
| Very tall tree (height > 10) | Dynamic target shrinks; below 50 tokens = passthrough |
| First append to new document | chars_per_token computed from that append's data |

## Acceptance Criteria

1. **Atomic leaves**: When `target_chunk_tokens` is `None`, `append_batch(["A", "B", "C"])` creates exactly 3 leaves
2. **No splitting**: Each leaf's text matches the input unit exactly (unless truncated)
3. **Dynamic targets**: Summary at height 2 targets ~1/4 the span's token count
4. **Passthrough floor**: Summaries targeting < 50 tokens pass through unchanged
5. **Backward compatible**: Existing configs with `target_chunk_tokens: 200` work identically
6. **Large unit handling**: Units > 50k chars are truncated with a warning logged
7. **Temporal auto-chunking**: Temporal documents always use client-managed chunking; `target_chunk_tokens` is ignored

## Non-Goals

- Timestamp fields on TreeNode (separate spec, relates to #306)
- Changes to the memory service transcript sync (that layer will adopt this mode)
- Changes to query/retrieval behavior
