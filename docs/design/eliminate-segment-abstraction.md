# Design Proposal: Eliminating the Segment Abstraction

## Executive Summary

This proposal outlines a simplification of the RagZoom architecture by eliminating the `Segment` abstraction and the associated `<<<MID>>>` delimiter system. Instead of treating internal nodes as splittable into LEFT/RIGHT segments, we propose treating all nodes as atomic units. Users can achieve their desired granularity by configuring the chunk size during indexing.

## Current State

### Segment-Based Architecture

The current system uses a two-level abstraction:

1. **Nodes**: Store the hierarchical summary tree
2. **Segments**: Represent portions of nodes that can be included in the final output
   - Leaf nodes → One segment (the full node)
   - Internal nodes → Two segments (LEFT and RIGHT, split at `mid_offset`)

### How Segments Work with `<<<MID>>>`

1. During summarization, the LLM inserts `<<<MID>>>` to mark the boundary between left and right child content
2. The delimiter position is stored as `mid_offset` 
3. For internal nodes, this enables creation of two segments:
   - LEFT segment: `text[:mid_offset]`
   - RIGHT segment: `text[mid_offset:]`

### Complexity Points

- Two different abstractions (Node vs Segment) to understand
- The DP algorithm operates on Segments, not Nodes
- Assembly must handle both full nodes (leaves) and half nodes (internal segments)

## Proposed Change

### Unified Node Model

Eliminate the `Segment` class and the `<<<MID>>>` delimiter system entirely:

```python
# Current model (simplified)
class TreeNode:
    id: str
    text: str
    mid_offset: Optional[int]  # Position of <<<MID>>> delimiter

class Segment:
    node_id: str
    side: Optional[Literal["LEFT", "RIGHT"]]  # None for leaves

# Proposed model
class TreeNode:
    id: str
    content: str  # Single content field for all nodes
    # mid_offset removed - no longer needed
    # Segment class eliminated
```

All nodes become atomic units - they are either included in full or not at all.

### Achieving Equivalent Granularity

The key insight: **granularity comes from chunk size, not from segment splitting**.

Current system with 200-token chunks:
```
P_AB (splittable via mid_offset)
/  \
A    B
(200) (200)
```

Proposed system with 100-token chunks:
```
    P_ALARBLBR
    /        \
  P_ALAR    P_BLBR  
  /    \    /    \
AL    AR  BL    BR
(100) (100) (100) (100)
```

Both achieve the same effective granularity - the ability to include ~100-token units in the final output.

### Algorithm Changes

#### Dynamic Programming Algorithm

The DP algorithm becomes simpler:

```python
# Current
def find_optimal_tiling(node, budget):
    if is_leaf(node):
        # Return full node or nothing
    else:
        # Option 1: Use LEFT + RIGHT segments
        # Option 2: Recurse to children
        
# Proposed  
def find_optimal_tiling(node, budget):
    if is_leaf(node):
        # Return full node or nothing
    else:
        # Option 1: Use this node's content
        # Option 2: Recurse to children
```

The binary choice at each level remains, but without segment complexity.

#### Assembly

Assembly becomes trivial:

```python
# Current
def assemble(segments):
    texts = []
    for segment in segments:
        node = get_node(segment.node_id)
        if segment.side == "LEFT":
            texts.append(node.text[:node.mid_offset])
        elif segment.side == "RIGHT":
            texts.append(node.text[node.mid_offset:])
        else:
            texts.append(node.text)
    return "\n\n".join(texts)

# Proposed
def assemble(nodes):
    return "\n\n".join(node.content for node in nodes)
```

## Configuration Changes

The `leaf_tokens` parameter (configurable via `RAGZOOM_LEAF_TOKENS`) remains unchanged. Users can set this to whatever chunk size works best for their use case.

The example showing 100-token chunks achieving the same granularity as 200-token chunks with segments was purely illustrative - there's no prescribed "correct" chunk size. Users should experiment to find what works best for their specific content and use cases.

## Benefits

1. **Conceptual Simplicity**: One abstraction (Node) instead of two (Node + Segment)
2. **Cleaner Data Model**: No `mid_offset`, no special delimiter handling
3. **Simpler Assembly**: Just concatenate node content (no text slicing logic)
4. **Easier Debugging**: Every piece of content is a complete, self-contained node
5. **Reduced Edge Cases**: No missing/malformed delimiters, no off-by-one errors in text slicing
6. **Simpler Summarization**: No need to instruct LLM about `<<<MID>>>` placement

## Tradeoffs

1. **Granularity vs Tree Depth**: Smaller chunks provide finer granularity but create deeper trees
2. **Indexing Cost**: More nodes mean more summarization API calls (though parallelism helps)
3. **Loss of Sub-Node Precision**: Can't include just half of a parent's summary (must include whole node or recurse to children)

## Migration Strategy

### Phase 1: Dual Mode Support

1. Add configuration flag: `use_segment_abstraction` (default: true)
2. Implement unified node indexing alongside current system
3. Both modes produce compatible output

### Phase 2: Migration Tools

1. Provide tool to re-index documents in unified mode
2. Add compatibility layer for reading old segment-based trees
3. Gradually migrate existing documents

### Phase 3: Deprecation

1. Switch default to unified mode
2. Mark segment-based code as deprecated
3. Eventually remove segment abstraction entirely

## Implementation Checklist

- [ ] Update `TreeNode` model to remove `mid_offset` field
- [ ] Modify `TreeBuilder` to remove `<<<MID>>>` delimiter prompt and parsing logic
- [ ] Remove `Segment` class entirely
- [ ] Simplify `DynamicTilingGenerator` to work with whole nodes only
- [ ] Simplify `Assembler` to just concatenate node content
- [ ] Update prompt templates to remove `<<<MID>>>` instructions
- [ ] Update tests for new structure
- [ ] Add feature flag for gradual rollout
- [ ] Create migration tools for existing indexed documents

## Validation Criteria

1. **Functional Equivalence**: The system should produce summaries of equivalent quality and granularity
2. **Performance**: Indexing time should remain reasonable despite more nodes
3. **Coverage**: No gaps in document coverage
4. **Budget Compliance**: Token budgets still respected

## Conclusion

Eliminating the segment abstraction significantly simplifies RagZoom's architecture while maintaining all functional capabilities. By adjusting chunk size, we achieve the same granularity through a cleaner, more intuitive model. The migration path allows for gradual adoption and validation of the new approach.