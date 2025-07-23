# Design Proposal: Eliminating the Segment Abstraction

## Executive Summary

This proposal outlines a simplification of the RagZoom architecture by eliminating the `Segment` abstraction and the associated `<<<MID>>>` delimiter system. Instead of treating internal nodes as splittable into LEFT/RIGHT segments, we propose treating all nodes as atomic units. Users can achieve their desired granularity by configuring the chunk size during indexing.

**Key Point**: This change maintains all current capabilities while significantly reducing complexity. The Dynamic Programming algorithm continues to guarantee perfect tiling (no gaps, no overlaps) under the token budget.

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

**Important**: The DP algorithm ensures perfect tiling (complete coverage, no gaps, no overlaps) in both systems. It never includes both a parent and its children - it chooses one or the other.

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

## Common Misconceptions Addressed

### "This will increase API costs"

API *call* count may increase with smaller chunks, but token costs (what you actually pay for) remain constant.

### "We lose semantic boundaries without <<<MID>>>"
**False**. The current system already splits at arbitrary token boundaries for leaf nodes. The `<<<MID>>>` delimiter is placed by the LLM in its generated summary, not at natural content boundaries. Both systems have the same level of semantic awareness.

### "This creates overlap between parent and child nodes"
**False**. The DP algorithm guarantees perfect tiling with no gaps and no overlaps. It never includes both a parent and its children in the same tiling - it chooses one or the other. This property is maintained in both systems.

### "Users will struggle with configuration"
**False**. Users already configure chunk size via `RAGZOOM_LEAF_TOKENS`. This doesn't add any new configuration burden.

### "The summarization process becomes more complex"
**False**. It becomes simpler:
- Old: "Summarize these two chunks and insert <<<MID>>> between them"
- New: "Summarize this chunk"

The target token size remains the same. There's no "overflow" or complexity - just simpler prompts.

### "Performance will degrade"
**False**. The system behaves identically to the current one. In fact, we could make it 100% backward compatible by simply using `effective_chunk_size = RAGZOOM_LEAF_TOKENS / 2` internally. No performance changes whatsoever.

### "Visualization will lose left/right distinction"
**False**. Visualization simply shifts from coloring segments to coloring nodes. Left child = left color, right child = right color. The visual distinction remains.

## Migration Strategy

Since this is a clean-break implementation with no existing production data to migrate, the strategy is simple:

1. **Development**: Implement all changes on a feature branch
2. **Testing**: Thoroughly test the new implementation
3. **Deployment**: Merge to main branch when ready
4. **Documentation**: Update all docs to reflect the new architecture

No dual-mode support or data migration is needed.

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

## FAQ

### Q: How do I get the same granularity as before?
**A**: Use smaller chunks. If you previously used 200-token chunks and relied on segments for finer control, try 100-token chunks. The granularity is equivalent, just structured differently in the tree.

### Q: Won't this create huge API costs?
**A**: Only if you choose smaller chunks. API costs scale linearly (not exponentially) with chunk count. Keep your current chunk size for the same costs, or adjust based on your needs.

### Q: What about documents with natural section boundaries?
**A**: The text splitter can be enhanced to respect section boundaries (e.g., chapters, paragraphs) while staying near the target chunk size. This is orthogonal to the segment elimination.

### Q: How does this affect retrieval quality?
**A**: Retrieval quality should be equivalent or better. The DP algorithm still finds optimal tilings, and simpler code means fewer bugs and edge cases.

## Detailed Implementation Roadmap

### Overview
This is a clean-break implementation - no backward compatibility with existing segment-based trees. Any existing indexed documents will need to be re-indexed.

### Phase 1: Data Model Changes (Foundation)
**Goal**: Update the core data structures

1. **Update SQLite Schema** (`ragzoom/store.py`)
   - Remove `mid_offset` column from `TreeNode` table definition
   - Update `add_node()` and `update_summary()` to remove `mid_offset` parameter
   - No migration needed - just change the schema directly

2. **Remove Segment Classes** (`ragzoom/dynamic_tiling.py`)
   - Delete `Segment` class
   - Delete `SegmentInfo` class  
   - Update `DPResult` to store node IDs directly instead of Segments

### Phase 2: Tree Building Simplification
**Goal**: Remove <<<MID>>> delimiter from indexing

1. **Update TreeBuilder** (`ragzoom/index.py`)
   - Remove <<<MID>>> from summarization prompt
   - Remove delimiter detection and retry logic
   - Remove `mid_offset` calculation
   - Simplify `_process_node_pair()` to just store summaries

2. **Update Tests**
   - Remove `tests/test_mid_delimiter.py` entirely
   - Update indexing tests to not expect <<<MID>>>

### Phase 3: Algorithm Simplification
**Goal**: Make DP algorithm work with atomic nodes

1. **Simplify DynamicTilingGenerator** (`ragzoom/dynamic_tiling.py`)
   - Update `_find_optimal_tiling_for_span()` to work with whole nodes
   - Remove LEFT/RIGHT segment creation logic
   - Simplify `_get_text_for_segment()` to always return full node text
   - Remove span calculation for half-nodes
   - Quality scoring: Use full node scores (no more half-scores for segments)

2. **Update Algorithm Tests**
   - Update `test_dp_assembly.py` to remove LEFT/RIGHT tests
   - Update `test_dp_integration.py` for atomic nodes

### Phase 4: Assembly and Presentation
**Goal**: Simplify text assembly and visualization

1. **Simplify Assembler** (`ragzoom/assemble.py`)
   - Remove segment-based text extraction
   - Simple concatenation of node content
   - Remove Segment import

2. **Update Visualization** (`ragzoom/tree_viz.py`)
   - Remove segment side handling
   - Simplify node lookup logic

3. **Update CLI** (`ragzoom/cli.py`)
   - Remove segment side from debug output
   - Simplify tiling display

### Phase 5: Validation and Cleanup
**Goal**: Remove segment-specific validation and clean up

1. **Update Validation** (`ragzoom/validate.py`)
   - Remove `mid_offset` validation
   - Remove segment span calculations
   - Simplify to whole-node validation

2. **Documentation Updates**
   - Update `docs/architecture.md`
   - Update `docs/deep-dives/tiling-algorithm.md`
   - Update docstrings throughout

3. **Final Cleanup**
   - Remove any remaining segment references
   - Update all remaining tests
   - Remove migration code after confirming working

### Testing Strategy

1. **Unit Tests First**: Update lowest-level tests (models, store) before integration tests
2. **Validation Suite**: Run comprehensive validation on test documents
3. **Integration Testing**: Test full indexing and retrieval workflows
4. **Functional Equivalence**: Verify that using half the chunk size produces identical results to the old system

### Risk Mitigation

1. **Branch Strategy**: Develop on feature branch with frequent commits
2. **Test Coverage**: Ensure all changes have corresponding test updates
3. **Manual Testing**: Test with various document types and chunk sizes
4. **Rollback Plan**: Keep original branch intact until fully validated

### Success Criteria

1. All tests pass with simplified model
2. Indexing and retrieval work correctly
3. No performance degradation
4. Code complexity metrics improve
5. Documentation is fully updated

## Critical Discovery: Tree Completeness Requirements

### The Issue

During implementation, we discovered a critical assumption in the node-based DP algorithm that wasn't immediately apparent: **the coverage tree must be a complete binary tree**. This means every internal node must have both children present in the coverage tree.

### Why This Matters

In the segment-based system, a parent node could contribute partial coverage through its LEFT or RIGHT segment even without both children. However, in the node-based system:

1. Nodes are atomic - included in full or not at all
2. To maintain coverage when recursing, we need both children
3. If only one child is present, we either:
   - Use the parent (potentially low relevance)
   - Recurse to one child (breaking coverage)

### Example

When running `ragzoom query "" -d smoke_test.txt --n-max 1`:
- Only one leaf node is selected (e.g., span 639-811 out of 0-1670)
- Its ancestors are added, creating a path to root
- But siblings at each level are missing
- The DP algorithm can't maintain full document coverage

### Requirements

1. **Indexed Tree**: Must be a full binary tree (every internal node has exactly 2 children)
2. **Coverage Tree**: Must be a complete binary subtree of the indexed tree
3. **Retriever**: Must ensure coverage tree completeness by including siblings

### Implementation Notes

- Add validation in DP algorithm to detect incomplete coverage trees
- Modify retriever to include siblings when building coverage trees
- Add validation during indexing to ensure full binary trees
- This maintains the perfect tiling guarantee (no gaps, no overlaps)

## Conclusion

Eliminating the segment abstraction significantly simplifies RagZoom's architecture while maintaining all functional capabilities. By adjusting chunk size, we achieve the same granularity through a cleaner, more intuitive model. The migration path allows for gradual adoption and validation of the new approach.

The discovery of the tree completeness requirement reinforces the importance of maintaining strict invariants in the data structure. With proper validation and coverage tree construction, the node-based system provides the same guarantees as the segment-based system while being conceptually simpler.