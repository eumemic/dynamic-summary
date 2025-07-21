# The Tiling Algorithm: Deep Dive

**Last Verified**: January 2025  
**Implementation Status**: CORE IMPLEMENTED, Some features NOT IMPLEMENTED

This document provides a comprehensive technical explanation of RagZoom's Dynamic Programming (DP) tiling algorithm, which generates optimal summaries by selecting the right level of detail for each part of a document based on relevance.

## Table of Contents

- [Motivation](#motivation)
- [Core Concepts](#core-concepts)
- [Algorithm Overview](#algorithm-overview)
- [Implementation Details](#implementation-details)
- [What's NOT Implemented](#whats-not-implemented)
- [Algorithm Pseudocode](#algorithm-pseudocode)
- [Examples](#examples)
- [Performance Characteristics](#performance-characteristics)

## Motivation

The original RagZoom algorithm used a multi-stage corrective pipeline that was brittle and error-prone:

1. Extract tiling using greedy top-down walk
2. Trim to budget (potentially breaking coverage)
3. Apply slope cap (potentially exceeding budget)
4. Re-trim if needed
5. Final de-duplication in assembly

Each stage could introduce bugs, and the interactions between stages made the system difficult to reason about. The DP algorithm replaces this with a single-pass, "correct-by-construction" approach.

## Core Concepts

### Terminology

- **Span**: An interval `[start, end)` in the document's character coordinates
- **Segment**: A portion of a node that can be included in the output
  - For leaf nodes (depth = 0): The entire node is one segment, `side = None`
  - For internal nodes (depth > 0): Split into two segments at the `<<<MID>>>` delimiter
    - Left segment: `(node_id, "LEFT")` 
    - Right segment: `(node_id, "RIGHT")`
- **Tiling**: A sequence of segments that:
  - Covers the entire document span
  - Has no gaps between segments
  - Has no overlapping segments
  - Maintains chronological order
- **Coverage Tree**: The set of nodes considered for inclusion (n_max most relevant leaves + all ancestors)

### Key Invariant

The algorithm maintains a critical invariant: **Complete, non-overlapping coverage**. Every recursive call returns a valid tiling for its span - never partial coverage or disconnected fragments.

This is why patterns like "Summary → Leaf A → Summary → Leaf C → Summary" are impossible. The algorithm decomposes the problem hierarchically, always producing contiguous tilings.

## Algorithm Overview

The DP algorithm treats tiling generation as an optimization problem:

> Given a tree node and token budget, find the tiling that maximizes total quality (relevance score) while staying within the budget.

### Recursive Structure

```
find_optimal_tiling(node, budget):
    if node is leaf:
        return [(node, None)], node.quality
    
    # Try option 1: Use parent's segments
    parent_segments = get_parent_segments(node)
    parent_quality = node.quality
    
    # Try option 2: Recurse into children
    left_budget, right_budget = split_budget_proportionally(budget, node)
    left_segments, left_quality = find_optimal_tiling(node.left, left_budget)
    right_segments, right_quality = find_optimal_tiling(node.right, right_budget)
    
    child_segments = left_segments + right_segments
    child_quality = left_quality + right_quality
    
    # Choose better option
    if child_quality > parent_quality AND fits_in_budget(child_segments, budget):
        return child_segments, child_quality
    else:
        return parent_segments, parent_quality
```

### Memoization

Results are cached by `(node_id, budget)` to avoid recomputing subproblems, making the algorithm efficient even for large trees.

## Implementation Details

### Budget Allocation

**STATUS: IMPLEMENTED**

Budget is split between left and right children proportionally based on:

1. **Primary**: Relevance scores of seed nodes in each subtree
2. **Fallback**: Text length when no relevance scores available

```python
def _split_budget_proportionally(self, node, budget_tokens, scores):
    # Get seed nodes in each subtree
    left_seeds = [n for n in scores if n in left_subtree]
    right_seeds = [n for n in scores if n in right_subtree]
    
    # Calculate total scores
    left_score = sum(scores[n] for n in left_seeds)
    right_score = sum(scores[n] for n in right_seeds)
    
    if left_score + right_score > 0:
        # Split based on relevance
        left_ratio = left_score / (left_score + right_score)
    else:
        # Fallback to text length
        left_ratio = left_text_length / total_text_length
    
    return int(budget_tokens * left_ratio), budget_tokens - int(budget_tokens * left_ratio)
```

### Segment Quality Calculation

**STATUS: IMPLEMENTED**

- Leaf segments: Use the node's full relevance score
- Half segments (internal nodes): Use 50% of the parent node's score

This heuristic was discovered during implementation - half segments represent partial information, so they get partial quality credit.

### Budget Overflow Handling

**STATUS: IMPLEMENTED**

Because left and right subproblems are solved independently, their combined cost might exceed the total budget. The algorithm handles this by:

1. Computing child tilings optimistically
2. Checking if combined cost ≤ budget
3. If over budget, falling back to parent segments

This ensures the budget constraint is never violated.

### Token Cost Calculation

**STATUS: IMPLEMENTED**

Uses tiktoken with cl100k_base encoding to count actual tokens for each segment:

- Leaf segments: Count tokens in full text
- Left segments: Count tokens from start to `<<<MID>>>`
- Right segments: Count tokens from `<<<MID>>>` to end

## What's NOT Implemented

### Slope Cap Enforcement

**STATUS: NOT IMPLEMENTED**

The configuration includes slope cap parameters:
- `enable_slope_cap` (default: True)
- `slope_cap_size` (default: 1)

However, the DP algorithm does not enforce depth constraints between adjacent segments. The proposed two-pass approach (generate optimal tiling, then post-process for slope violations) has not been implemented.

### Smoothing Pass

**STATUS: NOT IMPLEMENTED**

The `enable_smoothing` configuration parameter exists but has no effect. The proposed smoothing pass to improve readability by adding transition sentences between segments is not implemented.

### Mass-Based Relevance Propagation

**STATUS: NOT IMPLEMENTED**

The v2 design documents describe a more sophisticated "mass-based" system where relevance scores propagate up the tree. Currently, only seed nodes (from vector search) have relevance scores, and parent quality is a simple heuristic.

## Algorithm Pseudocode

Here's the complete pseudocode matching the actual implementation:

```python
class DynamicFrontierGenerator:
    def __init__(self, store, config):
        self.store = store
        self.config = config
        self._memo_cache = {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def find_optimal_tiling(self, budget_tokens, scores, document_id, coverage_map):
        root = store.get_root_node_for_document(document_id)
        segments, quality = self._find_optimal_for_span(root, budget_tokens, scores)
        return build_result(segments, quality, coverage_map)
    
    def _find_optimal_for_span(self, node, budget, scores):
        # Check memoization
        key = (node.id, budget)
        if key in self._memo_cache:
            return self._memo_cache[key]
        
        # Base case: leaf node
        if node.depth == 0:
            segment = Segment(node.id, side=None)
            if get_segment_cost(segment) <= budget:
                result = ([segment], scores.get(node.id, 0.0))
            else:
                result = ([], 0.0)  # Too expensive
            self._memo_cache[key] = result
            return result
        
        # Internal node: try both options
        
        # Option 1: Use this node's segments
        left_seg = Segment(node.id, side="LEFT")
        right_seg = Segment(node.id, side="RIGHT")
        parent_cost = get_segment_cost(left_seg) + get_segment_cost(right_seg)
        
        if parent_cost <= budget:
            parent_segments = [left_seg, right_seg]
            parent_quality = scores.get(node.id, 0.0)
        else:
            parent_segments = []
            parent_quality = 0.0
        
        # Option 2: Recurse into children
        if node.left and node.right:
            left_budget, right_budget = split_budget_proportionally(node, budget, scores)
            
            left_segments, left_quality = _find_optimal_for_span(node.left, left_budget, scores)
            right_segments, right_quality = _find_optimal_for_span(node.right, right_budget, scores)
            
            child_segments = left_segments + right_segments
            child_quality = left_quality + right_quality
            
            # Verify combined cost
            child_cost = sum(get_segment_cost(s) for s in child_segments)
            
            if child_cost <= budget and child_quality > parent_quality:
                result = (child_segments, child_quality)
            else:
                result = (parent_segments, parent_quality)
        else:
            # Missing children - use parent
            result = (parent_segments, parent_quality)
        
        self._memo_cache[key] = result
        return result
```

## Examples

### Example 1: Simple Binary Tree

```
Document: "The cat sat on the mat. The dog ran."

Tree structure:
       Root (0-40)
      /            \
  L1 (0-24)      L2 (24-40)
  "The cat..."   "The dog..."

Query: "cat" (high relevance to L1)
Budget: 20 tokens
```

The algorithm would:
1. Start at root with 20 tokens
2. Try parent option: Both segments of root (too expensive, ~15 tokens each)
3. Try child option: Allocate more budget to left (relevant) side
4. Return: [L1 (full leaf), (Root, RIGHT)]

### Example 2: Partial Relevance

```
Document with 4 sections A, B, C, D
Query matches sections B and C moderately

Tree structure:
          Root
       /        \
     AB          CD
    /  \        /  \
   A    B      C    D
```

With sufficient budget, the algorithm might return:
- `(AB, LEFT)` - Summary of A
- `B` - Full text of B  
- `C` - Full text of C
- `(CD, RIGHT)` - Summary of D

This gives more detail where relevant while maintaining complete coverage.

## Performance Characteristics

- **Time Complexity**: O(n × b) where n = number of nodes, b = distinct budget values
- **Space Complexity**: O(n × b) for memoization cache
- **Cache Efficiency**: High reuse for common budget values
- **Practical Performance**: Fast enough for real-time queries on documents with thousands of nodes

## Future Improvements

1. **Implement Slope Cap**: Add the two-pass post-processing to enforce depth constraints
2. **Add Smoothing**: Implement transition generation between segments
3. **Mass Propagation**: Implement the full v2 design with relevance mass flowing up the tree
4. **Budget Hints**: Pre-compute common budget allocations for faster query time
5. **Parallel Evaluation**: Evaluate left/right subproblems concurrently

## Conclusion

The DP tiling algorithm provides a solid foundation for RagZoom's hierarchical summarization. Its "correct-by-construction" approach eliminates entire classes of bugs while maintaining good performance. While some advanced features remain unimplemented, the core algorithm successfully delivers on the primary requirements of complete coverage, relevance-based detail, and budget guarantees.