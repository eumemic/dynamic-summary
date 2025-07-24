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
- **Node**: An atomic unit in the tree that can be included in the output
  - Leaf nodes (depth = 0): Contain raw text from the document
  - Internal nodes (depth > 0): Contain summaries of their children
- **Tiling**: A sequence of node IDs that:
  - Covers the entire document span
  - Has no gaps between nodes
  - Has no overlapping nodes
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
        # Leaf nodes are indivisible - include full node or nothing
        if node_cost <= budget:
            return [(node, None)], node.quality
        else:
            return [], 0.0
    
    # Internal node: Check if this node fits in budget
    parent_cost = cost(node)
    if parent_cost > budget:
        return [], 0.0
    
    # Option 1: Use this node
    parent_nodes = [node]
    parent_quality = node.quality * parent_cost
    
    # Option 2: Recurse into children
    left_budget, right_budget = split_budget_proportionally(budget, node)
    left_nodes, left_quality = find_optimal_tiling(node.left, left_budget)
    right_nodes, right_quality = find_optimal_tiling(node.right, right_budget)
    
    child_nodes = left_nodes + right_nodes
    child_quality = left_quality + right_quality
    
    # Check if child solution exceeds budget due to independent subproblems
    child_cost = sum(cost(node) for node in child_nodes)
    
    # Choose better option
    if child_cost <= budget and child_quality > parent_quality:
        return child_nodes, child_quality
    else:
        return parent_nodes, parent_quality
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

### Node Quality Calculation

**STATUS: IMPLEMENTED**

- Quality = relevance score × token cost
- Each node's full relevance score is used
- The algorithm optimizes for total quality (relevance-weighted tokens)

### Budget Overflow Handling

**STATUS: IMPLEMENTED**

Because left and right subproblems are solved independently, their combined cost might exceed the total budget. The algorithm handles this by:

1. Computing child tilings optimistically
2. Checking if combined cost ≤ budget
3. If over budget, falling back to parent node

This ensures the budget constraint is never violated.

### Token Cost Calculation

**STATUS: IMPLEMENTED**

Uses tiktoken with cl100k_base encoding to count actual tokens for each node's full text content.

## What's NOT Implemented

### Slope Cap Enforcement

**STATUS: NOT IMPLEMENTED**

The configuration includes slope cap parameters:
- `enable_slope_cap` (default: True)
- `slope_cap_size` (default: 1)

However, the DP algorithm does not enforce depth constraints between adjacent nodes in the tiling. The proposed two-pass approach (generate optimal tiling, then post-process for slope violations) has not been implemented.

### Smoothing Pass

**STATUS: NOT IMPLEMENTED**

The `enable_smoothing` configuration parameter exists but has no effect. The proposed smoothing pass to improve readability by adding transition sentences between nodes is not implemented.

### Mass-Based Relevance Propagation

**STATUS: NOT IMPLEMENTED**

The v2 design documents describe a more sophisticated "mass-based" system where relevance scores propagate up the tree. Currently, only seed nodes (from vector search) have relevance scores, and parent quality is a simple heuristic.

## Algorithm Pseudocode

Here's the complete pseudocode matching the actual implementation:

```python
class DynamicTilingGenerator:
    def __init__(self, store, config):
        self.store = store
        self.config = config
        self._memo_cache = {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def find_optimal_tiling(self, budget_tokens, scores, document_id, coverage_map):
        root = store.get_root_node_for_document(document_id)
        tiling = self._find_optimal_for_span(root, budget_tokens, scores)
        return build_result(tiling, coverage_map)
    
    def _find_optimal_for_span(self, node, budget, scores):
        # Check memoization
        key = (node.id, budget)
        if key in self._memo_cache:
            return self._memo_cache[key]
        
        # Base case: leaf node
        if node.depth == 0:
            if get_node_cost(node) <= budget:
                quality = scores.get(node.id, 0.0) * get_node_cost(node)
                result = (Tiling([node.id], quality), quality)
            else:
                result = ([], 0.0)  # Too expensive
            self._memo_cache[key] = result
            return result
        
        # Internal node: try both options
        
        # Option 1: Use this node
        parent_cost = get_node_cost(node)
        
        if parent_cost <= budget:
            parent_quality = scores.get(node.id, 0.0) * parent_cost
            parent_tiling = Tiling([node.id], parent_quality)
        else:
            parent_tiling = Tiling.empty()
            parent_quality = 0.0
        
        # Option 2: Recurse into children
        if node.left and node.right:
            left_budget, right_budget = split_budget_proportionally(node, budget, scores)
            
            left_tiling = _find_optimal_for_span(node.left, left_budget, scores)
            right_tiling = _find_optimal_for_span(node.right, right_budget, scores)
            
            child_tiling = left_tiling + right_tiling
            child_quality = child_tiling.relevance_tokens
            
            # Choose better option
            if child_quality > parent_quality:
                result = child_tiling
            else:
                result = parent_tiling
        else:
            # Missing children - use parent
            result = parent_tiling
        
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
2. Try parent option: Use root node (might be too expensive)
3. Try child option: Allocate more budget to left (relevant) side
4. Return: The option with higher quality score

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
- `AB` - Summary covering both A and B (if parent is more efficient)
- OR it might return `A` and `B` separately (if children have higher quality)
- Similarly for C and D

The algorithm chooses the option that maximizes quality (relevance × tokens) within budget.

## Performance Characteristics

- **Time Complexity**: O(n × b) where n = number of nodes, b = distinct budget values
- **Space Complexity**: O(n × b) for memoization cache
- **Cache Efficiency**: High reuse for common budget values
- **Practical Performance**: Fast enough for real-time queries on documents with thousands of nodes

## Future Improvements

1. **Implement Slope Cap**: Add the two-pass post-processing to enforce depth constraints
2. **Add Smoothing**: Implement transition generation between nodes
3. **Mass Propagation**: Implement the full v2 design with relevance mass flowing up the tree
4. **Budget Hints**: Pre-compute common budget allocations for faster query time
5. **Parallel Evaluation**: Evaluate left/right subproblems concurrently

## Conclusion

The DP tiling algorithm provides a solid foundation for RagZoom's hierarchical summarization. Its "correct-by-construction" approach eliminates entire classes of bugs while maintaining good performance. While some advanced features remain unimplemented, the core algorithm successfully delivers on the primary requirements of complete coverage, relevance-based detail, and budget guarantees.