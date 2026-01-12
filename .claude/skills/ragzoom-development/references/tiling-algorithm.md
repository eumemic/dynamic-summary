# The Tiling Algorithm: Deep Dive

**Last Verified**: December 2025
**Implementation Status**: CORE IMPLEMENTED, Some features NOT IMPLEMENTED

This document provides a comprehensive technical explanation of RagZoom's tiling algorithm, which generates summaries by selecting the right level of detail for each part of a document based on relevance.

## Table of Contents

- [Core Concepts](#core-concepts)
- [The Greedy Algorithm](#the-greedy-algorithm)
- [Budget Modes](#budget-modes)
- [Verbatim Budget](#verbatim-budget)
- [Implementation Details](#implementation-details)
- [What's NOT Implemented](#whats-not-implemented)
- [Algorithm Pseudocode](#algorithm-pseudocode)
- [Examples](#examples)
- [Performance Characteristics](#performance-characteristics)

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
- **Frontier**: The starting set of nodes before pruning (typically all leaves in the coverage tree)

### Key Invariant

The algorithm maintains a critical invariant: **Complete, non-overlapping coverage**. Every operation maintains a valid tiling for the document span - never partial coverage or disconnected fragments.

This is why patterns like "Summary → Leaf A → Summary → Leaf C → Summary" are impossible. The algorithm decomposes the problem hierarchically, always producing contiguous tilings.

## The Greedy Algorithm

RagZoom uses a greedy tiling algorithm that works **bottom-up**: it starts with all leaves (maximum detail) and iteratively rolls up sibling pairs until within budget.

### Algorithm Overview

```python
def find_greedy_tiling(root_ids, budget, scores, nodes):
    # Start with the full frontier (all leaves in coverage)
    frontier = build_frontier(nodes, root_ids)
    total_tokens = sum(nodes[nid].token_count for nid in frontier)

    # No budget constraint: return full frontier
    if budget is None:
        return frontier

    # If already within budget, done
    if total_tokens <= budget:
        return frontier

    # Initialize priority queue with roll-up candidates
    # Priority = quality_lost / tokens_saved (lower = better to roll up)
    queue = initialize_candidates(frontier, nodes, scores)

    # Iteratively roll up least-valuable sibling pairs
    while total_tokens > budget and queue:
        parent_id, left_id, right_id = pop_best_candidate(queue)

        # Replace children with parent in frontier
        pair_tokens = nodes[left_id].token_count + nodes[right_id].token_count
        parent_tokens = nodes[parent_id].token_count

        frontier.remove(left_id)
        frontier.remove(right_id)
        frontier.add(parent_id)
        total_tokens = total_tokens - pair_tokens + parent_tokens

        # Add grandparent as new candidate if eligible
        if nodes[parent_id].parent_id:
            enqueue_candidate(grandparent_id, queue, frontier, nodes, scores)

    return sorted(frontier, key=span_start)
```

### Roll-up Priority

Each candidate is scored by how much quality is lost per token saved:

```python
quality_lost = (left_relevance * left_tokens + right_relevance * right_tokens)
             - (parent_relevance * parent_tokens)
tokens_saved = (left_tokens + right_tokens) - parent_tokens
priority = quality_lost / tokens_saved  # Lower = better to roll up
```

Summaries that capture most of their children's relevance are prioritized for roll-up.

### Why Greedy Works Well

1. **Starts with maximum detail**: The frontier begins with all leaves, ensuring no relevant content is missed by premature summarization.

2. **Incremental decisions**: Each roll-up is a local decision based on actual token costs, not heuristic budget allocation.

3. **No budget allocation errors**: Unlike top-down approaches that must pre-allocate budget to subtrees, greedy makes all decisions with full knowledge of token costs.

4. **Predictable behavior**: Always produces the same result for the same inputs, with deterministic priority ordering.

## Budget Modes

The tiling algorithm supports three modes based on how `num_seeds` and `budget_tokens` are specified:

### Seeds-Only Mode (`num_seeds` specified, `budget_tokens` = None)

When only `num_seeds` is provided, the algorithm:
1. Uses vector search to find the top N most relevant seed nodes
2. Builds the coverage tree from those seeds
3. Returns the **full frontier** without any pruning

This mode is useful when you want maximum detail for a fixed number of relevant sections.

### Budget-Only Mode (`budget_tokens` specified, `num_seeds` = None)

When only `budget_tokens` is provided, the algorithm:
1. Calculates a conservative `num_seeds` from the budget (based on average leaf token cost)
2. Uses vector search to find those seed nodes
3. Builds the coverage tree and prunes to fit the budget

### Mixed Mode (both specified)

When both are provided:
1. Uses the specified `num_seeds` for vector search
2. Builds the coverage tree
3. Prunes to fit the specified `budget_tokens`

### Validation

At least one of `num_seeds` or `budget_tokens` must be specified. If neither is provided, the algorithm raises an error since there's no basis for determining how many seed nodes to retrieve.

## Verbatim Budget

The `recent_verbatim_token_budget` feature allows including recent content (rightmost leaves) without summarization. This is useful for conversation logs where the most recent messages should appear verbatim.

### How It Works

1. **Leaf Selection**: Starting from the rightmost leaf, select leaves moving left until the verbatim budget is exhausted
2. **Horizon Calculation**: The `span_start` of the leftmost selected leaf becomes the "verbatim horizon"
3. **Seed Filtering**: Vector search for seeds is restricted to `span_end < horizon` to prevent overlap
4. **Transient Pinning**: Selected verbatim leaves get `relevance=1.0`, making them strongly preferred by the tiling algorithm

### Example

```
Document: [Leaf1] [Leaf2] [Leaf3] [Leaf4] [Leaf5]
                                    ^-- verbatim budget selects Leaf4+Leaf5

Horizon = Leaf4.span_start
Seeds searched in: Leaf1, Leaf2, Leaf3 only
Tiling combines: relevance-based seeds + verbatim Leaf4+Leaf5
```

### Efficient Implementation

Verbatim leaf selection uses an efficient SQL window function query:

```sql
SELECT * FROM (
    SELECT *, SUM(token_count) OVER (ORDER BY span_end DESC) as cumsum
    FROM tree_nodes WHERE height = 0 AND document_id = ?
) WHERE cumsum - token_count < ?
ORDER BY span_start ASC
```

This avoids loading all leaves for large documents - only the leaves needed to fill the budget are fetched.

## Implementation Details

### Node Quality Calculation

**STATUS: IMPLEMENTED**

- Quality = relevance score × token cost
- Each node's full relevance score is used
- The algorithm optimizes for total quality (relevance-weighted tokens)

### Token Cost Calculation

**STATUS: IMPLEMENTED**

Uses tiktoken with cl100k_base encoding to count actual tokens for each node's full text content.

## What's NOT Implemented

### Slope Cap Enforcement

**STATUS: NOT IMPLEMENTED**

The configuration includes slope cap parameters:
- `enable_slope_cap` (default: True)
- `slope_cap_size` (default: 1)

However, the algorithm does not enforce depth constraints between adjacent nodes in the tiling.

### Smoothing Pass

**STATUS: NOT IMPLEMENTED**

The `enable_smoothing` configuration parameter exists but has no effect. The proposed smoothing pass to improve readability by adding transition sentences between nodes is not implemented.

### Mass-Based Relevance Propagation

**STATUS: NOT IMPLEMENTED**

The v2 design documents describe a more sophisticated "mass-based" system where relevance scores propagate up the tree. Currently, only seed nodes (from vector search) have relevance scores, and parent quality is a simple heuristic.

## Algorithm Pseudocode

Here's the complete pseudocode matching the actual implementation:

```python
class GreedyTilingGenerator:
    def __init__(self, config):
        self.config = config

    def find_optimal_tiling_over_roots(
        self,
        root_ids: Sequence[str],
        budget_tokens: int | None,
        scores: Mapping[str, float],
        nodes: Mapping[str, TreeNode],
    ) -> TilingResult:
        if not nodes:
            return TilingResult.empty()

        # Build frontier from leaves reachable from roots
        frontier = build_frontier(nodes, root_ids)

        # No budget constraint: return full frontier
        if budget_tokens is None:
            return build_result(frontier, scores, nodes)

        total_tokens = sum(nodes[nid].token_count for nid in frontier)

        # Already within budget: return as-is
        if total_tokens <= budget_tokens:
            return build_result(frontier, scores, nodes)

        # Iteratively replace least-relevant sibling pairs with parent
        frontier_set = set(frontier)
        queue, enqueued = initialize_candidate_queue(frontier_set, nodes, scores)

        while total_tokens > budget_tokens:
            replacement = pop_next_candidate(queue, enqueued, frontier_set, nodes, scores)
            if replacement is None:
                break  # No more valid candidates

            parent_id, left_id, right_id = replacement

            # Calculate token change
            pair_tokens = nodes[left_id].token_count
            if right_id != left_id:
                pair_tokens += nodes[right_id].token_count
            parent_tokens = nodes[parent_id].token_count

            # Update frontier
            frontier_set.remove(left_id)
            if right_id != left_id:
                frontier_set.remove(right_id)
            frontier_set.add(parent_id)
            total_tokens = total_tokens - pair_tokens + parent_tokens

            # Enqueue grandparent as potential new candidate
            grandparent_id = nodes[parent_id].parent_id
            if grandparent_id is not None:
                enqueue_candidate(grandparent_id, queue, enqueued, frontier_set, nodes, scores)

        ordered_frontier = sorted(frontier_set, key=lambda nid: nodes[nid].span_start)
        return build_result(ordered_frontier, scores, nodes)
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
1. Start with frontier = [L1, L2]
2. Calculate total tokens
3. If over budget, consider rolling up L1+L2 into Root
4. Since L1 has high relevance, the roll-up priority is high (much quality lost)
5. If budget forces the roll-up, Root is returned
6. Otherwise, [L1, L2] is returned

### Example 2: No Budget Constraint

```
Query with num_seeds=5 but no budget_tokens

The algorithm:
1. Finds 5 seed nodes via vector search
2. Builds coverage tree (seeds + all ancestors)
3. Returns full frontier without any pruning
4. Result may be large but maximally detailed
```

### Example 3: Partial Relevance

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

With budget constraint:
- Start with frontier [A, B, C, D]
- If over budget, find cheapest roll-ups
- Non-relevant A might roll up with B into AB
- Non-relevant D might roll up with C into CD
- Final tiling: [AB, CD] or [A, B, CD] depending on scores

## Performance Characteristics

- **Time Complexity**: O(n log n) where n = number of nodes (heap operations)
- **Space Complexity**: O(n) for frontier and heap
- **Deterministic**: Same inputs always produce same outputs
- **Practical Performance**: Fast enough for real-time queries on documents with thousands of nodes

## Future Improvements

1. **Implement Slope Cap**: Add post-processing to enforce depth constraints between adjacent nodes
2. **Add Smoothing**: Implement transition generation between nodes
3. **Mass Propagation**: Implement relevance score propagation from leaves to internal nodes

## Conclusion

The greedy tiling algorithm provides a solid foundation for RagZoom's hierarchical summarization. Its "correct-by-construction" approach eliminates entire classes of bugs while maintaining excellent performance. The bottom-up roll-up strategy avoids budget allocation problems inherent in top-down approaches, and the optional budget mode allows for maximum flexibility in how results are constrained.
