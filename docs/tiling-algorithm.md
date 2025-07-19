# The Tiling Algorithm: A Correct-by-Construction Approach to Dynamic Summarization

The core of RagZoom's retrieval and assembly is a Dynamic Programming (DP) algorithm that constructs a "tiling" of the source document. This document clarifies the key concepts and invariants of this algorithm. The term "frontier" is a holdover from a previous design; "tiling" is a more accurate description of the current process.

#### 1. Terminology

*   **Span:** An interval `[start, end)` in the document's token coordinates.
*   **Segment:** A half-node, either `(node_id, LEFT)` or `(node_id, RIGHT)`. Each segment has a defined span. For a leaf node, both LEFT and RIGHT segments represent the full content of the node.
*   **Tiling:** A gap-free, non-overlapping sequence of segments, ordered by their span start. A tiling has a span, which is the union of all of its segment spans. The purpose of the algorithm is to produce a tiling which spans the whole source document.
*   **Coverage Tree:** The set of nodes consisting of the `n_max` most relevant leaf nodes (seed nodes) and all of their ancestors up to the root.

#### 2. The Core Invariant: Complete, Non-Overlapping Coverage

The fundamental rule of the system is that the final output **must** be a valid Tiling that spans the entire document. This means:

*   **Completeness:** The spans of the selected Segments must cover the entire document from `span_start=0` to `span_end=document_length`. There can be no holes.
*   **Non-overlapping:** The span of any Segment in the Tiling cannot overlap with the span of any other Segment.

This "correct-by-construction" principle is the most important property of the algorithm.

#### 3. Algorithm High-Level Design

The algorithm is a recursive, memoized function, `find_optimal_tiling(node, budget)`, that computes the highest-quality tiling for the span covered by `node` that fits within the given `budget`.

The core logic is decomposed into solving two independent subproblems: finding the optimal tiling for the left child's span and the optimal tiling for the right child's span. The final tiling for the parent's span is constructed by concatenating the solutions to these two subproblems.

#### 4. Pseudocode and Logic Flow

The process can be broken down into three main components: a memoization wrapper, the core recursive logic, and a decision helper.

**A. Memoization Wrapper (`find_optimal_tiling`)**

This is the main entry point for the recursion. It uses a cache where the key is the state `(node_id, budget)` to avoid re-computing solutions for the same subproblem.

```pseudocode
memo_cache = {}
def find_optimal_tiling(node, budget):
    cache_key = (node.id if node else None, budget)
    if cache_key in memo_cache:
        return memo_cache[cache_key]

    result = find_optimal_tiling_unmemoized(node, budget)
    memo_cache[cache_key] = result
    return result
```

**B. Core Recursive Logic (`find_optimal_tiling_unmemoized`)**

This function orchestrates the decomposition.

```pseudocode
def find_optimal_tiling_unmemoized(node, budget):
    // Base Cases
    if not node: return ([], 0.0) // No node, empty tiling
    if budget < get_cost_of_node(node): return ([], 0.0) // Pruning heuristic

    // 1. Decompose Budget
    // The total budget for the parent's span is split proportionally
    // between the left and right child spans based on relevance.
    budget_L, budget_R = split_budget_proportionally(budget, node)

    // 2. Solve Subproblems
    // Independently find the best tiling for each side.
    tiling_L, quality_L = find_best_tiling_for_side('LEFT', node, budget_L)
    tiling_R, quality_R = find_best_tiling_for_side('RIGHT', node, budget_R)

    // 3. Combine Solutions
    // The final tiling for this node's span is the concatenation of the optimal solutions
    // for its left and right subproblems.
    final_tiling = tiling_L + tiling_R
    final_quality = quality_L + quality_R
    
    return (final_tiling, final_quality)
```

**C. Side Decision Helper (`find_best_tiling_for_side`)**

This is where the fundamental DP choice is made for each half of the parent's span.

```pseudocode
def find_best_tiling_for_side(side, parent_node, budget_for_side):
    // Option A: The low-resolution parent segment.
    parent_segment_tiling = [Segment(parent_node.id, side)]
    quality_A = calculate_quality(parent_segment_tiling)

    // Option B: The optimal high-resolution tiling from the child.
    child_node = get_child(parent_node, side)
    
    // **Crucially, this is a recursive call to the main memoized function.**
    child_tiling, quality_B = find_optimal_tiling(child_node, budget_for_side)

    // Compare and return the winner.
    if quality_B > quality_A:
        return (child_tiling, quality_B)
    else:
        return (parent_segment_tiling, quality_A)
```

#### 5. Why the Buggy Output is Impossible

The output `...Summary -> Leaf A -> Summary -> Leaf C -> Summary...` is impossible for a correctly implemented Tiling algorithm because it represents a **disconnected set of fragments**, not a single, contiguous Tiling. A valid Tiling must cover the *entire* document span. The buggy output has "holes" between the fragments that are not filled by any segment, violating the completeness invariant. The DP algorithm is designed to *never* produce such a result. 