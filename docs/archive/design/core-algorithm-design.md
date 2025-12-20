# Design Document: Zoom-Lens Core Algorithm

### Abstract

The **Zoom-Lens** algorithm provides a hierarchical, multi-resolution approach to managing and retrieving information from large text corpora for Large Language Models (LLMs). It transforms a single long document into a binary "summary tree," enabling a retrieval process that assembles a chronologically coherent, budget-constrained context. This context dynamically adjusts its level of detail, "zooming in" on relevant sections while summarizing peripheral content. The result is a single, continuous narrative stream sent to the LLM that maximizes relevance and completeness within a fixed token budget, improving the model's ability to answer questions that require both high-level thematic understanding and specific, low-level details.

-----

## 1\. Conceptual Overview

The algorithm comprises two distinct processes: a one-time **Index Building** phase and a per-query **Runtime Stitching** phase.

### 1.1. Index Building: Creating a Zoomable Map

The core idea is to pre-process a long document into a structure that is easy to navigate at different levels of detail.

1.  **Slicing**: The source text is first divided into "leaf" chunks of a consistent size (e.g., \~200 tokens), respecting sentence or paragraph boundaries where possible. These leaves form the highest-resolution layer of our index.
2.  **Summarization Tree**: The algorithm constructs a binary tree on top of these leaves. Adjacent pairs of leaf nodes are summarized into a parent node containing a more concise version of their combined text. This process is applied recursively up the hierarchy, with each level representing a more compressed summary. This continues until a single root node, containing a synopsis of the entire document, is created.
3.  **Embedding**: Every node in the tree—both leaf chunks and summary nodes—is processed through an embedding model. The resulting vector, along with metadata (ID, depth, text), is stored in a vector database for efficient similarity searching.

This results in a forest of perfect binary trees where each node provides a summary of the entire sub-tree beneath it. Because new data is simply appended as new leaves, index updates are efficient, typically requiring only $O(\\log n)$ operations per new chunk.

### 1.2. Query-Time Stitching: Crafting the Narrative

When a query is received, the algorithm dynamically assembles the most relevant context for the LLM.

1.  **Retrieve & Diversify**: The query is used to perform a similarity search against the vector database, fetching a generous set of candidate nodes from all levels of the tree. To avoid redundancy, this set is refined using a Maximal Marginal Relevance (MMR) algorithm, ensuring the final selection is both relevant and diverse.
2.  **Propagate Coverage**: The selected nodes, along with any user-pinned nodes and all their ancestors up to the root, are marked as "covered." This essentially highlights the relevant branches of the summary tree.
3.  **Walk the Frontier**: The algorithm performs an in-order traversal across the document's timeline, following the boundary of the "covered" region. It selects text from nodes on this frontier, creating a single, chronologically ordered narrative. This process naturally includes high-level summaries for irrelevant sections (from high-level covered nodes) and detailed text for relevant sections (from low-level covered nodes).
4.  **Budget & Polish**: The total length of the assembled text is checked against a predefined token budget. If it exceeds the budget, the least important nodes are evicted based on a priority score. Finally, an optional smoothing pass can use a lightweight LLM to polish the transitions between chunks of different detail levels.

The final stitched prompt is a single, coherent stream that allows the LLM to process the information efficiently, as if reading a story that slows down for important scenes and speeds through background information.

-----

## 2\. Detailed Algorithm Design

### 2.1. Definitions & Data Structures

| Symbol | Meaning | Typical Value / Structure |
| :--- | :--- | :--- |
| $L$ | Leaf size in tokens (target) | 180–220 tokens |
| $B$ | Budget – max tokens allowed in stitched summary | 8,000 tokens |
| $N\_{max}$ | Max kept hits after diversification, calculated as $N\_{max} = \\lfloor B / (2 \\cdot L) \\rfloor$ | ≈ 20 |
| **Node** | The core data structure for the tree | `{id, parent_id, depth, span_start, span_end, text, emb, pinned?}` |
| **VecDB** | Vector index storing embeddings for all nodes | Chroma / pgvector |
| **PQ** | Min-priority queue for budget eviction, keyed by `priority = sim · 0.9^Δturns` | Size ≤ few × $N\_{max}$ |

The tree structure is a **forest of perfect binary trees** and is **append-only**. The root is at `depth = 0`. The total depth is approximately $\\log\_2(\\text{\# of leaf chunks})$.

### 2.2. Index-Build Process (Append-Only)

The index is built by adding text chunks sequentially. The `add_chunk` function initiates the process, and `close_parents` recursively builds the summary hierarchy.

```python
def add_chunk(raw_text: str):
    # Create a new leaf node at the maximum depth
    leaf = Node(depth=max_depth, text=raw_text, emb=embed(raw_text))
    store(leaf) # Store in DB and VecDB
    close_parents(leaf)

def close_parents(child: Node):
    """Recursively create parent summaries when both children exist."""
    if child.is_left_child: # Still waiting for the right sibling to be added
        return

    # Right child has been added, so we can create the parent
    sibling = get_left_sibling(child)
    combined_text = sibling.text + " " + child.text

    # Summarize combined text to ~half the leaf size
    parent_text = summarise(combined_text, target=L//2)
    parent = Node(depth=child.depth - 1,
                  text=parent_text,
                  emb=embed(parent_text))

    link(sibling, child, parent) # Set parent/child pointers
    store(parent)
    close_parents(parent) # Recurse up the tree
```

**Complexity**: The cost of adding a new leaf is $O(\\log n)$, as it only requires creating or updating one node at each level of the tree up to the root.

### 2.3. Runtime Pipeline (Per User Turn)

This pipeline executes for each incoming query to generate the final context.

```python
def answer(query: str) -> str:
    ### 3.1 Retrieve
    # Fetch a set of candidates larger than the final target
    raw_hits = vecdb.similarity_search(query, k=2*N_max)
    # Diversify to get a relevant and non-redundant set
    hits = mmr_diversify(raw_hits, k=N_max, λ=0.7)

    ### 3.2 Propagate Coverage
    # All retrieved hits, their ancestors, and any pinned nodes form the covered set
    covered = set(hits) ∪ set(ancestors(hits)) ∪ pinned_nodes()
    propagate_upwards(covered) # OR-mark parents if any child is covered

    ### 3.3 Token-Budget Eviction (if using working-set queue)
    # Ensure the total tokens of the covered set fits within budget B
    ensure_budget(covered, budget=B)

    ### 3.4 Frontier Walk & Slope-Cap
    chunks = []
    # Traverse the frontier of the covered set in chronological order
    for node in inorder_frontier(covered):
        # To prevent abrupt jumps in detail, "bubble up" to a parent if the depth
        # gap between this node and the previous one is too large.
        if slope_cap and depth_gap(node, chunks[-1]) > 1:
            node = node.parent
        chunks.append(node.text)

    prompt = " ".join(chunks)

    ### 3.5 Optional Smoothing Pass
    # Replace boundary tags with natural language for better flow
    if cfg.smoothing.enabled:
        prompt = smooth(prompt, boundary_tags=True)

    # Send the final stitched context and original query to the LLM
    return gpt4o_generate(prompt, query)
```

### 2.4. Key Sub-routines

| Function | Details |
| :--- | :--- |
| **`mmr_diversify()`** | Implements greedy Maximal Marginal Relevance. A `λ` value of 0.7 provides a good balance between relevance to the query and diversity among the selected documents. |
| **`propagate_upwards()`** | A Breadth-First Search (BFS) starting from the initial `hits`. A parent node is marked as covered if any of its children are covered. |
| **`inorder_frontier()`**| A generator function that yields nodes in chronological order. A node is considered part of the frontier if its parent is covered but the node itself is not fully represented by deeper, covered children. |
| **`ensure_budget()`** | If `TotalTokens(covered) > B`, this function repeatedly pops the lowest-priority node from the priority queue (PQ), un-marks it, and repeats until the context fits the budget. |
| **`smooth()`** | An optional final pass. It inserts `<<UP>>` and `<<DOWN>>` tags at resolution boundaries and uses a cheap LLM (e.g., GPT-3.5) with a simple one-sentence prompt to replace them with transitional phrases. |

-----

## 3\. Configuration & Parameters

These parameters control the algorithm's behavior and can be tuned for different use cases.

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `leaf_tokens` | 200 | Target token size for the highest-resolution leaf nodes. |
| `budget_tokens` | 8000 | The maximum total tokens allowed in the final stitched context ($B$). |
| `mmr_lambda` | 0.7 | The diversity parameter for MMR; 1.0 is pure diversity, 0.0 is pure relevance. |
| `slope_cap` | `true` | Enables/disables the logic to prevent abrupt changes in detail level. |
| `adjacent_context_tokens`| 75 | The token budget for the summarizer's context prompt. |
| `smoothing.enabled` | `false` | Enables/disables the final LLM-based smoothing pass. |
| `smoothing.model` | `gpt-3.5-turbo` | The model used for the smoothing pass if enabled. |
| `pin_depth_max` | 2 | The maximum depth at which nodes can be pinned by the user. |
| `sliding_queue.enabled`| `true` | Enables the use of a priority queue for intelligent budget eviction. |
| `sliding_queue.decay` | 0.9 | Per-turn decay factor for node priority, favoring recently relevant items. |

-----

## 4\. Performance & Robustness

### 4.1. Complexity Analysis

| Step | Time Complexity | Space Complexity (Prompt Size) |
| :--- | :--- | :--- |
| Retrieval + MMR | $O(k \\log k)$ where $k \\approx 2 \\cdot N\_{max}$ | — |
| Coverage Propagation | $O(\\text{\#covered})$ | — |
| Frontier Concatenation | $O(N\_{max} \\cdot \\text{depth})$ | $\\le 2 \\cdot N\_{max} \\cdot L$ tokens |
| **Overall Worst-Case**| — | **≈ 8,000 tokens** (fits budget $B$) |

The algorithm is designed to have bounded costs. The prompt size is geometrically bounded and guaranteed to fit the budget $B$ through the eviction mechanism.

### 4.2. Failure Modes & Guardrails

  * **Hit Overflow**: If the initial set of covered nodes exceeds the token budget, the `ensure_budget()` loop guarantees the final context size fits by evicting the lowest-priority nodes.
  * **All Hits Evicted**: In the unlikely event that all relevant hits are evicted to meet the budget, the system falls back to a baseline context consisting of the root synopsis and the last K leaf chunks, ensuring some context is always provided.
  * **Edit/Delete Operations**: If a leaf node is edited or deleted, its entire parent chain is marked as "dirty." A background, asynchronous process is triggered to re-summarize the affected nodes up the tree.

### 4.3. Minimal Testing Checklist

  * **Unit Tests**: Verify core components like the text splitter, the `inorder_frontier` walk logic, and the `slope_cap` mechanism in isolation.
  * **Integration Test**: Build a summary tree from a multi-chapter document and verify that a query generates a stitched summary that is less than or equal to the budget $B$.
  * **Regression Test**: Ensure that running the same query twice (with no index changes) produces an identical stitched prompt.
  * **Stress Test**: Use a very large document (e.g., the Bible) and a common query term (e.g., "God"). Verify the process completes, the output respects the token budget $B$, and the coverage histogram shows that retrieved nodes span the entire document.