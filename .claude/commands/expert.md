# RagZoom Quick Expert Mode

Use extended thinking to internalize this codebase architecture. Read files in order, building your mental model as you go.

## Core Algorithm (MUST understand first)
1. Read `ragzoom/dynamic_tiling.py` - The heart of RagZoom. Focus on:
   - `_find_optimal_tiling_for_span()` - recursive DP with memoization
   - Budget splitting logic in `_split_budget_proportionally()`
   - Quality metric: relevance × tokens
   - Correct-by-construction: single pass, no corrections needed

2. Read `ragzoom/models.py` - Data structures that make it work:
   - `TreeNode` - binary tree with character spans
   - Parent nodes = summaries, leaf nodes = raw text
   - Every node tracks exact position in original document

## How Documents Become Trees
3. Read `ragzoom/tree_builder.py` focusing on `add_document()`:
   - Leaves created from text chunks (~200 tokens)
   - Bottom-up construction: pairs → summaries → pairs...
   - Character spans maintained throughout
   - Async with semaphore-controlled concurrency

## Query Flow (how retrieval actually works)
4. Read `ragzoom/retrieve.py` - The complete retrieval pipeline:
   - Vector search finds seeds → MMR for diversity
   - Coverage map includes ancestors + siblings (tree fullness)
   - ALL nodes get relevance scores (not just seeds!)
   - Returns node IDs + scores for DP algorithm

5. Skim `ragzoom/assembler.py` - Simple concatenation of selected nodes

## Storage Layer
6. Read `ragzoom/store.py` constructor and `get_node()`:
   - SQLite: tree structure + text
   - ChromaDB: embeddings for search
   - LRU cache (1000 nodes default)
   - Document isolation via foreign keys

## Testing Pattern (crucial for development)
7. Read `tests/conftest.py` - Dual testing strategy:
   - `SimpleMockStore` for fast unit tests (4.5x faster)
   - Real `Store` for integration tests
   - Tests parameterized to run both ways

## Key Insights
- Token budget is STRICT - algorithm guarantees it's never exceeded
- Tree fullness enforced everywhere (no missing children)
- Relevance propagation happens for entire coverage tree
- Memoization makes DP efficient: O(n×b) where n=nodes, b=budget
- "Correct-by-construction" means no post-processing needed

## Quick Validation
Run: `pytest tests/test_dynamic_tiling.py::test_budget_guarantee -xvs`
This test captures the essence of the system's guarantees.

Remember: The DP algorithm in `dynamic_tiling.py` is the crown jewel. Everything else exists to support it.