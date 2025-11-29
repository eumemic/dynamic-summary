---
allowed-tools: Read, Grep, Bash, Task
description: Develop deep expertise in codebase or specific area
argument-hint: [focus area]
---

# RagZoom Study Mode
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Project structure: !`find . -name "*.py" -path "*/ragzoom/*" | head -10 | sed 's|^\./||'`
- Key modules: !`ls ragzoom/*.py 2>/dev/null | grep -v __pycache__ | head -8`

Arguments: "$ARGUMENTS"

_Note: If no focus area is specified above, use conversation context to determine the area of expertise. If this is a fresh session, become an expert in the entire codebase._

Use extended thinking to internalize the codebase architecture. Read files systematically, building your mental model as you progress.

## Adaptive Expertise Development

Based on your focus area, adapt the learning path:

1. **Identify core modules** related to your focus
2. **Trace dependencies** - understand what your focus area relies on
3. **Find relevant tests** - they reveal intended behavior and edge cases
4. **Study data flow** - how information moves through your focus area

### Example Focus Adaptations:

- **"tiling algorithm"** → Start with `greedy_tiling.py` (default), understand roll-up strategy; also study `dynamic_tiling.py` for DP alternative
- **"tree structure"** → Focus on how nodes maintain spans, parent-child relationships, tree invariants
- **"performance"** → Study async patterns, batch operations, caching strategies
- **Any specific file** → Read it deeply, find its tests, trace its callers and dependencies

---

## Full Codebase Expertise Path

_Follow this path when developing comprehensive expertise:_

## Core Algorithm (MUST understand first)
1. Read `ragzoom/greedy_tiling.py` - The default tiling algorithm. Focus on:
   - `find_optimal_tiling_over_roots()` - starts with leaf frontier, rolls up to fit budget
   - `_RollupCandidate` priority: quality_lost / tokens_saved (lower = better to roll up)
   - Quality metric: relevance × tokens
   - Also read `ragzoom/dynamic_tiling.py` for the DP alternative (optimal but slower)

2. Read `ragzoom/store.py` - Data structures that make it work:
   - Study `TreeNode` class - binary tree with character spans
   - Notice: Parent nodes = summaries, leaf nodes = raw text
   - Understand how every node tracks exact position in original document

## How Documents Become Trees
3. Read `ragzoom/index.py` focusing on `add_document()`:
   - Observe leaves created from text chunks (~200 tokens)
   - Trace bottom-up construction: pairs → summaries → pairs...
   - Verify character spans maintained throughout
   - Study async patterns with semaphore-controlled concurrency

## Query Flow (how retrieval actually works)
4. Read `ragzoom/retrieve.py` - The complete retrieval pipeline:
   - Trace how vector search finds seeds → MMR for diversity
   - Understand coverage map includes ancestors + siblings (tree fullness)
   - Notice ALL nodes get relevance scores (not just seeds!)
   - See how it returns node IDs + scores for tiling algorithm

5. Skim `ragzoom/assembler.py` - Observe simple concatenation of selected nodes

## Storage Layer
6. Read `ragzoom/store.py` constructor and `get_node()`:
   - Examine SQLite: tree structure + text storage
   - Study ChromaDB: embeddings for search
   - Understand LRU cache (1000 nodes default)
   - Trace document isolation via foreign keys

## Testing Pattern (crucial for development)
7. Read `tests/conftest.py` - Master the dual testing strategy:
   - Study `SimpleMockStore` for fast unit tests (4.5x faster)
   - Understand real `Store` for integration tests
   - Learn how tests are parameterized to run both ways

## Key Insights
- Token budget is STRICT - algorithm guarantees it's never exceeded
- Tree fullness enforced everywhere (no missing children)
- Relevance propagation happens for entire coverage tree
- Greedy tiling is O(n log n) via heap; DP is O(n×b) with memoization
- "Correct-by-construction" means no post-processing needed

## Quick Validation
Run: `pytest tests/test_budget_guarantee.py -xvs`
This test captures the essence of the system's guarantees.

## Contextual Learning Tips

When developing expertise:
- **Start with what's relevant** - If working on a bug, trace that specific code path first
- **Use tests as documentation** - They show intended behavior better than comments
- **Follow the data** - Understand transformations at each step
- **Question assumptions** - Many design decisions optimize for specific constraints

Remember: The tiling algorithms (`greedy_tiling.py` default, `dynamic_tiling.py` alternative) are the crown jewels. Everything else exists to support them.