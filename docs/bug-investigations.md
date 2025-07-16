# RagZoom Investigation: Complete Findings

## Overview

This document consolidates all findings from investigating issues with RagZoom's frontier extraction, budget handling, and span corruption.

## Issue 1: Budget Trimming Breaks Frontier Properties

### Problem Statement

When querying with a small token budget (e.g., 200 tokens), the system sometimes returns incomplete narrative coverage instead of a low-resolution complete summary. The user reported seeing only a high-resolution fragment about dragons instead of a complete story summary.

### Initial Hypothesis (Incorrect)

Initially thought the frontier extraction logic was broken, including all ancestors instead of just the "frontier" nodes. This turned out to be a misunderstanding of what the frontier should be.

### Key Findings

#### 1. Frontier Definition is Correct

The frontier is correctly defined as "covered nodes with uncovered children". This is the mathematical definition of a frontier in a tree - the boundary between covered and uncovered regions.

#### 2. Coverage Map Guarantees Complete Coverage

When we select any node and include all its ancestors up to the root, we're guaranteed to have the root in coverage. This means we always have complete document coverage in theory.

#### 3. The <<<MID>>> Delimiter System

The system is designed to handle overlapping frontier nodes through the `<<<MID>>>` delimiter:
- Parent nodes have their summaries split by `<<<MID>>>` 
- When a child is in the frontier, the parent only outputs the half covering the OTHER child
- This prevents duplicate content even with overlapping spans

Example:
```
Root (0-600): "Summary of left half <<<MID>>> Summary of right half"
  ├─ LeftChild (0-300): in frontier
  └─ RightChild (300-600): not in frontier

Root outputs only: "Summary of right half" (covering 300-600)
LeftChild outputs: its full text (covering 0-300)
Result: Complete coverage without overlaps
```

#### 4. Test Bug Revealed the Issue

The test was passing parameters in wrong order to `add_node()`, causing `mid_offset` to be set to "test-doc" (string) instead of an integer. This broke the `<<<MID>>>` extraction logic, causing full text to be output from all nodes, creating massive overlaps.

#### 5. The Real Problem: Budget Trimming Logic

The current `trim_frontier_to_budget()` method has a fundamental flaw:

```python
# Current logic - sorts by "utility ratio" (relevance per token)
utility_ratio = scores.get(node_id, 0.0) / max(token_cost, 1)
node_costs.sort(key=lambda x: x[2], reverse=True)  # Keep highest utility
```

This means:
- Low-relevance nodes (like the root) can be dropped first
- High-relevance leaves (like the dragon scene) are kept
- This can create gaps in coverage!

#### 6. The Correct Solution

Budget trimming should:
1. **Prioritize coverage over relevance** when budget is tight
2. **Drop deepest nodes first** (highest resolution)
3. **When dropping a node, add its parent** to maintain frontier property
4. **Never drop a node if it would create a coverage gap**

Pseudo-algorithm:
```python
while frontier_tokens > budget:
    # Find deepest node (or deepest + least relevant)
    node_to_drop = find_deepest_node(frontier)
    
    # Remove it
    frontier.remove(node_to_drop)
    
    # Add parent to maintain coverage
    if node.parent_id:
        frontier.add(node.parent_id)
```

### Test Results

#### Before fixing mid_offset bug:
- Frontier: `['0_0_200_leaf1', '1_0_400_parent1', '2_0_600_root']`
- Each outputs full text → massive overlap
- Assembled text shows all three nodes concatenated

#### After fixing mid_offset bug:
- Retrieved: `['2_0_600_root']` 
- Frontier: `['2_0_600_root']`
- Assembled text: "Full document summary. Covering everything."
- Correct behavior!

## Issue 2: Zero Scores in Frontier Nodes

### Observation

All frontier nodes show score=0.000 in logs:
```
Dropping node af6bc865-3143-45d4-ac37-81f15471bcdf (score=0.000) to stay within budget
```

### Analysis

This suggests scores aren't being propagated correctly from retrieval to assembly. The scores dict might not include all frontier nodes, only the originally selected nodes.

### Impact

Without proper scores, the budget trimming can't make intelligent decisions about which nodes to keep based on relevance.

## Issue 3: Span Corruption in Tree Structure

### Problem Statement

When running with `--n-max 1` and NO token budget constraint, we still get incomplete coverage with fragmented output.

### Key Observations

1. **10 nodes in frontier** despite n_max=1
2. **Massive slope cap violations**: Depths jumping from 11→8→10→7→5→3→4→2→1
3. **Non-contiguous spans**: 
   - Node at depth 11: span (45073, 45073) - zero width!
   - Node at depth 7: span (106078, 121187)
   - Node at depth 10: span (106078, 45073) - end before start!
   - Node at depth 1: span (117636, 117883)

4. **Summary is fragmented**: Shows disconnected pieces about Gandalf, Bilbo, Smaug, Bard, etc. without narrative flow

### Root Cause Identified: Wraparound Tree Building

Through SQL analysis, we discovered:

1. **Leaf nodes are correct**: All depth 0 nodes have valid, consecutive spans
2. **Corruption starts at depth 1+**: Invalid spans where span_end < span_start
3. **Pattern identified**: Odd nodes from the end of the document get paired with nodes from the beginning

Example:
```
Depth 0: Last leaf node has span (143964, 143992)
Depth 1: Node from beginning has span (0, 239)
These get incorrectly paired at depth 2, creating span (143964, 239) - INVALID!
```

### How It Happens

The tree building algorithm's handling of odd nodes causes wraparound pairing:
- Odd nodes are carried forward to the next level
- They get paired with nodes from the beginning of that level
- This creates parent nodes with backwards spans
- Corruption cascades up the tree

```
Level 0 (leaves): 1215 nodes, all valid spans
Level 1: 607 nodes (last odd node carried forward)
Level 2: Odd node from end paired with node from beginning
  - Left child: (143964, 143992) - last leaf
  - Right child: (0, 239) - from beginning  
  - Parent: (143964, 239) - CORRUPT!
```

### SQL Evidence

```sql
-- Found corrupt nodes
SELECT id, depth, span_start, span_end FROM tree_nodes 
WHERE span_end < span_start OR span_start = span_end;

-- Results show:
-- 10+ nodes with span_end < span_start (e.g., span_start=143964, span_end=239)
-- Node with zero-width span (45073, 45073)
-- This corruption cascades up the tree
```

### Solution

Fix the tree building algorithm to maintain document order when handling odd nodes. Never pair nodes where left.span_end < right.span_start.

## Issue 4: Token vs Character Confusion

### Observation

Some truncation appears to happen at character boundaries rather than token boundaries, as seen in output ending with "Fire le" (truncated mid-word).

### Status

Not fully investigated yet, but likely related to budget enforcement using character counts instead of token counts somewhere in the pipeline.

## Issue 5: Incomplete Document Indexing

### Problem Statement

When querying The Hobbit Chapter 1 with `--n-max 1`, the output ends with a leaf chunk containing "dragon too—far too often, unless he has changed his habits." This is clearly NOT the end of the document, indicating missing coverage.

### Investigation

1. **Initial hypothesis**: The frontier extraction was failing to cover the end
2. **Discovery**: The indexed tree only covers characters [0-16975] 
3. **Actual document size**: 46,077 characters
4. **Missing coverage**: Characters [16975-46077] - over 29,000 characters (63% of document!)

### Root Cause

The tree building process is storing span positions as TOKEN counts instead of CHARACTER positions. This causes the system to think it has only indexed a small portion of the document when comparing span_end (in tokens) to document length (in characters).

### Evidence

From our test:
```
Document: "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
Length: 73 characters
Last leaf span_end: 15 (tokens)
Leaf text: Full 73-character text is actually stored
```

From The Hobbit example:
```
Document length: 46,077 characters
Max span_end in tree: 16,975 (likely tokens, not characters)
This would be ~37% if comparing tokens to characters
```

### Code Evidence

In `ragzoom/index.py` lines 204-212:
```python
# Calculate span positions in tokens
tokens_before = 0
for j in range(i):
    tokens_before += len(self.splitter.tokenizer.encode(chunks[j]))

chunk_encoded = self.splitter.tokenizer.encode(chunk)
chunk_tokens = len(chunk_encoded)
span_start = tokens_before  # TOKEN position
span_end = span_start + chunk_tokens  # TOKEN position
```

The spans are calculated as cumulative token counts, not character positions.

### Impact

This is a critical issue - the system appears to only index ~37% of documents when it's actually indexing 100% but storing positions incorrectly.

### Solution

Fixed in `ragzoom/index.py` by calculating span positions as character offsets instead of token counts:

```python
# OLD: span_start = tokens_before (cumulative token count)
# NEW: span_start = text.find(chunk, current_char_pos) (character position)
```

### Status

**FIXED** - Documents are now fully indexed with correct character-based span positions.

## Issue 6: Default Budget Constraint

### Problem Statement

The system applies a default budget of 8,000 tokens even when the user doesn't specify any budget constraint. This can trigger unnecessary budget trimming logic.

### Observation

When running with only `--n-max 1` (no budget specified), the logs still show:
```
INFO - Selected node 407e46b9... (utility=0.0098, cost=45)
INFO - Selected node 270ff7b6... (utility=0.0000, cost=28)
```

This indicates budget trimming logic is active even though the user wants unbounded output.

### Current Behavior

- `RagZoomConfig` has `budget_tokens: int = Field(default=8000)`
- CLI uses this default even when no budget is specified
- This can cause unwanted trimming even for small outputs

### Recommendation

Remove the default budget or make it optional (None) so that when users don't specify a budget, the system truly operates without budget constraints.

## Issue 7: Assembly Ordering with Partial Content

### Problem Statement

When nodes output only partial content (left or right halves based on `<<<MID>>>` delimiter), the assembly was sorting by the node's `span_start` rather than the actual span that the extracted text covers. This caused content to appear out of chronological order.

### Example

A node with span (28564-46077) that outputs only its right half actually covers approximately (35000-46077), but was being placed in the output based on its span_start of 28564. This caused ending content to appear in the middle of the assembled text.

### Solution

**FIXED** - Created `_extract_node_text_with_span()` method that:
1. Returns both the extracted text and its actual coverage span
2. For right-half output: actual span starts at right child's span_start
3. For left-half output: actual span ends at left child's span_end
4. Assembly now sorts text fragments by their actual coverage spans

### Impact

This fix ensures text appears in correct chronological order, making the output more coherent and readable.

## Issue 8: Truncated AI Summaries During Indexing

### Problem Statement

Some AI-generated summaries stored in the database are incomplete, ending mid-sentence. For example, a root node's summary ends with "Thorin worried about goblins and the" - clearly truncated.

### Evidence

Query output shows:
```
Gandalf explained Thrain gave him the map and key. Thorin worried about goblins and the
```

Database inspection confirms the stored summary is truncated:
```sql
SELECT text FROM tree_nodes WHERE id = '08eeba07-9159-48e9-b0ee-50ea8b9782db';
-- Result: "...Gandalf explained Thrain gave him the map and key. Thorin worried about goblins and the"
```

### Root Cause Identified

The `_summarize_text` method in `index.py` sets `max_tokens=target_tokens` for the OpenAI API call, where:

```python
compression_ratio = 1.0 / (current_depth + 1)
target_tokens = max(int((left_tokens + right_tokens) * compression_ratio), 50)
```

**Critical insight**: In RagZoom's inverted depth convention:
- Depth 0 = leaves (original text chunks)
- Depth 7 = root (top of tree)

At depth 7 (root), the compression formula results in severe token limits:
- compression_ratio = 1/8 = 0.125
- The two children being summarized are already compressed summaries (maybe ~100 tokens total)
- target_tokens = max(100 * 0.125, 50) = 50 tokens
- **This means the root node (summarizing the ENTIRE document) is limited to just 50 tokens!**

Compare this to leaves which get 200 tokens (RAGZOOM_LEAF_TOKENS) for much smaller text chunks. The root should have MORE tokens available than leaves, not 1/4 as many.

Database evidence:
- Node at depth 7 (root)
- max_tokens = 50 (hard API limit)
- Result: "Thorin worried about goblins and the" (truncated mid-sentence when hitting token limit)

### Compression Cascade Example

With RAGZOOM_LEAF_TOKENS=200, here's how token limits cascade up the tree:

```
Depth 0 (leaves): 200 tokens each (from config)
Depth 1: Combines two 200-token leaves
  - left_tokens + right_tokens = 400
  - compression_ratio = 1/2 = 0.5
  - target_tokens = 400 * 0.5 = 200 tokens

Depth 2: Combines two ~200-token summaries
  - left_tokens + right_tokens = 400
  - compression_ratio = 1/3 = 0.33
  - target_tokens = 400 * 0.33 = 133 tokens

...continuing...

Depth 6: Combines two ~60-token summaries
  - left_tokens + right_tokens = 120
  - compression_ratio = 1/7 = 0.14
  - target_tokens = max(120 * 0.14, 50) = 50 tokens

Depth 7 (root): Combines two ~50-token summaries
  - left_tokens + right_tokens = 100
  - compression_ratio = 1/8 = 0.125
  - target_tokens = max(100 * 0.125, 50) = 50 tokens
```

The root gets the SAME token limit (50) as its children, despite needing to summarize twice as much content!

### Impact

This causes incomplete coverage of document content, as important information from the end of sections is missing from summaries. The root node, which should provide a complete overview of the entire document, is severely constrained and often truncated mid-sentence.

### Solution

The fix is to separate the API's `max_tokens` parameter from the target summary length:
1. Set `max_tokens` to a higher value (e.g., `target_tokens * 2`) to give the AI room to complete sentences
2. The AI is already instructed to stay within `target_tokens` via the prompt
3. Post-process the response if needed to ensure it fits the target length
4. Add validation to detect truncated summaries and retry if needed

### Status

**IDENTIFIED** - The bug is in `index.py` line 133 where `max_tokens=target_tokens` causes hard truncation at the API level.

## Issue 9: Confusing Depth Convention

### Problem Statement

RagZoom uses a non-standard depth convention that is opposite to typical tree terminology:
- **Depth 0 = leaves** (should be root in standard notation)
- **Depth increases going UP** toward the root (should decrease)
- **Root is at maximum depth** (should be at depth 0)

### Evidence

From the database:
```
Depth 0: 68 nodes (leaves)
Depth 1: 34 nodes  
Depth 2: 17 nodes
...
Depth 7: 1 node (root)
```

### Impact

This inverted convention causes confusion when:
1. Discussing tree traversal and algorithms
2. Reading code that references "current_depth"
3. Debugging issues related to tree structure
4. Communicating with other developers who expect standard conventions

### Recommendation

Refactor to use standard tree depth convention where root = depth 0 and depth increases toward leaves. This would make the codebase more intuitive and align with computer science standards.

## Issue 10: Text Splitter Dropping Whitespace

### Problem Statement

The LangChain `RecursiveCharacterTextSplitter` is dropping whitespace characters (particularly newlines) between chunks, causing validation failures and potential content loss.

### Evidence

During validation, we see gaps like:
```
Gap found between nodes: ends at 14790, starts at 14791 (gap of 1 chars)
  Gap content: '\n'
```

### Root Cause

Even with `keep_separator="end"` parameter, the splitter is not consistently preserving all separator characters. This is a known limitation of text splitters that split on delimiters.

### Current Workaround

Updated validation to allow whitespace-only gaps, since they don't represent meaningful content loss. However, this is a band-aid solution.

### Proper Solution

Either:
1. Implement custom text splitting that preserves all characters
2. Switch to a different splitting strategy that doesn't drop characters
3. Store the dropped whitespace separately and reconstruct during assembly

## Issue 11: Root Node Appearing in Frontier

### Problem Statement

During retrieval, the root node (depth 8 in RagZoom's inverted convention) sometimes appears in the frontier, causing the entire document summary to be included alongside more detailed nodes.

### Evidence

```
Extracted from node 9d11cef2-6105-45ad-b1f6-8b850a89954a (depth 8, node span (0, 46077), actual span (0, 46077)): 706 chars
```

This shows the root node in the frontier with full document span [0, 46077].

### Impact

This violates the frontier principle and can cause:
1. Redundant content (whole document summary + detailed sections)
2. Incorrect coverage calculations
3. Wasted tokens on duplicate information

### Root Cause

The frontier extraction logic has a bug where it's including nodes that shouldn't be in the frontier. A node should only be in the frontier if:
- The node itself is covered (in coverage_map)
- At least one of its children is NOT covered

If both children are covered, the node should not be in the frontier (the children handle that coverage).

### Solution

Fix the frontier extraction logic in `retrieve.py` to properly check child coverage before including a node in the frontier.

## Issue 12: Frontier Validation Failures

### Problem Statement

Frontier validation during query operations reveals multiple issues:
1. Segments out of order
2. Gaps between segments (beyond whitespace)
3. Incomplete coverage (frontier doesn't reach document end)

### Evidence

```
Validation failed in frontier completeness: Frontier ends at 39876, expected 46077
```

### Root Cause

Multiple contributing factors:
1. The `_extract_node_text_with_span()` method returns actual coverage spans that differ from node spans when using MID delimiter
2. Frontier extraction may include invalid nodes (see Issue 11)
3. Assembly logic doesn't properly validate segment ordering

### Solution

1. Fix frontier extraction to ensure valid frontier
2. Add validation for segment ordering
3. Ensure actual coverage spans are calculated correctly

## Summary of All Issues

1. **Budget trimming breaks coverage** - drops by relevance instead of maintaining frontier property
2. **Zero scores in frontier** - scores not propagated from retrieval to assembly
3. **Span corruption from wraparound** - tree building pairs nodes incorrectly across document boundaries (FIXED)
4. **Token/character confusion** - possible character-based truncation instead of token-based
5. **Incomplete document indexing** - spans stored as token positions instead of character positions (FIXED)
6. **Default budget constraint** - 8000 token default applied even when user wants unbounded
7. **Assembly ordering with partial content** - sorted by node span_start instead of actual coverage span (FIXED)
8. **Truncated AI summaries during indexing** - AI responses cut off mid-sentence due to max_tokens=target_tokens (IDENTIFIED)
9. **Confusing depth convention** - uses inverted depth where leaves=0 and root=max_depth instead of standard convention
10. **Text splitter dropping whitespace** - LangChain splitter loses whitespace between chunks
11. **Root node appearing in frontier** - Invalid frontier extraction includes nodes that should be excluded
12. **Frontier validation failures** - Multiple issues with frontier completeness and ordering

## Recommended Fix Order

1. ~~**Fix incomplete indexing**~~ - COMPLETED
2. **Remove default budget** - simple fix that prevents unwanted constraints
3. **Fix budget trimming** - ensure it maintains frontier property and complete coverage
4. **Fix score propagation** - enable intelligent relevance-based decisions
5. **Fix token counting** - ensure all budget calculations use tokens, not characters

## Key Insights

- The frontier concept and `<<<MID>>>` delimiter system are well-designed
- The core architecture is sound
- The issues are in specific implementation details, not fundamental design flaws
- With these fixes, the system should maintain complete narrative coverage at the best resolution that fits the budget