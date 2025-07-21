# Token Position Resolver Algorithm Implementation

## 1. Overview

The TokenPositionResolver computes positions for nodes and segments in the output token space, where each segment's width is proportional to its token cost. This differs from character-based positioning where positions reflect the source document spans.

The key challenge is computing positions for covered but unselected nodes, which requires finding their selected descendants and aggregating their token costs while maintaining the invariant that parent width equals the sum of children widths.

**Implementation Status**: ✅ Complete and tested

## 2. Core Data Structures

```python
class TokenPositionResolver:
    def __init__(self, segments: list[Segment], store: Store, tokenizer):
        self.store = store
        self.tokenizer = tokenizer
        
        # Map from segment index to (start_pos, end_pos) in token space
        self.segment_positions: dict[int, tuple[float, float]] = {}
        
        # Map from (node_id, side) to segment index in tiling
        self.segment_lookup: dict[tuple[str, Optional[str]], int] = {}
        
        # Map from node_id to total token cost (for covered nodes)
        self.node_token_costs: dict[str, float] = {}
        
        # Map from node_id to start position in token space
        self.node_start_positions: dict[str, float] = {}
        
        # Total tokens in the output
        self.total_tokens: float = 0.0
```

## 3. Algorithm Phases

### Phase 1: Compute Segment Positions

This is straightforward - segments appear in order in the tiling, so we accumulate their token costs:

```python
def _compute_segment_positions(self, segments: list[Segment]):
    """Compute positions for segments in the tiling."""
    current_pos = 0.0
    
    for idx, segment in enumerate(segments):
        # Get token count for this segment
        token_count = self._get_segment_tokens(segment)
        
        # Record position
        self.segment_positions[idx] = (current_pos, current_pos + token_count)
        
        # Build reverse lookup
        key = (segment.node_id, segment.side)
        self.segment_lookup[key] = idx
        
        # Advance position
        current_pos += token_count
    
    self.total_tokens = current_pos
```

### Phase 2: Build Coverage Tree Structure

Before computing node costs, we need to understand the tree structure within the coverage map:

```python
def _build_coverage_structure(self, segments: list[Segment], coverage_map: dict[str, bool]):
    """Build parent-child relationships for covered nodes."""
    self.covered_nodes = set(coverage_map.keys())
    self.children_map: dict[str, tuple[Optional[str], Optional[str]]] = {}
    
    for node_id in self.covered_nodes:
        node = self.store.get_node(node_id)
        if node:
            left_child_id = node.left_child_id if node.left_child_id in self.covered_nodes else None
            right_child_id = node.right_child_id if node.right_child_id in self.covered_nodes else None
            self.children_map[node_id] = (left_child_id, right_child_id)
```

### Phase 3: Compute Node Token Costs (Bottom-Up)

This is the complex part. We compute token costs bottom-up, ensuring the parent width invariant:

```python
def _compute_node_token_costs(self):
    """Compute token costs for all covered nodes."""
    # Get all covered nodes sorted by depth (leaves first)
    covered_nodes = []
    for node_id in self.covered_nodes:
        node = self.store.get_node(node_id)
        if node:
            covered_nodes.append((node.depth, node_id))
    covered_nodes.sort()  # Sort by depth (ascending)
    
    # Process nodes bottom-up
    for depth, node_id in covered_nodes:
        node = self.store.get_node(node_id)
        if not node:
            continue
            
        if node.depth == 0:
            # Leaf node - check if it's selected
            self._compute_leaf_cost(node)
        else:
            # Internal node - aggregate from children or segments
            self._compute_internal_node_cost(node)

def _compute_leaf_cost(self, node: TreeNode):
    """Compute token cost for a leaf node."""
    # Check if this leaf is selected
    if (node.id, None) in self.segment_lookup:
        # It's selected - use actual segment tokens
        idx = self.segment_lookup[(node.id, None)]
        start, end = self.segment_positions[idx]
        self.node_token_costs[node.id] = end - start
    else:
        # Not selected - cost is 0
        self.node_token_costs[node.id] = 0.0

def _compute_internal_node_cost(self, node: TreeNode):
    """Compute token cost for an internal node."""
    left_child_id, right_child_id = self.children_map.get(node.id, (None, None))
    
    # Check if segments are selected
    left_selected = (node.id, "LEFT") in self.segment_lookup
    right_selected = (node.id, "RIGHT") in self.segment_lookup
    
    # Compute costs for each half
    if left_selected:
        # Use actual segment cost
        idx = self.segment_lookup[(node.id, "LEFT")]
        start, end = self.segment_positions[idx]
        left_cost = end - start
    elif left_child_id:
        # Use child's cost (recursive)
        left_cost = self.node_token_costs.get(left_child_id, 0.0)
    else:
        left_cost = 0.0
    
    if right_selected:
        # Use actual segment cost
        idx = self.segment_lookup[(node.id, "RIGHT")]
        start, end = self.segment_positions[idx]
        right_cost = end - start
    elif right_child_id:
        # Use child's cost (recursive)
        right_cost = self.node_token_costs.get(right_child_id, 0.0)
    else:
        right_cost = 0.0
    
    # Parent cost = sum of children costs
    self.node_token_costs[node.id] = left_cost + right_cost
```

### Phase 4: Compute Node Start Positions (Top-Down)

Once we have token costs, we compute start positions top-down:

```python
def _compute_node_start_positions(self):
    """Compute start positions for all covered nodes."""
    # Start with root at position 0
    root_nodes = [nid for nid in self.covered_nodes 
                  if self.store.get_node(nid).parent_id not in self.covered_nodes]
    
    for root_id in root_nodes:
        self._compute_positions_recursive(root_id, 0.0)

def _compute_positions_recursive(self, node_id: str, start_pos: float):
    """Recursively compute positions for a node and its children."""
    self.node_start_positions[node_id] = start_pos
    
    node = self.store.get_node(node_id)
    if not node or node.depth == 0:
        return
    
    left_child_id, right_child_id = self.children_map.get(node_id, (None, None))
    
    # Position children based on their costs
    if left_child_id:
        self._compute_positions_recursive(left_child_id, start_pos)
    
    if right_child_id:
        # Right child starts after left child's tokens
        left_cost = self._get_left_subtree_cost(node_id)
        right_start = start_pos + left_cost
        self._compute_positions_recursive(right_child_id, right_start)

def _get_left_subtree_cost(self, node_id: str) -> float:
    """Get the token cost of the left subtree."""
    node = self.store.get_node(node_id)
    if not node:
        return 0.0
    
    # Check if left segment is selected
    if (node_id, "LEFT") in self.segment_lookup:
        idx = self.segment_lookup[(node_id, "LEFT")]
        start, end = self.segment_positions[idx]
        return end - start
    
    # Otherwise, use left child's total cost
    left_child_id, _ = self.children_map.get(node_id, (None, None))
    if left_child_id:
        return self.node_token_costs.get(left_child_id, 0.0)
    
    return 0.0
```

## 4. Public Interface Implementation

```python
def get_extent(self) -> float:
    """Return total tokens in the output."""
    return self.total_tokens

def get_segment_position(self, segment: Segment, segment_index: int) -> tuple[float, float]:
    """Return position for a segment in the tiling."""
    return self.segment_positions.get(segment_index, (0.0, 0.0))

def get_node_position(self, node: TreeNode) -> tuple[float, float]:
    """Return position for a covered node."""
    if node.id not in self.node_token_costs:
        return (0.0, 0.0)
    
    start = self.node_start_positions.get(node.id, 0.0)
    width = self.node_token_costs.get(node.id, 0.0)
    return (start, start + width)
```

## 5. Optimization Strategies

### 5.1 Caching Token Counts

```python
def _get_segment_tokens(self, segment: Segment) -> float:
    """Get token count for a segment with caching."""
    cache_key = (segment.node_id, segment.side)
    if cache_key in self._token_cache:
        return self._token_cache[cache_key]
    
    node = self.store.get_node(segment.node_id)
    if not node or not node.text:
        return 0.0
    
    if node.depth == 0 or node.mid_offset is None:
        text = node.text
    elif segment.side == "LEFT":
        text = node.text[:node.mid_offset]
    else:  # RIGHT
        text = node.text[node.mid_offset:]
    
    token_count = float(len(self.tokenizer.encode(text.strip())))
    self._token_cache[cache_key] = token_count
    return token_count
```

### 5.2 Lazy Computation

Only compute positions for nodes that are actually queried:

```python
def get_node_position(self, node: TreeNode) -> tuple[float, float]:
    """Lazily compute position if not already cached."""
    if node.id not in self.node_start_positions:
        # Trigger computation for this branch
        self._ensure_node_computed(node.id)
    
    return self._get_cached_position(node.id)
```

## 6. Invariant Validation

For debugging, validate that parent width = sum of children widths:

```python
def _validate_invariants(self):
    """Validate that parent widths equal sum of children widths."""
    for node_id in self.covered_nodes:
        node = self.store.get_node(node_id)
        if not node or node.depth == 0:
            continue
        
        parent_cost = self.node_token_costs.get(node_id, 0.0)
        left_child_id, right_child_id = self.children_map.get(node_id, (None, None))
        
        child_sum = 0.0
        if (node_id, "LEFT") in self.segment_lookup:
            # Left segment selected
            idx = self.segment_lookup[(node_id, "LEFT")]
            start, end = self.segment_positions[idx]
            child_sum += end - start
        elif left_child_id:
            child_sum += self.node_token_costs.get(left_child_id, 0.0)
        
        if (node_id, "RIGHT") in self.segment_lookup:
            # Right segment selected
            idx = self.segment_lookup[(node_id, "RIGHT")]
            start, end = self.segment_positions[idx]
            child_sum += end - start
        elif right_child_id:
            child_sum += self.node_token_costs.get(right_child_id, 0.0)
        
        assert abs(parent_cost - child_sum) < 1e-6, \
            f"Node {node_id}: parent={parent_cost}, children={child_sum}"
```

## 7. Example Walkthrough

Consider a tree with the following tiling:
- Segment 0: (node_A, LEFT) - 100 tokens
- Segment 1: (leaf_B, None) - 50 tokens  
- Segment 2: (node_A, RIGHT) - 150 tokens

And coverage map includes: root, node_A, leaf_B, node_C (unselected child of node_A.RIGHT)

1. **Segment positions:**
   - Segment 0: [0, 100)
   - Segment 1: [100, 150)
   - Segment 2: [150, 300)

2. **Node token costs (bottom-up):**
   - leaf_B: 50 (selected)
   - node_C: 0 (not selected, no selected descendants)
   - node_A: 100 + 150 = 250 (sum of selected segments)
   - root: 250 (equals node_A since it's the only child)

3. **Node start positions (top-down):**
   - root: 0
   - node_A: 0 (same as root)
   - leaf_B: 100 (after node_A.LEFT)
   - node_C: 150 (after node_A.LEFT and leaf_B)

4. **Final positions:**
   - root: [0, 250)
   - node_A: [0, 250)
   - leaf_B: [100, 150)
   - node_C: [150, 150) (width=0, not selected)

## 8. Complexity Analysis

- **Time Complexity:**
  - Segment position computation: O(n) where n = number of segments
  - Node cost computation: O(m) where m = number of covered nodes
  - Node position computation: O(m)
  - Total: O(n + m)

- **Space Complexity:**
  - O(n + m) for storing positions and costs

## 9. Implementation Notes

1. **Floating point precision:** Use consistent precision to avoid rounding errors when validating invariants.

2. **Empty nodes:** Handle nodes with no selected descendants carefully - they should have width 0 but still have a valid position.

3. **Partial coverage:** The algorithm works correctly even when only part of the tree is covered, as long as the covered portion forms a connected subtree.

4. **Thread safety:** If used in concurrent contexts, protect the internal caches with appropriate locking.

5. **Memory optimization:** For very large trees, consider computing positions on-demand rather than pre-computing everything.

## 10. Key Implementation Decisions

### 10.1 Segment Ordering

The implementation sorts segments by their document position (span_start) before assigning token positions. This ensures the token visualization maintains left-to-right document flow:

```python
# Sort segments by their document order
sorted_infos = []
for idx, info in enumerate(segment_infos):
    node = store.get_node(info.segment.node_id)
    if node:
        # Calculate actual span_start for the segment
        if info.segment.side == "RIGHT":
            # RIGHT segment starts after left child
            seg_span_start = left_child.span_end
        else:
            seg_span_start = node.span_start
        sorted_infos.append((seg_span_start, idx, info))
sorted_infos.sort(key=lambda x: x[0])
```

### 10.2 Fallback for Unselected Nodes

For covered but unselected nodes with no selected descendants, the implementation uses the node's full text token count as a fallback:

```python
# If child is covered but has zero cost, use its full text cost
if left_cost == 0.0 and node.left_child_id in self.coverage_map:
    left_child = self.store.get_node(node.left_child_id)
    if left_child and left_child.text:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        left_cost = float(len(tokenizer.encode(left_child.text)))
```

### 10.3 MID Delimiter Handling

The implementation handles the MID delimiter for internal nodes when computing fallback costs:

```python
# For internal nodes, need to handle MID delimiter
if right_child.depth > 0 and right_child.mid_offset is not None:
    text = right_child.text.replace("<<<MID>>>", "")
    right_cost = float(len(tokenizer.encode(text)))
```

### 10.4 Integration with DP Algorithm

Rather than recomputing token costs, the implementation receives pre-computed costs from the DP algorithm via SegmentInfo objects. This ensures consistency and avoids redundant tokenization.