# Tree Visualization Coordinate Systems

## Overview

This document describes the implementation of multiple coordinate systems for the `build_ascii_tree()` visualization function. The system supports both character-based positioning (showing document coverage) and token-based positioning (showing budget allocation), with an extensible architecture for future coordinate systems.

**Implementation Status**: ✅ Complete and tested

## Motivation

The current character-based coordinate system shows where in the source document we have high vs low detail coverage. While useful, it has limitations:

- Leaf segments (high detail) appear as tiny slivers since they cover small character spans
- The visualization doesn't reflect how the token budget is actually distributed
- Segments have inconsistent widths, making labeling difficult

A token-based coordinate system would show how the output token budget is allocated, with each segment's width proportional to its token cost. This better reflects what the DP algorithm optimizes for.

## Design

### Position Resolver Interface

```python
from abc import ABC, abstractmethod
from typing import Optional

class PositionResolver(ABC):
    @abstractmethod
    def get_extent(self) -> float:
        """Return the total extent of the coordinate space.
        
        For character-based: total characters in document
        For token-based: total tokens in output summary
        """
        
    @abstractmethod
    def get_segment_position(self, segment: Segment, segment_index: int) -> tuple[float, float]:
        """Return (start, end) position for a segment in the tiling.
        
        Args:
            segment: The segment to position
            segment_index: Index of this segment in the tiling (for token-based ordering)
            
        Returns:
            (start, end) positions in coordinate space
        """
        
    @abstractmethod
    def get_node_position(self, node: TreeNode) -> tuple[float, float]:
        """Return (start, end) position for a covered but unselected node.
        
        For character-based: node's span positions
        For token-based: sum of selected descendants' token costs
        """
```

### Character-based Implementation

```python
class CharacterPositionResolver(PositionResolver):
    """Maps segments to their character positions in the source document."""
    
    def __init__(self, all_nodes: list[TreeNode], store: Store):
        self.store = store
        self.doc_start = min(node.span_start for node in all_nodes)
        self.doc_end = max(node.span_end for node in all_nodes)
    
    def get_extent(self) -> float:
        return float(self.doc_end - self.doc_start)
    
    def get_segment_position(self, segment: Segment, segment_index: int) -> tuple[float, float]:
        node = self.store.get_node(segment.node_id)
        if not node:
            return (0.0, 0.0)
        return (float(node.span_start - self.doc_start), 
                float(node.span_end - self.doc_start))
    
    def get_node_position(self, node: TreeNode) -> tuple[float, float]:
        return (float(node.span_start - self.doc_start), 
                float(node.span_end - self.doc_start))
```

### Token-based Implementation

```python
class TokenPositionResolver(PositionResolver):
    """Maps segments to positions based on output token allocation."""
    
    def __init__(self, segments: list[Segment], store: Store, tokenizer):
        self.store = store
        self.tokenizer = tokenizer
        self.segment_positions = {}
        self.node_token_costs = {}
        
        # Pre-compute segment positions based on tiling order
        current_pos = 0.0
        for idx, segment in enumerate(segments):
            token_count = self._get_segment_tokens(segment)
            self.segment_positions[idx] = (current_pos, current_pos + token_count)
            current_pos += token_count
        
        self.total_tokens = current_pos
        
        # Pre-compute token costs for all nodes in coverage map
        self._compute_node_costs(segments)
    
    def get_extent(self) -> float:
        return self.total_tokens
    
    def get_segment_position(self, segment: Segment, segment_index: int) -> tuple[float, float]:
        return self.segment_positions.get(segment_index, (0.0, 0.0))
    
    def get_node_position(self, node: TreeNode) -> tuple[float, float]:
        """For covered but unselected nodes, compute position based on selected descendants."""
        if node.id not in self.node_token_costs:
            return (0.0, 0.0)
        
        # Find position by summing tokens of selected descendants
        start_pos = self._find_start_position(node)
        token_cost = self.node_token_costs[node.id]
        return (start_pos, start_pos + token_cost)
    
    def _get_segment_tokens(self, segment: Segment) -> float:
        """Get token count for a segment."""
        node = self.store.get_node(segment.node_id)
        if not node or not node.text:
            return 0.0
            
        if node.depth == 0 or node.mid_offset is None:
            # Leaf node - use full text
            text = node.text
        elif segment.side == "LEFT":
            text = node.text[:node.mid_offset]
        else:  # RIGHT
            text = node.text[node.mid_offset:]
            
        return float(len(self.tokenizer.encode(text.strip())))
    
    def _compute_node_costs(self, segments: list[Segment]):
        """Pre-compute token costs for all nodes based on selected segments."""
        # Build segment lookup
        selected_segments = {(s.node_id, s.side) for s in segments}
        
        # Compute costs recursively...
        # (Implementation details omitted for brevity)
    
    def _find_start_position(self, node: TreeNode) -> float:
        """Find where this node's selected descendants begin in the tiling."""
        # (Implementation details omitted for brevity)
```

## Updated Function Signature

```python
def build_ascii_tree(
    segments: list[Segment],
    store: Store,
    document_id: str,
    width: int = 120,
    coverage_map: Optional[dict[str, bool]] = None,
    seed_node_ids: Optional[set[str]] = None,
    position_resolver: Optional[PositionResolver] = None,
) -> str:
    """Build an ASCII tree visualization.
    
    Args:
        segments: List of segments in the tiling
        store: Store instance
        document_id: Document to visualize
        width: Terminal width for visualization
        coverage_map: Optional dict of covered node IDs
        seed_node_ids: Optional set of seed node IDs (marked with *)
        position_resolver: Optional position resolver (defaults to character-based)
    """
    if position_resolver is None:
        all_nodes = store.get_all_nodes_for_document(document_id)
        position_resolver = CharacterPositionResolver(all_nodes, store)
    
    # Rest of implementation uses position_resolver.get_segment_position() etc.
```

## Usage Examples

```python
# Default character-based visualization
viz = build_ascii_tree(segments, store, doc_id)

# Token-based visualization
tokenizer = tiktoken.get_encoding("cl100k_base")
resolver = TokenPositionResolver(segments, store, tokenizer)
viz = build_ascii_tree(segments, store, doc_id, position_resolver=resolver)
```

## Benefits

1. **Extensibility**: Easy to add new coordinate systems without modifying visualization logic
2. **Better token visualization**: All segments have readable widths proportional to token cost
3. **Preserved invariants**: Parent width = sum of children widths in all coordinate systems
4. **Clean abstraction**: Visualization code doesn't need to understand coordinate details

## Implementation Notes

- The position resolver is responsible for all coordinate calculations
- For token-based positioning, unselected covered nodes show the sum of their selected descendants
- The visualization algorithm remains unchanged, just queries positions differently
- Character-based remains the default for backward compatibility
- **Segment Ordering**: In token-based view, segments are sorted by their document position (span_start) to maintain left-to-right document flow in the visualization
- **MID Delimiter Handling**: The DP algorithm includes <<<MID>>> tokens in its cost calculation, but the assembler strips them. TokenPositionResolver accounts for this by using the token costs from SegmentInfo
- **Coverage Visualization**: Fixed by painting covered nodes first (priority 0), then selected segments on top (priority 1)
- **Fallback for Unselected Nodes**: When computing positions for covered but unselected nodes with no selected descendants, the resolver uses the node's full text token count as a fallback

## Implementation Details

### Phase 1: Enhance DP Algorithm to Return Metadata

#### 1.1 Create New Data Classes

```python
# In ragzoom/dynamic_frontier.py

@dataclass
class SegmentInfo:
    """Extended segment information including token cost."""
    segment: Segment
    token_cost: int
    
@dataclass
class DPResult:
    """Complete result from DP frontier generation."""
    segments: list[Segment]
    segment_infos: list[SegmentInfo]
    total_quality: float
    coverage_map: dict[str, bool]  # All nodes in coverage tree
```

#### 1.2 Update DynamicFrontierGenerator

```python
class DynamicFrontierGenerator:
    def find_optimal_frontier(
        self, budget_tokens: int, scores: dict[str, float], document_id: Optional[str]
    ) -> DPResult:  # Changed return type
        logger.info("Using DP frontier generation")
        root_node = self.store.get_root_node_for_document(document_id)
        if not root_node:
            return DPResult([], [], 0.0, {})
        
        # Build coverage map from scores
        coverage_nodes = self.store.get_nodes(list(scores.keys()))
        coverage_map = {node.id: True for node in coverage_nodes}
        
        # Add ancestors to coverage map
        for node in coverage_nodes:
            current = node
            while current.parent_id:
                coverage_map[current.parent_id] = True
                current = self.store.get_node(current.parent_id)
                if not current:
                    break
        
        self._memo_cache = {}
        segments, quality = self._find_optimal_frontier_for_span(
            root_node, budget_tokens, scores
        )
        
        # Build segment infos with costs
        segment_infos = []
        for seg in segments:
            cost = self._get_segment_cost(seg)
            segment_infos.append(SegmentInfo(seg, cost))
        
        logger.info(
            f"DP frontier generated with total quality {quality:.3f} and {len(segments)} segments."
        )
        
        return DPResult(
            segments=segments,
            segment_infos=segment_infos,
            total_quality=quality,
            coverage_map=coverage_map
        )
```

### Phase 2: Update Retrieval Layer

#### 2.1 Update RetrievalResult

```python
# In ragzoom/retrieve.py

@dataclass
class RetrievalResult:
    """Result of the retrieval process."""
    coverage_tree_root: Optional[TreeNode]
    coverage_map: dict[str, bool]
    frontier_segments: list[Segment]
    segment_infos: list[SegmentInfo]  # NEW
    seed_node_ids: set[str]
    n_max: int
```

#### 2.2 Update retrieve_segments

```python
def retrieve_segments(
    query: str,
    store: Store,
    config: RagZoomConfig,
    document_id: Optional[str] = None,
    pinned_node_ids: Optional[list[str]] = None,
) -> RetrievalResult:
    # ... existing MMR and coverage logic ...
    
    # Generate frontier using DP
    generator = DynamicFrontierGenerator(config, store)
    dp_result = generator.find_optimal_frontier(
        config.budget_tokens, scores, document_id
    )
    
    return RetrievalResult(
        coverage_tree_root=coverage_tree_root,
        coverage_map=dp_result.coverage_map,  # Use DP's coverage map
        frontier_segments=dp_result.segments,
        segment_infos=dp_result.segment_infos,  # Pass through
        seed_node_ids=seed_node_ids,
        n_max=config.n_max,
    )
```

### Phase 3: Implement Position Resolvers

#### 3.1 Base Class and Character-based Resolver

```python
# In ragzoom/tree_viz.py

from abc import ABC, abstractmethod
from typing import Optional

class PositionResolver(ABC):
    """Abstract base class for coordinate system resolvers."""
    
    @abstractmethod
    def get_extent(self) -> float:
        """Return the total extent of the coordinate space."""
        pass
        
    @abstractmethod
    def get_segment_position(self, segment: Segment, segment_index: int) -> tuple[float, float]:
        """Return (start, end) position for a segment in the tiling."""
        pass
        
    @abstractmethod
    def get_node_position(self, node: TreeNode) -> tuple[float, float]:
        """Return (start, end) position for a covered but unselected node."""
        pass


class CharacterPositionResolver(PositionResolver):
    """Character-based positioning (current default behavior)."""
    
    def __init__(self, all_nodes: list[TreeNode], store: Store):
        self.store = store
        self.doc_start = min(node.span_start for node in all_nodes)
        self.doc_end = max(node.span_end for node in all_nodes)
        
    def get_extent(self) -> float:
        return float(self.doc_end - self.doc_start)
        
    def get_segment_position(self, segment: Segment, segment_index: int) -> tuple[float, float]:
        node = self.store.get_node(segment.node_id)
        if not node:
            return (0.0, 0.0)
        return (float(node.span_start - self.doc_start), 
                float(node.span_end - self.doc_start))
                
    def get_node_position(self, node: TreeNode) -> tuple[float, float]:
        return (float(node.span_start - self.doc_start), 
                float(node.span_end - self.doc_start))
```

#### 3.2 Token-based Resolver

```python
class TokenPositionResolver(PositionResolver):
    """Token-based positioning showing output budget allocation."""
    
    def __init__(self, segment_infos: list[SegmentInfo], coverage_map: dict[str, bool], store: Store):
        self.store = store
        self.segment_infos = segment_infos
        self.coverage_map = coverage_map
        
        # Build segment lookup for quick access
        self.segment_lookup = {(info.segment.node_id, info.segment.side): idx 
                              for idx, info in enumerate(segment_infos)}
        
        # Compute segment positions
        self.segment_positions = {}
        current_pos = 0.0
        for idx, info in enumerate(segment_infos):
            self.segment_positions[idx] = (current_pos, current_pos + info.token_cost)
            current_pos += info.token_cost
        self.total_tokens = current_pos
        
        # Compute positions for covered but unselected nodes
        self.node_positions = {}
        self._compute_node_positions()
    
    def get_extent(self) -> float:
        return self.total_tokens
    
    def get_segment_position(self, segment: Segment, segment_index: int) -> tuple[float, float]:
        return self.segment_positions.get(segment_index, (0.0, 0.0))
    
    def get_node_position(self, node: TreeNode) -> tuple[float, float]:
        return self.node_positions.get(node.id, (0.0, 0.0))
    
    def _compute_node_positions(self):
        """Compute positions for all covered nodes based on selected descendants."""
        # First pass: compute token costs for all nodes
        node_costs = {}
        
        def compute_cost(node_id: str) -> float:
            if node_id in node_costs:
                return node_costs[node_id]
            
            node = self.store.get_node(node_id)
            if not node:
                node_costs[node_id] = 0.0
                return 0.0
            
            # Check if this node has selected segments
            total_cost = 0.0
            has_selected = False
            
            # Check for leaf segment
            if (node_id, None) in self.segment_lookup:
                idx = self.segment_lookup[(node_id, None)]
                total_cost = self.segment_infos[idx].token_cost
                has_selected = True
            else:
                # Check for left/right segments
                left_cost = 0.0
                right_cost = 0.0
                
                if (node_id, "LEFT") in self.segment_lookup:
                    idx = self.segment_lookup[(node_id, "LEFT")]
                    left_cost = self.segment_infos[idx].token_cost
                    has_selected = True
                elif node.left_child_id:
                    left_cost = compute_cost(node.left_child_id)
                
                if (node_id, "RIGHT") in self.segment_lookup:
                    idx = self.segment_lookup[(node_id, "RIGHT")]
                    right_cost = self.segment_infos[idx].token_cost
                    has_selected = True
                elif node.right_child_id:
                    right_cost = compute_cost(node.right_child_id)
                
                total_cost = left_cost + right_cost
            
            node_costs[node_id] = total_cost
            return total_cost
        
        # Compute costs for all covered nodes
        for node_id in self.coverage_map:
            compute_cost(node_id)
        
        # Second pass: compute positions
        def compute_position(node_id: str) -> tuple[float, float]:
            if node_id in self.node_positions:
                return self.node_positions[node_id]
            
            node = self.store.get_node(node_id)
            if not node or node_costs.get(node_id, 0) == 0:
                self.node_positions[node_id] = (0.0, 0.0)
                return (0.0, 0.0)
            
            # If this node has selected segments, use their positions
            if (node_id, None) in self.segment_lookup:
                idx = self.segment_lookup[(node_id, None)]
                pos = self.segment_positions[idx]
                self.node_positions[node_id] = pos
                return pos
            
            # For internal nodes, compute based on children
            start_pos = float('inf')
            end_pos = 0.0
            
            # Check left side
            if (node_id, "LEFT") in self.segment_lookup:
                idx = self.segment_lookup[(node_id, "LEFT")]
                seg_start, seg_end = self.segment_positions[idx]
                start_pos = min(start_pos, seg_start)
                end_pos = max(end_pos, seg_end)
            elif node.left_child_id and node.left_child_id in self.coverage_map:
                child_start, child_end = compute_position(node.left_child_id)
                if child_end > child_start:  # Non-empty child
                    start_pos = min(start_pos, child_start)
                    end_pos = max(end_pos, child_end)
            
            # Check right side
            if (node_id, "RIGHT") in self.segment_lookup:
                idx = self.segment_lookup[(node_id, "RIGHT")]
                seg_start, seg_end = self.segment_positions[idx]
                start_pos = min(start_pos, seg_start)
                end_pos = max(end_pos, seg_end)
            elif node.right_child_id and node.right_child_id in self.coverage_map:
                child_start, child_end = compute_position(node.right_child_id)
                if child_end > child_start:  # Non-empty child
                    start_pos = min(start_pos, child_start)
                    end_pos = max(end_pos, child_end)
            
            # Handle case where no children have positions
            if start_pos == float('inf'):
                start_pos = 0.0
            
            self.node_positions[node_id] = (start_pos, end_pos)
            return (start_pos, end_pos)
        
        # Compute positions for all covered nodes
        for node_id in self.coverage_map:
            compute_position(node_id)
```

### Phase 4: Update build_ascii_tree

```python
def build_ascii_tree(
    segments: list[Segment],
    store: Store,
    document_id: str,
    width: int = 120,
    coverage_map: Optional[dict[str, bool]] = None,
    seed_node_ids: Optional[set[str]] = None,
    position_resolver: Optional[PositionResolver] = None,
) -> str:
    """Build an ASCII tree visualization using the specified coordinate system."""
    all_nodes = store.get_all_nodes_for_document(document_id)
    if not all_nodes:
        return "No nodes found for document"
    
    # Use character-based resolver by default
    if position_resolver is None:
        position_resolver = CharacterPositionResolver(all_nodes, store)
    
    # Get coordinate space extent
    extent = position_resolver.get_extent()
    if extent == 0:
        return "Empty coordinate space"
    
    # ... existing node organization code ...
    
    for depth in range(max_depth, -1, -1):
        if depth not in nodes_by_depth:
            continue
        nodes_to_show = nodes_by_depth[depth]
        
        # ... existing setup code ...
        
        for node in nodes_to_show:
            # Use resolver to get positions
            node_start, node_end = position_resolver.get_node_position(node)
            
            # Convert to pixel positions
            start_pos = int(node_start * actual_width / extent)
            end_pos = max(
                start_pos + 1,
                min(int(node_end * actual_width / extent), actual_width)
            )
            
            # ... rest of existing visualization logic ...
```

### Phase 5: Update CLI and API

#### 5.1 Update query command in cli.py

```python
@click.option(
    "--viz-coords",
    type=click.Choice(["chars", "tokens"]),
    default="chars",
    help="Coordinate system for tree visualization (chars=source position, tokens=output budget)",
)
def query(
    text: str,
    document_id: str,
    debug: bool,
    validate: bool,
    viz_width: int,
    viz_coords: str,
):
    # ... existing code ...
    
    if debug:
        # ... existing stats code ...
        
        # Create appropriate position resolver
        if viz_coords == "tokens":
            position_resolver = TokenPositionResolver(
                result.segment_infos,
                result.coverage_map,
                store
            )
        else:
            position_resolver = None  # Use default character-based
        
        # Build visualization
        tree_viz = build_ascii_tree(
            result.frontier_segments,
            store,
            document_id,
            width=viz_width,
            coverage_map=result.coverage_map,
            seed_node_ids=result.seed_node_ids,
            position_resolver=position_resolver,
        )
```

### Key Implementation Learnings

1. **Segment Ordering Solution**: Initially segments appeared out of order in token view. Fixed by sorting segments by their actual span_start positions before computing token positions. For RIGHT segments, we calculate the actual start position based on the left child's span_end.

2. **Coverage Painting Fix**: The initial implementation painted segments individually, which caused covered but unselected nodes to disappear in token view. Fixed by:
   - First painting all covered nodes with priority 0 (░)
   - Then painting selected segments on top with priority 1 (█)
   - Using a priority system to ensure selected segments always override covered nodes

3. **MID Delimiter Bug**: The _get_segment_cost method was including <<<MID>>> tokens for RIGHT segments, but the assembler strips them. Fixed by adding `text.replace("<<<MID>>>", "")` for RIGHT segments.

4. **Integration Approach**: Used the SegmentInfo approach (Option 1 from the integration plan) where DPResult includes segment_infos with pre-computed token costs. This avoided redundant tokenization and provided a clean data flow.

5. **Fallback Logic**: For covered but unselected nodes, when computing their token "cost" for positioning, we use their full text token count as a fallback. This ensures they still appear with appropriate width in the visualization.

### Testing Coverage

- Created comprehensive unit tests in `test_position_resolvers.py`
- Tests cover both CharacterPositionResolver and TokenPositionResolver
- Edge cases tested: empty nodes, single segments, complex trees
- Integration tested via manual validation with various queries