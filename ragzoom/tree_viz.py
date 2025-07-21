"""ASCII tree visualization for tiling display."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import tiktoken

from ragzoom.dynamic_tiling import Segment, SegmentInfo
from ragzoom.store import Store, TreeNode


class PositionResolver(ABC):
    """Abstract base class for coordinate system resolvers."""

    @abstractmethod
    def get_extent(self) -> float:
        """Return the total extent of the coordinate space."""
        pass

    @abstractmethod
    def get_segment_position(
        self, segment: Segment, segment_index: int
    ) -> tuple[float, float]:
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

    def get_segment_position(
        self, segment: Segment, segment_index: int
    ) -> tuple[float, float]:
        node = self.store.get_node(segment.node_id)
        if not node:
            return (0.0, 0.0)
        return (
            float(node.span_start - self.doc_start),
            float(node.span_end - self.doc_start),
        )

    def get_node_position(self, node: TreeNode) -> tuple[float, float]:
        return (
            float(node.span_start - self.doc_start),
            float(node.span_end - self.doc_start),
        )


class TokenPositionResolver(PositionResolver):
    """Token-based positioning showing output budget allocation."""

    def __init__(
        self,
        segment_infos: list[SegmentInfo],
        coverage_map: dict[str, bool],
        store: Store,
        tokenizer: Any = None,
    ):
        # Validate inputs
        if not segment_infos:
            raise ValueError("segment_infos cannot be empty")
        if not coverage_map:
            raise ValueError("coverage_map cannot be empty")

        self.store = store
        self.segment_infos = segment_infos
        self.coverage_map = coverage_map
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")

        # Build segment lookup for quick access
        self.segment_lookup = {
            (info.segment.node_id, info.segment.side): idx
            for idx, info in enumerate(segment_infos)
        }

        # Validate token costs are not null
        for info in segment_infos:
            if info.token_cost is None:
                raise ValueError(f"Null token cost for segment {info.segment}")

        # Sort segments by their document order (left-to-right)
        # This ensures the token visualization matches document flow
        sorted_infos = []
        for idx, info in enumerate(segment_infos):
            # Use the pre-computed span_start from SegmentInfo
            sorted_infos.append((info.span_start, idx, info))
        sorted_infos.sort(key=lambda x: x[0])

        # Compute segment positions in document order
        self.segment_positions = {}
        current_pos = 0.0
        for _, original_idx, info in sorted_infos:
            self.segment_positions[original_idx] = (
                current_pos,
                current_pos + info.token_cost,
            )
            current_pos += info.token_cost
        self.total_tokens = current_pos

        # Compute positions for covered but unselected nodes
        self.node_positions: dict[str, tuple[float, float]] = {}
        self._compute_node_positions()

    def get_extent(self) -> float:
        return self.total_tokens

    def get_segment_position(
        self, segment: Segment, segment_index: int
    ) -> tuple[float, float]:
        return self.segment_positions.get(segment_index, (0.0, 0.0))

    def get_node_position(self, node: TreeNode) -> tuple[float, float]:
        return self.node_positions.get(node.id, (0.0, 0.0))

    def _compute_node_positions(self) -> None:
        """Compute positions for all covered nodes based on selected descendants."""
        # First pass: compute token costs for all nodes
        node_costs: dict[str, float] = {}

        def compute_cost(node_id: str) -> float:
            if node_id in node_costs:
                return node_costs[node_id]

            node = self.store.get_node(node_id)
            if not node:
                # Handle missing node gracefully
                node_costs[node_id] = 0.0
                return 0.0

            # Check if this node has selected segments
            total_cost = 0.0

            # Check for leaf segment
            if (node_id, None) in self.segment_lookup:
                idx = self.segment_lookup[(node_id, None)]
                total_cost = self.segment_infos[idx].token_cost
            else:
                # Check for left/right segments
                left_cost = 0.0
                right_cost = 0.0

                if (node_id, "LEFT") in self.segment_lookup:
                    idx = self.segment_lookup[(node_id, "LEFT")]
                    left_cost = self.segment_infos[idx].token_cost
                elif node.left_child_id:
                    left_cost = compute_cost(node.left_child_id)
                    # If child is covered but has zero cost, use its full text cost
                    if left_cost == 0.0 and node.left_child_id in self.coverage_map:
                        left_child = self.store.get_node(node.left_child_id)
                        if left_child and left_child.text:
                            left_cost = float(
                                len(self.tokenizer.encode(left_child.text))
                            )

                if (node_id, "RIGHT") in self.segment_lookup:
                    idx = self.segment_lookup[(node_id, "RIGHT")]
                    right_cost = self.segment_infos[idx].token_cost
                elif node.right_child_id:
                    right_cost = compute_cost(node.right_child_id)
                    # If child is covered but has zero cost, use its full text cost
                    if right_cost == 0.0 and node.right_child_id in self.coverage_map:
                        right_child = self.store.get_node(node.right_child_id)
                        if right_child and right_child.text:
                            right_cost = float(
                                len(self.tokenizer.encode(right_child.text))
                            )

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
            start_pos = float("inf")
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
            if start_pos == float("inf"):
                start_pos = 0.0

            self.node_positions[node_id] = (start_pos, end_pos)
            return (start_pos, end_pos)

        # Compute positions for all covered nodes
        for node_id in self.coverage_map:
            compute_position(node_id)


def build_ascii_tree(
    segments: list[Segment],
    store: Store,
    document_id: str,
    width: int = 120,
    coverage_map: Optional[dict[str, bool]] = None,
    seed_node_ids: Optional[set[str]] = None,
    position_resolver: Optional[PositionResolver] = None,
    segment_infos: Optional[list[SegmentInfo]] = None,
    use_token_coords: bool = False,
    preloaded_nodes: Optional[dict[str, "TreeNode"]] = None,
) -> str:
    """Build an ASCII tree visualization showing the tiling structure.

    Args:
        segments: List of segments to visualize
        store: Store instance
        document_id: Document to visualize
        width: Terminal width for visualization
        coverage_map: Optional dict of covered node IDs
        seed_node_ids: Optional set of seed node IDs (marked with *)
        position_resolver: Optional position resolver (deprecated, use use_token_coords)
        segment_infos: Segment metadata including token costs (required if use_token_coords=True)
        use_token_coords: If True, use token-based positioning; if False, use character-based
    """
    # Use pre-loaded nodes if available
    if preloaded_nodes:
        all_nodes = [
            node for node in preloaded_nodes.values() if node.document_id == document_id
        ]
        if not all_nodes:
            return "No nodes found in preloaded nodes"
    elif coverage_map:
        # Load only nodes that are in the coverage map
        all_nodes = []
        for node_id in coverage_map:
            node = store.get_node(node_id)
            if node and node.document_id == document_id:
                all_nodes.append(node)
        if not all_nodes:
            return "No nodes found in coverage map"
    else:
        # Fallback to all nodes only if no coverage map provided
        all_nodes = store.get_all_nodes_for_document(document_id)
        if not all_nodes:
            return "No nodes found for document"

    # Handle backward compatibility with position_resolver parameter
    if position_resolver is not None:
        # Use the provided resolver
        pass
    elif use_token_coords:
        # Create token-based resolver
        if not segment_infos:
            raise ValueError("segment_infos required when use_token_coords=True")
        tokenizer = tiktoken.get_encoding("cl100k_base")
        position_resolver = TokenPositionResolver(
            segment_infos, coverage_map or {}, store, tokenizer
        )
    else:
        # Default to character-based resolver
        position_resolver = CharacterPositionResolver(all_nodes, store)

    # Get coordinate space extent
    extent = position_resolver.get_extent()
    if extent == 0:
        return "Empty coordinate space"

    # Group nodes by height (distance to furthest leaf)
    nodes_by_height: dict[int, list[TreeNode]] = {}
    max_height = 0
    for node in all_nodes:
        height = store.get_node_height(node.id)
        if height not in nodes_by_height:
            nodes_by_height[height] = []
        nodes_by_height[height].append(node)
        max_height = max(max_height, height)
    for height in nodes_by_height:
        nodes_by_height[height].sort(key=lambda n: n.span_start)

    selected_segments: set[tuple[str, Optional[str]]] = set()
    segment_labels: dict[tuple[str, Optional[str]], str] = {}
    for idx, seg in enumerate(segments):
        key = (seg.node_id, seg.side)
        selected_segments.add(key)
        # Add asterisk to label if this is a seed node
        label = str(idx)
        if seed_node_ids and seg.node_id in seed_node_ids:
            label += "*"
        segment_labels[key] = label

    lines = []

    # Iterate from root (max height) down to leaves (height 0)
    for height in range(max_height, -1, -1):
        if height not in nodes_by_height:
            continue
        nodes_to_show = nodes_by_height[height]

        level_prefix = f"H{height} "
        prefix_len = len(level_prefix)
        actual_width = width - prefix_len
        # Priority array: -1 = blank, 0 = covered, 1 = selected
        pixels: list[int] = [-1] * actual_width
        label_spans = []  # (start, end, label, is_selected)

        def paint(lo: int, hi: int, priority: int) -> None:
            lo = max(0, min(lo, actual_width - 1))
            hi = max(lo + 1, min(hi, actual_width))
            for i in range(lo, hi):
                if priority > pixels[i]:
                    pixels[i] = priority

        for node in nodes_to_show:
            # Use resolver to get positions
            node_start, node_end = position_resolver.get_node_position(node)

            # Convert to pixel positions
            start_pos = int(node_start * actual_width / extent)
            end_pos = max(
                start_pos + 1, min(int(node_end * actual_width / extent), actual_width)
            )
            if end_pos <= start_pos:
                end_pos = start_pos + 1
            is_covered = coverage_map and node.id in coverage_map
            # Leaf node
            if store.is_leaf_node(node.id):
                char_priority = (
                    1
                    if (node.id, None) in selected_segments
                    else 0 if is_covered else -1
                )
                if char_priority >= 0:
                    paint(start_pos, end_pos, char_priority)
                # Label for selected leaf
                if (node.id, None) in selected_segments:
                    label = segment_labels.get((node.id, None), "")
                    if label:
                        mid_pos = (start_pos + end_pos) // 2
                        label_spans.append((mid_pos, label, True))
            # Internal node
            elif node.left_child_id and node.right_child_id:
                # For token coordinates, we need to handle segments differently
                if isinstance(position_resolver, TokenPositionResolver):
                    # First, if the node is covered, paint the entire node as covered
                    if is_covered:
                        paint(start_pos, end_pos, 0)

                    # Then paint selected segments on top
                    has_left = (node.id, "LEFT") in selected_segments
                    has_right = (node.id, "RIGHT") in selected_segments

                    if has_left or has_right:
                        # Find the actual segment positions
                        for idx, seg in enumerate(segments):
                            if seg.node_id == node.id:
                                seg_start, seg_end = (
                                    position_resolver.get_segment_position(seg, idx)
                                )
                                seg_start_pos = int(seg_start * actual_width / extent)
                                seg_end_pos = max(
                                    seg_start_pos + 1,
                                    min(
                                        int(seg_end * actual_width / extent),
                                        actual_width,
                                    ),
                                )

                                if seg.side == "LEFT" and has_left:
                                    paint(seg_start_pos, seg_end_pos, 1)
                                    label = segment_labels.get((node.id, "LEFT"), "")
                                    if label:
                                        label_pos = (seg_start_pos + seg_end_pos) // 2
                                        label_spans.append((label_pos, label, True))
                                elif seg.side == "RIGHT" and has_right:
                                    paint(seg_start_pos, seg_end_pos, 1)
                                    label = segment_labels.get((node.id, "RIGHT"), "")
                                    if label:
                                        label_pos = (seg_start_pos + seg_end_pos) // 2
                                        label_spans.append((label_pos, label, True))
                else:
                    # Character-based positioning - use the original logic
                    left_child = store.get_node(node.left_child_id)
                    right_child = store.get_node(node.right_child_id)
                    if left_child and right_child:
                        # Get left child's end position using resolver
                        left_start, left_end = position_resolver.get_node_position(
                            left_child
                        )
                        mid_pos = int(left_end * actual_width / extent)
                        mid_pos = max(0, min(mid_pos, actual_width - 1))
                        # Left segment
                        left_priority = (
                            1
                            if (node.id, "LEFT") in selected_segments
                            else 0 if is_covered else -1
                        )
                        if left_priority >= 0:
                            paint(start_pos, mid_pos, left_priority)
                        if (node.id, "LEFT") in selected_segments:
                            label = segment_labels.get((node.id, "LEFT"), "")
                            if label:
                                label_pos = (start_pos + mid_pos) // 2
                                label_spans.append((label_pos, label, True))
                        # Right segment
                        right_priority = (
                            1
                            if (node.id, "RIGHT") in selected_segments
                            else 0 if is_covered else -1
                        )
                        if right_priority >= 0:
                            paint(mid_pos, end_pos, right_priority)
                        if (node.id, "RIGHT") in selected_segments:
                            label = segment_labels.get((node.id, "RIGHT"), "")
                            if label:
                                label_pos = (mid_pos + end_pos) // 2
                                label_spans.append((label_pos, label, True))
        # Convert pixels to characters
        line = ["█" if p == 1 else "░" if p == 0 else " " for p in pixels]
        lines.append(level_prefix + "".join(line))
        label_line = [" "] * actual_width
        if label_spans:
            label_spans.sort(key=lambda x: x[0])
            last_end = -1
            for pos, label, is_selected in label_spans:
                start = max(0, pos - len(label) // 2)
                if start <= last_end:
                    start = last_end + 2
                if start + len(label) > actual_width:
                    if start < actual_width:
                        truncated_label = label[: max(0, actual_width - start)]
                        for i, char in enumerate(truncated_label):
                            if start + i < len(label_line):
                                label_line[start + i] = char
                        last_end = start + len(truncated_label) - 1
                else:
                    for i, char in enumerate(label):
                        if start + i < len(label_line):
                            label_line[start + i] = char
                    last_end = start + len(label) - 1
        lines.append(" " * prefix_len + "".join(label_line))
    return "\n".join(lines)
