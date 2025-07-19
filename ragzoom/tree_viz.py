"""ASCII tree visualization for tiling display."""

from typing import Optional

from ragzoom.dynamic_frontier import Segment
from ragzoom.store import Store, TreeNode


def build_ascii_tree(
    segments: list[Segment], store: Store, document_id: str, width: int = 120
) -> str:
    """Build an ASCII tree visualization showing the tiling structure.

    Args:
        segments: List of segments in the tiling
        store: Store instance to get node information
        document_id: Document ID to visualize
        width: Width of the ASCII art in characters

    Returns:
        ASCII art string showing the tree with highlighted segments
    """
    # Get all nodes for the document
    all_nodes = store.get_all_nodes_for_document(document_id)
    if not all_nodes:
        return "No nodes found for document"

    # Find document span
    doc_start = min(node.span_start for node in all_nodes)
    doc_end = max(node.span_end for node in all_nodes)
    doc_span = doc_end - doc_start

    # Group nodes by depth
    nodes_by_depth: dict[int, list[TreeNode]] = {}
    max_depth = 0
    for node in all_nodes:
        if node.depth not in nodes_by_depth:
            nodes_by_depth[node.depth] = []
        nodes_by_depth[node.depth].append(node)
        max_depth = max(max_depth, node.depth)

    # Sort nodes at each depth by span_start
    for depth in nodes_by_depth:
        nodes_by_depth[depth].sort(key=lambda n: n.span_start)

    # Create a set of (node_id, side) tuples for selected segments
    selected_segments: set[tuple[str, Optional[str]]] = set()
    segment_labels: dict[tuple[str, Optional[str]], str] = {}

    for seg in segments:
        key = (seg.node_id, seg.side)
        selected_segments.add(key)
        # Create short label
        if seg.side:
            segment_labels[key] = f"{seg.node_id[:8]}-{seg.side[0]}"
        else:
            segment_labels[key] = f"{seg.node_id[:8]}"

    # Build the ASCII art
    lines = []

    # Header
    lines.append(f"Document span: {doc_start}-{doc_end} ({doc_span} chars)")
    lines.append("")

    # Process each depth level from highest (root) to lowest (leaves)
    for depth in range(max_depth, -1, -1):
        if depth not in nodes_by_depth:
            continue

        # For levels with many nodes, only show selected ones
        nodes_to_show = []
        if len(nodes_by_depth[depth]) > 30:
            # Only show nodes that have selected segments
            for node in nodes_by_depth[depth]:
                if (
                    (node.id, "LEFT") in selected_segments
                    or (node.id, "RIGHT") in selected_segments
                    or (node.id, None) in selected_segments
                ):
                    nodes_to_show.append(node)
            # If no selected nodes at this level, skip it
            if not nodes_to_show:
                continue
        else:
            nodes_to_show = nodes_by_depth[depth]

        # Calculate the actual drawing width accounting for level prefix
        level_prefix = f"Level {depth}: "
        prefix_len = len(level_prefix)
        actual_width = width - prefix_len

        # Create the line for this depth
        line = [" "] * actual_width
        labels_to_place = []  # Store labels with their positions

        for node in nodes_to_show:
            # Calculate positions - ensure they stay within bounds
            start_pos = int((node.span_start - doc_start) * actual_width / doc_span)
            end_pos = int((node.span_end - doc_start) * actual_width / doc_span)

            # Clamp positions to valid range
            start_pos = max(0, min(start_pos, actual_width - 1))
            end_pos = max(start_pos + 1, min(end_pos, actual_width))

            # Ensure minimum width
            if end_pos <= start_pos:
                end_pos = start_pos + 1

            # Determine if this node has selected segments
            has_left_selected = (node.id, "LEFT") in selected_segments
            has_right_selected = (node.id, "RIGHT") in selected_segments
            has_none_selected = (node.id, None) in selected_segments  # For leaf nodes

            # For leaf nodes
            if node.depth == 0:
                if has_none_selected:
                    # Fill with '█' for selected leaf nodes
                    for i in range(start_pos, min(end_pos, len(line))):
                        if i < len(line):
                            line[i] = "█"
                    # Add label
                    label = segment_labels.get((node.id, None), "")
                    if label:
                        mid_pos = (start_pos + end_pos) // 2
                        labels_to_place.append(
                            (mid_pos, label, True)
                        )  # True = selected
                else:
                    # Draw borders for unselected leaf nodes
                    if 0 <= start_pos < len(line):
                        line[start_pos] = "│"
                    if 0 <= end_pos - 1 < len(line):
                        line[end_pos - 1] = "│"
                    for i in range(max(0, start_pos + 1), min(end_pos - 1, len(line))):
                        if 0 <= i < len(line) and line[i] == " ":
                            line[i] = "─"
            else:
                # Internal node - handle left and right segments separately
                if node.left_child_id and node.right_child_id:
                    # Get child nodes to determine segment boundaries
                    left_child = store.get_node(node.left_child_id)
                    right_child = store.get_node(node.right_child_id)

                    if left_child and right_child:
                        # Calculate mid position based on child boundaries
                        mid_pos = int(
                            (left_child.span_end - doc_start) * actual_width / doc_span
                        )
                        mid_pos = max(0, min(mid_pos, actual_width - 1))

                        # Draw left segment
                        if has_left_selected:
                            for i in range(start_pos, min(mid_pos, len(line))):
                                if 0 <= i < len(line):
                                    line[i] = "█"
                            # Add label
                            label = segment_labels.get((node.id, "LEFT"), "")
                            if label:
                                label_pos = (start_pos + mid_pos) // 2
                                labels_to_place.append((label_pos, label, True))
                        else:
                            if 0 <= start_pos < len(line):
                                line[start_pos] = "│"
                            for i in range(
                                max(0, start_pos + 1), min(mid_pos, len(line))
                            ):
                                if 0 <= i < len(line) and line[i] == " ":
                                    line[i] = "─"

                        # Draw right segment
                        if has_right_selected:
                            for i in range(mid_pos, min(end_pos, len(line))):
                                if 0 <= i < len(line):
                                    line[i] = "█"
                            # Add label
                            label = segment_labels.get((node.id, "RIGHT"), "")
                            if label:
                                label_pos = (mid_pos + end_pos) // 2
                                labels_to_place.append((label_pos, label, True))
                        else:
                            if 0 <= mid_pos < len(line):
                                line[mid_pos] = "│"
                            if 0 <= end_pos - 1 < len(line):
                                line[end_pos - 1] = "│"
                            for i in range(
                                max(0, mid_pos + 1), min(end_pos - 1, len(line))
                            ):
                                if 0 <= i < len(line) and line[i] == " ":
                                    line[i] = "─"
                else:
                    # Shouldn't happen with properly formed tree
                    if 0 <= start_pos < len(line):
                        line[start_pos] = "│"
                    if 0 <= end_pos - 1 < len(line):
                        line[end_pos - 1] = "│"
                    for i in range(max(0, start_pos + 1), min(end_pos - 1, len(line))):
                        if 0 <= i < len(line) and line[i] == " ":
                            line[i] = "─"

        # Add the main line with level prefix
        level_prefix = f"Level {depth}: "
        lines.append(f"{level_prefix}{''.join(line)}")

        # Add labels on a separate line if there are any
        if labels_to_place:
            label_line = [" "] * actual_width
            # Sort labels by position to handle overlaps
            labels_to_place.sort(key=lambda x: x[0])

            # Place labels, avoiding overlaps
            last_end = -1
            for pos, label, is_selected in labels_to_place:
                # Center the label around the position
                start = max(0, pos - len(label) // 2)
                # Avoid overlap with previous label
                if start <= last_end:
                    start = last_end + 2
                # Make sure it fits within width
                if start + len(label) > actual_width:
                    # If label doesn't fit, truncate or skip
                    if start < actual_width:
                        # Truncate label to fit
                        truncated_label = label[: min(len(label), actual_width - start)]
                        for i, char in enumerate(truncated_label):
                            if start + i < len(label_line):
                                label_line[start + i] = char
                        last_end = start + len(truncated_label) - 1
                else:
                    # Label fits completely
                    for i, char in enumerate(label):
                        if start + i < len(label_line):
                            label_line[start + i] = char
                    last_end = start + len(label) - 1

            # Add label line with spacing to align with level content
            lines.append(f"{' ' * prefix_len}{''.join(label_line)}")
            lines.append("")  # Empty line for spacing

    return "\n".join(lines)
