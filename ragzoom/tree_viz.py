"""ASCII tree visualization for tiling display."""

from typing import Optional

from ragzoom.dynamic_frontier import Segment
from ragzoom.store import Store, TreeNode


def build_ascii_tree(
    segments: list[Segment],
    store: Store,
    document_id: str,
    width: int = 120,
    coverage_map: Optional[dict[str, bool]] = None,
    seed_node_ids: Optional[set[str]] = None,
) -> str:
    """Build an ASCII tree visualization showing the tiling structure (no node boundary markers)."""
    all_nodes = store.get_all_nodes_for_document(document_id)
    if not all_nodes:
        return "No nodes found for document"

    doc_start = min(node.span_start for node in all_nodes)
    doc_end = max(node.span_end for node in all_nodes)
    doc_span = doc_end - doc_start

    nodes_by_depth: dict[int, list[TreeNode]] = {}
    max_depth = 0
    for node in all_nodes:
        if node.depth not in nodes_by_depth:
            nodes_by_depth[node.depth] = []
        nodes_by_depth[node.depth].append(node)
        max_depth = max(max_depth, node.depth)
    for depth in nodes_by_depth:
        nodes_by_depth[depth].sort(key=lambda n: n.span_start)

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

    # Add a top border spanning the full width
    lines.append("─" * width)

    for depth in range(max_depth, -1, -1):
        if depth not in nodes_by_depth:
            continue
        nodes_to_show = []
        if len(nodes_by_depth[depth]) > 30:
            for node in nodes_by_depth[depth]:
                if (
                    (node.id, "LEFT") in selected_segments
                    or (node.id, "RIGHT") in selected_segments
                    or (node.id, None) in selected_segments
                ):
                    nodes_to_show.append(node)
            if not nodes_to_show:
                continue
        else:
            nodes_to_show = nodes_by_depth[depth]

        level_prefix = f"L{depth} "
        prefix_len = len(level_prefix)
        actual_width = width - prefix_len
        # Priority array: -1 = blank, 0 = covered, 1 = selected
        pixels: list[int] = [-1] * actual_width
        label_spans = []  # (start, end, label, is_selected)

        def paint(lo: int, hi: int, priority: int):
            lo = max(0, min(lo, actual_width - 1))
            hi = max(lo + 1, min(hi, actual_width))
            for i in range(lo, hi):
                if priority > pixels[i]:
                    pixels[i] = priority

        for node in nodes_to_show:
            start_pos = int((node.span_start - doc_start) * actual_width / doc_span)
            end_pos = max(
                start_pos + 1,
                min(
                    int((node.span_end - doc_start) * actual_width / doc_span),
                    actual_width,
                ),
            )
            if end_pos <= start_pos:
                end_pos = start_pos + 1
            is_covered = coverage_map and node.id in coverage_map
            # Leaf node
            if node.depth == 0:
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
                left_child = store.get_node(node.left_child_id)
                right_child = store.get_node(node.right_child_id)
                if left_child and right_child:
                    mid_pos = int(
                        (left_child.span_end - doc_start) * actual_width / doc_span
                    )
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
        # Add horizontal border after each level except the last, spanning the full width
        if depth > 0:
            lines.append("─" * width)
    # Add a bottom border spanning the full width
    lines.append("─" * width)
    return "\n".join(lines)
