"""Validation functions for RagZoom to ensure correctness of indexing and retrieval."""

import logging
from typing import Callable, Optional

from ragzoom.store import Store, TreeNode

logger = logging.getLogger(__name__)


# Global flag to control validation
_validate_enabled = False


def set_validation_enabled(enabled: bool) -> None:
    """Set global validation flag."""
    global _validate_enabled
    _validate_enabled = enabled
    if enabled:
        logger.info("🔍 Validation enabled")


def validate(validation_fn: Callable[[], Optional[str]], context: str = "") -> None:
    """Run validation function if validation is enabled.

    Args:
        validation_fn: Function that returns None if valid, error message if invalid
        context: Optional context for error messages

    Raises:
        SystemExit: If validation fails (exits with code 1)
    """
    if not _validate_enabled:
        return

    error = validation_fn()
    if error:
        full_error = f"Validation failed{f' in {context}' if context else ''}: {error}"
        logger.error(f"❌ {full_error}")
        import os

        os._exit(1)  # Use os._exit to ensure immediate termination


def validate_document_coverage(
    original_text: str, leaf_nodes: list[TreeNode]
) -> Optional[str]:
    """Validate that leaf nodes cover the entire document.

    Args:
        original_text: The original document text
        leaf_nodes: List of leaf nodes (depth 0)

    Returns:
        Error message if invalid, None if valid
    """
    if not leaf_nodes:
        return "No leaf nodes found"

    # Sort by span_start
    sorted_leaves = sorted(leaf_nodes, key=lambda n: n.span_start)

    # Check first node starts at 0
    if sorted_leaves[0].span_start != 0:
        return f"First leaf node starts at {sorted_leaves[0].span_start}, expected 0"

    # Check last node ends at document length
    if sorted_leaves[-1].span_end != len(original_text):
        return (
            f"Last leaf node ends at {sorted_leaves[-1].span_end}, "
            f"expected {len(original_text)} (document length)"
        )

    # Check that chunks are contiguous (with our gap reconstruction, they should be)
    for i in range(len(sorted_leaves) - 1):
        current = sorted_leaves[i]
        next_node = sorted_leaves[i + 1]

        # Chunks should be exactly adjacent with no gaps
        if next_node.span_start != current.span_end:
            return (
                f"Non-contiguous chunks found: {current.id} ends at {current.span_end}, "
                f"{next_node.id} starts at {next_node.span_start}"
            )

    logger.info(
        f"✓ Document coverage validated: {len(sorted_leaves)} leaf nodes cover entire document"
    )
    return None


def validate_chunk_sizes(
    leaf_nodes: list[TreeNode], target_tokens: int, tolerance: float = 0.2
) -> Optional[str]:
    """Validate that chunk sizes are within tolerance of target.

    Args:
        leaf_nodes: List of leaf nodes
        target_tokens: Target size in tokens (RAGZOOM_LEAF_TOKENS)
        tolerance: Acceptable deviation (default 20%)

    Returns:
        Error message if invalid, None if valid
    """
    from tiktoken import get_encoding

    encoding = get_encoding("cl100k_base")

    min_allowed = int(target_tokens * (1 - tolerance))
    max_allowed = int(target_tokens * (1 + tolerance))

    oversized = []
    undersized = []

    for i, node in enumerate(leaf_nodes):
        tokens = len(encoding.encode(node.text))

        # Last chunk can be smaller
        if i == len(leaf_nodes) - 1 and tokens < min_allowed:
            continue

        if tokens > max_allowed:
            oversized.append((node.id, tokens))
        elif tokens < min_allowed:
            undersized.append((node.id, tokens))

    if oversized:
        logger.warning(
            f"Found {len(oversized)} oversized chunks (>{max_allowed} tokens)"
        )
        for node_id, tokens in oversized[:5]:  # Show first 5
            logger.warning(f"  {node_id}: {tokens} tokens")

    if undersized:
        logger.warning(
            f"Found {len(undersized)} undersized chunks (<{min_allowed} tokens)"
        )
        for node_id, tokens in undersized[:5]:  # Show first 5
            logger.warning(f"  {node_id}: {tokens} tokens")

    if not oversized and not undersized:
        logger.info(
            f"✓ Chunk sizes validated: all within ±{int(tolerance*100)}% of {target_tokens} tokens"
        )

    return None  # No errors


def validate_tree_structure(
    store: Store, document_id: str, original_text: Optional[str] = None
) -> Optional[str]:
    """Validate tree structure integrity.

    Args:
        store: Storage instance
        document_id: Document to validate

    Returns:
        Error message if invalid, None if valid
    """
    # Get all nodes for document
    with store.SessionLocal() as session:
        from ragzoom.store import TreeNode as TreeNodeModel

        nodes = session.query(TreeNodeModel).filter_by(document_id=document_id).all()

    if not nodes:
        return "No nodes found for document"

    errors = []

    # Check each node
    for node in nodes:
        # Validate span
        if node.span_start >= node.span_end:
            errors.append(
                f"Node {node.id}: Invalid span [{node.span_start}, {node.span_end})"
            )

        # Check parent-child relationships
        if node.left_child_id or node.right_child_id:
            # Parent node checks
            if store.is_leaf_node(node.id):
                errors.append(f"Node {node.id}: Leaf node has children")

            if node.left_child_id:
                left_child = store.get_node(node.left_child_id)
                if not left_child:
                    errors.append(
                        f"Node {node.id}: Left child {node.left_child_id} not found"
                    )
                elif left_child.span_start != node.span_start:
                    errors.append(
                        f"Node {node.id}: Span start {node.span_start} doesn't match "
                        f"left child start {left_child.span_start}"
                    )

            if node.right_child_id:
                right_child = store.get_node(node.right_child_id)
                if not right_child:
                    errors.append(
                        f"Node {node.id}: Right child {node.right_child_id} not found"
                    )
                elif right_child.span_end != node.span_end:
                    errors.append(
                        f"Node {node.id}: Span end {node.span_end} doesn't match "
                        f"right child end {right_child.span_end}"
                    )

            # Check for gaps between children
            if node.left_child_id and node.right_child_id:
                left_child = store.get_node(node.left_child_id)
                right_child = store.get_node(node.right_child_id)
                if left_child and right_child:
                    if left_child.span_end < right_child.span_start:
                        right_child.span_start - left_child.span_end
                        # If we have original text, check if gap is only whitespace
                        if original_text:
                            gap_text = original_text[
                                left_child.span_end : right_child.span_start
                            ]
                            if not gap_text.isspace():
                                errors.append(
                                    f"Node {node.id}: Non-whitespace gap between children - left ends at {left_child.span_end}, "
                                    f"right starts at {right_child.span_start}, gap content: {repr(gap_text)}"
                                )
                            # else: whitespace gap is allowed
                        else:
                            # Without original text, report all gaps
                            errors.append(
                                f"Node {node.id}: Gap between children - left ends at {left_child.span_end}, "
                                f"right starts at {right_child.span_start}"
                            )

        # Validate summaries for non-leaf nodes
        if not store.is_leaf_node(node.id):
            if not node.summary:
                errors.append(f"Node {node.id}: Non-leaf node missing summary")

            # For non-leaf nodes, a valid mid_offset is required.
            if node.mid_offset is None or node.mid_offset < 0:
                errors.append(
                    f"Node {node.id}: Invalid or missing mid_offset: {node.mid_offset}"
                )

    if errors:
        for error in errors[:10]:  # Show first 10 errors
            logger.error(error)
        return f"Tree structure validation failed with {len(errors)} errors"

    logger.info(f"✓ Tree structure validated: {len(nodes)} nodes")
    return None


def validate_frontier_completeness(
    frontier_segments: list[tuple[str, str, int, int]],
    document_span: tuple[int, int],
    original_text: Optional[str] = None,
) -> Optional[str]:
    """Validate that frontier provides complete coverage with no gaps.

    Args:
        frontier_segments: List of (node_id, text, span_start, span_end) tuples
        document_span: (start, end) of entire document
        original_text: Original document text to check if gaps are whitespace

    Returns:
        Error message if invalid, None if valid
    """
    if not frontier_segments:
        return "Frontier is empty"

    # First check if segments are properly ordered
    for i in range(len(frontier_segments) - 1):
        current = frontier_segments[i]
        next_seg = frontier_segments[i + 1]
        if current[2] > next_seg[2]:  # span_start of current > span_start of next
            return (
                f"Frontier segments out of order: segment {i} starts at {current[2]}, "
                f"segment {i+1} starts at {next_seg[2]}"
            )

    # Sort by span_start (in case caller didn't sort)
    sorted_segments = sorted(frontier_segments, key=lambda x: x[2])

    # Log segment details for debugging
    logger.debug(f"Validating frontier with {len(sorted_segments)} segments:")
    for i, (node_id, text, start, end) in enumerate(sorted_segments[:5]):
        logger.debug(f"  Segment {i}: [{start}, {end}) - {node_id}")
    if len(sorted_segments) > 5:
        logger.debug(f"  ... and {len(sorted_segments) - 5} more")

    # Check coverage starts at document start
    if sorted_segments[0][2] != document_span[0]:
        return (
            f"Frontier starts at {sorted_segments[0][2]}, expected {document_span[0]}"
        )

    # Check coverage ends at document end
    if sorted_segments[-1][3] != document_span[1]:
        logger.debug(
            f"Last segment: [{sorted_segments[-1][2]}, {sorted_segments[-1][3]})"
        )
        return f"Frontier ends at {sorted_segments[-1][3]}, expected {document_span[1]}"

    # Check for gaps
    for i in range(len(sorted_segments) - 1):
        current = sorted_segments[i]
        next_seg = sorted_segments[i + 1]

        if current[3] != next_seg[2]:
            gap = next_seg[2] - current[3]
            # Check if gap contains only whitespace
            if original_text and gap > 0:
                gap_text = original_text[current[3] : next_seg[2]]
                if gap_text.isspace():
                    logger.debug(
                        f"Allowing whitespace gap in frontier: [{current[3]}, {next_seg[2]}) "
                        f"contains {repr(gap_text)}"
                    )
                    continue

            # Show details about the gap
            gap_info = f"Gap in frontier: segment {i} ends at {current[3]}, segment {i+1} starts at {next_seg[2]}"
            if gap > 0:
                gap_info += f" (gap of {gap} chars)"
                if original_text:
                    gap_text = original_text[current[3] : next_seg[2]]
                    gap_info += f"\n  Gap content: {repr(gap_text[:50])}{'...' if len(gap_text) > 50 else ''}"
            else:
                gap_info += f" (overlap of {-gap} chars)"

            return gap_info

    logger.info(
        f"✓ Frontier completeness validated: {len(sorted_segments)} segments cover document"
    )
    return None


def validate_no_overlap(
    frontier_segments: list[tuple[str, str, int, int]],
) -> Optional[str]:
    """Validate that frontier segments don't overlap.

    Args:
        frontier_segments: List of (node_id, text, span_start, span_end) tuples

    Returns:
        Error message if invalid, None if valid
    """
    # Sort by span_start
    sorted_segments = sorted(frontier_segments, key=lambda x: x[2])

    for i in range(len(sorted_segments) - 1):
        current = sorted_segments[i]
        next_seg = sorted_segments[i + 1]

        if current[3] > next_seg[2]:
            return (
                f"Overlapping segments: {current[0]} [{current[2]}, {current[3]}) "
                f"overlaps with {next_seg[0]} [{next_seg[2]}, {next_seg[3]})"
            )

    logger.info("✓ No overlap validated: all frontier segments are non-overlapping")
    return None


def validate_extraction_rule(
    node: TreeNode, covered_set: set[str], store: Store, expected_contribution: str
) -> Optional[str]:
    """Validate that node follows the extraction rule based on child coverage.

    Args:
        node: Node to validate
        covered_set: Set of covered node IDs
        store: Storage instance
        expected_contribution: Expected contribution ('none', 'left', 'right', 'full')

    Returns:
        Error message if invalid, None if valid
    """
    left_covered = node.left_child_id in covered_set if node.left_child_id else False
    right_covered = node.right_child_id in covered_set if node.right_child_id else False

    # Determine what should be contributed based on rule
    if left_covered and right_covered:
        expected = "none"
    elif left_covered:
        expected = "right"
    elif right_covered:
        expected = "left"
    else:
        expected = "full"

    if expected != expected_contribution:
        return (
            f"Node {node.id}: Expected {expected} contribution based on child coverage, "
            f"but got {expected_contribution}"
        )

    return None


def validate_tiling(
    segments, store: Store, document_id: str, original_text: Optional[str] = None
) -> Optional[str]:
    """Validate that a tiling of Segments has no overlaps, no duplicates, and (optionally) covers the document."""
    if not segments:
        return "Tiling is empty"

    # Build list of (segment, span_start, span_end)
    seen_segments = set()
    segment_spans = []
    for seg in segments:
        key = (seg.node_id, seg.side)
        if key in seen_segments:
            return f"Duplicate segment: node {seg.node_id} side {seg.side}"
        seen_segments.add(key)
        node = store.get_node(seg.node_id)
        if not node:
            return f"Node {seg.node_id} not found in store"

        # Validate side invariant
        is_leaf = store.is_leaf_node(node.id)
        if is_leaf or node.mid_offset is None:
            if seg.side is not None:
                return f"Node {seg.node_id} is a leaf (is_leaf={is_leaf}, mid_offset={node.mid_offset}) but segment has side={seg.side}, expected None"
            # Leaf or unsplit node: full span
            span_start, span_end = node.span_start, node.span_end
        else:
            if seg.side not in {"LEFT", "RIGHT"}:
                return f"Node {seg.node_id} is internal (is_leaf={is_leaf}, mid_offset={node.mid_offset}) but segment has side={seg.side}, expected LEFT or RIGHT"
            # For internal nodes, segment spans match child spans
            if seg.side == "LEFT":
                left_child = store.get_node(node.left_child_id)
                if not left_child:
                    return f"Node {seg.node_id} has no left child"
                span_start, span_end = left_child.span_start, left_child.span_end
            else:  # RIGHT
                right_child = store.get_node(node.right_child_id)
                if not right_child:
                    return f"Node {seg.node_id} has no right child"
                span_start, span_end = right_child.span_start, right_child.span_end
        segment_spans.append((seg, span_start, span_end))

    # Sort by span_start
    segment_spans.sort(key=lambda x: x[1])

    # Check for overlaps
    for i in range(len(segment_spans) - 1):
        _, start1, end1 = segment_spans[i]
        _, start2, end2 = segment_spans[i + 1]
        if end1 > start2:
            return f"Overlapping segments: {segment_spans[i][0]} [{start1},{end1}) overlaps with {segment_spans[i+1][0]} [{start2},{end2})"

    # Optionally, check for complete coverage
    doc_nodes = store.get_all_nodes_for_document(document_id)
    if doc_nodes:
        doc_start = min(n.span_start for n in doc_nodes)
        doc_end = max(n.span_end for n in doc_nodes)
        if segment_spans[0][1] != doc_start:
            return f"Tiling does not start at document start: {segment_spans[0][1]} != {doc_start}"
        if segment_spans[-1][2] != doc_end:
            return f"Tiling does not end at document end: {segment_spans[-1][2]} != {doc_end}"
        # Check for gaps
        for i in range(len(segment_spans) - 1):
            if segment_spans[i][2] != segment_spans[i + 1][1]:
                gap = segment_spans[i + 1][1] - segment_spans[i][2]
                if gap > 0:
                    if original_text:
                        gap_text = original_text[
                            segment_spans[i][2] : segment_spans[i + 1][1]
                        ]
                        if not gap_text.isspace():
                            return f"Non-whitespace gap in tiling: {segment_spans[i][2]} to {segment_spans[i + 1][1]}"
                    else:
                        return f"Gap in tiling: {segment_spans[i][2]} to {segment_spans[i + 1][1]}"

    return None  # Valid tiling
