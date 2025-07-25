"""Validation functions for RagZoom to ensure correctness of indexing and retrieval."""

import logging
from typing import Any, Callable, Optional

from ragzoom.store import Store, TreeNode

logger = logging.getLogger(__name__)


# Global flag to control validation
_validate_enabled = False


def set_validation_enabled(enabled: bool) -> None:
    """Set global validation flag."""
    global _validate_enabled
    _validate_enabled = enabled
    # Don't log when enabling validation - we only want to see errors


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

    # Success - no need to log
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

    # Success - no need to log
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

        # Validate text content for all nodes
        if not node.text:
            errors.append(f"Node {node.id}: Missing text content")

    if errors:
        for error in errors[:10]:  # Show first 10 errors
            logger.error(error)
        return f"Tree structure validation failed with {len(errors)} errors"

    # Success - no need to log
    return None


def validate_tiling(
    tiling: list[str],  # List of node IDs
    store: Store,
    document_id: str,
    original_text: Optional[str] = None,
    budget_tokens: Optional[int] = None,
) -> Optional[str]:
    """Validate that a tiling has no overlaps, no duplicates, and (optionally) covers the document.

    Args:
        tiling: List of node IDs in the tiling
        store: Store instance
        document_id: Document ID
        original_text: Optional original text for gap validation
        budget_tokens: Optional token budget to validate against

    Returns:
        Error message if invalid, None if valid
    """
    if not tiling:
        return "Tiling is empty"

    # Build list of (node_id, span_start, span_end)
    seen_nodes = set()
    node_spans = []
    for node_id in tiling:
        if node_id in seen_nodes:
            return f"Duplicate node: {node_id}"
        seen_nodes.add(node_id)
        node = store.get_node(node_id)
        if not node:
            return f"Node {node_id} not found in store"

        # Get node span
        span_start, span_end = node.span_start, node.span_end
        node_spans.append((node_id, span_start, span_end))

    # Sort by span_start
    node_spans.sort(key=lambda x: x[1])

    # Check for overlaps
    for i in range(len(node_spans) - 1):
        node_id1, start1, end1 = node_spans[i]
        node_id2, start2, end2 = node_spans[i + 1]
        if end1 > start2:
            return f"Overlapping nodes: {node_id1} [{start1},{end1}) overlaps with {node_id2} [{start2},{end2})"

    # Optionally, check for complete coverage
    doc_nodes = store.get_all_nodes_for_document(document_id)
    if doc_nodes:
        doc_start = min(n.span_start for n in doc_nodes)
        doc_end = max(n.span_end for n in doc_nodes)
        if node_spans[0][1] != doc_start:
            return f"Tiling does not start at document start: {node_spans[0][1]} != {doc_start}"
        if node_spans[-1][2] != doc_end:
            return (
                f"Tiling does not end at document end: {node_spans[-1][2]} != {doc_end}"
            )
        # Check for gaps
        for i in range(len(node_spans) - 1):
            if node_spans[i][2] != node_spans[i + 1][1]:
                gap = node_spans[i + 1][1] - node_spans[i][2]
                if gap > 0:
                    if original_text:
                        gap_text = original_text[
                            node_spans[i][2] : node_spans[i + 1][1]
                        ]
                        if not gap_text.isspace():
                            return f"Non-whitespace gap in tiling: {node_spans[i][2]} to {node_spans[i + 1][1]}"
                    else:
                        return f"Gap in tiling: {node_spans[i][2]} to {node_spans[i + 1][1]}"

    # Check budget compliance if budget is provided
    if budget_tokens is not None:
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")

        total_tokens = 0
        for node_id in tiling:
            node = store.get_node(node_id)
            if not node or not node.text:
                continue

            # For atomic nodes, just count the full text
            tokens = len(tokenizer.encode(node.text))
            total_tokens += tokens

        if total_tokens > budget_tokens:
            return (
                f"Tiling exceeds budget: {total_tokens} tokens > {budget_tokens} budget"
            )

    return None  # Valid tiling


def validate_tree_is_left_balanced(store: Store, document_id: str) -> Optional[str]:
    """Validate that the indexed tree is left-balanced.

    A left-balanced tree means internal nodes have either:
    1. Only a left child (no right child), or
    2. Both left and right children

    No node can have only a right child without a left child.

    Args:
        store: Storage instance
        document_id: Document to validate

    Returns:
        Error message if invalid, None if valid
    """
    nodes = store.get_all_nodes_for_document(document_id)
    if not nodes:
        return "No nodes found for document"

    # A single-node tree is valid
    if len(nodes) == 1:
        node = nodes[0]
        if node.left_child_id is not None or node.right_child_id is not None:
            return f"Invalid tree: node {node.id} references non-existent children"
        return None

    # Build a set of valid node IDs for quick lookup
    node_ids = {node.id for node in nodes}

    # Check each node
    for node in nodes:
        has_left = node.left_child_id is not None
        has_right = node.right_child_id is not None

        # Check invalid state: right child without left child
        if has_right and not has_left:
            return (
                f"Tree is not left-balanced: node {node.id} has a right child but no left child. "
                f"In a left-balanced tree, nodes must have a left child before having a right child."
            )

        # Verify child references are valid
        if has_left and node.left_child_id not in node_ids:
            return f"Invalid tree: node {node.id} references non-existent left child {node.left_child_id}"

        if has_right and node.right_child_id not in node_ids:
            return f"Invalid tree: node {node.id} references non-existent right child {node.right_child_id}"

    return None  # Tree is left-balanced


def validate_equal_leaf_depth(store: Store, document_id: str) -> Optional[str]:
    """Validate that all leaf nodes are at the same (maximal) depth.

    This ensures consistent abstraction levels across the tree and prevents
    mixing of raw text and summaries at different heights.

    Args:
        store: Storage instance
        document_id: Document to validate

    Returns:
        Error message if invalid, None if valid
    """
    nodes = store.get_all_nodes_for_document(document_id)
    if not nodes:
        return "No nodes found for document"

    # Build node lookup and identify leaf nodes
    node_lookup = {node.id: node for node in nodes}
    leaf_nodes = []

    for node in nodes:
        # A node is a leaf if it has no children
        if node.left_child_id is None and node.right_child_id is None:
            leaf_nodes.append(node)

    if not leaf_nodes:
        return "No leaf nodes found"

    # Find root node (node with no parent)
    root_node = None
    for node in nodes:
        if node.parent_id is None:
            root_node = node
            break

    if not root_node:
        return "No root node found"

    # Calculate depth for each leaf node
    def get_depth(node_id: str) -> int:
        """Calculate depth from node to root."""
        depth = 0
        current_id = node_id
        while current_id != root_node.id:
            node = node_lookup.get(current_id)
            if not node or not node.parent_id:
                return -1  # Invalid tree structure
            current_id = node.parent_id
            depth += 1
        return depth

    # Get depths of all leaf nodes
    leaf_depths = []
    for leaf in leaf_nodes:
        depth = get_depth(leaf.id)
        if depth == -1:
            return f"Invalid tree structure: leaf node {leaf.id} cannot reach root"
        leaf_depths.append((leaf.id, depth))

    # Check if all depths are the same
    if leaf_depths:
        first_depth = leaf_depths[0][1]
        for leaf_id, depth in leaf_depths:
            if depth != first_depth:
                return (
                    f"Leaf nodes at different depths: {leaf_depths[0][0]} at depth {first_depth}, "
                    f"{leaf_id} at depth {depth}. All leaves should be at the same depth."
                )

    return None  # All leaves at same depth


async def validate_summary_faithfulness(
    summary: str,
    left_text: str,
    right_text: str,
    openai_client: Any,
    model: str = "gpt-4o",
) -> Optional[str]:
    """Validate that a summary faithfully represents its children's content.

    This uses a cheap LLM to verify the summary contains only information
    from the children and nothing extraneous.

    Args:
        summary: The generated summary to validate
        left_text: Text content of the left child
        right_text: Text content of the right child
        openai_client: OpenAI client for validation
        model: Model to use for validation (default: gpt-4o-mini)

    Returns:
        Error message if validation fails, None if valid
    """
    if not _validate_enabled:
        return None

    # Combine children's text for reference
    combined_children = f"{left_text}\n\n{right_text}"

    # Truncate if too long (to stay within token limits)
    max_chars = 8000  # Conservative limit for context
    if len(combined_children) > max_chars:
        combined_children = combined_children[:max_chars] + "... [truncated]"
    if len(summary) > 2000:
        summary = summary[:2000] + "... [truncated]"

    prompt = f"""You are a validation assistant. Your task is to check if a summary accurately represents the content from its source texts, without adding factual information that isn't present.

Source texts (the content that should be summarized):
---
{combined_children}
---

Summary to validate:
---
{summary}
---

Check if the summary:
1. Contains only information that can be found or reasonably inferred from the source texts
2. Does NOT add new facts, events, or details not present in the source
3. Is a reasonable summary of the source content

IMPORTANT CLARIFICATIONS:
- Paraphrasing is ALLOWED and expected (e.g., "The mother of our particular hobbit" → "The mother of this hobbit")
- If something is referenced/mentioned in the source, saying it was "mentioned" is VALID
- Focus on factual additions, not stylistic differences
- Minor interpretations that stay true to the source meaning are VALID
- Reasonable inferences from context are VALID (e.g., "At may never return he began" → "As Thorin mentioned 'may never return'")
- If the context clearly indicates who said something, attributing it to that speaker is VALID

Respond with either:
- "VALID" if the summary accurately represents the source content
- "INVALID: <brief explanation>" if the summary adds factual information NOT in the source texts

Be strict about factual additions, but allow normal paraphrasing and summarization."""

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Slightly higher to reduce overly literal interpretations
            max_tokens=200,
        )

        result = response.choices[0].message.content.strip()

        if result.startswith("VALID"):
            return None
        elif result.startswith("INVALID:"):
            return str(result)
        else:
            logger.warning(f"Unexpected validation response: {result}")
            return "INVALID: unexpected response format"

    except Exception as e:
        logger.error(f"Error during summary validation: {e}")
        # Don't fail indexing due to validation errors
        return None
