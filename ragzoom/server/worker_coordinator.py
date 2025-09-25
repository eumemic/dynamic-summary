"""Worker coordination utilities for server-managed summarization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ragzoom.contracts.tree_node import TreeNode
from ragzoom.document_store import DocumentStore


@dataclass(frozen=True, slots=True)
class ReadyParentCandidate:
    """Payload describing a parent node that should be built."""

    document_id: str
    left_child_id: str
    right_child_id: str | None
    height: int
    span_start: int
    span_end: int


def _span_bounds(nodes: Iterable[TreeNode]) -> int:
    """Return the maximum span end for a collection of nodes."""

    max_end = 0
    for node in nodes:
        max_end = max(max_end, int(getattr(node, "span_end", 0)))
    return max_end


def compute_ready_parent_candidates(store: DocumentStore) -> list[ReadyParentCandidate]:
    """Identify child nodes whose parents need to be (re)constructed.

    The function inspects the current document tree and selects left-child nodes
    that meet the readiness predicate described in the server-worker design:

    1. `parent_id` is NULL.
    2. Either the node has a following neighbour, or it already spans the entire
       document (meaning it will become the root once summarised).
    3. The following neighbour, when present, also lacks a parent, sits at the
       same height, and points back to the current node via
       `preceding_neighbor_id`.

    Returns:
        A list of `ReadyParentCandidate` objects ordered by span start.
    """

    document_id = store.document_id or ""
    nodes = store.nodes.get_all()
    if not nodes:
        return []

    by_id: dict[str, TreeNode] = {node.id: node for node in nodes}
    doc_span_end = _span_bounds(nodes)

    candidates: list[ReadyParentCandidate] = []
    claimed: set[str] = set()

    for node in sorted(nodes, key=lambda n: int(getattr(n, "span_start", 0))):
        node_id = node.id
        if node_id in claimed:
            continue
        if node.parent_id is not None:
            continue

        span_start = int(getattr(node, "span_start", 0))
        span_end = int(getattr(node, "span_end", 0))
        height = int(getattr(node, "height", 0))

        right_id = getattr(node, "following_neighbor_id", None)
        if right_id:
            right_node = by_id.get(right_id)
            if (
                right_node
                and right_node.parent_id is None
                and int(getattr(right_node, "height", -1)) == height
                and getattr(right_node, "preceding_neighbor_id", None) == node_id
            ):
                candidates.append(
                    ReadyParentCandidate(
                        document_id=document_id,
                        left_child_id=node_id,
                        right_child_id=right_id,
                        height=height,
                        span_start=span_start,
                        span_end=int(getattr(right_node, "span_end", span_end)),
                    )
                )
                claimed.add(node_id)
                claimed.add(right_id)
                continue

        # If no right neighbour exists, check if this node spans the full document.
        if span_start == 0 and span_end == doc_span_end:
            candidates.append(
                ReadyParentCandidate(
                    document_id=document_id,
                    left_child_id=node_id,
                    right_child_id=None,
                    height=height,
                    span_start=span_start,
                    span_end=span_end,
                )
            )
            claimed.add(node_id)

    return candidates


__all__ = ["ReadyParentCandidate", "compute_ready_parent_candidates"]
