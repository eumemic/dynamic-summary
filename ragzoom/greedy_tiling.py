"""Greedy tiling generator that maintains coverage while trimming to budget."""

from __future__ import annotations

import heapq
import logging
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from ragzoom.config import QueryConfig
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.dynamic_tiling import DPResult, NodeInfo
from ragzoom.tiling import Tiling

logger = logging.getLogger(__name__)


class GreedyTilingGenerator:
    """Greedy alternative to DP: iteratively roll up least-relevant sibling pairs."""

    def __init__(self, config: QueryConfig):
        self.config = config

    def find_optimal_tiling_over_roots(
        self,
        root_ids: Sequence[str],
        budget_tokens: int,
        scores: Mapping[str, float],
        nodes: Mapping[str, TreeNode],
    ) -> DPResult:
        """Generate a tiling by pruning the frontier until within budget.

        Args:
            root_ids: Root node IDs to start traversal from
            budget_tokens: Token budget for the tiling
            scores: Relevance scores for each node
            nodes: All nodes in the coverage tree
        """
        if not nodes:
            return DPResult(Tiling.empty(), [], 0.0, {})

        frontier = _build_frontier(nodes, root_ids)
        total_tokens = sum(nodes[node_id].token_count for node_id in frontier)

        # Already within budget: return as-is.
        if total_tokens <= budget_tokens:
            return _build_result(frontier, scores, nodes)

        # Iteratively replace the least-relevant sibling pair with its parent.
        frontier_set = set(frontier)
        queue, enqueued = _initialize_candidate_queue(frontier_set, nodes, scores)
        while total_tokens > budget_tokens:
            replacement = _pop_next_candidate(
                queue, enqueued, frontier_set, nodes, scores
            )
            if replacement is None:
                logger.debug(
                    "Greedy tiling could not reduce budget further "
                    "(tokens=%d, budget=%d, frontier=%d)",
                    total_tokens,
                    budget_tokens,
                    len(frontier_set),
                )
                break

            parent_id, left_id, right_id = replacement
            pair_tokens = nodes[left_id].token_count
            if right_id != left_id:
                pair_tokens += nodes[right_id].token_count
            parent_tokens = nodes[parent_id].token_count

            frontier_set.remove(left_id)
            if right_id != left_id:
                frontier_set.remove(right_id)
            frontier_set.add(parent_id)
            total_tokens = total_tokens - pair_tokens + parent_tokens

            grandparent_id = nodes[parent_id].parent_id
            if grandparent_id is not None:
                _enqueue_candidate(
                    grandparent_id,
                    queue,
                    enqueued,
                    frontier_set,
                    nodes,
                    scores,
                )

        ordered_frontier = sorted(
            frontier_set,
            key=lambda nid: (int(getattr(nodes[nid], "span_start", 0)), nid),
        )
        return _build_result(ordered_frontier, scores, nodes)


def _build_frontier(
    nodes: Mapping[str, TreeNode], root_ids: Sequence[str]
) -> list[str]:
    """Return frontier nodes reachable from the supplied roots."""

    visited: set[str] = set()
    stack: list[str] = [rid for rid in root_ids if rid in nodes]
    if not stack:
        stack = list(nodes.keys())

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        node = nodes[current]
        if node.left_child_id:
            if node.left_child_id in nodes:
                stack.append(node.left_child_id)
        if node.right_child_id and node.right_child_id != node.left_child_id:
            if node.right_child_id in nodes:
                stack.append(node.right_child_id)

    frontier: list[str] = []
    for node_id in visited:
        node = nodes[node_id]
        has_left = node.left_child_id in visited
        has_right = node.right_child_id in visited
        if not has_left and not has_right:
            frontier.append(node_id)
    frontier.sort(key=lambda nid: (int(getattr(nodes[nid], "span_start", 0)), nid))
    return frontier


class _RollupCandidate(NamedTuple):
    """Roll-up option ordered by priority for the frontier queue."""

    priority: float  # quality lost per token saved (lower is better)
    neg_tokens_saved: int  # tie-breaker: save more tokens first
    parent_id: str
    left_id: str
    right_id: str


def _compute_candidate(
    parent_id: str,
    frontier: set[str],
    nodes: Mapping[str, TreeNode],
    scores: Mapping[str, float],
) -> _RollupCandidate | None:
    """Return a roll-up candidate if the parent is eligible."""
    if parent_id not in nodes:
        return None
    parent = nodes[parent_id]
    left_id = parent.left_child_id
    right_id = parent.right_child_id

    has_left = left_id is not None and left_id in nodes
    has_right = right_id is not None and right_id in nodes

    if has_left and has_right:
        if left_id not in frontier or right_id not in frontier:
            return None
        child_ids = (left_id, right_id)
    elif has_left:
        if left_id not in frontier:
            return None
        child_ids = (left_id, left_id)
    elif has_right:
        if right_id not in frontier:
            return None
        child_ids = (right_id, right_id)
    else:
        return None

    left_child_id, right_child_id = child_ids

    pair_tokens = nodes[left_child_id].token_count
    if right_child_id != left_child_id:
        pair_tokens += nodes[right_child_id].token_count

    tokens_saved = pair_tokens - parent.token_count

    parent_mass = scores.get(parent_id, 0.0) * parent.token_count
    pair_mass = scores.get(left_child_id, 0.0) * nodes[left_child_id].token_count
    if right_child_id != left_child_id:
        pair_mass += scores.get(right_child_id, 0.0) * nodes[right_child_id].token_count
    quality_lost = pair_mass - parent_mass

    priority = quality_lost / tokens_saved if tokens_saved != 0 else float("inf")

    return _RollupCandidate(
        priority,
        -tokens_saved,
        parent_id,
        left_child_id,
        right_child_id,
    )


def _enqueue_candidate(
    parent_id: str,
    queue: list[_RollupCandidate],
    enqueued: set[str],
    frontier: set[str],
    nodes: Mapping[str, TreeNode],
    scores: Mapping[str, float],
) -> None:
    """Add a parent to the priority queue if it just became eligible."""
    if parent_id in enqueued:
        return
    candidate = _compute_candidate(parent_id, frontier, nodes, scores)
    if candidate is None:
        return
    heapq.heappush(queue, candidate)
    enqueued.add(parent_id)


def _initialize_candidate_queue(
    frontier: set[str],
    nodes: Mapping[str, TreeNode],
    scores: Mapping[str, float],
) -> tuple[list[_RollupCandidate], set[str]]:
    """Seed the roll-up queue with all eligible parents from the initial frontier."""
    queue: list[_RollupCandidate] = []
    enqueued: set[str] = set()
    for node_id in frontier:
        parent_id = nodes[node_id].parent_id
        if parent_id is None:
            continue
        _enqueue_candidate(parent_id, queue, enqueued, frontier, nodes, scores)
    return queue, enqueued


def _pop_next_candidate(
    queue: list[_RollupCandidate],
    enqueued: set[str],
    frontier: set[str],
    nodes: Mapping[str, TreeNode],
    scores: Mapping[str, float],
) -> tuple[str, str, str] | None:
    """Return the best still-valid candidate from the queue."""
    while queue:
        candidate = heapq.heappop(queue)
        enqueued.discard(candidate.parent_id)
        current = _compute_candidate(candidate.parent_id, frontier, nodes, scores)
        if current is None:
            continue
        return current.parent_id, current.left_id, current.right_id
    return None


def _build_result(
    frontier: Sequence[str],
    scores: Mapping[str, float],
    nodes: Mapping[str, TreeNode],
) -> DPResult:
    """Convert frontier node ids into a DPResult shape."""

    tiling_score = sum(
        scores.get(node_id, 0.0) * nodes[node_id].token_count for node_id in frontier
    )
    tiling = Tiling(node_ids=list(frontier), relevance_tokens=tiling_score)

    node_infos = [
        NodeInfo(
            node_id=node_id,
            token_cost=nodes[node_id].token_count,
            span_start=nodes[node_id].span_start,
            span_end=nodes[node_id].span_end,
        )
        for node_id in frontier
    ]

    coverage_map = {node_id: True for node_id in nodes}
    return DPResult(
        tiling=tiling,
        node_infos=node_infos,
        total_quality=tiling.relevance_tokens,
        coverage_map=coverage_map,
    )
