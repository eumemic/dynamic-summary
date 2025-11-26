"""Greedy tiling generator that maintains coverage while trimming to budget."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

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
        """Generate a tiling by pruning the frontier until within budget."""

        if not nodes:
            return DPResult(Tiling.empty(), [], 0.0, {})

        frontier = _build_frontier(nodes, root_ids)
        total_tokens = sum(nodes[node_id].token_count for node_id in frontier)

        # Already within budget: return as-is.
        if total_tokens <= budget_tokens:
            return _build_result(frontier, scores, nodes)

        # Iteratively replace the least-relevant sibling pair with its parent.
        frontier_set = set(frontier)
        while total_tokens > budget_tokens:
            replacement = _select_replacement(frontier_set, scores, nodes)
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
            if right_id != left_id and right_id in nodes:
                pair_tokens += nodes[right_id].token_count
            parent_tokens = nodes[parent_id].token_count

            if left_id in frontier_set:
                frontier_set.remove(left_id)
            if right_id != left_id and right_id in frontier_set:
                frontier_set.remove(right_id)
            frontier_set.add(parent_id)
            total_tokens = total_tokens - pair_tokens + parent_tokens

        ordered_frontier = sorted(
            frontier_set,
            key=lambda nid: (int(getattr(nodes[nid], "span_start", 0)), nid),
        )
        return _build_result(ordered_frontier, scores, nodes)


def _build_frontier(
    nodes: Mapping[str, TreeNode], root_ids: Sequence[str]
) -> list[str]:
    """Return frontier nodes (those without covered children)."""

    frontier: list[str] = []
    node_ids = set(nodes.keys())
    for node_id in node_ids:
        node = nodes[node_id]
        has_left = node.left_child_id in node_ids
        has_right = node.right_child_id in node_ids
        if not has_left and not has_right:
            frontier.append(node_id)
    frontier.sort(key=lambda nid: (int(getattr(nodes[nid], "span_start", 0)), nid))
    return frontier


def _select_replacement(
    frontier: set[str],
    scores: Mapping[str, float],
    nodes: Mapping[str, TreeNode],
) -> tuple[str, str, str] | None:
    """Pick the roll-up with the smallest quality loss per token saved."""

    best: tuple[float, float, tuple[str, str, str]] | None = None
    for node_id in list(frontier):
        node = nodes[node_id]
        parent_id = node.parent_id
        if parent_id is None or parent_id not in nodes:
            continue
        parent = nodes[parent_id]
        parent_mass = scores.get(parent_id, 0.0) * parent.token_count

        # Identify sibling (may not exist for single-child parents)
        if parent.left_child_id == node_id:
            sib_id = parent.right_child_id
        else:
            sib_id = parent.left_child_id

        # Single-child roll-up: collapse lone child into parent if it reduces tokens.
        if sib_id is None or sib_id not in frontier:
            tokens_saved = node.token_count - parent.token_count
            if tokens_saved > 0:
                child_mass = scores.get(node_id, 0.0) * node.token_count
                quality_lost = child_mass - parent_mass
                candidate = (
                    quality_lost / tokens_saved,
                    -tokens_saved,
                    (parent_id, node_id, node_id),
                )
                if best is None or candidate < best:
                    best = candidate
            continue

        # Sibling pair roll-up
        sibling = nodes[sib_id]
        pair_tokens = node.token_count + sibling.token_count
        tokens_saved = pair_tokens - parent.token_count
        # No benefit if parent is same or larger
        if tokens_saved <= 0:
            continue
        pair_mass = (
            scores.get(node_id, 0.0) * node.token_count
            + scores.get(sib_id, 0.0) * sibling.token_count
        )
        quality_lost = pair_mass - parent_mass
        candidate = (
            quality_lost / tokens_saved,
            -tokens_saved,
            (parent_id, node_id, sib_id),
        )
        if best is None or candidate < best:
            best = candidate

    if best is None:
        return None
    return best[2]


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
