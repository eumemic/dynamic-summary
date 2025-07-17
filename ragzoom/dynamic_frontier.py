import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import tiktoken

from ragzoom.config import RagZoomConfig
from ragzoom.store import Store

if TYPE_CHECKING:
    from ragzoom.store import TreeNode

logger = logging.getLogger(__name__)


@dataclass
class SummarySegment:
    """Represents a segment of the summary, which is always a half-node."""

    node_id: str
    side: Literal["LEFT", "RIGHT"]


class DynamicFrontierGenerator:
    """Generates a frontier using a dynamic programming approach."""

    def __init__(self, config: RagZoomConfig, store: Store):
        self.config = config
        self.store = store
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._memo_cache: dict[
            tuple[Optional[str], int], tuple[list[SummarySegment], float]
        ] = {}

    def find_optimal_frontier(
        self, budget_tokens: int, scores: dict[str, float], document_id: Optional[str]
    ) -> list["SummarySegment"]:
        logger.info("Using DP frontier generation")
        root_node = self.store.get_root_node_for_document(document_id)
        if not root_node:
            return []

        self._memo_cache = {}
        segments, quality = self._find_optimal_frontier_for_span(
            root_node, budget_tokens, scores
        )

        logger.info(
            f"DP frontier generated with total quality {quality:.3f} and {len(segments)} segments."
        )
        return segments

    def _get_segment_cost(self, segment: "SummarySegment") -> int:
        node = self.store.get_node(segment.node_id)
        if not node or not node.text:
            return 0
        if node.depth == 0 or node.mid_offset is None:
            return len(self.tokenizer.encode(node.text))
        if segment.side == "LEFT":
            text = node.text[: node.mid_offset]
        else:
            text = node.text[node.mid_offset :]
        return len(self.tokenizer.encode(text.strip()))

    def _calculate_segment_quality(
        self, segment: "SummarySegment", scores: dict[str, float]
    ) -> float:
        segment_node = self.store.get_node(segment.node_id)
        if not segment_node:
            return 0.0
        child = self.store.get_child(segment.node_id, segment.side)
        if not child:
            if segment_node.depth == 0:
                return scores.get(segment_node.id, 0.0)
            return scores.get(segment_node.id, 0.0) / 2.0

        segment_span_start = child.span_start
        segment_span_end = child.span_end
        seed_nodes = self.store.get_nodes(list(scores.keys()))
        covered_seeds = self._get_nodes_in_span(
            segment_span_start, segment_span_end, seed_nodes
        )
        return sum(scores.get(seed.id, 0.0) for seed in covered_seeds)

    def _get_nodes_in_span(
        self, span_start: int, span_end: int, nodes: list["TreeNode"]
    ) -> list["TreeNode"]:
        return [
            node
            for node in nodes
            if node.span_start >= span_start and node.span_end <= span_end
        ]

    def _split_budget_proportionally(
        self, budget: int, node: "TreeNode", scores: dict[str, float]
    ) -> tuple[int, int]:
        left_child, right_child = self.store.get_children(node.id)
        if not left_child or not right_child:
            return budget // 2, budget // 2
        seed_nodes = self.store.get_nodes(list(scores.keys()))
        seeds_left = self._get_nodes_in_span(
            left_child.span_start, left_child.span_end, seed_nodes
        )
        seeds_right = self._get_nodes_in_span(
            right_child.span_start, right_child.span_end, seed_nodes
        )
        relevance_left = sum(scores.get(s.id, 0.0) for s in seeds_left)
        relevance_right = sum(scores.get(s.id, 0.0) for s in seeds_right)
        total_relevance = relevance_left + relevance_right
        if total_relevance == 0:
            len_left = len(left_child.text) if left_child.text else 1
            len_right = len(right_child.text) if right_child.text else 1
            total_len = len_left + len_right
            budget_l = int(budget * (len_left / total_len))
        else:
            budget_l = int(budget * (relevance_left / total_relevance))
        budget_r = budget - budget_l
        return budget_l, budget_r

    def _find_best_choice_for_side(
        self,
        parent_node: "TreeNode",
        side: Literal["LEFT", "RIGHT"],
        budget_for_side: int,
        scores: dict[str, float],
    ) -> tuple[list[SummarySegment], float]:
        frontier_1 = [SummarySegment(parent_node.id, side)]
        quality_1 = self._calculate_segment_quality(frontier_1[0], scores)
        child_node = self.store.get_child(parent_node.id, side)
        frontier_2, quality_2 = self._find_optimal_frontier_for_span(
            child_node, budget_for_side, scores
        )
        if quality_2 > quality_1:
            return (frontier_2, quality_2)
        else:
            return (frontier_1, quality_1)

    def _find_optimal_frontier_for_span_unmemoized(
        self, node: Optional["TreeNode"], budget: int, scores: dict[str, float]
    ) -> tuple[list[SummarySegment], float]:
        if not node:
            return ([], 0.0)
        cost_of_this_node = self._get_segment_cost(
            SummarySegment(node.id, "LEFT")
        ) + self._get_segment_cost(SummarySegment(node.id, "RIGHT"))
        if budget < cost_of_this_node:
            return ([], 0.0)
        budget_l, budget_r = self._split_budget_proportionally(budget, node, scores)
        best_frontier_left, best_quality_left = self._find_best_choice_for_side(
            node, "LEFT", budget_l, scores
        )
        best_frontier_right, best_quality_right = self._find_best_choice_for_side(
            node, "RIGHT", budget_r, scores
        )
        final_frontier = best_frontier_left + best_frontier_right
        final_quality = best_quality_left + best_quality_right
        final_cost = sum(self._get_segment_cost(seg) for seg in final_frontier)
        if final_cost > budget:
            quality = self._calculate_segment_quality(
                SummarySegment(node.id, "LEFT"), scores
            ) + self._calculate_segment_quality(
                SummarySegment(node.id, "RIGHT"), scores
            )
            return (
                [SummarySegment(node.id, "LEFT"), SummarySegment(node.id, "RIGHT")],
                quality,
            )
        return (final_frontier, final_quality)

    def _find_optimal_frontier_for_span(
        self, node: Optional["TreeNode"], budget: int, scores: dict[str, float]
    ) -> tuple[list[SummarySegment], float]:
        if budget is None:
            budget = self.config.budget_tokens
        node_id = node.id if node else None
        cache_key = (node_id, budget)
        if cache_key in self._memo_cache:
            return self._memo_cache[cache_key]
        result = self._find_optimal_frontier_for_span_unmemoized(node, budget, scores)
        self._memo_cache[cache_key] = result
        return result
