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
class Segment:
    """Represents a segment of the summary.

    For internal nodes (depth > 0): side is "LEFT" or "RIGHT" (half-node).
    For leaf nodes (depth == 0): side is None (full node).
    """

    node_id: str
    side: Optional[Literal["LEFT", "RIGHT"]]


@dataclass
class SegmentInfo:
    """Extended segment information including token cost and span."""

    segment: Segment
    token_cost: int
    span_start: int
    span_end: int


@dataclass
class DPResult:
    """Complete result from DP tiling generation."""

    segments: list[Segment]
    segment_infos: list[SegmentInfo]
    total_quality: float
    coverage_map: dict[str, bool]


class DynamicTilingGenerator:
    """Generates a tiling using a dynamic programming approach."""

    def __init__(self, config: RagZoomConfig, store: Store):
        self.config = config
        self.store = store
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._memo_cache: dict[
            tuple[Optional[str], int], tuple[list[Segment], float]
        ] = {}

    def find_optimal_tiling(
        self,
        budget_tokens: int,
        scores: dict[str, float],
        document_id: Optional[str],
        coverage_map: dict[str, bool],
    ) -> DPResult:
        logger.info("Using DP tiling generation")
        root_node = self.store.get_root_node_for_document(document_id)
        if not root_node:
            return DPResult([], [], 0.0, coverage_map)

        self._memo_cache = {}
        segments, quality = self._find_optimal_tiling_for_span(
            root_node, budget_tokens, scores
        )

        # Build segment infos with costs and spans
        segment_infos = []
        for seg in segments:
            cost = self._get_segment_cost(seg)
            span_start, span_end = self._get_segment_span(seg)
            segment_infos.append(SegmentInfo(seg, cost, span_start, span_end))

        logger.info(
            f"DP tiling generated with total quality {quality:.3f} and {len(segments)} segments."
        )

        return DPResult(
            segments=segments,
            segment_infos=segment_infos,
            total_quality=quality,
            coverage_map=coverage_map,
        )

    def _get_segment_cost(self, segment: "Segment") -> int:
        node = self.store.get_node(segment.node_id)
        if not node or not node.text:
            return 0
        if self.store.is_leaf_node(node.id) or node.mid_offset is None:
            # Leaf node - side should be None
            return len(self.tokenizer.encode(node.text))
        # Internal node - side should be "LEFT" or "RIGHT"
        if segment.side == "LEFT":
            text = node.text[: node.mid_offset]
        else:  # segment.side == "RIGHT"
            text = node.text[node.mid_offset :]
        return len(self.tokenizer.encode(text.strip()))

    def _get_segment_span(self, segment: "Segment") -> tuple[int, int]:
        """Get the actual character span for a segment."""
        node = self.store.get_node(segment.node_id)
        if not node:
            return (0, 0)

        # For leaf nodes, the span is the node's span
        if self.store.is_leaf_node(node.id) or segment.side is None:
            return (node.span_start, node.span_end)

        # For internal nodes, we need to determine the actual segment span
        if segment.side == "LEFT":
            # LEFT segment spans from node start to left child end
            if node.left_child_id:
                left_child = self.store.get_node(node.left_child_id)
                if left_child:
                    return (node.span_start, left_child.span_end)
            # Fallback to half of node
            return (node.span_start, (node.span_start + node.span_end) // 2)
        else:  # RIGHT
            # RIGHT segment spans from right child start to node end
            if node.right_child_id:
                right_child = self.store.get_node(node.right_child_id)
                if right_child:
                    return (right_child.span_start, node.span_end)
            # Fallback to half of node
            return ((node.span_start + node.span_end) // 2, node.span_end)

    def _calculate_segment_quality(
        self, segment: "Segment", scores: dict[str, float]
    ) -> float:
        base_score = scores.get(segment.node_id, 0.0)
        node = self.store.get_node(segment.node_id)
        if node and not self.store.is_leaf_node(segment.node_id):
            return base_score / 2.0
        return base_score

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
    ) -> tuple[list[Segment], float]:
        tiling_1 = [Segment(parent_node.id, side)]
        quality_1 = self._calculate_segment_quality(tiling_1[0], scores)
        child_node = self.store.get_child(parent_node.id, side)
        tiling_2, quality_2 = self._find_optimal_tiling_for_span(
            child_node, budget_for_side, scores
        )
        if tiling_2 and quality_2 > quality_1:
            return (tiling_2, quality_2)
        else:
            return (tiling_1, quality_1)

    def _find_optimal_tiling_for_span_unmemoized(
        self, node: Optional["TreeNode"], budget: int, scores: dict[str, float]
    ) -> tuple[list[Segment], float]:
        if not node:
            return ([], 0.0)

        # NEW: leaf is indivisible
        if self.store.is_leaf_node(node.id) or node.mid_offset is None:
            seg = Segment(node.id, None)
            cost = self._get_segment_cost(seg)
            if budget < cost:
                return ([], 0.0)  # cannot afford
            quality = self._calculate_segment_quality(seg, scores)
            return ([seg], quality)

        # Internal node logic continues as before
        cost_of_this_node = self._get_segment_cost(
            Segment(node.id, "LEFT")
        ) + self._get_segment_cost(Segment(node.id, "RIGHT"))
        if budget < cost_of_this_node:
            return ([], 0.0)
        budget_l, budget_r = self._split_budget_proportionally(budget, node, scores)
        best_tiling_left, best_quality_left = self._find_best_choice_for_side(
            node, "LEFT", budget_l, scores
        )
        best_tiling_right, best_quality_right = self._find_best_choice_for_side(
            node, "RIGHT", budget_r, scores
        )
        final_tiling = best_tiling_left + best_tiling_right
        final_quality = best_quality_left + best_quality_right
        final_cost = sum(self._get_segment_cost(seg) for seg in final_tiling)
        if final_cost > budget:
            quality = self._calculate_segment_quality(
                Segment(node.id, "LEFT"), scores
            ) + self._calculate_segment_quality(Segment(node.id, "RIGHT"), scores)
            return (
                [Segment(node.id, "LEFT"), Segment(node.id, "RIGHT")],
                quality,
            )
        return (final_tiling, final_quality)

    def _find_optimal_tiling_for_span(
        self, node: Optional["TreeNode"], budget: int, scores: dict[str, float]
    ) -> tuple[list[Segment], float]:
        node_id = node.id if node else None
        cache_key = (node_id, budget)
        if cache_key in self._memo_cache:
            return self._memo_cache[cache_key]
        result = self._find_optimal_tiling_for_span_unmemoized(node, budget, scores)
        self._memo_cache[cache_key] = result
        return result
