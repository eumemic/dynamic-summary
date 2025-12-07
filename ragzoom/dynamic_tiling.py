import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ragzoom.config import QueryConfig
from ragzoom.tiling import Tiling
from ragzoom.utils.tokenization import tokenizer

if TYPE_CHECKING:
    from ragzoom.contracts.tree_node import TreeNode

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a node in the tiling."""

    node_id: str
    token_cost: int
    span_start: int
    span_end: int


@dataclass
class DPResult:
    """Complete result from DP tiling generation."""

    tiling: Tiling  # The optimal tiling found
    node_infos: list[NodeInfo]
    total_quality: float
    coverage_map: dict[str, bool]


class BaseDynamicTilingGenerator:
    """Base class for DP tiling generators with shared utility methods."""

    def __init__(self, config: QueryConfig):
        self.config = config
        self.tokenizer = tokenizer
        self._subtree_relevance_cache: dict[str, float] = {}
        self._nodes: Mapping[str, TreeNode] = {}  # Will be set per tiling request
        self._min_cover_cost_cache: dict[str, int] = {}

    def _get_node_cost(self, node: "TreeNode") -> int:
        """Get the token cost of a node."""
        if not node:
            return 0
        return node.token_count

    def _get_min_cover_cost(self, node: "TreeNode") -> int:
        """Compute the minimum token cost needed to cover the node's full span."""

        if node.id in self._min_cover_cost_cache:
            return self._min_cover_cost_cache[node.id]

        left_child = self._nodes.get(node.left_child_id) if node.left_child_id else None
        right_child = (
            self._nodes.get(node.right_child_id) if node.right_child_id else None
        )

        if not left_child and not right_child:
            cost = self._get_node_cost(node)
            self._min_cover_cost_cache[node.id] = cost
            return cost

        children_cost = 0
        if left_child:
            children_cost += self._get_min_cover_cost(left_child)
        if right_child:
            children_cost += self._get_min_cover_cost(right_child)

        if children_cost == 0:
            children_cost = self._get_node_cost(node)

        cost = min(self._get_node_cost(node), children_cost)
        self._min_cover_cost_cache[node.id] = cost
        return cost

    def _get_subtree_relevance(
        self, node: "TreeNode", scores: dict[str, float]
    ) -> float:
        """Recursively sum all relevance scores in a subtree with memoization."""
        if node.id in self._subtree_relevance_cache:
            return self._subtree_relevance_cache[node.id]

        # Get this node's score
        node_score = scores.get(node.id, 0.0)
        total = node_score

        # Add children's scores recursively
        # Only traverse children that exist in our nodes dict
        if node.left_child_id and node.left_child_id in self._nodes:
            left_child = self._nodes[node.left_child_id]
            total += self._get_subtree_relevance(left_child, scores)

        if node.right_child_id and node.right_child_id in self._nodes:
            right_child = self._nodes[node.right_child_id]
            total += self._get_subtree_relevance(right_child, scores)

        self._subtree_relevance_cache[node.id] = total
        return total

    def _split_budget_proportionally(
        self, budget: int, node: "TreeNode", scores: dict[str, float]
    ) -> tuple[int, int] | None:
        """Split budget between left and right children based on their relevance."""
        # Get children from our nodes dict
        left_child = self._nodes.get(node.left_child_id) if node.left_child_id else None
        right_child = (
            self._nodes.get(node.right_child_id) if node.right_child_id else None
        )

        if not left_child or not right_child:
            return None

        allocations = self._allocate_budget_for_nodes(
            [left_child, right_child], budget, scores
        )
        if allocations is None:
            return None

        return allocations[0], allocations[1]

    @staticmethod
    def _distribute_budget(weights: Sequence[float], total: int) -> list[int]:
        """Distribute integer budget according to weights."""

        count = len(weights)
        if count == 0:
            return []
        if total <= 0:
            return [0] * count

        weight_sum = sum(w for w in weights if w > 0)
        if weight_sum <= 0:
            normalized = [1.0 / count for _ in range(count)]
        else:
            normalized = [max(0.0, w) / weight_sum for w in weights]

        allocations: list[int] = []
        assigned = 0
        running = 0.0
        for idx, weight in enumerate(normalized):
            if idx == count - 1:
                alloc = total - assigned
            else:
                running += weight * total
                alloc = int(round(running)) - assigned
            if alloc < 0:
                alloc = 0
            allocations.append(alloc)
            assigned += alloc

        if allocations:
            diff = total - sum(allocations)
            if diff != 0:
                allocations[-1] += diff
        return allocations

    def _allocate_budget_for_nodes(
        self,
        nodes: Sequence["TreeNode"],
        budget: int,
        scores: dict[str, float],
    ) -> list[int] | None:
        """Allocate budget across nodes while respecting minimum cover cost."""

        if not nodes:
            return []

        min_costs = [self._get_min_cover_cost(node) for node in nodes]
        required = sum(min_costs)

        if budget < required:
            return None

        allocations = min_costs[:]
        remaining = budget - required
        if remaining <= 0:
            return allocations

        weights: list[float] = []
        for node in nodes:
            relevance = self._get_subtree_relevance(node, scores)
            if relevance <= 0:
                text_len = len(node.text) if node.text else 1
                relevance = max(1.0, float(text_len))
            weights.append(float(relevance))

        extras = self._distribute_budget(weights, remaining)
        return [base + extra for base, extra in zip(allocations, extras)]

    def _build_result(
        self, tiling: Tiling, nodes: Mapping[str, "TreeNode"]
    ) -> DPResult:
        """Build DPResult from tiling and nodes."""
        node_infos = []
        for node_id in tiling.node_ids:
            if node_id in nodes:
                node = nodes[node_id]
                cost = self._get_node_cost(node)
                node_infos.append(
                    NodeInfo(node_id, cost, node.span_start, node.span_end)
                )

        coverage_map = {node_id: True for node_id in nodes}

        return DPResult(
            tiling=tiling,
            node_infos=node_infos,
            total_quality=tiling.relevance_tokens,
            coverage_map=coverage_map,
        )


class DynamicTilingGenerator(BaseDynamicTilingGenerator):
    """Generates a tiling using a dynamic programming approach."""

    def __init__(self, config: QueryConfig):
        super().__init__(config)
        self._memo_cache: dict[tuple[str | None, int], Tiling] = {}

    def find_optimal_tiling(
        self,
        budget_tokens: int,
        scores: dict[str, float],
        nodes: Mapping[str, "TreeNode"],
        root_id: str,
    ) -> DPResult:
        result = self.find_optimal_tiling_over_roots(
            [root_id], budget_tokens, scores, nodes
        )
        logger.info(
            "DP tiling generated with total quality %.3f and %d nodes.",
            result.total_quality,
            len(result.tiling.node_ids),
        )
        return result

    def find_optimal_tiling_over_roots(
        self,
        root_ids: Sequence[str],
        budget_tokens: int,
        scores: dict[str, float],
        nodes: Mapping[str, "TreeNode"],
        pinned_ids: set[str] | None = None,
    ) -> DPResult:
        logger.info("Using DP tiling generation across %d roots", len(root_ids))

        if not nodes:
            return DPResult(Tiling.empty(), [], 0.0, {})

        # Apply transient pinning: override scores for pinned nodes
        effective_scores = dict(scores)
        if pinned_ids:
            for node_id in pinned_ids:
                if node_id in effective_scores:
                    effective_scores[node_id] = 1.0

        unique_root_ids: list[str] = []
        seen: set[str] = set()
        for root_id in root_ids:
            if root_id in nodes and root_id not in seen:
                unique_root_ids.append(root_id)
                seen.add(root_id)

        if not unique_root_ids:
            return DPResult(Tiling.empty(), [], 0.0, {})

        root_nodes = sorted(
            (nodes[root_id] for root_id in unique_root_ids),
            key=lambda node: (int(getattr(node, "span_start", 0)), node.id),
        )

        self._nodes = nodes
        self._memo_cache = {}
        self._subtree_relevance_cache = {}
        self._min_cover_cost_cache = {}

        budgets = self._allocate_budget_for_nodes(
            root_nodes, budget_tokens, effective_scores
        )
        if budgets is None:
            logger.debug(
                "Budget %d cannot cover forest; returning empty tiling",
                budget_tokens,
            )
            return self._build_result(Tiling.empty(), nodes)
        combined = Tiling.empty()

        for root_node, root_budget in zip(root_nodes, budgets):
            if root_budget <= 0:
                continue
            combined += self._find_optimal_tiling_for_span(
                root_node, root_budget, effective_scores
            )

        return self._build_result(combined, nodes)

    # jscpd:ignore-start - Legitimate async/sync wrapper pattern for DP algorithm
    def _find_optimal_tiling_for_span_unmemoized(
        self, node: Optional["TreeNode"], budget: int, scores: dict[str, float]
    ) -> Tiling:
        if not node:
            return Tiling.empty()

        node_cost = self._get_node_cost(node)

        # If budget can't afford this node, return empty
        if budget < node_cost:
            return Tiling.empty()

        # Get this node's relevance score
        node_relevance = scores.get(node.id, 0.0)
        node_quality = node_relevance * node_cost

        # Check if this is a leaf node (no left child = leaf in perfect binary tree)
        is_leaf = node.left_child_id not in self._nodes

        # For leaf nodes, we can only use the whole node
        if is_leaf:
            return Tiling(node_ids=[node.id], relevance_tokens=node_quality)

        # For internal nodes, we have two options:
        # Option 1: Use this node
        option1 = Tiling(node_ids=[node.id], relevance_tokens=node_quality)

        # Option 2: Recurse to children
        # Get children from our nodes dict
        left_child = self._nodes.get(node.left_child_id) if node.left_child_id else None
        right_child = (
            self._nodes.get(node.right_child_id) if node.right_child_id else None
        )

        if left_child and not right_child:
            # Single left child case - give entire budget to left child
            left_tiling = self._find_optimal_tiling_for_span(left_child, budget, scores)
            option2 = left_tiling
        elif left_child and right_child:
            split = self._split_budget_proportionally(budget, node, scores)
            if split is None:
                option2 = Tiling.empty()
            else:
                budget_l, budget_r = split
                left_tiling = self._find_optimal_tiling_for_span(
                    left_child, budget_l, scores
                )
                right_tiling = self._find_optimal_tiling_for_span(
                    right_child, budget_r, scores
                )
                option2 = left_tiling + right_tiling
        else:
            # No children - this should be caught by is_leaf check above
            raise ValueError(f"Internal node {node.id} has no children")

        # Return the option with higher quality
        return (
            option2 if option2.relevance_tokens > option1.relevance_tokens else option1
        )

    # jscpd:ignore-end

    # jscpd:ignore-start - Legitimate async/sync wrapper pattern for memoization
    def _find_optimal_tiling_for_span(
        self, node: Optional["TreeNode"], budget: int, scores: dict[str, float]
    ) -> Tiling:
        node_id = node.id if node else None
        cache_key = (node_id, budget)
        if cache_key in self._memo_cache:
            return self._memo_cache[cache_key]
        result = self._find_optimal_tiling_for_span_unmemoized(node, budget, scores)
        self._memo_cache[cache_key] = result
        return result

    # jscpd:ignore-end


class AsyncDynamicTilingGenerator(BaseDynamicTilingGenerator):
    """Async version of DynamicTilingGenerator with parallel subtree processing."""

    def __init__(self, config: QueryConfig, min_nodes_for_parallel: int = 10):
        super().__init__(config)
        self.min_nodes_for_parallel = min_nodes_for_parallel
        self._memo_cache: dict[tuple[str | None, int], Tiling] = {}
        self._memo_lock: asyncio.Lock | None = None
        # Threading lock for relevance cache will be created lazily

    def _get_memo_lock(self) -> asyncio.Lock:
        """Lazily create the memo lock in async context."""
        if self._memo_lock is None:
            self._memo_lock = asyncio.Lock()
        return self._memo_lock

    async def find_optimal_tiling(
        self,
        budget_tokens: int,
        scores: dict[str, float],
        nodes: Mapping[str, "TreeNode"],
        root_id: str,
    ) -> DPResult:
        result = await self.find_optimal_tiling_over_roots(
            [root_id], budget_tokens, scores, nodes
        )
        logger.info(
            "Async DP tiling generated with total quality %.3f and %d nodes.",
            result.total_quality,
            len(result.tiling.node_ids),
        )
        return result

    # jscpd:ignore-start - async variant mirrors sync implementation for parity
    async def find_optimal_tiling_over_roots(
        self,
        root_ids: Sequence[str],
        budget_tokens: int,
        scores: dict[str, float],
        nodes: Mapping[str, "TreeNode"],
        pinned_ids: set[str] | None = None,
    ) -> DPResult:
        logger.info("Using async DP tiling generation across %d roots", len(root_ids))

        if not nodes:
            return DPResult(Tiling.empty(), [], 0.0, {})

        # Apply transient pinning: override scores for pinned nodes
        effective_scores = dict(scores)
        if pinned_ids:
            for node_id in pinned_ids:
                if node_id in effective_scores:
                    effective_scores[node_id] = 1.0

        unique_root_ids: list[str] = []
        seen: set[str] = set()
        for root_id in root_ids:
            if root_id in nodes and root_id not in seen:
                unique_root_ids.append(root_id)
                seen.add(root_id)

        if not unique_root_ids:
            return DPResult(Tiling.empty(), [], 0.0, {})

        root_nodes = sorted(
            (nodes[root_id] for root_id in unique_root_ids),
            key=lambda node: (int(getattr(node, "span_start", 0)), node.id),
        )

        self._nodes = nodes
        self._memo_cache = {}
        self._subtree_relevance_cache = {}
        self._min_cover_cost_cache = {}

        budgets = self._allocate_budget_for_nodes(
            root_nodes, budget_tokens, effective_scores
        )
        if budgets is None:
            logger.debug(
                "Budget %d cannot cover forest; returning empty tiling",
                budget_tokens,
            )
            return self._build_result(Tiling.empty(), nodes)
        combined = Tiling.empty()

        for root_node, root_budget in zip(root_nodes, budgets):
            if root_budget <= 0:
                continue
            combined += await self._find_optimal_tiling_for_span(
                root_node, root_budget, effective_scores
            )

        return self._build_result(combined, nodes)

    # jscpd:ignore-end

    def _count_subtree_nodes(self, node: "TreeNode") -> int:
        """Count total nodes in a subtree."""
        count = 1
        if node.left_child_id and node.left_child_id in self._nodes:
            left_child = self._nodes[node.left_child_id]
            count += self._count_subtree_nodes(left_child)
        if node.right_child_id and node.right_child_id in self._nodes:
            right_child = self._nodes[node.right_child_id]
            count += self._count_subtree_nodes(right_child)
        return count

    def _should_parallelize(
        self, left_child: "TreeNode", right_child: "TreeNode"
    ) -> bool:
        """Determine if subtrees are large enough to warrant parallelization."""
        left_nodes = self._count_subtree_nodes(left_child)
        right_nodes = self._count_subtree_nodes(right_child)
        return (left_nodes + right_nodes) >= self.min_nodes_for_parallel

    # jscpd:ignore-start - Legitimate async/sync wrapper pattern for DP algorithm
    async def _find_optimal_tiling_for_span_unmemoized(
        self, node: Optional["TreeNode"], budget: int, scores: dict[str, float]
    ) -> Tiling:
        if not node:
            return Tiling.empty()

        node_cost = self._get_node_cost(node)

        # If budget can't afford this node, return empty
        if budget < node_cost:
            return Tiling.empty()

        # Get this node's relevance score
        node_relevance = scores.get(node.id, 0.0)
        node_quality = node_relevance * node_cost

        # Check if this is a leaf node (no left child = leaf in perfect binary tree)
        is_leaf = node.left_child_id not in self._nodes

        # For leaf nodes, we can only use the whole node
        if is_leaf:
            return Tiling(node_ids=[node.id], relevance_tokens=node_quality)

        # For internal nodes, we have two options:
        # Option 1: Use this node
        option1 = Tiling(node_ids=[node.id], relevance_tokens=node_quality)

        # Option 2: Recurse to children
        # Get children from our nodes dict
        left_child = self._nodes.get(node.left_child_id) if node.left_child_id else None
        right_child = (
            self._nodes.get(node.right_child_id) if node.right_child_id else None
        )

        if left_child and not right_child:
            # Single left child case - give entire budget to left child
            left_tiling = await self._find_optimal_tiling_for_span(
                left_child, budget, scores
            )
            option2 = left_tiling
        elif left_child and right_child:
            split = self._split_budget_proportionally(budget, node, scores)
            if split is None:
                option2 = Tiling.empty()
            else:
                budget_l, budget_r = split

                # Decide whether to parallelize based on subtree size
                if self._should_parallelize(left_child, right_child):
                    try:
                        # Run both recursive calls in parallel
                        left_task = asyncio.create_task(
                            self._find_optimal_tiling_for_span(
                                left_child, budget_l, scores
                            )
                        )
                        right_task = asyncio.create_task(
                            self._find_optimal_tiling_for_span(
                                right_child, budget_r, scores
                            )
                        )
                        results = await asyncio.gather(
                            left_task, right_task, return_exceptions=True
                        )

                        # Handle exceptions gracefully - if one subtree fails, fall back to sequential
                        if isinstance(results[0], Exception) or isinstance(
                            results[1], Exception
                        ):
                            errors = [r for r in results if isinstance(r, Exception)]
                            logger.warning(
                                f"Parallel execution failed for node {node.id} with errors: {errors}, falling back to sequential"
                            )
                            left_tiling = await self._find_optimal_tiling_for_span(
                                left_child, budget_l, scores
                            )
                            right_tiling = await self._find_optimal_tiling_for_span(
                                right_child, budget_r, scores
                            )
                        else:
                            left_tiling = results[0]  # type: ignore[assignment]
                            right_tiling = results[1]  # type: ignore[assignment]
                    except Exception as e:
                        logger.warning(
                            f"Parallelization failed for node {node.id}: {e}, using sequential"
                        )
                        left_tiling = await self._find_optimal_tiling_for_span(
                            left_child, budget_l, scores
                        )
                        right_tiling = await self._find_optimal_tiling_for_span(
                            right_child, budget_r, scores
                        )
                else:
                    # Sequential processing for small subtrees
                    left_tiling = await self._find_optimal_tiling_for_span(
                        left_child, budget_l, scores
                    )
                    right_tiling = await self._find_optimal_tiling_for_span(
                        right_child, budget_r, scores
                    )

                option2 = left_tiling + right_tiling
        else:
            # No children - this should be caught by is_leaf check above
            raise ValueError(f"Internal node {node.id} has no children")

        # Return the option with higher quality
        return (
            option2 if option2.relevance_tokens > option1.relevance_tokens else option1
        )

    # jscpd:ignore-end

    # jscpd:ignore-start - Legitimate async/sync wrapper pattern for memoization
    async def _find_optimal_tiling_for_span(
        self, node: Optional["TreeNode"], budget: int, scores: dict[str, float]
    ) -> Tiling:
        node_id = node.id if node else None
        cache_key = (node_id, budget)

        # Thread-safe cache access
        async with self._get_memo_lock():
            if cache_key in self._memo_cache:
                return self._memo_cache[cache_key]

        result = await self._find_optimal_tiling_for_span_unmemoized(
            node, budget, scores
        )

        # Thread-safe cache write
        async with self._get_memo_lock():
            self._memo_cache[cache_key] = result

        return result

    # jscpd:ignore-end
