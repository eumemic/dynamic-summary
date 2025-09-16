import asyncio
import logging
from collections.abc import Mapping
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

    def _get_node_cost(self, node: "TreeNode") -> int:
        """Get the token cost of a node."""
        if not node:
            return 0
        return node.token_count

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
    ) -> tuple[int, int]:
        """Split budget between left and right children based on their relevance."""
        # Get children from our nodes dict
        left_child = self._nodes.get(node.left_child_id) if node.left_child_id else None
        right_child = (
            self._nodes.get(node.right_child_id) if node.right_child_id else None
        )

        if not left_child or not right_child:
            return budget // 2, budget // 2

        # Get minimum costs for each child
        min_left = self._get_node_cost(left_child)
        min_right = self._get_node_cost(right_child)
        min_total = min_left + min_right

        # If budget can't even cover minimum, split proportionally by min costs
        if budget <= min_total:
            budget_l = int(budget * (min_left / min_total))
            budget_r = budget - budget_l
            return budget_l, budget_r

        # Compute total relevance mass in each subtree
        relevance_left = self._get_subtree_relevance(left_child, scores)
        relevance_right = self._get_subtree_relevance(right_child, scores)
        total_relevance = relevance_left + relevance_right

        # Direct proportional split based on relevance
        if total_relevance == 0:
            # Fall back to text length-based allocation
            len_left = len(left_child.text) if left_child.text else 1
            len_right = len(right_child.text) if right_child.text else 1
            total_len = len_left + len_right
            target_left = int(budget * (len_left / total_len))
        else:
            target_left = int(budget * (relevance_left / total_relevance))

        target_right = budget - target_left

        # Ensure minimums are met
        if target_left < min_left:
            # Left is under minimum, give it what it needs
            budget_l = min_left
            budget_r = budget - budget_l
        elif target_right < min_right:
            # Right is under minimum, give it what it needs
            budget_r = min_right
            budget_l = budget - budget_r
        else:
            # Both meet minimums, use the targets
            budget_l = target_left
            budget_r = target_right

        return budget_l, budget_r

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
        logger.info("Using DP tiling generation")

        if not nodes or root_id not in nodes:
            return DPResult(Tiling.empty(), [], 0.0, {})

        self._nodes = nodes
        self._memo_cache = {}
        self._subtree_relevance_cache = {}

        root_node = nodes[root_id]
        tiling = self._find_optimal_tiling_for_span(root_node, budget_tokens, scores)

        logger.info(
            f"DP tiling generated with total quality {tiling.relevance_tokens:.3f} and {len(tiling.node_ids)} nodes."
        )

        return self._build_result(tiling, nodes)

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

        # Check if this is a leaf node (no left child = leaf due to left-balanced property)
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
            # Both children exist - split budget proportionally
            budget_l, budget_r = self._split_budget_proportionally(budget, node, scores)
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
        logger.info("Using async DP tiling generation")

        if not nodes or root_id not in nodes:
            return DPResult(Tiling.empty(), [], 0.0, {})

        self._nodes = nodes
        self._memo_cache = {}
        self._subtree_relevance_cache = {}

        root_node = nodes[root_id]
        tiling = await self._find_optimal_tiling_for_span(
            root_node, budget_tokens, scores
        )

        logger.info(
            f"Async DP tiling generated with total quality {tiling.relevance_tokens:.3f} and {len(tiling.node_ids)} nodes."
        )

        return self._build_result(tiling, nodes)

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

        # Check if this is a leaf node (no left child = leaf due to left-balanced property)
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
            # Both children exist - split budget proportionally
            budget_l, budget_r = self._split_budget_proportionally(budget, node, scores)

            # Decide whether to parallelize based on subtree size
            if self._should_parallelize(left_child, right_child):
                try:
                    # Run both recursive calls in parallel
                    left_task = asyncio.create_task(
                        self._find_optimal_tiling_for_span(left_child, budget_l, scores)
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
                        # Both results are successful Tiling objects
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
