"""Greedy tiling keeps top seeds that fit the budget."""

from dataclasses import dataclass
from typing import cast

from ragzoom.config import QueryConfig
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.greedy_tiling import GreedyTilingGenerator


@dataclass
class FakeNode:
    """Minimal TreeNode stand-in for tiling tests."""

    id: str
    token_count: int
    span_start: int
    span_end: int
    height: int
    level_index: int
    document_id: str | None = None
    parent_id: str | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None
    text: str = "x"
    is_pinned: bool = False
    preceding_neighbor_id: str | None = None
    following_neighbor_id: str | None = None

    def is_root(self) -> bool:
        return self.parent_id is None


def test_greedy_keeps_high_score_seeds_when_they_fit_budget() -> None:
    """Greedy roll-up should keep the top-scoring leaves when the budget fits them."""

    # All nodes 200 tokens
    root = FakeNode("root", 200, 0, 1000, 2, 0)
    l1 = FakeNode("l1", 200, 0, 100, 0, 0, parent_id="L")
    l2 = FakeNode("l2", 200, 100, 200, 0, 1, parent_id="L")
    left = FakeNode(
        "L",
        200,
        0,
        200,
        1,
        0,
        parent_id="root",
        left_child_id="l1",
        right_child_id="l2",
    )
    right = FakeNode("R", 200, 200, 300, 0, 1, parent_id="root")
    root.left_child_id = left.id
    root.right_child_id = right.id

    nodes = {n.id: cast(TreeNode, n) for n in (root, left, l1, l2, right)}
    # Top scores on the leaves we want to keep
    scores = {"l1": 10.0, "l2": 9.0, "L": 0.5, "R": 8.0, "root": 0.0}

    gen = GreedyTilingGenerator(QueryConfig(tiling_strategy="greedy"))
    result = gen.find_optimal_tiling_over_roots([root.id], 600, scores, nodes)

    assert result.tiling.node_ids == ["l1", "l2", "R"]
    assert sum(nodes[nid].token_count for nid in result.tiling.node_ids) == 600


def test_greedy_prunes_by_min_quality_loss_per_token() -> None:
    """Greedy should drop the pair with the smallest relevance loss per saved token."""

    # Balanced binary tree, all nodes 200 tokens
    root = FakeNode("root", 200, 0, 800, 2, 0)
    a = FakeNode("A", 200, 0, 100, 0, 0, parent_id="L")
    b = FakeNode("B", 200, 100, 200, 0, 1, parent_id="L")
    left = FakeNode(
        "L",
        200,
        0,
        200,
        1,
        0,
        parent_id="root",
        left_child_id="A",
        right_child_id="B",
    )
    c = FakeNode("C", 200, 200, 300, 0, 0, parent_id="R")
    d = FakeNode("D", 200, 300, 400, 0, 1, parent_id="R")
    right = FakeNode(
        "R",
        200,
        200,
        400,
        1,
        1,
        parent_id="root",
        left_child_id="C",
        right_child_id="D",
    )
    root.left_child_id = left.id
    root.right_child_id = right.id

    nodes = {n.id: cast(TreeNode, n) for n in (root, left, right, a, b, c, d)}

    # Left children have high relevance but their parent is almost as good (small loss).
    # Right children are low relevance and roll-up would lose proportionally more.
    scores = {
        "A": 10.0,
        "B": 10.0,
        "L": 19.0,
        "C": 1.0,
        "D": 1.0,
        "R": 0.0,
        "root": 0.0,
    }

    gen = GreedyTilingGenerator(QueryConfig(tiling_strategy="greedy"))
    result = gen.find_optimal_tiling_over_roots([root.id], 600, scores, nodes)

    # Should roll up A/B (smallest quality loss per saved token) not C/D.
    assert result.tiling.node_ids == ["L", "C", "D"]
    assert sum(nodes[nid].token_count for nid in result.tiling.node_ids) == 600
