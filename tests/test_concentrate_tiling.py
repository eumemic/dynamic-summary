"""Tests for the concentrate retrieval mode in GreedyTilingGenerator.

Concentrate mode is the top-k-over-the-tree's-leaves variant: it ranks the
verbatim leaves (height == 0) by query relevance, admits the highest-scoring
leaves until the token budget is exhausted, performs NO roll-up into summary
nodes, and emits the selected leaves in document order. It intentionally does
NOT guarantee whole-range coverage.

These tests are fully in-process: the tiling algorithm operates on plain
``nodes``/``scores`` mappings, so no server or sqlite backend is required.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragzoom.config import QueryConfig
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.greedy_tiling import GreedyTilingGenerator


@dataclass
class FakeNode:
    """Minimal in-memory TreeNode for tiling unit tests."""

    id: str
    document_id: str | None
    parent_id: str | None
    left_child_id: str | None
    right_child_id: str | None
    span_start: int
    span_end: int
    text: str
    token_count: int
    height: int
    is_pinned: bool | int
    preceding_neighbor_id: str | None
    following_neighbor_id: str | None
    level_index: int
    preceding_context: str | None
    preceding_context_summary: str | None
    embedding: bytes | None
    time_start: float | None
    time_end: float | None

    def is_leaf(self) -> bool:
        return self.height == 0

    def is_root(self) -> bool:
        return self.parent_id is None

    def get_depth(self) -> int:
        return 0


def _leaf(
    node_id: str,
    span_start: int,
    token_count: int,
    *,
    parent_id: str,
    level_index: int,
) -> FakeNode:
    return FakeNode(
        id=node_id,
        document_id="doc",
        parent_id=parent_id,
        left_child_id=None,
        right_child_id=None,
        span_start=span_start,
        span_end=span_start + 20,
        text=f"text-{node_id}",
        token_count=token_count,
        height=0,
        is_pinned=False,
        preceding_neighbor_id=None,
        following_neighbor_id=None,
        level_index=level_index,
        preceding_context=None,
        preceding_context_summary=None,
        embedding=None,
        time_start=None,
        time_end=None,
    )


def _inner(
    node_id: str,
    span_start: int,
    span_end: int,
    token_count: int,
    *,
    left_child_id: str,
    right_child_id: str,
    level_index: int,
) -> FakeNode:
    return FakeNode(
        id=node_id,
        document_id="doc",
        parent_id=None,
        left_child_id=left_child_id,
        right_child_id=right_child_id,
        span_start=span_start,
        span_end=span_end,
        text=f"summary-{node_id}",
        token_count=token_count,
        height=1,
        is_pinned=False,
        preceding_neighbor_id=None,
        following_neighbor_id=None,
        level_index=level_index,
        preceding_context=None,
        preceding_context_summary=None,
        embedding=None,
        time_start=None,
        time_end=None,
    )


def _build_tree() -> tuple[dict[str, TreeNode], list[str]]:
    """Two-tree forest with four leaves of 10 tokens each under two roots.

    Layout (span order): leaf-a, leaf-b under root-left; leaf-c, leaf-d under
    root-right. Each leaf is 10 tokens; each root is 8 tokens.
    """
    leaf_a = _leaf("leaf-a", 0, 10, parent_id="root-left", level_index=0)
    leaf_b = _leaf("leaf-b", 20, 10, parent_id="root-left", level_index=1)
    root_left = _inner(
        "root-left",
        0,
        40,
        8,
        left_child_id="leaf-a",
        right_child_id="leaf-b",
        level_index=0,
    )
    leaf_c = _leaf("leaf-c", 40, 10, parent_id="root-right", level_index=2)
    leaf_d = _leaf("leaf-d", 60, 10, parent_id="root-right", level_index=3)
    root_right = _inner(
        "root-right",
        40,
        80,
        8,
        left_child_id="leaf-c",
        right_child_id="leaf-d",
        level_index=1,
    )

    nodes: dict[str, TreeNode] = {
        n.id: n for n in (leaf_a, leaf_b, root_left, leaf_c, leaf_d, root_right)
    }
    root_ids = ["root-left", "root-right"]
    return nodes, root_ids


def test_concentrate_returns_only_leaves_ranked_by_relevance() -> None:
    """Concentrate admits the highest-scoring leaves, leaves only, no roll-up."""
    nodes, root_ids = _build_tree()
    # leaf-c most relevant, then leaf-a, then leaf-d, then leaf-b.
    scores = {
        "leaf-a": 0.8,
        "leaf-b": 0.1,
        "leaf-c": 0.9,
        "leaf-d": 0.4,
        "root-left": 0.0,
        "root-right": 0.0,
    }

    generator = GreedyTilingGenerator(QueryConfig(budget_tokens=20))
    # Budget 20 tokens => exactly two 10-token leaves: the two best, c and a.
    result = generator.find_optimal_tiling_over_roots(
        root_ids, 20, scores, nodes, mode="concentrate"
    )

    selected = result.tiling.node_ids
    # Only verbatim leaves, never summary/roll-up nodes.
    assert all(nodes[nid].height == 0 for nid in selected)
    assert "root-left" not in selected
    assert "root-right" not in selected
    # The two highest-scoring leaves admitted, emitted in document order.
    assert selected == ["leaf-a", "leaf-c"]


def test_concentrate_respects_budget_boundary() -> None:
    """A leaf that would exceed budget is excluded, not partially admitted."""
    nodes, root_ids = _build_tree()
    scores = {
        "leaf-a": 0.9,
        "leaf-b": 0.7,
        "leaf-c": 0.5,
        "leaf-d": 0.3,
        "root-left": 0.0,
        "root-right": 0.0,
    }

    # Budget 25: two leaves (20 tokens) fit; a third (30) would exceed -> excluded.
    generator = GreedyTilingGenerator(QueryConfig(budget_tokens=25))
    result = generator.find_optimal_tiling_over_roots(
        root_ids, 25, scores, nodes, mode="concentrate"
    )

    selected = result.tiling.node_ids
    assert selected == ["leaf-a", "leaf-b"]
    total_tokens = sum(nodes[nid].token_count for nid in selected)
    assert total_tokens <= 25


def test_concentrate_skips_unaffordable_leaf_but_admits_smaller_one() -> None:
    """A high-scoring oversized leaf is skipped; a smaller affordable one fits.

    Greedy admit-in-score-order: once budget is too tight for the next-best
    leaf, a later, cheaper leaf that still fits is admitted.
    """
    nodes, _ = _build_tree()
    # Make leaf-a huge so it cannot fit; leaf-b is small.
    big_leaf = _leaf("leaf-a", 0, 100, parent_id="root-left", level_index=0)
    nodes["leaf-a"] = big_leaf
    scores = {
        "leaf-a": 0.99,  # most relevant but too big
        "leaf-b": 0.5,
        "leaf-c": 0.4,
        "leaf-d": 0.3,
        "root-left": 0.0,
        "root-right": 0.0,
    }

    generator = GreedyTilingGenerator(QueryConfig(budget_tokens=15))
    result = generator.find_optimal_tiling_over_roots(
        ["root-left", "root-right"], 15, scores, nodes, mode="concentrate"
    )

    selected = result.tiling.node_ids
    # leaf-a (100 tokens) cannot fit in budget 15; leaf-b (10) is admitted.
    assert "leaf-a" not in selected
    assert selected == ["leaf-b"]


def test_concentrate_differs_from_coverage_on_same_tree() -> None:
    """Concentrate and coverage produce different, characterizable tilings.

    With budget that forces compression, coverage rolls leaves up into summary
    roots (covering the whole range), while concentrate keeps only the top
    verbatim leaves (no roll-up, partial range).
    """
    nodes, root_ids = _build_tree()
    scores = {
        "leaf-a": 0.2,
        "leaf-b": 0.2,
        "leaf-c": 0.9,
        "leaf-d": 0.2,
        "root-left": 0.2,
        "root-right": 0.5,
    }

    # Budget 16 tokens: too small for any two 10-token leaves together, but the
    # two 8-token roots (16 total) fit -> coverage rolls up to the roots.
    coverage = GreedyTilingGenerator(QueryConfig(budget_tokens=16))
    coverage_result = coverage.find_optimal_tiling_over_roots(
        root_ids, 16, scores, nodes, mode="coverage"
    )
    concentrate = GreedyTilingGenerator(QueryConfig(budget_tokens=16))
    concentrate_result = concentrate.find_optimal_tiling_over_roots(
        root_ids, 16, scores, nodes, mode="concentrate"
    )

    coverage_ids = set(coverage_result.tiling.node_ids)
    concentrate_ids = set(concentrate_result.tiling.node_ids)

    # Coverage rolls up to summary roots; concentrate keeps the single best leaf.
    assert coverage_ids == {"root-left", "root-right"}
    assert concentrate_ids == {"leaf-c"}
    assert coverage_ids != concentrate_ids
    # Concentrate emits only verbatim leaves.
    assert all(nodes[nid].height == 0 for nid in concentrate_result.tiling.node_ids)
    # Coverage emits only summary (rolled-up) nodes here.
    assert all(nodes[nid].height > 0 for nid in coverage_result.tiling.node_ids)


def test_concentrate_no_whole_range_coverage_guarantee() -> None:
    """Concentrate does not span the full document range when budget is tight."""
    nodes, root_ids = _build_tree()
    scores = {
        "leaf-a": 0.1,
        "leaf-b": 0.1,
        "leaf-c": 0.95,
        "leaf-d": 0.1,
        "root-left": 0.0,
        "root-right": 0.0,
    }

    generator = GreedyTilingGenerator(QueryConfig(budget_tokens=10))
    result = generator.find_optimal_tiling_over_roots(
        root_ids, 10, scores, nodes, mode="concentrate"
    )

    selected = result.tiling.node_ids
    assert selected == ["leaf-c"]
    # The selected span (40-60) does not cover the document start (0) or end (80).
    covered_start = min(nodes[nid].span_start for nid in selected)
    covered_end = max(nodes[nid].span_end for nid in selected)
    assert covered_start > 0
    assert covered_end < 80


def test_concentrate_coverage_map_marks_selected_leaves() -> None:
    """coverage_map reflects exactly the selected leaves in concentrate mode."""
    nodes, root_ids = _build_tree()
    scores = {
        "leaf-a": 0.8,
        "leaf-b": 0.1,
        "leaf-c": 0.9,
        "leaf-d": 0.2,
        "root-left": 0.0,
        "root-right": 0.0,
    }

    generator = GreedyTilingGenerator(QueryConfig(budget_tokens=20))
    result = generator.find_optimal_tiling_over_roots(
        root_ids, 20, scores, nodes, mode="concentrate"
    )

    selected = set(result.tiling.node_ids)
    assert set(result.coverage_map) == selected
    assert all(result.coverage_map[nid] for nid in selected)
    # node_infos line up with the selected leaves.
    assert {info.node_id for info in result.node_infos} == selected


def test_coverage_mode_is_default_and_unchanged() -> None:
    """Omitting mode preserves the existing coverage behavior."""
    nodes, root_ids = _build_tree()
    scores = {node_id: 1.0 for node_id in nodes}

    generator = GreedyTilingGenerator(QueryConfig(budget_tokens=64))
    explicit = generator.find_optimal_tiling_over_roots(
        root_ids, 64, scores, nodes, mode="coverage"
    )
    default = generator.find_optimal_tiling_over_roots(root_ids, 64, scores, nodes)

    assert explicit.tiling.node_ids == default.tiling.node_ids
