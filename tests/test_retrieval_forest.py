"""Forest-specific retrieval invariants for coverage and DP tiling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import pytest

from ragzoom.config import QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.document_store import DocumentStore
from ragzoom.dynamic_tiling import DPResult, DynamicTilingGenerator
from ragzoom.greedy_tiling import GreedyTilingGenerator
from ragzoom.retrieval.coverage_builder import CoverageBuilder


@dataclass
class ForestNode:
    node_id: str
    text: str
    span_start: int
    span_end: int
    parent_id: str | None
    left_child_id: str | None
    right_child_id: str | None
    token_count: int
    height: int
    level_index: int


def _seed_forest(document_store: DocumentStore) -> list[str]:
    """Seed the document store with a simple two-tree forest."""
    nodes = [
        ForestNode(
            node_id="leaf-left",
            text="Left leaf text",
            span_start=0,
            span_end=20,
            parent_id="root-left",
            left_child_id=None,
            right_child_id=None,
            token_count=5,
            height=0,
            level_index=0,
        ),
        ForestNode(
            node_id="leaf-right",
            text="Right leaf text",
            span_start=20,
            span_end=40,
            parent_id="root-left",
            left_child_id=None,
            right_child_id=None,
            token_count=5,
            height=0,
            level_index=1,
        ),
        ForestNode(
            node_id="root-left",
            text="Left subtree summary",
            span_start=0,
            span_end=40,
            parent_id=None,
            left_child_id="leaf-left",
            right_child_id="leaf-right",
            token_count=10,
            height=1,
            level_index=0,
        ),
        ForestNode(
            node_id="leaf-mid",
            text="Mid leaf text",
            span_start=40,
            span_end=60,
            parent_id="root-right",
            left_child_id=None,
            right_child_id=None,
            token_count=5,
            height=0,
            level_index=2,
        ),
        ForestNode(
            node_id="leaf-far",
            text="Far leaf text",
            span_start=60,
            span_end=80,
            parent_id="root-right",
            left_child_id=None,
            right_child_id=None,
            token_count=5,
            height=0,
            level_index=3,
        ),
        ForestNode(
            node_id="root-right",
            text="Right subtree summary",
            span_start=40,
            span_end=80,
            parent_id=None,
            left_child_id="leaf-mid",
            right_child_id="leaf-far",
            token_count=10,
            height=1,
            level_index=1,
        ),
    ]

    payload: list[dict[str, str | int | float | bool | list[float] | None]] = []
    for node in nodes:
        payload_item: dict[str, str | int | float | bool | list[float] | None] = {
            "node_id": node.node_id,
            "text": node.text,
            "span_start": node.span_start,
            "span_end": node.span_end,
            "parent_id": node.parent_id,
            "left_child_id": node.left_child_id,
            "right_child_id": node.right_child_id,
            "document_id": document_store.document_id,
            "token_count": node.token_count,
            "height": node.height,
            "level_index": node.level_index,
            "embedding": [0.0, 0.0, 0.0],
        }
        payload.append(payload_item)

    document_store.nodes.add_batch(payload)  # type: ignore[arg-type]

    # Update parent references for children (None entries become roots)
    updates: list[tuple[str, str | None]] = []
    for node in nodes:
        if node.parent_id is not None:
            updates.append((node.node_id, node.parent_id))
    if updates:
        document_store.nodes.update_parent_references_batch(updates)

    return [node.node_id for node in nodes if node.parent_id is None]


@pytest.fixture
def forest_store(storage_backend: StorageBackend) -> DocumentStore:
    storage_backend.clear_document("forest-test")
    store = storage_backend.for_document("forest-test")
    store.set_metadata(
        file_path="forest.txt",
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )
    return store


def test_forest_root_injection(forest_store: DocumentStore) -> None:
    root_ids = _seed_forest(forest_store)
    coverage_builder = CoverageBuilder(forest_store)

    left_leaf = "leaf-left"
    coverage = coverage_builder.build_complete_coverage_map([left_leaf])
    assert root_ids, "Seed helper should produce forest roots"
    for root_id in root_ids:
        assert root_id in coverage, "Coverage must include every root in the forest"


def test_forest_roots_included_with_empty_selection(
    forest_store: DocumentStore,
) -> None:
    """Roots should be included in coverage even with no selections."""
    root_ids = _seed_forest(forest_store)
    coverage_builder = CoverageBuilder(forest_store)

    # Empty selection - roots should still be included for forest support
    result = coverage_builder.build_complete_coverage([])
    assert root_ids, "Seed helper should produce forest roots"
    for root_id in root_ids:
        assert (
            root_id in result.coverage_map
        ), "Roots must be in coverage even with empty selection"


def test_dp_accepts_multiple_roots(forest_store: DocumentStore) -> None:
    root_ids = _seed_forest(forest_store)
    nodes = {node.id: node for node in forest_store.nodes.get_all()}

    dp = DynamicTilingGenerator(QueryConfig(budget_tokens=64))
    func = getattr(dp, "find_optimal_tiling_over_roots", None)
    assert callable(func), "DynamicTilingGenerator must support forest root tiling"

    solver = cast(
        Callable[[Sequence[str], int, dict[str, float], dict[str, TreeNode]], DPResult],
        func,
    )
    scores = {node_id: 1.0 for node_id in nodes}

    result = solver(root_ids, 64, scores, nodes)
    assert result.tiling.node_ids
    assert set(result.coverage_map) >= set(root_ids)


def test_forest_budget_insufficient_returns_empty(forest_store: DocumentStore) -> None:
    root_ids = _seed_forest(forest_store)
    nodes = {node.id: node for node in forest_store.nodes.get_all()}

    dp = DynamicTilingGenerator(QueryConfig(budget_tokens=10))
    scores = {node_id: 1.0 for node_id in nodes}

    result = dp.find_optimal_tiling_over_roots(root_ids, 10, scores, nodes)
    assert not result.tiling.node_ids


def test_greedy_handles_multiple_roots(forest_store: DocumentStore) -> None:
    root_ids = _seed_forest(forest_store)
    nodes = {node.id: node for node in forest_store.nodes.get_all()}
    scores = {node_id: 1.0 for node_id in nodes}

    greedy = GreedyTilingGenerator(
        QueryConfig(budget_tokens=64, tiling_strategy="greedy")
    )
    result = greedy.find_optimal_tiling_over_roots(root_ids, 64, scores, nodes)

    assert result.tiling.node_ids
    # Should include at least one node from each root subtree
    seen_roots = set(result.coverage_map) & set(root_ids)
    assert seen_roots == set(root_ids)
