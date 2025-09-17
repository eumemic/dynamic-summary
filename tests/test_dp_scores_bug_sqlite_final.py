"""SQLite-based tests for DP algorithm score handling bugs.

These tests ensure the DP tiling algorithm only uses scores from nodes within
the coverage tree using the real in-memory
SQLite backend.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.config import QueryConfig
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.document_store import DocumentStore
from ragzoom.dynamic_tiling import DynamicTilingGenerator
from ragzoom.retrieve import RetrievalResult


@pytest.mark.usefixtures("sqlite_backend")
class TestDPScoresBugSQLite:
    """Test that DP algorithm correctly respects coverage tree boundaries."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("doc1")

    def test_dp_uses_scores_outside_coverage_tree(
        self, doc_store: DocumentStore
    ) -> None:
        """Demonstrate that DP uses any node with a score, ignoring coverage tree."""
        # Create a tree structure:
        #          root
        #         /    \
        #     node_a   node_b
        #      / \      / \
        #    a1  a2   b1  b2

        # Seed nodes with explicit token counts for DP cost control
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            # Leaf nodes
            {
                "node_id": "a1",
                "text": "Leaf a1 content",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 250,
                "document_id": "doc1",
                "token_count": 25,
                "height": 0,
                "parent_id": "node_a",
                "path": "00",
            },
            {
                "node_id": "a2",
                "text": "Leaf a2 content",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 250,
                "span_end": 500,
                "document_id": "doc1",
                "token_count": 25,
                "height": 0,
                "parent_id": "node_a",
                "path": "01",
            },
            {
                "node_id": "b1",
                "text": "Leaf b1 content",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 500,
                "span_end": 750,
                "document_id": "doc1",
                "token_count": 25,
                "height": 0,
                "parent_id": "node_b",
                "path": "10",
            },
            {
                "node_id": "b2",
                "text": "Leaf b2 content",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 750,
                "span_end": 1000,
                "document_id": "doc1",
                "token_count": 25,
                "height": 0,
                "parent_id": "node_b",
                "path": "11",
            },
            # Internal nodes
            {
                "node_id": "node_a",
                "text": "Node A summary",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 500,
                "document_id": "doc1",
                "token_count": 50,
                "height": 1,
                "left_child_id": "a1",
                "right_child_id": "a2",
                "parent_id": "root",
                "path": "0",
            },
            {
                "node_id": "node_b",
                "text": "Node B summary",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 500,
                "span_end": 1000,
                "document_id": "doc1",
                "token_count": 50,
                "height": 1,
                "left_child_id": "b1",
                "right_child_id": "b2",
                "parent_id": "root",
                "path": "1",
            },
            # Root
            {
                "node_id": "root",
                "text": "Root summary of document",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 1000,
                "document_id": "doc1",
                "token_count": 100,
                "height": 2,
                "left_child_id": "node_a",
                "right_child_id": "node_b",
                "path": "",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("a1", "node_a"),
                ("a2", "node_a"),
                ("b1", "node_b"),
                ("b2", "node_b"),
                ("node_a", "root"),
                ("node_b", "root"),
            ]
        )

        # Create config and DP generator
        query_config = QueryConfig(budget_tokens=10000)  # Large budget
        dp_generator = DynamicTilingGenerator(query_config)

        # Pass in a full coverage tree (all nodes)
        coverage_tree = {"a1", "a2", "b1", "b2", "node_a", "node_b", "root"}

        scores = {
            "a1": 0.9,  # Selected node
            "a2": 0.8,  # Sibling
            "b1": 0.85,  # Sibling
            "b2": 0.7,  # Sibling
            "node_a": 0.5,
            "node_b": 0.5,
            "root": 0.3,
        }

        # Load nodes from coverage map
        nodes_map: dict[str, TreeNode] = {}
        for nid in coverage_tree:
            node = doc_store.nodes.get_node(nid)
            if node:
                nodes_map[nid] = node

        # Find root node
        root_id = "root"

        dp_result = dp_generator.find_optimal_tiling(
            budget_tokens=10000,
            scores=scores,
            nodes=nodes_map,
            root_id=root_id,
        )
        tiling = dp_result.tiling

        # Check results
        leaf_node_ids = {
            node_id
            for node_id in tiling.node_ids
            if (node := doc_store.nodes.get_node(node_id))
            and node.left_child_id is None
            and node.right_child_id is None
        }

        # With our fix, all leaf nodes in tiling must be in the coverage tree
        leaf_violations = [
            node_id for node_id in leaf_node_ids if node_id not in coverage_tree
        ]
        assert (
            len(leaf_violations) == 0
        ), f"Found leaf nodes outside coverage tree: {leaf_violations}"

    def test_retrieval_result_demonstrates_bug(self, doc_store: DocumentStore) -> None:
        """Test using actual RetrievalResult to show the bug."""
        # Seed nodes with simple tree structure
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "leaf1",
                "text": "Leaf 1",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 500,
                "document_id": "doc1",
                "token_count": 50,
                "height": 0,
                "parent_id": "root",
                "path": "0",
            },
            {
                "node_id": "leaf2",
                "text": "Leaf 2",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 500,
                "span_end": 1000,
                "document_id": "doc1",
                "token_count": 50,
                "height": 0,
                "parent_id": "root",
                "path": "1",
            },
            {
                "node_id": "root",
                "text": "Root",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 1000,
                "document_id": "doc1",
                "token_count": 100,
                "height": 1,
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
                "path": "",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("leaf1", "root"), ("leaf2", "root")]
        )

        query_config = QueryConfig(budget_tokens=10000)
        dp_generator = DynamicTilingGenerator(query_config)

        # Pass in a full coverage tree (root and both leaves)
        result = RetrievalResult(
            node_ids=["leaf1"],  # Only 1 selected
            scores={
                "leaf1": 0.9,  # Selected
                "leaf2": 0.8,  # Sibling
                "root": 0.5,
            },
            coverage_map={"leaf1": True, "leaf2": True, "root": True},
            tiling=None,
        )

        # Load nodes from coverage map
        nodes_map: dict[str, TreeNode] = {}
        for node_id in result.coverage_map:
            node = doc_store.nodes.get_node(node_id)
            if node:
                nodes_map[node_id] = node

        # Find root node
        root_id = "root"

        dp_result = dp_generator.find_optimal_tiling(
            budget_tokens=10000,
            scores=result.scores,
            nodes=nodes_map,
            root_id=root_id,
        )
        tiling = dp_result.tiling

        # Check results
        leaf_node_ids = {
            node_id
            for node_id in tiling.node_ids
            if (node := doc_store.nodes.get_node(node_id))
            and node.left_child_id is None
            and node.right_child_id is None
        }

        # With our fix: leaf2 should NOT appear in tiling unless it is in the coverage map
        assert (
            "leaf2" in result.coverage_map
        ), "leaf2 must be in coverage map for DP to consider it"
        assert (
            "leaf2" in leaf_node_ids or "leaf1" in leaf_node_ids
        ), "At least one leaf should be in the tiling"
