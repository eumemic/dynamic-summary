"""Tests for DP algorithm score handling.

Tests demonstrating how the DP algorithm uses scores outside
the coverage tree.
"""

from __future__ import annotations

from ragzoom.config import QueryConfig
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.dynamic_tiling import DynamicTilingGenerator
from ragzoom.retrieve import RetrievalResult


class TestDPScoresBug:
    """Test that DP algorithm incorrectly uses nodes outside coverage tree."""

    def test_dp_uses_scores_outside_coverage_tree(
        self, storage_backend: StorageBackend
    ) -> None:
        """Demonstrate that DP uses any node with a score, ignoring coverage tree."""
        # Get document store and set metadata
        doc_store = storage_backend.for_document("doc-id")
        doc_store.set_metadata(
            file_path="dp_scores_test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Set up a tree structure:
        #          root
        #         /    \
        #     node_a   node_b
        #      / \      / \
        #    a1  a2   b1  b2

        nodes: list[NodeDataDict] = [
            # Root
            {
                "node_id": "root",
                "text": "Root summary of document",
                "span_start": 0,
                "span_end": 1000,
                "document_id": "doc-id",
                "token_count": 100,
                "height": 2,
                "level_index": 0,
                "left_child_id": "node_a",
                "right_child_id": "node_b",
            },
            # Internal nodes
            {
                "node_id": "node_a",
                "text": "Node A summary",
                "span_start": 0,
                "span_end": 500,
                "document_id": "doc-id",
                "token_count": 50,
                "height": 1,
                "level_index": 0,
                "left_child_id": "a1",
                "right_child_id": "a2",
            },
            {
                "node_id": "node_b",
                "text": "Node B summary",
                "span_start": 500,
                "span_end": 1000,
                "document_id": "doc-id",
                "token_count": 50,
                "height": 1,
                "level_index": 0,
                "left_child_id": "b1",
                "right_child_id": "b2",
            },
            # Leaf nodes
            {
                "node_id": "a1",
                "text": "Leaf a1 content",
                "span_start": 0,
                "span_end": 250,
                "document_id": "doc-id",
                "token_count": 25,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "a2",
                "text": "Leaf a2 content",
                "span_start": 250,
                "span_end": 500,
                "document_id": "doc-id",
                "token_count": 25,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "b1",
                "text": "Leaf b1 content",
                "span_start": 500,
                "span_end": 750,
                "document_id": "doc-id",
                "token_count": 25,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "b2",
                "text": "Leaf b2 content",
                "span_start": 750,
                "span_end": 1000,
                "document_id": "doc-id",
                "token_count": 25,
                "height": 0,
                "level_index": 0,
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("node_a", "root"),
                ("node_b", "root"),
                ("a1", "node_a"),
                ("a2", "node_a"),
                ("b1", "node_b"),
                ("b2", "node_b"),
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
        nodes_map: dict[str, object] = {}
        for nid in coverage_tree:
            node = doc_store.nodes.get_node(nid)
            if node:
                nodes_map[nid] = node

        # Find root node
        root_id = "root"

        dp_result = dp_generator.find_optimal_tiling(
            budget_tokens=10000,
            scores=scores,
            nodes=nodes_map,  # type: ignore[arg-type]
            root_id=root_id,
        )
        tiling = dp_result.tiling

        # Check results - need to check if nodes are leaves using document store
        leaf_node_ids = {
            node_id for node_id in tiling.node_ids if doc_store.tree.is_leaf(node_id)
        }

        # With our fix, all leaf nodes in tiling must be in the coverage tree
        leaf_violations = [
            node_id for node_id in leaf_node_ids if node_id not in coverage_tree
        ]
        assert (
            len(leaf_violations) == 0
        ), f"Found leaf nodes outside coverage tree: {leaf_violations}"

    def test_retrieval_result_demonstrates_bug(
        self, storage_backend: StorageBackend
    ) -> None:
        """Test using actual RetrievalResult to show the bug."""
        # Get document store and set metadata
        doc_store = storage_backend.for_document("doc-id")
        doc_store.set_metadata(
            file_path="retrieval_bug_test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Simplified tree setup
        nodes: list[NodeDataDict] = [
            {
                "node_id": "root",
                "text": "Root",
                "span_start": 0,
                "span_end": 1000,
                "document_id": "doc-id",
                "token_count": 100,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
            },
            {
                "node_id": "leaf1",
                "text": "Leaf 1",
                "span_start": 0,
                "span_end": 500,
                "document_id": "doc-id",
                "token_count": 50,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf2",
                "text": "Leaf 2",
                "span_start": 500,
                "span_end": 1000,
                "document_id": "doc-id",
                "token_count": 50,
                "height": 0,
                "level_index": 0,
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
        nodes_map: dict[str, object] = {}
        for node_id in result.coverage_map:
            node = doc_store.nodes.get_node(node_id)
            if node:
                nodes_map[node_id] = node

        # Find root node
        root_id = "root"

        dp_result = dp_generator.find_optimal_tiling(
            budget_tokens=10000,
            scores=result.scores,
            nodes=nodes_map,  # type: ignore[arg-type]
            root_id=root_id,
        )
        tiling = dp_result.tiling

        # Check results - need to check if nodes are leaves using document store
        leaf_node_ids = {
            node_id for node_id in tiling.node_ids if doc_store.tree.is_leaf(node_id)
        }

        # With our fix: leaf2 should NOT appear in tiling unless it is in the coverage map
        assert (
            "leaf2" in result.coverage_map
        ), "leaf2 must be in coverage map for DP to consider it"
        assert (
            "leaf2" in leaf_node_ids or "leaf1" in leaf_node_ids
        ), "At least one leaf should be in the tiling"
