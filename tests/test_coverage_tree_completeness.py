"""SQLite-based tests for coverage tree completeness requirements.

This file converts the core coverage tree completeness tests from
test_coverage_tree_completeness.py to use the real in-memory SQLite backend
providing higher fidelity testing of the
coverage tree validation functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore

if TYPE_CHECKING:
    from ragzoom.greedy_tiling import GreedyTilingGenerator
    from ragzoom.retrieve import Retriever


class TestCoverageTreeCompletenessSQLite:
    """Tests that ensure coverage trees work correctly with forest of perfect binary trees."""

    @pytest.fixture
    def doc_store(self, storage_backend: StorageBackend) -> DocumentStore:
        """Create a document-scoped store for test-doc."""
        doc_store = storage_backend.for_document("test-doc")
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        return doc_store

    @pytest.fixture
    def setup_incomplete_tree(
        self, doc_store: DocumentStore
    ) -> tuple[IndexConfig, DocumentStore, Retriever, GreedyTilingGenerator]:
        """Set up a system with a tree that will produce incomplete coverage."""
        index_config = IndexConfig.load(
            target_chunk_tokens=100, preceding_summary_budget_tokens=50
        )
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Use the doc_store fixture (already scoped to test-doc)
        document_store = doc_store

        # Create a simple tree structure directly in SQLite:
        #         root
        #        /    \
        #      P1      P2
        #     /  \    /  \
        #    L1  L2  L3  L4

        nodes: list[NodeDataDict] = [
            # Leaf nodes
            {
                "node_id": "L1",
                "text": "Chapter 1 content",
                "span_start": 0,
                "span_end": 20,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "L2",
                "text": "Chapter 2 content",
                "span_start": 20,
                "span_end": 40,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "L3",
                "text": "Chapter 3 content",
                "span_start": 40,
                "span_end": 60,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "L4",
                "text": "Chapter 4 content",
                "span_start": 60,
                "span_end": 80,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 0,
                "level_index": 0,
            },
            # Parent nodes
            {
                "node_id": "P1",
                "text": "Summary of chapters 1-2",
                "span_start": 0,
                "span_end": 40,
                "document_id": "test-doc",
                "height": 1,
                "token_count": 10,
                "level_index": 0,
                "left_child_id": "L1",
                "right_child_id": "L2",
            },
            {
                "node_id": "P2",
                "text": "Summary of chapters 3-4",
                "span_start": 40,
                "span_end": 80,
                "document_id": "test-doc",
                "height": 1,
                "token_count": 10,
                "level_index": 0,
                "left_child_id": "L3",
                "right_child_id": "L4",
            },
            # Root node
            {
                "node_id": "root",
                "text": "Full document summary",
                "span_start": 0,
                "span_end": 80,
                "document_id": "test-doc",
                "height": 2,
                "token_count": 20,
                "level_index": 0,
                "left_child_id": "P1",
                "right_child_id": "P2",
            },
        ]

        document_store.nodes.add_batch(nodes)

        # Update parent references
        document_store.nodes.update_parent_references_batch(
            [
                ("L1", "P1"),
                ("L2", "P1"),
                ("L3", "P2"),
                ("L4", "P2"),
                ("P1", "root"),
                ("P2", "root"),
            ]
        )

        from ragzoom.vector_factory import create_vector_index
        from tests.utils import create_retriever

        vi = create_vector_index(
            "python", "sqlite:///:memory:", query_config.embedding_model
        )
        retriever = create_retriever(
            query_config,
            doc_store,
            document_id="test-doc",
            api_key=operational_config.openai_api_key.get_secret_value(),
            vector_index=vi,
        )
        tiling_generator = retriever.tiling_generator

        return index_config, document_store, retriever, tiling_generator

    def test_partial_coverage_map_handling(
        self,
        setup_incomplete_tree: tuple[
            IndexConfig, DocumentStore, Retriever, GreedyTilingGenerator
        ],
    ) -> None:
        """Test that left-balanced trees with single children are handled correctly."""
        index_config, document_store, retriever, tiling_generator = (
            setup_incomplete_tree
        )

        # Simulate what happens with --num-seeds 1: only L3 is selected
        # This creates a partial coverage map where P2's subtree is incomplete
        coverage_map = {"L3": True}

        # Add ancestors (this is what current retriever does)
        current_id = "L3"
        node = document_store.nodes.get_node(current_id)
        while node and node.parent_id:
            parent = document_store.nodes.get_node(node.parent_id)
            if parent:
                coverage_map[parent.id] = True
                current_id = parent.id
                node = parent
            else:
                break

        # Load nodes from coverage map
        nodes = {}
        for node_id in coverage_map:
            node = document_store.nodes.get_node(node_id)
            if node:
                nodes[node_id] = node

        # Coverage map has L3, P2, and root, but not L4 (sibling of L3)
        assert "L3" in nodes
        assert "P2" in nodes
        assert "root" in nodes
        assert "L4" not in nodes  # Not in coverage map (sibling not selected)

        # The DP algorithm handles partial coverage maps correctly
        # Provide scores for all nodes in coverage to ensure L3 is selected
        scores = {node_id: 0.1 for node_id in nodes}  # Base score for all
        scores["L3"] = 1.0  # L3 has high relevance

        result = tiling_generator.find_optimal_tiling_over_roots(
            root_ids=["root"],
            budget_tokens=1000,
            scores=scores,
            nodes=nodes,
        )

        # The DP algorithm makes the optimal choice based on relevance and budget
        assert result.tiling.node_ids  # Should have some result
        assert result.total_quality >= 0  # Should have non-negative quality

    def test_complete_coverage_tree_works(
        self,
        setup_incomplete_tree: tuple[
            IndexConfig, DocumentStore, Retriever, GreedyTilingGenerator
        ],
    ) -> None:
        """Test that complete coverage trees work correctly."""
        index_config, document_store, retriever, tiling_generator = (
            setup_incomplete_tree
        )

        # Create a complete coverage tree by including all nodes
        nodes = {}
        for node_id in ["root", "P1", "P2", "L1", "L2", "L3", "L4"]:
            node = document_store.nodes.get_node(node_id)
            if node:
                nodes[node_id] = node

        # This should work without errors
        result = tiling_generator.find_optimal_tiling_over_roots(
            root_ids=["root"],
            budget_tokens=1000,
            scores={"L3": 1.0},  # L3 is most relevant
            nodes=nodes,
        )

        # Should produce a valid tiling
        assert result.tiling is not None
        assert len(result.tiling.node_ids) > 0

    def test_coverage_tree_with_siblings_included(
        self,
        setup_incomplete_tree: tuple[
            IndexConfig, DocumentStore, Retriever, GreedyTilingGenerator
        ],
    ) -> None:
        """Test the correct way to build coverage tree with siblings."""
        index_config, document_store, retriever, tiling_generator = (
            setup_incomplete_tree
        )

        # Start with selected node
        selected_nodes = ["L3"]
        coverage_nodes = set(selected_nodes)

        # Build complete coverage tree
        # Step 1: Add all ancestors
        for node_id in selected_nodes:
            current_id = node_id
            while current_id:
                coverage_nodes.add(current_id)
                node = document_store.nodes.get_node(current_id)
                if node and node.parent_id:
                    current_id = node.parent_id
                else:
                    break

        # Step 2: For each node in coverage, ensure both children are included
        # This ensures completeness
        nodes_to_check = list(coverage_nodes)
        while nodes_to_check:
            node_id = nodes_to_check.pop(0)
            node = document_store.nodes.get_node(node_id)
            if node:
                # If this node has children, both must be in coverage
                if node.left_child_id:
                    if node.left_child_id not in coverage_nodes:
                        coverage_nodes.add(node.left_child_id)
                        nodes_to_check.append(node.left_child_id)
                if node.right_child_id:
                    if node.right_child_id not in coverage_nodes:
                        coverage_nodes.add(node.right_child_id)
                        nodes_to_check.append(node.right_child_id)

        # Load all nodes
        nodes = {}
        for node_id in coverage_nodes:
            node = document_store.nodes.get_node(node_id)
            if node:
                nodes[node_id] = node

        # Should have complete subtree
        assert "L3" in nodes
        assert "L4" in nodes  # Sibling included
        assert "P2" in nodes
        assert "P1" in nodes  # Sibling of P2
        assert "L1" in nodes  # Children of P1
        assert "L2" in nodes
        assert "root" in nodes

        # This should work without errors
        result = tiling_generator.find_optimal_tiling_over_roots(
            root_ids=["root"],
            budget_tokens=1000,
            scores={"L3": 1.0},
            nodes=nodes,
        )

        assert result.tiling is not None

    def test_coverage_tree_completeness_validation(
        self,
        setup_incomplete_tree: tuple[
            IndexConfig, DocumentStore, Retriever, GreedyTilingGenerator
        ],
    ) -> None:
        """Test validation logic for coverage tree completeness."""
        index_config, document_store, retriever, tiling_generator = (
            setup_incomplete_tree
        )

        # Test incomplete coverage tree (missing sibling)
        incomplete_nodes = {}
        for node_id in ["root", "P2", "L3"]:  # Missing L4, P1, L1, L2
            node = document_store.nodes.get_node(node_id)
            if node:
                incomplete_nodes[node_id] = node

        # This represents a partial coverage map where P2's subtree is incomplete
        # DP algorithm should handle this gracefully
        scores = {"L3": 1.0, "P2": 0.5, "root": 0.3}

        result = tiling_generator.find_optimal_tiling_over_roots(
            root_ids=["root"],
            budget_tokens=1000,
            scores=scores,
            nodes=incomplete_nodes,
        )

        # Should produce a valid result without errors
        assert result.tiling is not None
        assert result.total_quality >= 0

    def test_single_leaf_coverage_tree(
        self,
        setup_incomplete_tree: tuple[
            IndexConfig, DocumentStore, Retriever, GreedyTilingGenerator
        ],
    ) -> None:
        """Test coverage tree with single leaf node and its ancestors."""
        index_config, document_store, retriever, tiling_generator = (
            setup_incomplete_tree
        )

        # Build minimal coverage tree with just one leaf and ancestors
        coverage_nodes = set()

        # Start with L1
        current_id = "L1"
        while current_id:
            coverage_nodes.add(current_id)
            node = document_store.nodes.get_node(current_id)
            if node and node.parent_id:
                current_id = node.parent_id
            else:
                break

        # Load nodes
        nodes = {}
        for node_id in coverage_nodes:
            node = document_store.nodes.get_node(node_id)
            if node:
                nodes[node_id] = node

        # Should have L1, P1, and root
        assert "L1" in nodes
        assert "P1" in nodes
        assert "root" in nodes
        assert "L2" not in nodes  # Sibling not included

        # Test DP algorithm with this minimal coverage
        scores = {"L1": 1.0, "P1": 0.5, "root": 0.3}

        result = tiling_generator.find_optimal_tiling_over_roots(
            root_ids=["root"],
            budget_tokens=1000,
            scores=scores,
            nodes=nodes,
        )

        # Should handle incomplete subtree gracefully
        assert result.tiling is not None
        assert result.total_quality >= 0
