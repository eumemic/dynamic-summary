"""Test that demonstrates the current bug in Retriever - creates incomplete coverage trees."""

import asyncio
from collections.abc import Generator
from typing import Any

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from tests.mock_store import SimpleMockStore


class TestRetrieverBug:
    """Tests that show the current Retriever creates incomplete coverage trees."""

    @pytest.fixture
    def setup_tree_for_bug_demo(
        self,
    ) -> Generator[tuple[Any, SimpleMockStore, Any], None, None]:
        """Set up a system with a tree structure to demonstrate the bug."""
        index_config = IndexConfig.load(
            target_chunk_tokens=100, preceding_context_tokens=50
        )
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Create a simple config object with properties for backward compatibility
        class Config:
            def __init__(self) -> None:
                self.target_chunk_tokens = index_config.target_chunk_tokens
                self.preceding_context_tokens = index_config.preceding_context_tokens
                self.openai_api_key = (
                    operational_config.openai_api_key.get_secret_value()
                )

        config = Config()
        store = SimpleMockStore(config=config)

        # Create same tree structure as before
        #         root
        #        /    \
        #      P1      P2
        #     /  \    /  \
        #    L1  L2  L3  L4

        # Add all nodes (same as in other test)
        store.add_node(
            node_id="L1",
            text="Chapter 1 content",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=20,
            document_id="test-doc",
            parent_id="P1",
        )
        store.add_node(
            node_id="L2",
            text="Chapter 2 content",
            embedding=[0.2] * 1536,
            span_start=20,
            span_end=40,
            document_id="test-doc",
            parent_id="P1",
        )
        store.add_node(
            node_id="L3",
            text="Chapter 3 content",
            embedding=[0.3] * 1536,
            span_start=40,
            span_end=60,
            document_id="test-doc",
            parent_id="P2",
        )
        store.add_node(
            node_id="L4",
            text="Chapter 4 content",
            embedding=[0.4] * 1536,
            span_start=60,
            span_end=80,
            document_id="test-doc",
            parent_id="P2",
        )

        store.add_node(
            node_id="P1",
            text="Summary of chapters 1-2",
            embedding=[0.15] * 1536,
            span_start=0,
            span_end=40,
            document_id="test-doc",
            parent_id="root",
            left_child_id="L1",
            right_child_id="L2",
        )
        store.add_node(
            node_id="P2",
            text="Summary of chapters 3-4",
            embedding=[0.35] * 1536,
            span_start=40,
            span_end=80,
            document_id="test-doc",
            parent_id="root",
            left_child_id="L3",
            right_child_id="L4",
        )

        store.add_node(
            node_id="root",
            text="Full document summary",
            embedding=[0.25] * 1536,
            span_start=0,
            span_end=80,
            document_id="test-doc",
            left_child_id="P1",
            right_child_id="P2",
        )

        # Update paths after tree construction is complete
        store.update_node_paths_from_tree_structure()

        from tests.utils import create_retriever

        retriever = create_retriever(
            query_config=query_config,
            store=store,
            document_id="test-doc",  # Specify the document we're working with
            api_key=operational_config.openai_api_key.get_secret_value(),
        )
        return config, store, retriever

    def test_retriever_bug_with_num_seeds_1(
        self, setup_tree_for_bug_demo: tuple[Any, SimpleMockStore, Any]
    ) -> None:
        """Test that the retriever should build complete coverage trees but currently doesn't."""
        config, store, retriever = setup_tree_for_bug_demo

        # Mock the vector search to return only L3
        # This simulates what happens with --num-seeds 1 when L3 is most relevant
        def mock_search_similar(
            embedding: list[float], n_results: int, where: Any = None
        ) -> list[tuple[str, float, dict[str, Any]]]:
            return [("L3", 0.95, {})]

        store.search_similar = mock_search_similar
        # Also mock MMR to return L3 as selected
        store.compute_mmr_diverse_results = lambda *args: ["L3"]

        # Mock the query embedding generation
        retriever.embedding_service.get_query_embedding = (
            lambda query, document_id=None: [0.3] * 1536
        )

        # This SHOULD work without errors if the retriever built complete coverage trees
        # But it currently raises an error because of the bug
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",  # Query text doesn't matter with our mock
                num_seeds=1,  # Only select 1 node
                budget_tokens=1000,
                document_id="test-doc",
            )
        )

        # If we got here, the retriever should have produced a valid tiling
        assert result.tiling is not None
        assert len(result.tiling) > 0

    def test_retriever_builds_complete_coverage_trees(
        self, setup_tree_for_bug_demo: tuple[Any, SimpleMockStore, Any]
    ) -> None:
        """Test that retriever should include siblings to build complete coverage trees."""
        config, store, retriever = setup_tree_for_bug_demo

        # Mock vector search to return only L3
        def mock_search_similar(
            embedding: list[float], n_results: int, where: Any = None
        ) -> list[tuple[str, float, dict[str, Any]]]:
            return [("L3", 0.95, {})]

        store.search_similar = mock_search_similar
        # Also mock MMR to return L3 as selected
        store.compute_mmr_diverse_results = lambda *args: ["L3"]

        # Mock the query embedding generation
        retriever.embedding_service.get_query_embedding = (
            lambda query, document_id=None: [0.3] * 1536
        )

        # Patch to capture what nodes the DP algorithm receives
        captured_nodes = {}
        original_find_optimal = retriever.dp_generator.find_optimal_tiling

        def capture_and_pass_through(
            budget_tokens: int,
            scores: dict[str, float],
            nodes: dict[str, Any],
            root_id: str,
        ) -> Any:
            captured_nodes.update(nodes)
            return original_find_optimal(budget_tokens, scores, nodes, root_id)

        retriever.dp_generator.find_optimal_tiling = capture_and_pass_through

        # Run retriever - should work without errors
        result = asyncio.run(
            retriever.retrieve_async(
                query="test",
                num_seeds=1,
                budget_tokens=1000,
                document_id="test-doc",
            )
        )

        # Verify the retriever built a coverage tree with the right nodes
        # When L3 is selected, we need its sibling L4 to maintain completeness
        assert "L3" in captured_nodes  # Selected leaf
        assert (
            "L4" in captured_nodes
        )  # Sibling MUST be included (P2 needs both children)
        assert "P2" in captured_nodes  # Parent of L3 and L4
        assert (
            "P1" in captured_nodes
        )  # Sibling of P2 MUST be included (root needs both children)
        assert "root" in captured_nodes  # Root

        # P1's children (L1, L2) should NOT be included
        # P1 can be a leaf in the coverage tree
        assert "L1" not in captured_nodes
        assert "L2" not in captured_nodes

        # The result should be valid
        assert result.tiling is not None
        assert len(result.tiling) > 0
