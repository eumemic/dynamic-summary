"""Test that verifies the num_seeds constraint fix works correctly."""

from unittest.mock import Mock, patch

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.retrieve import Retriever
from tests.mock_store import SimpleMockStore


class TestNumSeedsFix:
    """Test that the fix for num_seeds constraint works correctly."""

    def test_retrieve_respects_coverage_tree(self):
        """Test that retrieve() only passes coverage tree nodes to DP."""
        # Create a mock store
        store = SimpleMockStore()

        # Build a simple tree
        store.add_node(
            node_id="root",
            text="Root summary document",
            span_start=0,
            span_end=1000,
            parent_id=None,
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="nodeA",
            right_child_id="nodeB",
        )

        store.add_node(
            node_id="nodeA",
            text="Node A content",
            span_start=0,
            span_end=500,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="leaf1",
            right_child_id="leaf2",
        )

        store.add_node(
            node_id="nodeB",
            text="Node B content",
            span_start=500,
            span_end=1000,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="leaf3",
            right_child_id="leaf4",
        )

        # Add leaf nodes - all with high similarity to query
        for i, (nid, start, end, parent) in enumerate(
            [
                ("leaf1", 0, 250, "nodeA"),
                ("leaf2", 250, 500, "nodeA"),
                ("leaf3", 500, 750, "nodeB"),
                ("leaf4", 750, 1000, "nodeB"),
            ]
        ):
            store.add_node(
                node_id=nid,
                text=f"Leaf {i} with dragon content",
                span_start=start,
                span_end=end,
                parent_id=parent,
                document_id="doc1",
                embedding=[0.9] * 1536,  # High similarity
            )

        # Mock the search_similar to return all leaves as candidates
        # Note: search_similar now returns (id, similarity, metadata) tuples
        store.search_similar = Mock(
            return_value=[
                ("leaf1", 0.9, {}),  # High similarity, empty metadata
                ("leaf2", 0.9, {}),
                ("leaf3", 0.9, {}),
                ("leaf4", 0.9, {}),
            ]
        )

        # Mock compute_mmr_diverse_results to select only leaf1
        store.compute_mmr_diverse_results = Mock(return_value=["leaf1"])

        # Mock get_ancestors to return proper ancestors
        def mock_get_ancestors(node_ids):
            ancestors = []
            for nid in node_ids:
                if nid == "leaf1":
                    ancestors.extend([Mock(id="nodeA"), Mock(id="root")])
            return ancestors

        store.get_ancestors = Mock(side_effect=mock_get_ancestors)

        # Create config and retriever
        index_config = IndexConfig()
        query_config = QueryConfig(budget_tokens=10000)
        operational_config = OperationalConfig(openai_api_key="test-key")

        # Mock OpenAI client
        with patch("ragzoom.retrieve.OpenAI") as mock_client:
            mock_embeddings = Mock()
            mock_embeddings.create = Mock(
                return_value=Mock(data=[Mock(embedding=[0.9] * 1536)])
            )
            mock_instance = Mock()
            mock_instance.embeddings = mock_embeddings
            mock_client.return_value = mock_instance

            retriever = Retriever(
                query_config=query_config,
                index_config=index_config,
                store=store,
                api_key=operational_config.openai_api_key,
                tree_builder=None,
            )

            # Retrieve with num_seeds=1
            result = retriever.retrieve("dragon", num_seeds=1, document_id="doc1")

        # Verify selected nodes
        assert result.node_ids == ["leaf1"]

        # Verify coverage map contains selected + ancestors + siblings to maintain full binary tree
        # Since leaf1 is included and nodeA is its parent, leaf2 (sibling) must be included
        # Since nodeA is included and root is its parent, nodeB (sibling) must be included
        expected_coverage = {"leaf1", "leaf2", "nodeA", "nodeB", "root"}
        assert set(result.coverage_map.keys()) == expected_coverage

        # CRITICAL: Verify scores only contain nodes from coverage map
        assert set(result.scores.keys()).issubset(expected_coverage), (
            f"Scores contain nodes outside coverage map! "
            f"Scores: {set(result.scores.keys())}, "
            f"Coverage: {expected_coverage}"
        )

        # Verify tiling only uses nodes from coverage tree
        if result.tiling:
            tiling_nodes = set(result.tiling)  # tiling is now a list of node IDs
            assert tiling_nodes.issubset(expected_coverage), (
                f"Tiling contains nodes outside coverage tree! "
                f"Tiling: {tiling_nodes}, Coverage: {expected_coverage}"
            )

            # Count leaf nodes in tiling
            leaf_count = sum(
                1 for node_id in result.tiling if store.is_leaf_node(node_id)
            )
            # Since we have to include leaf2 to maintain coverage property, the DP algorithm
            # might choose to use both leaves instead of their parent
            assert leaf_count <= 2, f"Expected at most 2 leaf nodes, got {leaf_count}"
