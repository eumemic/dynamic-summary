import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.retrieve import Retriever
from tests.mock_store import SimpleMockStore


class _TestConfig:
    """Test configuration wrapper for compatibility."""

    def __init__(self, index_config, query_config, operational_config):
        self.index_config = index_config
        self.query_config = query_config
        self.operational_config = operational_config

    @property
    def target_chunk_tokens(self):
        return self.index_config.target_chunk_tokens


class TestDPTiling:
    """Tests for the new DP-based tiling generation."""

    @pytest.fixture
    def setup_system(self):
        """Set up a complete system with DP mode enabled and a mock store."""
        index_config = IndexConfig.load(target_chunk_tokens=100)
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(
            openai_api_key="test-key",
            database_url="postgresql:///:memory:",
        )
        config = _TestConfig(index_config, query_config, operational_config)
        store = SimpleMockStore(config=config)
        retriever = Retriever(
            query_config=query_config,
            store=store,
            api_key=operational_config.openai_api_key,
        )
        assembler = Assembler(store)
        dp_generator = retriever.dp_generator
        return config, store, retriever, assembler, dp_generator

    def test_dp_single_node_tree(self, setup_system):
        """Test the DP algorithm on a tree with only a single node."""
        config, store, retriever, assembler, dp_generator = setup_system

        # Manually create a single-node tree
        store.add_node(
            node_id="root",
            text="single node",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            document_id="test-doc-single",
        )
        store.set_mock_scores({"root": 1.0})

        # We need to manually call the DP generator for now
        coverage_map = {"root": True}

        # Load nodes from coverage map
        nodes = {}
        for node_id in coverage_map:
            node = store.get_node(node_id)
            if node:
                nodes[node_id] = node

        # Find root node
        root_id = "root"  # We know it's root in this test

        dp_result = dp_generator.find_optimal_tiling(
            1000, {"root": 1.0}, nodes, root_id
        )
        tiling = dp_result.tiling

        assert tiling, "DP tiling should not be empty for single node tree"
        assert len(tiling.node_ids) == 1
        assert tiling.node_ids[0] == "root"
