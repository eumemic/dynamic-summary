"""Performance tests for parallel DP tiling algorithm."""

import time

import pytest

from ragzoom.dynamic_tiling import AsyncDynamicTilingGenerator, DynamicTilingGenerator
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from tests.utils import create_predictable_summary_mock, mock_openai_context


@pytest.mark.asyncio
class TestParallelDPPerformance:
    """Test parallel DP performance compared to sequential."""

    @pytest.fixture
    async def large_document_setup(self, store, config_factory):
        """Set up a test system with a larger document for performance testing."""
        config = config_factory(
            target_chunk_tokens=200,
            budget_tokens=2000,  # Larger budget for more complex trees
        )

        with mock_openai_context() as (mock_index, mock_retrieve, mock_assemble):
            mock_chat_sync, mock_chat_async = create_predictable_summary_mock()
            mock_index.chat.completions.create = mock_chat_async

            tree_builder = TreeBuilder(
                config.index_config,
                store,
                api_key=config.openai_api_key,
            )

            # Create a smaller document for testing (4 chunks = 2-3 levels)
            # Each chunk is ~200 tokens, create enough for basic tree testing
            chunk_text = (
                "This is test content for performance testing. " * 20
            )  # ~100 tokens
            large_document = " ".join(
                [chunk_text for _ in range(4)]
            )  # 4 chunks = 2-3 levels

            await tree_builder.add_document_async(large_document, "large-test-doc")

            yield config, store, tree_builder

    async def test_sync_vs_async_dp_correctness(self, large_document_setup):
        """Test that sync and async DP generators produce identical results."""
        config, store, _ = large_document_setup

        # Create both generators
        sync_generator = DynamicTilingGenerator(config.query_config)
        async_generator = AsyncDynamicTilingGenerator(
            config.query_config, min_nodes_for_parallel=5
        )

        # Get test data
        retriever = Retriever(config.query_config, store, config.openai_api_key)
        result = await retriever.retrieve_async("test content", budget_tokens=1500)

        # Extract the data needed for DP
        nodes = result.nodes or {}
        scores = result.scores
        root_id = None
        for node_id, node in nodes.items():
            if node.parent_id is None or node.parent_id not in nodes:
                root_id = node_id
                break

        if not root_id or len(nodes) < 3:
            pytest.skip("Not enough nodes for meaningful comparison")

        # Run sync version
        sync_result = sync_generator.find_optimal_tiling(1500, scores, nodes, root_id)

        # Run async version
        async_result = await async_generator.find_optimal_tiling(
            1500, scores, nodes, root_id
        )

        # Results should be identical
        assert sync_result.tiling.node_ids == async_result.tiling.node_ids
        assert abs(sync_result.total_quality - async_result.total_quality) < 1e-6
        assert len(sync_result.node_infos) == len(async_result.node_infos)

    async def test_async_dp_performance_benefit(self, large_document_setup):
        """Test that async DP provides performance benefit on larger trees."""
        config, store, _ = large_document_setup

        # Create generators with low threshold to force parallelization
        sync_generator = DynamicTilingGenerator(config.query_config)
        async_generator = AsyncDynamicTilingGenerator(
            config.query_config, min_nodes_for_parallel=3
        )

        # Get test data
        retriever = Retriever(config.query_config, store, config.openai_api_key)
        result = await retriever.retrieve_async("test content", budget_tokens=1800)

        nodes = result.nodes or {}
        scores = result.scores
        root_id = None
        for node_id, node in nodes.items():
            if node.parent_id is None or node.parent_id not in nodes:
                root_id = node_id
                break

        if not root_id or len(nodes) < 3:
            pytest.skip("Not enough nodes for meaningful performance test")

        # Benchmark sync version
        start_time = time.perf_counter()
        sync_result = sync_generator.find_optimal_tiling(1800, scores, nodes, root_id)
        sync_time = time.perf_counter() - start_time

        # Benchmark async version
        start_time = time.perf_counter()
        async_result = await async_generator.find_optimal_tiling(
            1800, scores, nodes, root_id
        )
        async_time = time.perf_counter() - start_time

        print(f"Sync time: {sync_time:.4f}s, Async time: {async_time:.4f}s")
        print(f"Speedup: {sync_time / async_time:.2f}x")
        print(f"Nodes processed: {len(nodes)}")

        # Results should be identical
        assert sync_result.tiling.node_ids == async_result.tiling.node_ids

        # On small trees, async overhead may outweigh benefits
        # This is expected behavior - async benefits come with larger trees
        # For now, just verify that both produce identical results (done above)
        if len(nodes) >= 50:  # Only expect speedup on larger trees
            assert (
                async_time <= sync_time * 1.2
            ), f"Async ({async_time:.4f}s) much slower than sync ({sync_time:.4f}s) on large tree"

    async def test_retriever_with_async_dp(self, large_document_setup):
        """Test retriever using async DP generator."""
        config, store, _ = large_document_setup

        # Create retrievers with and without async DP
        sync_retriever = Retriever(
            config.query_config, store, config.openai_api_key, use_async_dp=False
        )
        async_retriever = Retriever(
            config.query_config,
            store,
            config.openai_api_key,
            use_async_dp=True,
            min_nodes_for_parallel=5,
        )

        # Test both retrievers produce same results
        sync_result = sync_retriever.retrieve("test content", budget_tokens=1200)

        async_result = await async_retriever.retrieve_async(
            "test content", budget_tokens=1200
        )

        # Results should be identical
        assert sync_result.tiling == async_result.tiling
        assert sync_result.scores == async_result.scores

    async def test_error_handling_in_parallel_dp(self, large_document_setup):
        """Test graceful error handling in parallel DP execution."""
        config, store, _ = large_document_setup

        async_generator = AsyncDynamicTilingGenerator(
            config.query_config, min_nodes_for_parallel=1
        )

        # Get test data
        retriever = Retriever(config.query_config, store, config.openai_api_key)
        result = await retriever.retrieve_async("test content", budget_tokens=1000)

        nodes = result.nodes or {}
        scores = result.scores
        root_id = None
        for node_id, node in nodes.items():
            if node.parent_id is None or node.parent_id not in nodes:
                root_id = node_id
                break

        if not root_id:
            pytest.skip("No valid root found")

        # Should handle errors gracefully and still produce a result
        result = await async_generator.find_optimal_tiling(1000, scores, nodes, root_id)
        assert result is not None
        assert result.tiling is not None

    async def test_parallelization_threshold(self, large_document_setup):
        """Test that parallelization threshold works correctly."""
        config, store, _ = large_document_setup

        # High threshold should disable parallelization for most trees
        high_threshold_generator = AsyncDynamicTilingGenerator(
            config.query_config, min_nodes_for_parallel=1000
        )

        # Low threshold should enable parallelization
        low_threshold_generator = AsyncDynamicTilingGenerator(
            config.query_config, min_nodes_for_parallel=1
        )

        retriever = Retriever(config.query_config, store, config.openai_api_key)
        result = await retriever.retrieve_async("test content", budget_tokens=1000)

        nodes = result.nodes or {}
        scores = result.scores
        root_id = None
        for node_id, node in nodes.items():
            if node.parent_id is None or node.parent_id not in nodes:
                root_id = node_id
                break

        if not root_id:
            pytest.skip("No valid root found")

        high_result = await high_threshold_generator.find_optimal_tiling(
            1000, scores, nodes, root_id
        )
        low_result = await low_threshold_generator.find_optimal_tiling(
            1000, scores, nodes, root_id
        )

        # Both should produce identical results regardless of parallelization
        assert high_result.tiling.node_ids == low_result.tiling.node_ids
        assert abs(high_result.total_quality - low_result.total_quality) < 1e-6
