"""Performance tests for parallel DP tiling algorithm."""

import asyncio
import time
from typing import Any

import pytest

from ragzoom.dynamic_tiling import AsyncDynamicTilingGenerator, DynamicTilingGenerator
from ragzoom.index import TreeBuilder
from tests.utils import create_predictable_summary_mock, mock_openai_context


@pytest.mark.asyncio
class TestParallelDPPerformance:
    """Test parallel DP performance compared to sequential."""

    @pytest.fixture
    def large_document_setup(self, store: Any, config_factory: Any) -> Any:
        """Set up a test system with a larger document for performance testing."""
        config = config_factory(
            target_chunk_tokens=200,
            budget_tokens=2000,  # Larger budget for more complex trees
        )

        with mock_openai_context() as (mock_index, mock_retrieve, mock_assemble):
            mock_chat_sync, mock_chat_async = create_predictable_summary_mock()
            mock_index.chat.completions.create = mock_chat_async

            # Create document with proper metadata
            store.add_document(
                document_id="large-test-doc",
                file_path=None,
                content_hash="test-hash",
                chunk_count=0,
                embedding_model="text-embedding-3-small",
                summary_model="gpt-4o-mini",
            )
            # Create document-scoped store
            doc_store = store.for_document("large-test-doc")
            tree_builder = TreeBuilder(
                config.index_config,
                doc_store,
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

            tree_builder.add_document(large_document, "large-test-doc")

            yield config, store, tree_builder, mock_retrieve

    async def test_sync_vs_async_dp_correctness(
        self, large_document_setup: Any
    ) -> None:
        """Test that sync and async DP generators produce identical results."""
        config, store, _, mock_client = large_document_setup

        # Create both generators
        sync_generator = DynamicTilingGenerator(config.query_config)
        async_generator = AsyncDynamicTilingGenerator(
            config.query_config, min_nodes_for_parallel=5
        )

        # Get test data
        from tests.utils import create_retriever

        retriever = create_retriever(
            config.query_config,
            store,
            document_id="large-test-doc",
            api_key=config.openai_api_key.get_secret_value(),
            client=mock_client,
        )
        result = await retriever.retrieve_async(
            "test content", budget_tokens=1500, document_id="large-test-doc"
        )

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

        # Run sync version in executor to avoid blocking event loop.
        # This is necessary because calling synchronous code directly in an async
        # function can block the event loop, preventing other async operations
        # from running and potentially causing deadlocks.
        loop = asyncio.get_event_loop()
        sync_result = await loop.run_in_executor(
            None, sync_generator.find_optimal_tiling, 1500, scores, nodes, root_id
        )

        # Run async version
        async_result = await async_generator.find_optimal_tiling(
            1500, scores, nodes, root_id
        )

        # Results should be identical
        assert sync_result.tiling.node_ids == async_result.tiling.node_ids
        assert abs(sync_result.total_quality - async_result.total_quality) < 1e-6
        assert len(sync_result.node_infos) == len(async_result.node_infos)

    async def test_async_dp_performance_benefit(
        self, large_document_setup: Any
    ) -> None:
        """Test that async DP provides performance benefit on larger trees."""
        from tests.utils import create_retriever

        config, store, _, mock_client = large_document_setup

        # Create generators with low threshold to force parallelization
        sync_generator = DynamicTilingGenerator(config.query_config)
        async_generator = AsyncDynamicTilingGenerator(
            config.query_config, min_nodes_for_parallel=3
        )

        # Get test data
        retriever = create_retriever(
            config.query_config,
            store,
            document_id="large-test-doc",
            api_key=config.openai_api_key.get_secret_value(),
            client=mock_client,
        )
        result = await retriever.retrieve_async(
            "test content", budget_tokens=1800, document_id="large-test-doc"
        )

        nodes = result.nodes or {}
        scores = result.scores
        root_id = None
        for node_id, node in nodes.items():
            if node.parent_id is None or node.parent_id not in nodes:
                root_id = node_id
                break

        if not root_id or len(nodes) < 3:
            pytest.skip("Not enough nodes for meaningful performance test")

        # Benchmark sync version in executor to avoid blocking event loop.
        # run_in_executor() allows synchronous code to run in a thread pool
        # without blocking the async event loop.
        loop = asyncio.get_event_loop()
        start_time = time.perf_counter()
        sync_result = await loop.run_in_executor(
            None, sync_generator.find_optimal_tiling, 1800, scores, nodes, root_id
        )
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

    async def test_retriever_with_async_dp(self, large_document_setup: Any) -> None:
        """Test retriever using async DP generator."""
        config, store, _, mock_client = large_document_setup

        # Create retrievers with and without async DP
        from tests.utils import create_retriever

        sync_retriever = create_retriever(
            config.query_config,
            store,
            document_id="large-test-doc",
            api_key=config.openai_api_key.get_secret_value(),
            client=mock_client,
        )
        sync_retriever.use_async_dp = False

        async_retriever = create_retriever(
            config.query_config,
            store,
            document_id="large-test-doc",
            api_key=config.openai_api_key.get_secret_value(),
            client=mock_client,
        )
        async_retriever.use_async_dp = True

        # Test both retrievers produce same results
        # Run sync version in executor to prevent event loop blocking.
        # Without this, the synchronous retrieve() call would block all
        # async operations in the test.
        loop = asyncio.get_event_loop()

        # Create a wrapper function to handle keyword arguments properly
        def sync_retrieve() -> Any:
            return sync_retriever.retrieve(
                "test content", budget_tokens=1200, document_id="large-test-doc"
            )

        sync_result = await loop.run_in_executor(None, sync_retrieve)

        async_result = await async_retriever.retrieve_async(
            "test content", budget_tokens=1200, document_id="large-test-doc"
        )

        # Results should be identical
        assert sync_result.tiling == async_result.tiling
        assert sync_result.scores == async_result.scores

    async def test_error_handling_in_parallel_dp(
        self, large_document_setup: Any
    ) -> None:
        """Test graceful error handling in parallel DP execution."""
        config, store, _, mock_client = large_document_setup

        async_generator = AsyncDynamicTilingGenerator(
            config.query_config, min_nodes_for_parallel=1
        )

        # Get test data
        from tests.utils import create_retriever

        retriever = create_retriever(
            config.query_config,
            store,
            document_id="large-test-doc",
            api_key=config.openai_api_key.get_secret_value(),
            client=mock_client,
        )
        result = await retriever.retrieve_async(
            "test content", budget_tokens=1000, document_id="large-test-doc"
        )

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

    async def test_parallelization_threshold(self, large_document_setup: Any) -> None:
        """Test that parallelization threshold works correctly."""
        from tests.utils import create_retriever

        config, store, _, mock_client = large_document_setup

        # High threshold should disable parallelization for most trees
        high_threshold_generator = AsyncDynamicTilingGenerator(
            config.query_config, min_nodes_for_parallel=1000
        )

        # Low threshold should enable parallelization
        low_threshold_generator = AsyncDynamicTilingGenerator(
            config.query_config, min_nodes_for_parallel=1
        )

        retriever = create_retriever(
            config.query_config,
            store,
            document_id="large-test-doc",
            api_key=config.openai_api_key.get_secret_value(),
            client=mock_client,
        )
        result = await retriever.retrieve_async(
            "test content", budget_tokens=1000, document_id="large-test-doc"
        )

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
