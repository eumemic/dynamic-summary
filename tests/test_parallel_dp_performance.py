"""Performance tests for parallel DP tiling algorithm."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from openai import OpenAI

    from ragzoom.retrieve import RetrievalResult

import pytest

from ragzoom.config import IndexConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.dynamic_tiling import AsyncDynamicTilingGenerator, DynamicTilingGenerator
from tests.chunk_size_regression_harness import configure_runtime
from tests.conftest import IndexerRuntimeHarness
from tests.utils import create_predictable_summary_mock, mock_openai_context
from tests.vector_index_stubs import RecordingVectorIndex


@pytest.mark.asyncio
class TestParallelDPPerformance:
    """Test parallel DP performance compared to sequential."""

    DOCUMENT_ID = "large-test-doc"

    @staticmethod
    def _bind_vector_index(
        harness: IndexerRuntimeHarness, vector_index: RecordingVectorIndex
    ) -> None:
        harness.runtime._vector_index_factory = lambda _model_id: vector_index
        harness.worker_coordinator._vector_index_factory = (
            lambda _document_id: vector_index
        )

    @staticmethod
    async def _index_document(
        harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
        document_id: str,
        text: str,
    ) -> None:
        await harness.append(
            document_id,
            text,
            replace_existing=True,
            file_path=f"{document_id}.txt",
        )
        await harness.wait_for_idle(document_id)

    @staticmethod
    def _prepare_defaults() -> (
        tuple[IndexConfig, QueryConfig, RecordingVectorIndex, str]
    ):
        index_config = IndexConfig.load(
            target_chunk_tokens=200,
            preceding_context_tokens=25,
        )
        query_config = QueryConfig(budget_tokens=2000)
        vector_index = RecordingVectorIndex()
        chunk_text = "This is test content for performance testing. " * 20
        large_document = " ".join([chunk_text for _ in range(4)])
        return index_config, query_config, vector_index, large_document

    async def test_sync_vs_async_dp_correctness(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that sync and async DP generators produce identical results."""

        index_config, query_config, vector_index, large_document = (
            self._prepare_defaults()
        )
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        with mock_openai_context() as (mock_index, mock_retrieve, _):
            _mock_sync, mock_chat_async = create_predictable_summary_mock()
            mock_index.chat.completions.create = mock_chat_async

            await self._index_document(
                indexer_runtime_harness,
                storage_backend,
                self.DOCUMENT_ID,
                large_document,
            )

            doc_store = storage_backend.for_document(self.DOCUMENT_ID)

            # Create both generators
            sync_generator = DynamicTilingGenerator(query_config)
            async_generator = AsyncDynamicTilingGenerator(
                query_config, min_nodes_for_parallel=5
            )

            from tests.utils import create_retriever

            retriever = create_retriever(
                query_config,
                doc_store,
                client=cast("OpenAI", mock_retrieve),
                vector_index=vector_index,
            )
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
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that async DP provides performance benefit on larger trees."""

        index_config, query_config, vector_index, large_document = (
            self._prepare_defaults()
        )
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        with mock_openai_context() as (mock_index, mock_retrieve, _):
            _mock_sync, mock_chat_async = create_predictable_summary_mock()
            mock_index.chat.completions.create = mock_chat_async

            await self._index_document(
                indexer_runtime_harness,
                storage_backend,
                self.DOCUMENT_ID,
                large_document,
            )

            doc_store = storage_backend.for_document(self.DOCUMENT_ID)

            sync_generator = DynamicTilingGenerator(query_config)
            async_generator = AsyncDynamicTilingGenerator(
                query_config, min_nodes_for_parallel=3
            )

            from tests.utils import create_retriever

            retriever = create_retriever(
                query_config,
                doc_store,
                client=cast("OpenAI", mock_retrieve),
                vector_index=vector_index,
            )
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

    async def test_retriever_with_async_dp(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test retriever using async DP generator."""

        from tests.utils import create_retriever

        index_config, query_config, vector_index, large_document = (
            self._prepare_defaults()
        )
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        with mock_openai_context() as (mock_index, mock_retrieve, _):
            _mock_sync, mock_chat_async = create_predictable_summary_mock()
            mock_index.chat.completions.create = mock_chat_async

            await self._index_document(
                indexer_runtime_harness,
                storage_backend,
                self.DOCUMENT_ID,
                large_document,
            )

            doc_store = storage_backend.for_document(self.DOCUMENT_ID)

            sync_retriever = create_retriever(
                query_config,
                doc_store,
                client=cast("OpenAI", mock_retrieve),
                vector_index=vector_index,
            )
            sync_retriever.use_async_dp = False

            async_retriever = create_retriever(
                query_config,
                doc_store,
                client=cast("OpenAI", mock_retrieve),
                vector_index=vector_index,
            )
            async_retriever.use_async_dp = True

            loop = asyncio.get_event_loop()

            def sync_retrieve() -> RetrievalResult:
                return sync_retriever.retrieve("test content", budget_tokens=1200)

            sync_result = await loop.run_in_executor(None, sync_retrieve)

            async_result = await async_retriever.retrieve_async(
                "test content", budget_tokens=1200
            )

            assert sync_result.tiling == async_result.tiling
            assert sync_result.scores == async_result.scores

    async def test_error_handling_in_parallel_dp(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test graceful error handling in parallel DP execution."""

        from tests.utils import create_retriever

        index_config, query_config, vector_index, large_document = (
            self._prepare_defaults()
        )
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        async_generator = AsyncDynamicTilingGenerator(
            query_config, min_nodes_for_parallel=1
        )

        with mock_openai_context() as (mock_index, mock_retrieve, _):
            _mock_sync, mock_chat_async = create_predictable_summary_mock()
            mock_index.chat.completions.create = mock_chat_async

            await self._index_document(
                indexer_runtime_harness,
                storage_backend,
                self.DOCUMENT_ID,
                large_document,
            )

            doc_store = storage_backend.for_document(self.DOCUMENT_ID)
            retriever = create_retriever(
                query_config,
                doc_store,
                client=cast("OpenAI", mock_retrieve),
                vector_index=vector_index,
            )
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

            dp_result = await async_generator.find_optimal_tiling(
                1000, scores, nodes, root_id
            )
            assert dp_result is not None
            assert dp_result.tiling is not None

    async def test_parallelization_threshold(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that parallelization threshold works correctly."""

        from tests.utils import create_retriever

        index_config, query_config, vector_index, large_document = (
            self._prepare_defaults()
        )
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        high_threshold_generator = AsyncDynamicTilingGenerator(
            query_config, min_nodes_for_parallel=1000
        )
        low_threshold_generator = AsyncDynamicTilingGenerator(
            query_config, min_nodes_for_parallel=1
        )

        with mock_openai_context() as (mock_index, mock_retrieve, _):
            _mock_sync, mock_chat_async = create_predictable_summary_mock()
            mock_index.chat.completions.create = mock_chat_async

            await self._index_document(
                indexer_runtime_harness,
                storage_backend,
                self.DOCUMENT_ID,
                large_document,
            )

            doc_store = storage_backend.for_document(self.DOCUMENT_ID)
            retriever = create_retriever(
                query_config,
                doc_store,
                client=cast("OpenAI", mock_retrieve),
                vector_index=vector_index,
            )
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

            assert high_result.tiling.node_ids == low_result.tiling.node_ids
            assert abs(high_result.total_quality - low_result.total_quality) < 1e-6
