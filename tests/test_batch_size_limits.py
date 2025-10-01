"""Test handling of large embedding batches."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ragzoom.config import IndexConfig
from ragzoom.services.llm_service import LLMService
from ragzoom.utils.tokenization import tokenizer


class TestBatchSizeLimits:
    """Test that large embedding batches are automatically split."""

    @pytest.fixture
    def config(self) -> IndexConfig:
        """Create test configuration."""

        return IndexConfig.load(
            target_chunk_tokens=10,
            embedding_batch_size=100,
        )

    @pytest.fixture
    def llm_service(self, config: IndexConfig) -> LLMService:
        """Instantiate LLMService with a mocked OpenAI client."""

        service = LLMService(config, api_key="test-key")
        mock_client = Mock()
        mock_embeddings = Mock()
        mock_embeddings.create = AsyncMock()
        mock_client.embeddings = mock_embeddings
        service.client = mock_client
        return service

    @pytest.mark.asyncio
    async def test_small_batch_no_splitting(self, llm_service: LLMService) -> None:
        """Test that small batches are processed normally."""
        mock_response = Mock()
        mock_response.data = [
            SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(100)
        ]
        llm_service.client.embeddings.create.return_value = mock_response  # type: ignore[attr-defined]

        texts = [f"text {i}" for i in range(100)]
        result = await llm_service._get_embeddings_batch(texts)

        assert llm_service.client.embeddings.create.call_count == 1  # type: ignore[attr-defined]
        assert len(result) == 100

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(2.0)
    async def test_large_batch_automatic_splitting(
        self, llm_service: LLMService
    ) -> None:
        """Test that large batches are automatically split."""

        def mock_create(**kwargs: object) -> Mock:
            batch_size = len(cast(list[str], kwargs["input"]))
            mock_response = Mock()
            mock_response.data = [
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(batch_size)
            ]
            return mock_response

        llm_service.client.embeddings.create.side_effect = mock_create  # type: ignore[attr-defined]

        texts = [f"text {i}" for i in range(2500)]
        result = await llm_service._get_embeddings_batch(texts)

        assert llm_service.client.embeddings.create.call_count == 3  # type: ignore[attr-defined]
        assert len(result) == 2500

    @pytest.mark.asyncio
    async def test_exactly_max_batch_size(self, llm_service: LLMService) -> None:
        """Test batch exactly at the limit."""
        mock_response = Mock()
        mock_response.data = [
            SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(1000)
        ]
        llm_service.client.embeddings.create.return_value = mock_response  # type: ignore[attr-defined]

        texts = [f"text {i}" for i in range(1000)]
        result = await llm_service._get_embeddings_batch(texts)

        assert llm_service.client.embeddings.create.call_count == 1  # type: ignore[attr-defined]
        assert len(result) == 1000

    @pytest.mark.asyncio
    async def test_batch_size_limit_constant(self, llm_service: LLMService) -> None:
        """Test that the batch size limit is set correctly."""

        def mock_create(**kwargs: object) -> Mock:
            batch_size = len(cast(list[str], kwargs["input"]))
            mock_response = Mock()
            mock_response.data = [
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(batch_size)
            ]
            return mock_response

        llm_service.client.embeddings.create.side_effect = mock_create  # type: ignore[attr-defined]

        texts = [f"text {i}" for i in range(1001)]
        result = await llm_service._get_embeddings_batch(texts)

        assert llm_service.client.embeddings.create.call_count == 2  # type: ignore[attr-defined]
        assert len(result) == 1001

    @pytest.mark.asyncio
    async def test_empty_text_validation_still_works(
        self, llm_service: LLMService
    ) -> None:
        """Test that empty text validation still works after batch splitting."""
        texts = ["valid text", "", "another valid text"]

        with pytest.raises(
            ValueError, match="Empty text at index 1 in embedding batch"
        ):
            await llm_service._get_embeddings_batch(texts)

    @pytest.mark.asyncio
    async def test_empty_batch_handling(self, llm_service: LLMService) -> None:
        """Test that empty batches are handled correctly."""
        result = await llm_service._get_embeddings_batch([])
        assert result == []
        assert llm_service.client.embeddings.create.call_count == 0  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_token_budget_splitting(self, llm_service: LLMService) -> None:
        """Batches should split when aggregate token budget exceeds provider limit."""

        original_limit = llm_service._embedding_batch_token_limit
        llm_service._embedding_batch_token_limit = 100

        def mock_create(**kwargs: object) -> Mock:
            batch_size = len(cast(list[str], kwargs["input"]))
            mock_response = Mock()
            mock_response.data = [
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(batch_size)
            ]
            return mock_response

        llm_service.client.embeddings.create.side_effect = mock_create  # type: ignore[attr-defined]

        texts = ["chunk-a", "chunk-b", "chunk-c", "chunk-d"]
        token_sequence = [60, 60, 20, 20]

        try:
            with patch.object(tokenizer, "count_tokens", side_effect=token_sequence):
                result = await llm_service._get_embeddings_batch(texts)
        finally:
            llm_service._embedding_batch_token_limit = original_limit

        assert llm_service.client.embeddings.create.call_count == 2  # type: ignore[attr-defined]
        assert len(result) == len(texts)

    @pytest.mark.asyncio
    async def test_dynamic_batch_sizes_follow_token_capacity(
        self, llm_service: LLMService
    ) -> None:
        """Embedding batches should pack as many items as token budget allows."""

        llm_service._embedding_batch_token_limit = 5000
        llm_service._provider_max_embedding_batch_size = 1000

        async def mock_create(**kwargs: object) -> Mock:
            batch_size = len(cast(list[str], kwargs["input"]))
            mock_response = Mock()
            mock_response.data = [
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(batch_size)
            ]
            return mock_response

        llm_service.client.embeddings.create.side_effect = mock_create  # type: ignore[attr-defined]

        texts = [f"short-{i}" for i in range(400)]
        token_sequence = [10] * len(texts)

        with patch.object(tokenizer, "count_tokens", side_effect=token_sequence):
            result = await llm_service._get_embeddings_batch(texts)

        assert llm_service.client.embeddings.create.call_count == 1  # type: ignore[attr-defined]
        assert len(result) == len(texts)

    @pytest.mark.asyncio
    async def test_embedding_worker_packs_batches_by_tokens(
        self,
        config: IndexConfig,
        llm_service: LLMService,
    ) -> None:
        """Embedding worker should consolidate queue batches by token capacity."""

        from ragzoom.dataflow.core import BatchAwareQueue, embedding_worker
        from ragzoom.dataflow.domain import DomainNode

        llm_service._embedding_batch_token_limit = 5000
        llm_service._provider_max_embedding_batch_size = 1000

        call_sizes: list[int] = []

        async def fake_batch(texts: list[str]) -> list[list[float]]:
            call_sizes.append(len(texts))
            return [[0.1, 0.2, 0.3] for _ in texts]

        mocked_get_batch = AsyncMock(side_effect=fake_batch)

        queue = BatchAwareQueue(batch_size=config.embedding_batch_size)
        shutdown = asyncio.Event()

        with patch.object(
            llm_service,
            "_get_embeddings_batch",
            mocked_get_batch,
        ):
            worker = asyncio.create_task(
                embedding_worker(
                    0,
                    queue,
                    llm_service,
                    shutdown,
                    reporter=None,
                    progress=None,
                )
            )

            texts = [f"leaf-{i}" for i in range(400)]
            token_sequence = [10] * (len(texts) + 1)

            with patch.object(tokenizer, "count_tokens", side_effect=token_sequence):
                for idx, text in enumerate(texts):
                    node = DomainNode(
                        id=f"node-{idx}",
                        document_id="doc",
                        parent_id=None,
                        left_child_id=None,
                        right_child_id=None,
                        span_start=0,
                        span_end=0,
                        text=text,
                        token_count=0,
                        height=0,
                        is_pinned=False,
                        depth=1,
                        preceding_neighbor_id=None,
                        following_neighbor_id=None,
                        embedding=None,
                    )
                    await queue.put(node)

            await asyncio.sleep(0.1)
            shutdown.set()
            await worker
            await queue.join()

        assert call_sizes == [400]
        assert mocked_get_batch.await_count == 1

    @pytest.mark.asyncio
    async def test_embedding_worker_accepts_zero_token_nodes(
        self,
        config: IndexConfig,
        llm_service: LLMService,
    ) -> None:
        """Zero-token nodes should embed without blowing up the worker."""

        from ragzoom.dataflow.core import BatchAwareQueue, embedding_worker
        from ragzoom.dataflow.domain import DomainNode

        llm_service._embedding_batch_token_limit = 100
        llm_service._provider_max_embedding_batch_size = 10

        call_sizes: list[int] = []

        async def fake_batch(texts: list[str]) -> list[list[float]]:
            call_sizes.append(len(texts))
            return [[0.1, 0.2, 0.3] for _ in texts]

        mock_get_batch = AsyncMock(side_effect=fake_batch)

        queue = BatchAwareQueue(batch_size=config.embedding_batch_size)
        shutdown = asyncio.Event()

        zero_nodes = [
            DomainNode(
                id=f"zero-{idx}",
                document_id="doc",
                parent_id=None,
                left_child_id=None,
                right_child_id=None,
                span_start=0,
                span_end=0,
                text="",
                token_count=0,
                height=0,
                is_pinned=False,
                depth=1,
                preceding_neighbor_id=None,
                following_neighbor_id=None,
                embedding=None,
            )
            for idx in range(3)
        ]

        with patch.object(
            llm_service,
            "_get_embeddings_batch",
            mock_get_batch,
        ):
            with patch.object(tokenizer, "count_tokens", return_value=0):
                worker = asyncio.create_task(
                    embedding_worker(
                        0,
                        queue,
                        llm_service,
                        shutdown,
                        reporter=None,
                        progress=None,
                    )
                )

                for node in zero_nodes:
                    await queue.put(node)

                await asyncio.sleep(0.05)
                shutdown.set()
                await worker
                await queue.join()

        assert call_sizes == [3]
        assert mock_get_batch.await_count == 1
