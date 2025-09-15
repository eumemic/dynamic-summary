"""Test handling of large embedding batches."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ragzoom.config import IndexConfig
from ragzoom.index import TreeBuilder


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
    def tree_builder(self, config: IndexConfig) -> TreeBuilder:
        """Create tree builder with mocked dependencies."""
        with patch("ragzoom.document_store.DocumentStore"):
            mock_doc_store = Mock()
            mock_doc_store.document_id = "test-doc"
            mock_doc_store.compute_content_hash = Mock(return_value="hash123")
            mock_doc_store.set_metadata = Mock()
            mock_doc_store.session_local = Mock
            mock_doc_store.node_cache = {}
            mock_doc_store.cache_order = []
            # Provide a minimal vector index via factory for TreeBuilder
            from ragzoom.vector_factory import create_vector_index

            vi = create_vector_index(
                "python", "sqlite:///:memory:", config.embedding_model
            )
            builder = TreeBuilder(config, mock_doc_store, vi, api_key="test-key")

            # Mock the OpenAI client on the LLM service
            builder.llm_service.client = Mock()
            builder.llm_service.client.embeddings.create = AsyncMock()

            return builder

    @pytest.mark.asyncio
    async def test_small_batch_no_splitting(self, tree_builder: TreeBuilder) -> None:
        """Test that small batches are processed normally."""
        # Mock response for a small batch
        mock_response = Mock()
        # Lightweight items to avoid heavy Mock creation
        mock_response.data = [
            SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(100)
        ]
        tree_builder.llm_service.client.embeddings.create.return_value = mock_response  # type: ignore[attr-defined]

        texts = [f"text {i}" for i in range(100)]
        result = await tree_builder.llm_service._get_embeddings_batch(texts)

        # Should call API once
        assert tree_builder.llm_service.client.embeddings.create.call_count == 1  # type: ignore[attr-defined]
        assert len(result) == 100

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(2.0)
    async def test_large_batch_automatic_splitting(
        self, tree_builder: TreeBuilder
    ) -> None:
        """Test that large batches are automatically split."""

        # Mock response that returns embeddings matching the input batch size
        def mock_create(**kwargs: object) -> Mock:
            batch_size = len(cast(list[str], kwargs["input"]))
            mock_response = Mock()
            mock_response.data = [
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(batch_size)
            ]
            return mock_response

        tree_builder.llm_service.client.embeddings.create.side_effect = mock_create  # type: ignore[attr-defined]

        # Create a batch larger than the limit (1000)
        texts = [f"text {i}" for i in range(2500)]
        result = await tree_builder.llm_service._get_embeddings_batch(texts)

        # Should split into 3 batches: 1000, 1000, 500
        assert tree_builder.llm_service.client.embeddings.create.call_count == 3  # type: ignore[attr-defined]
        assert len(result) == 2500

    @pytest.mark.asyncio
    async def test_exactly_max_batch_size(self, tree_builder: TreeBuilder) -> None:
        """Test batch exactly at the limit."""
        mock_response = Mock()
        mock_response.data = [
            SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(1000)
        ]
        tree_builder.llm_service.client.embeddings.create.return_value = mock_response  # type: ignore[attr-defined]

        texts = [f"text {i}" for i in range(1000)]
        result = await tree_builder.llm_service._get_embeddings_batch(texts)

        # Should call API once (exactly at limit)
        assert tree_builder.llm_service.client.embeddings.create.call_count == 1  # type: ignore[attr-defined]
        assert len(result) == 1000

    @pytest.mark.asyncio
    async def test_batch_size_limit_constant(self, tree_builder: TreeBuilder) -> None:
        """Test that the batch size limit is set correctly."""

        # Mock response that returns embeddings matching the input batch size
        def mock_create(**kwargs: object) -> Mock:
            batch_size = len(cast(list[str], kwargs["input"]))
            mock_response = Mock()
            mock_response.data = [
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(batch_size)
            ]
            return mock_response

        tree_builder.llm_service.client.embeddings.create.side_effect = mock_create  # type: ignore[attr-defined]

        texts = [f"text {i}" for i in range(1001)]
        result = await tree_builder.llm_service._get_embeddings_batch(texts)

        # Should split into 2 batches: 1000, 1
        assert tree_builder.llm_service.client.embeddings.create.call_count == 2  # type: ignore[attr-defined]
        assert len(result) == 1001

    @pytest.mark.asyncio
    async def test_empty_text_validation_still_works(
        self, tree_builder: TreeBuilder
    ) -> None:
        """Test that empty text validation still works after batch splitting."""
        texts = ["valid text", "", "another valid text"]

        with pytest.raises(
            ValueError, match="Empty text at index 1 in embedding batch"
        ):
            await tree_builder.llm_service._get_embeddings_batch(texts)

    @pytest.mark.asyncio
    async def test_empty_batch_handling(self, tree_builder: TreeBuilder) -> None:
        """Test that empty batches are handled correctly."""
        result = await tree_builder.llm_service._get_embeddings_batch([])
        assert result == []
        assert tree_builder.llm_service.client.embeddings.create.call_count == 0  # type: ignore[attr-defined]
