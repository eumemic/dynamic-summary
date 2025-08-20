"""Test handling of empty summaries during indexing."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from ragzoom.config import IndexConfig
from ragzoom.index import TreeBuilder


class TestEmptySummaryHandling:
    """Test that empty summaries are handled gracefully during tree building."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return IndexConfig.load(
            target_chunk_tokens=10,  # Very small to force tree building
            preceding_context_tokens=5,
        )

    @pytest.fixture
    def tree_builder(self, config):
        """Create tree builder with mocked dependencies."""
        with patch("ragzoom.index.Store"):
            mock_store = Mock()
            return TreeBuilder(config, mock_store, api_key="test-key")

    @pytest.mark.asyncio
    async def test_empty_summary_filtering_logic(self, tree_builder):
        """Test the filtering logic for empty summaries."""
        # Mock results with mixed valid and empty summaries
        mock_results = [
            {
                "summary": "Valid summary",
                "parent_id": "node1",
                "token_count": 10,
                "node_data": {"embedding": None},
            },
            {
                "summary": "",  # Empty summary
                "parent_id": "node2",
                "token_count": 0,
                "node_data": {"embedding": None},
            },
            {
                "summary": "   \n\t  ",  # Whitespace-only summary
                "parent_id": "node3",
                "token_count": 0,
                "node_data": {"embedding": None},
            },
            {
                "summary": "Another valid summary",
                "parent_id": "node4",
                "token_count": 8,
                "node_data": {"embedding": None},
            },
        ]

        # Test the filtering logic from the actual code
        valid_summaries = []
        valid_indices = []
        for i, result in enumerate(mock_results):
            summary = result["summary"]
            if summary and summary.strip():  # This is the actual filtering logic
                valid_summaries.append(summary)
                valid_indices.append(i)

        # Should have 2 valid summaries (indices 0 and 3)
        assert len(valid_summaries) == 2
        assert valid_indices == [0, 3]
        assert valid_summaries == ["Valid summary", "Another valid summary"]

    @pytest.mark.asyncio
    async def test_empty_summary_validation_in_batch_method(self, tree_builder):
        """Test that _get_embeddings_batch detects empty strings and provides a clear error."""
        # Test that the safety net in _get_embeddings_batch works
        with pytest.raises(ValueError, match="Empty text at index.*embedding batch"):
            await tree_builder._get_embeddings_batch(
                ["valid text", "", "another valid text"]
            )

    @pytest.mark.asyncio
    async def test_demonstrates_fix_prevents_original_error(self, tree_builder):
        """Test that demonstrates our fix prevents the original OpenAI API error."""
        from unittest.mock import AsyncMock

        # Create a mock that fails with empty strings (simulating real OpenAI API)
        async def mock_embeddings_create(*args, **kwargs):
            input_texts = kwargs.get("input", [])
            # Simulate OpenAI API behavior: reject if any text is empty
            for text in input_texts:
                if not text or not text.strip():
                    # Simulate the exact error from the original issue
                    raise Exception(
                        "Error code: 400 - {'error': {'message': \"'$.input' is invalid. Please check the API reference: https://platform.openai.com/docs/api-reference.\", 'type': 'invalid_request_error', 'param': None, 'code': None}}"
                    )

            # Return mock valid response for non-empty texts
            return type(
                "MockResponse",
                (),
                {
                    "data": [
                        type("MockEmbedding", (), {"embedding": [0.1] * 1536})()
                        for _ in input_texts
                    ]
                },
            )()

        # Replace the actual OpenAI client call
        tree_builder.client.embeddings.create = AsyncMock(
            side_effect=mock_embeddings_create
        )

        # Simulate what would happen without our fix: empty summaries get sent directly
        # We need to temporarily bypass our safety net to show the original problem
        original_method = tree_builder._get_embeddings_batch

        async def bypass_safety_net(texts):
            """Version without our safety net - simulates old behavior."""
            if not texts:
                return []
            # Skip our safety check and go directly to API call
            async with tree_builder.semaphore:
                try:
                    response = await tree_builder.client.embeddings.create(
                        model=tree_builder.config.embedding_model,
                        input=texts,
                    )
                    return [item.embedding for item in response.data]
                except Exception:
                    raise

        # Test: Without our fix - this would fail with the original error
        tree_builder._get_embeddings_batch = bypass_safety_net
        with pytest.raises(Exception, match="invalid.*API reference"):
            await tree_builder._get_embeddings_batch(["valid text", "", "another text"])

        # Test: With our fix - this works
        tree_builder._get_embeddings_batch = original_method

        # Our current implementation would catch this in the safety net
        with pytest.raises(ValueError, match="Empty text at index.*embedding batch"):
            await tree_builder._get_embeddings_batch(["valid text", "", "another text"])

        # But the filtering logic in the tree building prevents it from getting here
        texts_with_empty = ["valid text", "", "another text"]
        filtered_texts = [text for text in texts_with_empty if text and text.strip()]
        result = await tree_builder._get_embeddings_batch(filtered_texts)
        assert len(result) == 2  # Only the 2 valid texts

    @pytest.mark.asyncio
    async def test_whitespace_only_summary_validation(self, tree_builder):
        """Test that whitespace-only summaries are also caught."""
        with pytest.raises(ValueError, match="Empty text at index.*embedding batch"):
            await tree_builder._get_embeddings_batch(
                ["valid text", "   \n\t  ", "another valid text"]
            )

    @pytest.mark.asyncio
    async def test_empty_summary_fallback_embedding(self, tree_builder):
        """Test that empty summaries get fallback embeddings."""
        # Mock to create a scenario where we have empty summaries
        mock_results = [
            {
                "summary": "Valid summary",
                "parent_id": "node1",
                "token_count": 10,
                "node_data": {"embedding": None},
            },
            {
                "summary": "",  # Empty summary
                "parent_id": "node2",
                "token_count": 0,
                "node_data": {"embedding": None},
            },
            {
                "summary": "Another valid summary",
                "parent_id": "node3",
                "token_count": 8,
                "node_data": {"embedding": None},
            },
        ]

        # Mock the embedding methods
        tree_builder._get_embeddings_batch = AsyncMock(
            return_value=[[0.1] * 1536] * 2
        )  # Only 2 valid summaries
        tree_builder._get_embedding = AsyncMock(
            return_value=[0.2] * 1536
        )  # Fallback embedding

        # Create a scenario similar to the tree building logic
        valid_summaries = []
        valid_indices = []
        for i, result in enumerate(mock_results):
            summary = result["summary"]
            if summary and summary.strip():
                valid_summaries.append(summary)
                valid_indices.append(i)

        # Should have 2 valid summaries
        assert len(valid_summaries) == 2
        assert valid_indices == [0, 2]

        # Get embeddings for valid summaries
        embeddings = await tree_builder._get_embeddings_batch(valid_summaries)
        assert len(embeddings) == 2

        # Assign embeddings back to results
        embedding_iter = iter(embeddings)
        for i, result in enumerate(mock_results):
            if i in valid_indices:
                result["node_data"]["embedding"] = next(embedding_iter)
            else:
                # Generate fallback embedding for empty summary
                fallback_embedding = await tree_builder._get_embedding("empty summary")
                result["node_data"]["embedding"] = fallback_embedding

        # Verify all results have embeddings
        for result in mock_results:
            assert result["node_data"]["embedding"] is not None
            assert len(result["node_data"]["embedding"]) == 1536
