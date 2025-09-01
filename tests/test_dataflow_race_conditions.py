"""Tests for race conditions in dataflow implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragzoom.dataflow.core import build_tree_dataflow


class TestDataflowRaceConditions:
    """Test race conditions that can occur in concurrent dataflow processing."""

    @pytest.mark.asyncio
    async def test_embedding_workers_wait_for_all_summaries(self) -> None:
        """Test that embedding workers don't exit before processing summaries.

        This test simulates the race condition where:
        1. Leaf embeddings are processed quickly
        2. Summary generation is slow
        3. Embedding workers might exit before summary embeddings are queued
        """
        mock_llm_service = MagicMock()

        # Track what gets embedded
        embedded_nodes = []

        async def slow_summary(*args: object, **kwargs: object) -> tuple[str, int, int]:
            """Simulate slow summary generation."""
            await asyncio.sleep(0.05)  # Slow enough to expose race condition
            return ("Summary text", 1, 10)

        async def fast_embeddings(texts: list[str]) -> list[list[float]]:
            """Track what gets embedded."""
            embedded_nodes.extend(texts)
            await asyncio.sleep(0.001)  # Fast embedding
            return [[0.1] * 10 for _ in texts]

        mock_llm_service._summarize_text = AsyncMock(side_effect=slow_summary)
        mock_llm_service._get_embeddings_batch = AsyncMock(side_effect=fast_embeddings)

        # Create a tree that will have internal nodes
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"]

        # Build tree
        result = await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=2,
            max_embedding_concurrency=2,
            embedding_batch_size=10,  # Large batch to process leaves quickly
        )

        # Verify ALL nodes got embeddings
        for node in result:
            assert node.embedding is not None, (
                f"Node {node.id} (height={node.height}) "
                f"missing embedding. Text: {node.text[:20] if node.text else None}"
            )

        # Verify we embedded all texts (leaves + summaries)
        assert len(embedded_nodes) == len(
            result
        ), f"Expected {len(result)} embeddings, got {len(embedded_nodes)}"

    @pytest.mark.asyncio
    async def test_summary_queuing_before_embedding_exit(self) -> None:
        """Test that summaries get queued for embedding before workers exit.

        This tests the specific race where:
        - Embedding workers process all leaves
        - Counter shows more work pending (for summaries)
        - But queue is empty (summaries not yet generated)
        - Workers must wait, not exit
        """
        mock_llm_service = MagicMock()

        summary_generated = asyncio.Event()
        embedding_calls = []

        async def delayed_summary(
            *args: object, **kwargs: object
        ) -> tuple[str, int, int]:
            """Summary that signals when complete."""
            await asyncio.sleep(0.1)  # Significant delay
            summary_generated.set()
            return ("Delayed summary", 1, 10)

        async def track_embeddings(texts: list[str]) -> list[list[float]]:
            """Track when embeddings are called."""
            embedding_calls.append(len(texts))
            # If this is called with summary text, the race was avoided
            if any("summary" in t.lower() for t in texts):
                assert (
                    summary_generated.is_set()
                ), "Embedding summary before it was generated!"
            return [[0.1] * 10 for _ in texts]

        mock_llm_service._summarize_text = AsyncMock(side_effect=delayed_summary)
        mock_llm_service._get_embeddings_batch = AsyncMock(side_effect=track_embeddings)

        # Simple 2-node tree (will create 1 parent)
        chunks = ["Chunk 1", "Chunk 2"]

        result = await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=1,
            max_embedding_concurrency=1,
            embedding_batch_size=10,
        )

        # Should have 3 nodes total (2 leaves + 1 parent)
        assert len(result) == 3

        # All should have embeddings
        for node in result:
            assert node.embedding is not None

        # Should have made embedding calls for all nodes
        total_embedded = sum(embedding_calls)
        assert total_embedded == 3, f"Expected 3 embeddings, got {total_embedded}"
