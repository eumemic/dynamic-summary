"""Tests for dataflow core implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragzoom.dataflow.core import (
    AtomicCounter,
    build_tree_dataflow,
    poke,
)
from ragzoom.models import TreeNode


class TestAtomicCounter:
    """Test atomic counter for tracking pending work."""

    @pytest.mark.asyncio
    async def test_atomic_counter_basic(self):
        """Test basic atomic counter operations."""
        counter = AtomicCounter(10)
        assert counter.value == 10

        counter.decrement(3)
        assert counter.value == 7

        counter.decrement(7)
        assert counter.value == 0

    @pytest.mark.asyncio
    async def test_atomic_counter_thread_safe(self):
        """Test that atomic counter is thread-safe."""
        counter = AtomicCounter(100)

        async def decrement_many():
            for _ in range(10):
                counter.decrement(1)
                await asyncio.sleep(0)  # Yield to other tasks

        # Run multiple tasks concurrently
        await asyncio.gather(*[decrement_many() for _ in range(10)])

        assert counter.value == 0


class TestPokeMechanism:
    """Test the poke mechanism for dependency checking."""

    @pytest.mark.asyncio
    async def test_poke_with_all_dependencies_ready(self):
        """Test poke when all dependencies are ready."""
        # Create a simple lookup dict
        lookup = {}
        queue = asyncio.Queue()

        # Create nodes with satisfied dependencies
        left_child = TreeNode(
            id="left",
            text="Left text",
            height=0,
            span_start=0,
            span_end=5,
            path="0",
            document_id="doc1",
            embedding=[],
            token_count=10,
        )
        right_child = TreeNode(
            id="right",
            text="Right text",
            height=0,
            span_start=5,
            span_end=10,
            path="1",
            document_id="doc1",
            embedding=[],
            token_count=10,
        )
        parent = TreeNode(
            id="parent",
            text="",  # Empty string for not yet processed
            height=1,
            span_start=0,
            span_end=10,
            path="",
            document_id="doc1",
            left_child_id="left",
            right_child_id="right",
            embedding=[],
            token_count=0,
        )

        lookup = {"left": left_child, "right": right_child, "parent": parent}

        # Poke the parent - should be queued since children have text
        await poke("parent", lookup, queue)

        assert queue.qsize() == 1
        queued_id = await queue.get()
        assert queued_id == "parent"

    @pytest.mark.asyncio
    async def test_poke_with_missing_dependencies(self):
        """Test poke when dependencies are not ready."""
        lookup = {}
        queue = asyncio.Queue()

        # Create nodes where left child has no text yet
        left_child = TreeNode(
            id="left",
            text="",  # Empty string means not ready
            height=0,
            span_start=0,
            span_end=5,
            path="0",
            document_id="doc1",
            embedding=[],
            token_count=0,
        )
        right_child = TreeNode(
            id="right",
            text="Right text",
            height=0,
            span_start=5,
            span_end=10,
            path="1",
            document_id="doc1",
            embedding=[],
            token_count=10,
        )
        parent = TreeNode(
            id="parent",
            text="",
            height=1,
            span_start=0,
            span_end=10,
            path="",
            document_id="doc1",
            left_child_id="left",
            right_child_id="right",
            embedding=[],
            token_count=0,
        )

        lookup = {"left": left_child, "right": right_child, "parent": parent}

        # Poke the parent - should NOT be queued since left child has no text
        await poke("parent", lookup, queue)

        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_poke_with_preceding_neighbor_dependency(self):
        """Test poke with preceding neighbor dependency."""
        lookup = {}
        queue = asyncio.Queue()

        # Create nodes with preceding neighbor dependency
        node1 = TreeNode(
            id="node1",
            text="First node",
            height=1,
            span_start=0,
            span_end=5,
            path="0",
            document_id="doc1",
            left_child_id="child1",
            right_child_id="child2",
            embedding=[],
            token_count=10,
        )
        node2 = TreeNode(
            id="node2",
            text="",
            height=1,
            span_start=5,
            span_end=10,
            path="1",
            document_id="doc1",
            left_child_id="child3",
            right_child_id="child4",
            preceding_neighbor_id="node1",
            embedding=[],
            token_count=0,
        )

        # Add children with text so they're not blocking
        for i in range(1, 5):
            child = TreeNode(
                id=f"child{i}",
                text=f"Child {i} text",
                height=0,
                span_start=0,
                span_end=5,
                path=str(i),
                document_id="doc1",
                embedding=[],
                token_count=10,
            )
            lookup[f"child{i}"] = child

        lookup["node1"] = node1
        lookup["node2"] = node2

        # Poke node2 - should be queued since all dependencies are ready
        await poke("node2", lookup, queue)

        assert queue.qsize() == 1


class TestDataflowIntegration:
    """Test the complete dataflow implementation."""

    @pytest.mark.asyncio
    async def test_build_tree_dataflow_simple(self):
        """Test building a simple tree with dataflow."""
        # Create mock LLM service
        mock_llm_service = MagicMock()
        mock_llm_service.generate_summary_async = AsyncMock(
            return_value=("Summary text", 1, 10)
        )

        # Mock should return embeddings for each text in the batch
        async def mock_embeddings(texts):
            return [[0.1] * 10 for _ in texts]

        mock_llm_service.generate_embeddings_batch_async = AsyncMock(
            side_effect=mock_embeddings
        )

        # Create simple chunks for testing
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"]

        # Build tree with dataflow
        result = await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=2,
            max_embedding_concurrency=2,
            embedding_batch_size=2,
        )

        # Check that we got nodes with embeddings
        assert len(result) > 0
        for node in result:
            assert node.embedding is not None
            assert len(node.embedding) > 0

    @pytest.mark.asyncio
    async def test_dataflow_respects_concurrency_limits(self):
        """Test that dataflow respects concurrency limits."""
        call_count = {"summary": 0, "embedding": 0}
        max_concurrent_summary = {"value": 0}
        max_concurrent_embedding = {"value": 0}
        current_summary = {"value": 0}
        current_embedding = {"value": 0}

        async def mock_summary(*args, **kwargs):
            call_count["summary"] += 1
            current_summary["value"] += 1
            max_concurrent_summary["value"] = max(
                max_concurrent_summary["value"], current_summary["value"]
            )
            await asyncio.sleep(0.01)  # Simulate work
            current_summary["value"] -= 1
            return "Summary", 1, 10

        async def mock_embeddings(*args, **kwargs):
            call_count["embedding"] += 1
            current_embedding["value"] += 1
            max_concurrent_embedding["value"] = max(
                max_concurrent_embedding["value"], current_embedding["value"]
            )
            await asyncio.sleep(0.01)  # Simulate work
            current_embedding["value"] -= 1
            return [[0.1] * 10] * len(args[0])

        mock_llm_service = MagicMock()
        mock_llm_service.generate_summary_async = mock_summary
        mock_llm_service.generate_embeddings_batch_async = mock_embeddings

        # Create tree with 8 chunks (will create multiple levels)
        chunks = [f"Chunk {i}" for i in range(8)]

        # Build with limited concurrency
        await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=2,  # Limit to 2 concurrent summaries
            max_embedding_concurrency=1,  # Limit to 1 concurrent embedding batch
            embedding_batch_size=4,
        )

        # Check concurrency was respected
        assert max_concurrent_summary["value"] <= 2
        assert max_concurrent_embedding["value"] <= 1

    @pytest.mark.asyncio
    async def test_dataflow_error_handling(self):
        """Test that dataflow handles errors appropriately."""
        mock_llm_service = MagicMock()
        mock_llm_service.generate_summary_async = AsyncMock(
            side_effect=Exception("API error")
        )

        # Mock should return embeddings for each text in the batch
        async def mock_embeddings(texts):
            return [[0.1] * 10 for _ in texts]

        mock_llm_service.generate_embeddings_batch_async = AsyncMock(
            side_effect=mock_embeddings
        )

        chunks = ["Chunk 1", "Chunk 2"]

        # Should raise exception on summary error
        with pytest.raises(Exception, match="API error"):
            await build_tree_dataflow(
                chunks=chunks,
                document_id="test-doc",
                llm_service=mock_llm_service,
                max_summary_concurrency=1,
                max_embedding_concurrency=1,
                embedding_batch_size=2,
            )

    @pytest.mark.asyncio
    async def test_dataflow_produces_complete_tree(self):
        """Test that dataflow produces a complete tree with all nodes."""
        mock_llm_service = MagicMock()
        mock_llm_service.generate_summary_async = AsyncMock(
            return_value=("Summary", 1, 10)
        )

        # Mock should return embeddings for each text in the batch
        async def mock_embeddings(texts):
            return [[0.1] * 10 for _ in texts]

        mock_llm_service.generate_embeddings_batch_async = AsyncMock(
            side_effect=mock_embeddings
        )

        # Create tree with 4 chunks
        chunks = ["A", "B", "C", "D"]

        result = await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=2,
            max_embedding_concurrency=2,
            embedding_batch_size=10,
        )

        # Should have all nodes (4 leaves + 2 parents + 1 root = 7)
        assert len(result) == 7

        # All nodes should have embeddings
        for node in result:
            assert node.embedding is not None
            assert len(node.embedding) == 10
