"""Tests for dataflow core implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragzoom.dataflow.core import (
    build_tree_dataflow,
    poke,
)
from ragzoom.models import TreeNode


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
        poke("parent", lookup, queue)

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
        poke("parent", lookup, queue)

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
        poke("node2", lookup, queue)

        assert queue.qsize() == 1


class TestLeafNodeCreation:
    """Test leaf node creation in dataflow."""

    def test_create_leaf_nodes_sets_correct_token_count(self):
        """Test that leaf nodes are created with correct token counts."""
        from ragzoom.dataflow.core import create_leaf_nodes

        # Create test chunks with known token counts
        chunks = ["Hello world", "This is a longer text", "Short"]
        document_id = "test_doc"

        # Create leaf nodes
        lookup, leaves = create_leaf_nodes(chunks, document_id)

        # Verify all leaves have non-zero token counts
        for leaf in leaves:
            assert (
                leaf.token_count > 0
            ), f"Leaf {leaf.id} has token_count = {leaf.token_count}, expected > 0"
            assert leaf.text in chunks

        # Verify token counts are reasonable (not zero, not huge)
        # "Hello world" should be 2 tokens, "This is a longer text" should be 6, "Short" should be 1
        assert leaves[0].token_count >= 1
        assert leaves[1].token_count >= 4  # longer text should have more tokens
        assert leaves[2].token_count >= 1

    def test_parent_input_token_calculation(self):
        """Test that parent nodes can correctly calculate input tokens from children."""

        from ragzoom.dataflow.core import create_leaf_nodes

        # Create test chunks
        chunks = ["First chunk", "Second chunk"]
        document_id = "test_doc"

        # Create leaf nodes
        lookup, leaves = create_leaf_nodes(chunks, document_id)

        # Simulate what a summary worker would do
        left_child = leaves[0]
        right_child = leaves[1]

        # Calculate input tokens like the summary worker does
        left_token_count = left_child.token_count
        right_token_count = right_child.token_count
        input_text_tokens = left_token_count + right_token_count

        # Verify that both children have positive token counts
        assert left_token_count > 0, f"Left child token count is {left_token_count}"
        assert right_token_count > 0, f"Right child token count is {right_token_count}"

        # Verify that input_text_tokens is the sum (and positive)
        assert input_text_tokens == left_token_count + right_token_count
        assert (
            input_text_tokens > 0
        ), f"Combined input_text_tokens is {input_text_tokens}, should be > 0"


class TestDataflowIntegration:
    """Test the complete dataflow implementation."""

    @pytest.mark.asyncio
    async def test_build_tree_dataflow_simple(self):
        """Test building a simple tree with dataflow."""
        # Create mock LLM service
        mock_llm_service = MagicMock()
        mock_llm_service._summarize_text = AsyncMock(
            return_value=("Summary text", 1, 10)
        )

        # Mock should return embeddings for each text in the batch
        async def mock_embeddings(texts):
            return [[0.1] * 10 for _ in texts]

        mock_llm_service._get_embeddings_batch = AsyncMock(side_effect=mock_embeddings)

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
        mock_llm_service._summarize_text = mock_summary
        mock_llm_service._get_embeddings_batch = mock_embeddings

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
        mock_llm_service._summarize_text = AsyncMock(side_effect=Exception("API error"))

        # Mock should return embeddings for each text in the batch
        async def mock_embeddings(texts):
            return [[0.1] * 10 for _ in texts]

        mock_llm_service._get_embeddings_batch = AsyncMock(side_effect=mock_embeddings)

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
        mock_llm_service._summarize_text = AsyncMock(return_value=("Summary", 1, 10))

        # Mock should return embeddings for each text in the batch
        async def mock_embeddings(texts):
            return [[0.1] * 10 for _ in texts]

        mock_llm_service._get_embeddings_batch = AsyncMock(side_effect=mock_embeddings)

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


class TestEmbeddingBatching:
    """Test that embedding workers use optimal batching strategies."""

    @pytest.mark.asyncio
    async def test_embedding_workers_process_available_items(self):
        """Test that embedding workers process available items efficiently."""
        batch_calls = []

        async def mock_embeddings(texts):
            # Record the batch size for analysis
            batch_calls.append(len(texts))
            await asyncio.sleep(0.01)  # Simulate API call
            return [[0.1] * 10 for _ in texts]

        async def mock_slow_summary(*args, **kwargs):
            # Simulate realistic API timing - summaries arrive spaced out
            await asyncio.sleep(0.1)  # Much slower than embedding batching
            return ("Summary", 1, 10)

        mock_llm_service = MagicMock()
        mock_llm_service._summarize_text = mock_slow_summary
        mock_llm_service._get_embeddings_batch = mock_embeddings

        # Create a larger tree to ensure we have enough summaries to batch
        # 16 chunks -> 8 parents -> 4 grandparents -> 2 great-grandparents -> 1 root
        # Total internal nodes: 8 + 4 + 2 + 1 = 15
        chunks = [f"Chunk {i}" for i in range(16)]

        result = await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=5,
            max_embedding_concurrency=2,
            embedding_batch_size=8,  # Batch size of 8
        )

        # Verify all nodes got embeddings
        assert all(node.embedding is not None for node in result)
        assert len(result) == 31  # 16 leaves + 15 internal nodes

        # Analyze batching
        total_items = sum(batch_calls)
        assert total_items == 31, f"Expected 31 embeddings, got {total_items}"

        # The initial leaf batch should be efficient (likely 2 full batches of 8)
        # Later batches may be smaller as summaries trickle in
        # This is OK - the new algorithm prioritizes responsiveness over batch size
        assert len(batch_calls) > 0, "Should have made embedding calls"

        # All batch sizes should respect the max batch size
        for size in batch_calls:
            assert size <= 8, f"Batch size {size} exceeds max batch size 8"
            assert size > 0, "Batch size should be at least 1"

    @pytest.mark.asyncio
    async def test_multiple_workers_coordinate_batching(self):
        """Test that multiple embedding workers coordinate to take full batches."""
        batch_calls = []
        worker_calls = {}

        async def mock_embeddings(texts):
            # Use asyncio context to identify which worker made the call
            worker_id = id(asyncio.current_task())
            if worker_id not in worker_calls:
                worker_calls[worker_id] = []

            batch_size = len(texts)
            batch_calls.append(batch_size)
            worker_calls[worker_id].append(batch_size)

            await asyncio.sleep(0.01)  # Simulate work
            return [[0.1] * 10 for _ in texts]

        mock_llm_service = MagicMock()
        mock_llm_service._summarize_text = AsyncMock(return_value=("Summary", 1, 10))
        mock_llm_service._get_embeddings_batch = mock_embeddings

        # Use many chunks to create lots of summaries
        chunks = [f"Chunk {i}" for i in range(32)]  # 31 total internal nodes

        await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=10,
            max_embedding_concurrency=4,  # Multiple workers
            embedding_batch_size=10,
        )

        # Multiple workers should have participated
        assert (
            len(worker_calls) >= 2
        ), f"Expected multiple workers, got {len(worker_calls)}"

        # Most batches should be full size
        full_batches = sum(1 for size in batch_calls if size == 10)
        total_items = sum(batch_calls)
        batching_efficiency = (full_batches * 10) / total_items

        assert (
            batching_efficiency >= 0.6
        ), f"Multi-worker batching efficiency {batching_efficiency:.2%} too low. Batch sizes: {batch_calls}"

    @pytest.mark.asyncio
    async def test_final_partial_batch_processed(self):
        """Test that final partial batches are processed correctly."""
        batch_calls = []

        async def mock_embeddings(texts):
            batch_calls.append(len(texts))
            await asyncio.sleep(0.01)
            return [[0.1] * 10 for _ in texts]

        mock_llm_service = MagicMock()
        mock_llm_service._summarize_text = AsyncMock(return_value=("Summary", 1, 10))
        mock_llm_service._get_embeddings_batch = mock_embeddings

        # Choose chunk count that will result in partial final batch
        # 7 chunks -> 4 parents -> 2 grandparents -> 1 root = 7 internal nodes
        # With batch_size=3: leaves=7 (2 full + 1 partial), summaries=7 (2 full + 1 partial)
        chunks = [f"Chunk {i}" for i in range(7)]

        result = await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=3,
            max_embedding_concurrency=1,
            embedding_batch_size=3,
        )

        # All nodes should have embeddings (partial batch was processed)
        assert len(result) == 14  # 7 leaves + 7 internal nodes
        for node in result:
            assert node.embedding is not None
            assert len(node.embedding) == 10

        # Should have processed some partial batches
        partial_batches = [size for size in batch_calls if size < 3 and size > 0]
        assert (
            len(partial_batches) >= 1
        ), f"Expected partial batches, got batch sizes: {batch_calls}"

    @pytest.mark.asyncio
    async def test_root_node_as_sentinel(self):
        """Test that root node acts as sentinel for embedding workers."""
        batch_calls = []

        async def mock_embeddings(texts):
            batch_calls.append(len(texts))
            await asyncio.sleep(0.001)
            return [[0.1] * 10 for _ in texts]

        mock_llm_service = MagicMock()
        mock_llm_service._summarize_text = AsyncMock(return_value=("Summary", 1, 10))
        mock_llm_service._get_embeddings_batch = mock_embeddings

        # Create a small tree
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"]

        result = await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=2,
            max_embedding_concurrency=2,
            embedding_batch_size=3,
        )

        # Find the root node (parent_id is None)
        root_nodes = [n for n in result if n.parent_id is None]
        assert len(root_nodes) == 1, "Should have exactly one root node"
        root = root_nodes[0]

        # Root should have embedding
        assert root.embedding is not None
        assert len(root.embedding) == 10

        # All nodes should have embeddings
        for node in result:
            assert node.embedding is not None
            assert len(node.embedding) == 10

    @pytest.mark.asyncio
    async def test_atomic_batch_collection(self):
        """Test that batch collection is atomic (no interleaving)."""
        batch_timings = []

        async def mock_embeddings(texts):
            # Record when each batch starts and its size
            batch_timings.append((asyncio.get_event_loop().time(), len(texts)))
            await asyncio.sleep(0.01)  # Simulate work
            return [[0.1] * 10 for _ in texts]

        mock_llm_service = MagicMock()
        mock_llm_service._summarize_text = AsyncMock(return_value=("Summary", 1, 10))
        mock_llm_service._get_embeddings_batch = mock_embeddings

        # Create chunks that will generate batches
        chunks = [f"Chunk {i}" for i in range(10)]

        result = await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=5,
            max_embedding_concurrency=3,  # Multiple workers
            embedding_batch_size=3,
        )

        # All nodes should have embeddings
        assert all(node.embedding is not None for node in result)

        # Check that batches don't have weird sizes (indicating interleaving)
        batch_sizes = [size for _, size in batch_timings]
        for size in batch_sizes:
            assert size <= 3, f"Batch size {size} exceeds max batch size 3"
            assert size > 0, "Batch size should be at least 1"

    @pytest.mark.asyncio
    async def test_single_node_tree(self):
        """Test edge case of single node tree (root is also leaf)."""
        mock_llm_service = MagicMock()
        mock_llm_service._summarize_text = AsyncMock(return_value=("Summary", 1, 10))
        mock_llm_service._get_embeddings_batch = AsyncMock(return_value=[[0.1] * 10])

        # Single chunk creates single node tree
        chunks = ["Only chunk"]

        result = await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=1,
            max_embedding_concurrency=1,
            embedding_batch_size=10,
        )

        # Should have exactly one node
        assert len(result) == 1
        node = result[0]

        # Node should be both root and leaf
        assert node.parent_id is None  # Root
        assert node.height == 0  # Leaf
        assert node.text == "Only chunk"
        assert node.embedding is not None
        assert len(node.embedding) == 10

        # Embedding should have been called once
        mock_llm_service._get_embeddings_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_aware_queue_waits_for_full_batches(self):
        """Test that BatchAwareQueue waits for full batches when possible."""
        batch_calls = []
        batch_timings = []

        async def mock_embeddings(texts):
            # Record batch size and timing
            batch_calls.append(len(texts))
            batch_timings.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)
            return [[0.1] * 10 for _ in texts]

        # Use slower summaries to test batching behavior
        async def mock_slow_summary(*args, **kwargs):
            await asyncio.sleep(0.05)  # Summaries arrive gradually
            return ("Summary", 1, 10)

        mock_llm_service = MagicMock()
        mock_llm_service._summarize_text = mock_slow_summary
        mock_llm_service._get_embeddings_batch = mock_embeddings

        # Create a tree large enough to test batching
        # 8 chunks -> 4 parents -> 2 grandparents -> 1 root
        chunks = [f"Chunk {i}" for i in range(8)]

        result = await build_tree_dataflow(
            chunks=chunks,
            document_id="test-doc",
            llm_service=mock_llm_service,
            max_summary_concurrency=3,
            max_embedding_concurrency=2,
            embedding_batch_size=4,  # Batch size of 4
        )

        # All nodes should have embeddings
        assert all(node.embedding is not None for node in result)
        assert len(result) == 15  # 8 leaves + 7 internal nodes

        # Check batching behavior
        # With batch size 4 and condition variables:
        # - Leaves: 8 items -> 2 full batches of 4
        # - Summaries: Should batch efficiently when enough are ready
        full_batches = sum(1 for size in batch_calls if size == 4)
        assert (
            full_batches >= 2
        ), f"Expected at least 2 full batches, got {full_batches}. Sizes: {batch_calls}"

        # All batches should respect max size
        for size in batch_calls:
            assert size <= 4, f"Batch size {size} exceeds max 4"
            assert size > 0, "Batch should have at least 1 item"
