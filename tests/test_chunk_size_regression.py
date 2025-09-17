"""SQLite-based regression test for chunk size issue where chunks were 4x larger than configured.

SQLite-based testing with focus on assembly and
chunk size behavior validation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol
from ragzoom.document_store import DocumentStore
from ragzoom.index import TreeBuilder
from ragzoom.splitter import TextSplitter
from ragzoom.utils.tokenization import tokenizer


@pytest.mark.usefixtures("sqlite_backend")
class TestChunkSizeRegressionSQLite:
    """Test that chunks are created at the correct token size using SQLite backend."""

    @pytest.fixture
    def config(self) -> IndexConfig:
        """Create test config with specific leaf token size."""
        return IndexConfig.load(
            target_chunk_tokens=200,
        )

    def test_splitter_creates_correct_chunk_size(self, config: IndexConfig) -> None:
        """Test that text splitter creates chunks of approximately the configured token size."""
        splitter = TextSplitter(config)
        # Use shared tokenizer instead of creating new instance

        # Create a test document with known content
        # Moby Dick opening - should be split into multiple chunks
        test_text = (
            """
        Call me Ishmael. Some years ago—never mind how long precisely—having
        little or no money in my purse, and nothing particular to interest me
        on shore, I thought I would sail about a little and see the watery part
        of the world. It is a way I have of driving off the spleen and
        regulating the circulation. Whenever I find myself growing grim about
        the mouth; whenever it is a damp, drizzly November in my soul; whenever
        I find myself involuntarily pausing before coffin warehouses, and
        bringing up the rear of every funeral I meet; and especially whenever
        my hypos get such an upper hand of me, that it requires a strong moral
        principle to prevent me from deliberately stepping into the street, and
        methodically knocking people's hats off—then, I account it high time to
        get to sea as soon as I can. This is my substitute for pistol and ball.
        With a philosophical flourish Cato throws himself upon his sword; I
        quietly take to the ship. There is nothing surprising in this. If they
        but knew it, almost all men in their degree, some time or other,
        cherish very nearly the same feelings towards the ocean with me.

        There now is your insular city of the Manhattoes, belted round by
        wharves as Indian isles by coral reefs—commerce surrounds it with her
        surf. Right and left, the streets take you waterward. Its extreme
        downtown is the battery, where that noble mole is washed by waves, and
        cooled by breezes, which a few hours previous were out of sight of
        land. Look at the crowds of water-gazers there.
        """
            * 5
        )  # Repeat to ensure we get multiple chunks

        chunks = splitter.split_text(test_text)

        # Check that chunks are created
        assert len(chunks) > 1, "Text should be split into multiple chunks"

        # Check each chunk's token size
        for i, chunk in enumerate(chunks):
            token_count = tokenizer.count_tokens(chunk)

            # Allow more tolerance due to boundary splitting
            # The splitter respects sentence boundaries, so chunks can be smaller
            min_tokens = int(config.target_chunk_tokens * 0.4)  # Allow 60% smaller
            max_tokens = int(config.target_chunk_tokens * 1.2)  # Allow 20% larger

            # Last chunk can be even smaller
            if i < len(chunks) - 1:
                assert (
                    min_tokens <= token_count <= max_tokens
                ), f"Chunk {i} has {token_count} tokens, expected ~{config.target_chunk_tokens}"
            else:
                # Last chunk just needs to be non-empty
                assert (
                    token_count > 0 and token_count <= max_tokens
                ), f"Last chunk {i} has {token_count} tokens, should be > 0 and <= {max_tokens}"

        # Check average chunk size - should be reasonable but can be lower due to boundaries
        avg_tokens = sum(tokenizer.count_tokens(chunk) for chunk in chunks) / len(
            chunks
        )
        assert (
            50 <= avg_tokens <= config.target_chunk_tokens * 1.2
        ), f"Average chunk size {avg_tokens} tokens is outside reasonable range (expected 50-{int(config.target_chunk_tokens * 1.2)})"

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(2.0)
    async def test_indexed_chunks_have_correct_size(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
        monkeypatch: pytest.MonkeyPatch,
        vector_index: _VectorIndexProtocol,
    ) -> None:
        """Test that indexed chunks in the database have the correct token size."""
        # Set up test environment
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create separate configs
        index_config = IndexConfig.load(target_chunk_tokens=200)
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test-key"),
        )

        # Create document store
        doc_store = sqlite_store_factory("test-doc")

        # Create test document with more varied content
        test_doc = (
            """Once upon a time in a distant kingdom, there lived a wise old king who ruled with fairness and justice.
        The kingdom prospered under his reign, with fertile lands yielding abundant harvests and trade routes bringing wealth from far and wide.
        The people were happy and content, living in peace and harmony. Children played in the streets without fear, and merchants conducted their business honestly.
        However, not all was perfect in this idyllic realm. In the shadows lurked those who envied the king's success and plotted against him.
        They whispered in dark corners and made secret alliances, waiting for the right moment to strike.
        The king, aware of these threats, surrounded himself with loyal advisors and brave knights who would defend the kingdom with their lives.
        """
            * 50
        )  # Create a larger, more realistic document

        # Mock API responses
        mock_async_client = MagicMock()

        # Mock embeddings (one per chunk) - needs to be async
        async def mock_embeddings(*args: object, **kwargs: object) -> MagicMock:
            # Get the actual number of input texts
            input_texts = kwargs.get("input", [])
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            num_embeddings = len(cast(list[str], input_texts))
            return MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536) for _ in range(num_embeddings)]
            )

        mock_async_client.embeddings.create.side_effect = mock_embeddings

        # Mock summaries with mid delimiters - needs to be async
        async def mock_chat_completion(*args: object, **kwargs: object) -> MagicMock:
            response = MagicMock()
            response.choices[0].message.content = "Summary of part 1 and part 2"
            return response

        mock_async_client.chat.completions.create.side_effect = mock_chat_completion

        # Mock the document setup in the store
        doc_store.set_metadata(
            file_path=None,
            content_hash=doc_store.compute_content_hash(test_doc),
            chunk_count=0,
            embedding_model=index_config.embedding_model,
            summary_model=index_config.summary_model,
        )

        # Index the document
        with patch(
            "ragzoom.services.llm_service.AsyncOpenAI", return_value=mock_async_client
        ):
            builder = TreeBuilder(
                index_config,
                doc_store,
                vector_index,
                api_key=operational_config.openai_api_key,
            )
            await builder.add_document_async(test_doc)

        # Get all leaf nodes for this document
        leaf_nodes = doc_store.nodes.get_leaves()

        assert len(leaf_nodes) > 0, "Should have created leaf nodes"

        # Check each leaf node's token count
        for i, node in enumerate(leaf_nodes):
            token_count = tokenizer.count_tokens(node.text)

            # Allow more flexibility for the last chunk which might be smaller
            if i == len(leaf_nodes) - 1:
                # Last chunk can be smaller
                min_tokens = 20  # Minimum viable chunk
                max_tokens = int(index_config.target_chunk_tokens * 1.2)
            else:
                # Regular chunks should be close to configured size
                min_tokens = int(
                    index_config.target_chunk_tokens * 0.7
                )  # Allow 30% smaller
                max_tokens = int(
                    index_config.target_chunk_tokens * 1.3
                )  # Allow 30% larger

            assert (
                min_tokens <= token_count <= max_tokens
            ), f"Leaf node {node.id} (chunk {i+1}/{len(leaf_nodes)}) has {token_count} tokens, expected ~{index_config.target_chunk_tokens}"

        # Check that parent nodes (summaries) are also reasonable size
        all_nodes = doc_store.nodes.get_all()
        parent_nodes = [
            node
            for node in all_nodes
            if hasattr(node, "left_child_id")
            and hasattr(node, "right_child_id")
            and (node.left_child_id is not None or node.right_child_id is not None)
        ]

        for node in parent_nodes:
            token_count = tokenizer.count_tokens(node.text)

            # Parent summaries should also be roughly the same size
            max_summary_tokens = (
                index_config.target_chunk_tokens * 2
            )  # Allow up to 2x for summaries

            assert (
                token_count <= max_summary_tokens
            ), f"Parent node {node.id} has {token_count} tokens, too large"

    @pytest.fixture
    def doc_store_with_chunks(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        """Create a document store with example chunk nodes for testing."""
        doc_store = sqlite_store_factory("chunk-test-doc")

        # Create nodes with various token sizes to test chunk size behavior
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            # Small chunks (under target)
            {
                "node_id": "small1",
                "text": "Short text.",
                "embedding": np.zeros(1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 11,
                "document_id": "chunk-test-doc",
                "token_count": 2,
                "height": 0,
                "path": "00",
            },
            {
                "node_id": "small2",
                "text": "Another short text.",
                "embedding": np.zeros(1536, dtype=np.float64),
                "span_start": 12,
                "span_end": 31,
                "document_id": "chunk-test-doc",
                "token_count": 3,
                "height": 0,
                "path": "01",
            },
            # Target-size chunks (around 200 tokens)
            {
                "node_id": "target1",
                "text": " ".join(["word"] * 200),  # Approximately 200 tokens
                "embedding": np.zeros(1536, dtype=np.float64),
                "span_start": 32,
                "span_end": 1232,
                "document_id": "chunk-test-doc",
                "token_count": 200,
                "height": 0,
                "path": "10",
            },
            {
                "node_id": "target2",
                "text": " ".join(["token"] * 195),  # Approximately 195 tokens
                "embedding": np.zeros(1536, dtype=np.float64),
                "span_start": 1233,
                "span_end": 2408,
                "document_id": "chunk-test-doc",
                "token_count": 195,
                "height": 0,
                "path": "11",
            },
            # Parent nodes (summaries)
            {
                "node_id": "left_parent",
                "text": "Summary of small chunks.",
                "embedding": np.zeros(1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 31,
                "document_id": "chunk-test-doc",
                "height": 1,
                "left_child_id": "small1",
                "right_child_id": "small2",
                "token_count": 4,
                "path": "0",
            },
            {
                "node_id": "right_parent",
                "text": "Summary of target-size chunks.",
                "embedding": np.zeros(1536, dtype=np.float64),
                "span_start": 32,
                "span_end": 2408,
                "document_id": "chunk-test-doc",
                "height": 1,
                "left_child_id": "target1",
                "right_child_id": "target2",
                "token_count": 5,
                "path": "1",
            },
            {
                "node_id": "root",
                "text": "Overall summary of all chunks.",
                "embedding": np.zeros(1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 2408,
                "document_id": "chunk-test-doc",
                "height": 2,
                "left_child_id": "left_parent",
                "right_child_id": "right_parent",
                "token_count": 6,
                "path": "",
            },
        ]

        doc_store.nodes.add_batch(nodes)

        # Update parent references
        doc_store.nodes.update_parent_references_batch(
            [
                ("small1", "left_parent"),
                ("small2", "left_parent"),
                ("target1", "right_parent"),
                ("target2", "right_parent"),
                ("left_parent", "root"),
                ("right_parent", "root"),
            ]
        )

        return doc_store

    def test_chunk_size_analysis(self, doc_store_with_chunks: DocumentStore) -> None:
        """Test analysis of chunk sizes in the document store."""
        # Get all leaf nodes
        leaf_nodes = doc_store_with_chunks.nodes.get_leaves()

        assert len(leaf_nodes) == 4, "Should have 4 leaf nodes"

        # Verify chunk sizes are as expected
        small_chunks = [node for node in leaf_nodes if node.token_count < 10]
        target_chunks = [node for node in leaf_nodes if node.token_count >= 190]

        assert len(small_chunks) == 2, "Should have 2 small chunks"
        assert len(target_chunks) == 2, "Should have 2 target-size chunks"

        # Verify parent nodes exist and have reasonable sizes
        all_nodes = doc_store_with_chunks.nodes.get_all()
        parent_nodes = [
            node
            for node in all_nodes
            if hasattr(node, "left_child_id")
            and hasattr(node, "right_child_id")
            and (node.left_child_id is not None or node.right_child_id is not None)
        ]

        assert len(parent_nodes) == 3, "Should have 3 parent nodes"

        # Verify all parent nodes have reasonable token counts
        for node in parent_nodes:
            assert (
                node.token_count <= 50
            ), f"Parent node {node.id} should have reasonable token count"

    def test_chunk_distribution_analysis(
        self, doc_store_with_chunks: DocumentStore
    ) -> None:
        """Test analysis of chunk size distribution."""
        leaf_nodes = doc_store_with_chunks.nodes.get_leaves()

        token_counts = [node.token_count for node in leaf_nodes]

        # Calculate distribution metrics
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / len(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0

        # Verify distribution characteristics
        assert total_tokens == 400, f"Expected total tokens 400, got {total_tokens}"
        assert avg_tokens == 100.0, f"Expected average tokens 100.0, got {avg_tokens}"
        assert min_tokens == 2, f"Expected min tokens 2, got {min_tokens}"
        assert max_tokens == 200, f"Expected max tokens 200, got {max_tokens}"

        # Test specific chunk characteristics
        small_chunks = [node for node in leaf_nodes if node.token_count < 10]
        large_chunks = [node for node in leaf_nodes if node.token_count >= 190]

        assert len(small_chunks) == 2, "Should have 2 small chunks"
        assert len(large_chunks) == 2, "Should have 2 large chunks"

        # Verify no chunks are excessively large (4x target would be 800 tokens)
        oversized_chunks = [node for node in leaf_nodes if node.token_count > 400]
        assert len(oversized_chunks) == 0, "Should have no oversized chunks"
