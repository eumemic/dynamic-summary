"""Regression test for chunk size issue where chunks were 4x larger than configured."""

from unittest.mock import MagicMock, patch

import pytest
import tiktoken

from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.splitter import TextSplitter
from ragzoom.store import Store


class TestChunkSizeRegression:
    """Test that chunks are created at the correct token size."""

    @pytest.fixture
    def config(self):
        """Create test config with specific leaf token size."""
        return RagZoomConfig(
            leaf_tokens=200,
            leaf_overlap_tokens=20,
        )

    def test_splitter_creates_correct_chunk_size(self, config):
        """Test that text splitter creates chunks of approximately the configured token size."""
        splitter = TextSplitter(config)
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Create a test document with known content
        # Moby Dick opening - should be split into multiple chunks
        test_text = """
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
        """ * 5  # Repeat to ensure we get multiple chunks

        chunks = splitter.split_text(test_text)

        # Check that chunks are created
        assert len(chunks) > 1, "Text should be split into multiple chunks"

        # Check each chunk's token size
        for i, chunk in enumerate(chunks):
            token_count = len(tokenizer.encode(chunk))

            # Allow more tolerance due to boundary splitting
            # The splitter respects sentence boundaries, so chunks can be smaller
            min_tokens = int(config.leaf_tokens * 0.4)  # Allow 60% smaller
            max_tokens = int(config.leaf_tokens * 1.2)  # Allow 20% larger

            # Last chunk can be even smaller
            if i < len(chunks) - 1:
                assert min_tokens <= token_count <= max_tokens, \
                    f"Chunk {i} has {token_count} tokens, expected ~{config.leaf_tokens}"
            else:
                # Last chunk just needs to be non-empty
                assert token_count > 0 and token_count <= max_tokens, \
                    f"Last chunk {i} has {token_count} tokens, should be > 0 and <= {max_tokens}"

        # Check average chunk size - should be reasonable but can be lower due to boundaries
        avg_tokens = sum(len(tokenizer.encode(chunk)) for chunk in chunks) / len(chunks)
        assert 50 <= avg_tokens <= config.leaf_tokens * 1.2, \
            f"Average chunk size {avg_tokens} tokens is outside reasonable range (expected 50-{int(config.leaf_tokens * 1.2)})"

    @pytest.mark.asyncio
    async def test_indexed_chunks_have_correct_size(self, tmp_path, monkeypatch):
        """Test that indexed chunks in the database have the correct token size."""
        # Set up test environment
        monkeypatch.setenv("RAGZOOM_SQLITE_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
        monkeypatch.setenv("RAGZOOM_CHROMA_DB_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("RAGZOOM_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("RAGZOOM_LEAF_TOKENS", "200")

        config = RagZoomConfig()
        store = Store(config)
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Create test document with more varied content
        test_doc = """Once upon a time in a distant kingdom, there lived a wise old king who ruled with fairness and justice.
        The kingdom prospered under his reign, with fertile lands yielding abundant harvests and trade routes bringing wealth from far and wide.
        The people were happy and content, living in peace and harmony. Children played in the streets without fear, and merchants conducted their business honestly.
        However, not all was perfect in this idyllic realm. In the shadows lurked those who envied the king's success and plotted against him.
        They whispered in dark corners and made secret alliances, waiting for the right moment to strike.
        The king, aware of these threats, surrounded himself with loyal advisors and brave knights who would defend the kingdom with their lives.
        """ * 50  # Create a larger, more realistic document
        test_file = tmp_path / "test.txt"
        test_file.write_text(test_doc)

        # Mock API responses
        mock_async_client = MagicMock()

        # Mock embeddings (one per chunk) - needs to be async
        async def mock_embeddings(*args, **kwargs):
            return MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536) for _ in range(5)]
            )
        mock_async_client.embeddings.create.side_effect = mock_embeddings

        # Mock summaries with mid delimiters - needs to be async
        async def mock_chat_completion(*args, **kwargs):
            response = MagicMock()
            response.choices[0].message.content = "Summary part 1 <<<MID>>> Summary part 2"
            return response

        mock_async_client.chat.completions.create.side_effect = mock_chat_completion

        # Index the document
        with patch("ragzoom.index.AsyncOpenAI", return_value=mock_async_client):
            builder = TreeBuilder(config, store)
            await builder.add_document_async(str(test_file))

        # Check leaf node sizes
        leaf_nodes = []
        with store.SessionLocal() as session:
            from ragzoom.store import TreeNode
            leaf_nodes = session.query(TreeNode).filter(TreeNode.depth == 0).all()

        assert len(leaf_nodes) > 0, "Should have created leaf nodes"

        # Check each leaf node's token count
        for i, node in enumerate(leaf_nodes):
            token_count = len(tokenizer.encode(node.text))

            # Allow more flexibility for the last chunk which might be smaller
            if i == len(leaf_nodes) - 1:
                # Last chunk can be smaller
                min_tokens = 20  # Minimum viable chunk
                max_tokens = int(config.leaf_tokens * 1.2)
            else:
                # Regular chunks should be close to configured size
                min_tokens = int(config.leaf_tokens * 0.7)  # Allow 30% smaller
                max_tokens = int(config.leaf_tokens * 1.3)  # Allow 30% larger

            assert min_tokens <= token_count <= max_tokens, \
                f"Leaf node {node.id} (chunk {i+1}/{len(leaf_nodes)}) has {token_count} tokens, expected ~{config.leaf_tokens}"

        # Check that parent nodes (summaries) are also reasonable size
        parent_nodes = []
        with store.SessionLocal() as session:
            parent_nodes = session.query(TreeNode).filter(TreeNode.depth > 0).all()

        for node in parent_nodes:
            token_count = len(tokenizer.encode(node.text))

            # Parent summaries should also be roughly the same size
            max_summary_tokens = config.leaf_tokens * 2  # Allow up to 2x for summaries

            assert token_count <= max_summary_tokens, \
                f"Parent node at depth {node.depth} has {token_count} tokens, too large"

        # Close store to prevent file handle leaks
        store.close()

    def test_token_budget_not_exceeded_due_to_large_chunks(self, config):
        """Test that retrieval token budget is not massively exceeded due to chunk size issues."""
        # This tests the symptom we saw: 11k tokens returned for 2k budget
        # With proper chunk sizes, the assembly should not be 5x over budget

        from ragzoom.assemble import Assembler

        # Create a mock store with nodes
        store = MagicMock()

        # Create mock nodes - if chunks are sized correctly, these should be ~200 tokens each
        mock_nodes = []
        for i in range(10):
            node = MagicMock()
            node.id = f"node_{i}"
            node.text = "This is a properly sized chunk. " * 40  # ~200 tokens
            node.depth = 0
            node.span_start = i * 200
            node.span_end = (i + 1) * 200
            node.mid_offset = None
            mock_nodes.append(node)

        store.get_node.side_effect = lambda node_id: next(
            (n for n in mock_nodes if n.id == node_id), None
        )

        # Mock get_children to return no children (these are leaf nodes)
        store.get_children.return_value = (None, None)

        # Mock retrieval result with 10 nodes (should be ~2000 tokens)
        from ragzoom.retrieve import RetrievalResult
        retrieval_result = RetrievalResult(
            node_ids=[n.id for n in mock_nodes],
            scores={n.id: 0.9 for n in mock_nodes},
            coverage_map={n.id: True for n in mock_nodes},
            frontier_nodes=[n.id for n in mock_nodes],
        )

        # Assemble with budget
        assembler = Assembler(config, store)
        assembled_text, token_count = assembler.assemble_with_budget(
            retrieval_result, token_budget=2000
        )

        # With properly sized chunks, assembly shouldn't be wildly over budget
        # Allow up to 50% over due to assembly overhead
        assert token_count <= 3000, \
            f"Assembly produced {token_count} tokens, way over 2000 budget"

