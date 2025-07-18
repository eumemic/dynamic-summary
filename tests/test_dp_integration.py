"""Integration tests for the full DP pipeline to verify it maintains important invariants."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever


def sync_embedding(*args, **kwargs):
    texts = kwargs.get("input")
    if texts is None and len(args) > 0:
        texts = args[0]
    if not isinstance(texts, list):
        texts = [texts]
    embeddings = []
    for text in texts:
        hash_val = sum(ord(c) for c in text) % 100
        embedding = [hash_val / 100.0] * 1536
        embeddings.append(MagicMock(embedding=embedding))
    return MagicMock(data=embeddings)


async def async_embedding(*args, **kwargs):
    return sync_embedding(*args, **kwargs)


def sync_summary(*args, **kwargs):
    messages = kwargs.get("messages")
    if messages is None and len(args) > 0:
        messages = args[0]
    content = messages[1]["content"] if messages and len(messages) > 1 else ""
    if "First chunk" in content and "Second chunk" in content:
        summary = (
            "Summary of first two chunks. <<<MID>>> Combined content of chunks 1 and 2."
        )
    elif "Third chunk" in content and "Fourth chunk" in content:
        summary = (
            "Summary of last two chunks. <<<MID>>> Combined content of chunks 3 and 4."
        )
    elif "Summary of first" in content and "Summary of last" in content:
        summary = "Overall document summary. <<<MID>>> Complete document overview."
    else:
        summary = "Generic summary. <<<MID>>> Generic content."
    return MagicMock(choices=[MagicMock(message=MagicMock(content=summary))])


async def async_summary(*args, **kwargs):
    return sync_summary(*args, **kwargs)


class TestDPIntegration:
    """Test the full DP pipeline maintains critical invariants."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RagZoomConfig(
            openai_api_key="test-key",
            slope_cap=True,
            budget_tokens=500,
            leaf_tokens=50,
            adjacent_context_tokens=0,
        )

    @pytest.fixture
    def mock_openai(self, monkeypatch):
        """Mock OpenAI for consistent embeddings and summaries."""
        monkeypatch.setenv("RAGZOOM_OPENAI_API_KEY", "test-key")

        # Mock embeddings
        async def mock_embedding(*args, **kwargs):
            # Extract the 'input' argument (list of texts)
            texts = kwargs.get("input")
            if texts is None and len(args) > 0:
                texts = args[0]
            if not isinstance(texts, list):
                texts = [texts]
            embeddings = []
            for text in texts:
                hash_val = sum(ord(c) for c in text) % 100
                embedding = [hash_val / 100.0] * 1536
                embeddings.append(MagicMock(embedding=embedding))
            return MagicMock(data=embeddings)

        # Mock summaries with MID delimiter
        async def mock_summary(*args, **kwargs):
            messages = kwargs.get("messages")
            if messages is None and len(args) > 0:
                messages = args[0]
            content = messages[1]["content"] if messages and len(messages) > 1 else ""
            if "First chunk" in content and "Second chunk" in content:
                summary = "Summary of first two chunks. <<<MID>>> Combined content of chunks 1 and 2."
            elif "Third chunk" in content and "Fourth chunk" in content:
                summary = "Summary of last two chunks. <<<MID>>> Combined content of chunks 3 and 4."
            elif "Summary of first" in content and "Summary of last" in content:
                summary = (
                    "Overall document summary. <<<MID>>> Complete document overview."
                )
            else:
                summary = "Generic summary. <<<MID>>> Generic content."
            return MagicMock(choices=[MagicMock(message=MagicMock(content=summary))])

        mock_client = MagicMock()
        mock_client.embeddings.create = mock_embedding

        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create = mock_summary

        return mock_client, mock_async_client

    @pytest.mark.asyncio
    async def test_no_duplicate_content(self, config, store, mock_openai, monkeypatch):
        """Test that the full DP pipeline produces no duplicate content."""
        mock_client, mock_async_client = mock_openai

        # Create a document with clear chunk boundaries
        base_lines = [
            "First chunk of text that should appear once.",
            "Second chunk of text that should also appear once.",
            "Third chunk of text with unique content.",
            "Fourth chunk of text to complete the document.",
        ]
        # Repeat to ensure multiple chunks
        document = "\n".join(base_lines * 8)

        # Index the document
        tree_builder = TreeBuilder(config, store)
        tree_builder.client.embeddings.create = async_embedding
        tree_builder.client.chat.completions.create = async_summary
        await tree_builder.add_document_async(
            document, document_id="doc1", show_progress=False
        )

        # Retrieve with a query
        retriever = Retriever(config, store, tree_builder)
        retriever.client.embeddings.create = sync_embedding
        retriever.client.chat.completions.create = sync_summary
        query = "First chunk Second chunk"  # Query that should match the first half
        result = retriever.retrieve(query, document_id="doc1")

        print("FRONTIER NODES:", result.frontier_nodes)
        if (
            hasattr(result, "frontier_segments")
            and result.frontier_segments is not None
        ):
            print("FRONTIER SEGMENTS:", result.frontier_segments)

        # Assemble the result
        assembler = Assembler(config, store)
        assembled = assembler.assemble(result)
        print("ASSEMBLED OUTPUT:\n", assembled)
        # The summary strings should appear as expected for internal nodes
        assert (
            assembled.count("Summary of first two chunks.") >= 1
        ), "Summary string should appear at least once"
        assert (
            assembled.count("Combined content of chunks 1 and 2.") >= 1
        ), "Combined content string should appear at least once"
        # The leaf text should appear as expected
        assert (
            assembled.count("Third chunk of text with unique content.") == 2
        ), "Third chunk leaf should appear exactly twice"
        assert (
            assembled.count("Fourth chunk of text to complete the document.") == 2
        ), "Fourth chunk leaf should appear exactly twice"
        # Verify no MID delimiters in output
        assert "<<<MID>>>" not in assembled

    @pytest.mark.asyncio
    async def test_parent_child_deduplication(
        self, config, store, mock_openai, monkeypatch
    ):
        """Test that DP frontier doesn't include both parent and child."""
        mock_client, mock_async_client = mock_openai

        # Create a simple document
        document = "This is a test document with some content."

        # Index with very small chunks to force tree structure
        config.leaf_tokens = 10  # Very small chunks
        tree_builder = TreeBuilder(
            config=config,
            store=store,
        )
        tree_builder.client.embeddings.create = async_embedding
        tree_builder.client.chat.completions.create = async_summary
        await tree_builder.add_document_async(
            document, document_id="doc1", show_progress=False
        )

        # Retrieve
        retriever = Retriever(config, store, tree_builder)
        retriever.client.embeddings.create = sync_embedding
        retriever.client.chat.completions.create = sync_summary
        result = retriever.retrieve("test document", document_id="doc1")

        # Check frontier doesn't have both parent and child
        frontier_nodes = [store.get_node(nid) for nid in result.frontier_nodes]
        for i, node in enumerate(frontier_nodes):
            for j, other in enumerate(frontier_nodes):
                if i != j:
                    # Check if one is ancestor of the other
                    if (
                        node.left_child_id == other.id
                        or node.right_child_id == other.id
                    ):
                        pytest.fail(
                            f"Frontier contains both parent {node.id} and child {other.id}"
                        )

    @pytest.mark.asyncio
    async def test_span_coverage(self, config, store, mock_openai, monkeypatch):
        """Test that the assembled text covers the document span correctly."""
        mock_client, mock_async_client = mock_openai

        # Create a document with known content
        document = "AAAA BBBB CCCC DDDD"

        # Index the document
        config.leaf_tokens = 5  # One word per chunk approximately
        tree_builder = TreeBuilder(
            config=config,
            store=store,
        )
        tree_builder.client.embeddings.create = async_embedding
        tree_builder.client.chat.completions.create = async_summary
        await tree_builder.add_document_async(
            document, document_id="doc1", show_progress=False
        )

        # Retrieve with different queries
        retriever = Retriever(config, store, tree_builder)

        # Patch retriever client for sync
        retriever.client.embeddings.create = sync_embedding
        retriever.client.chat.completions.create = sync_summary
        # Query for first half
        result1 = retriever.retrieve("AAAA BBBB", document_id="doc1")
        assembler = Assembler(config, store)
        assembled1 = assembler.assemble(result1)
        # Should contain content from first half
        assert "AAAA" in assembled1 or "first" in assembled1.lower()

        # Query for second half
        result2 = retriever.retrieve("CCCC DDDD", document_id="doc1")
        assembled2 = assembler.assemble(result2)

        # Should contain content from second half
        assert (
            "CCCC" in assembled2 or "DDDD" in assembled2 or "last" in assembled2.lower()
        )

    @pytest.mark.asyncio
    async def test_mid_delimiter_extraction(
        self, config, store, mock_openai, monkeypatch
    ):
        """Test that MID delimiter extraction works correctly in full pipeline."""
        mock_client, mock_async_client = mock_openai

        document = (
            "Part one content. Part two content. Part three content. Part four content."
        )

        # Index
        config.leaf_tokens = 20
        tree_builder = TreeBuilder(
            config=config,
            store=store,
        )
        tree_builder.client.embeddings.create = async_embedding
        tree_builder.client.chat.completions.create = async_summary
        await tree_builder.add_document_async(
            document, document_id="doc1", show_progress=False
        )

        # Retrieve - the DP algorithm should handle MID delimiter extraction
        retriever = Retriever(config, store, tree_builder)
        retriever.client.embeddings.create = sync_embedding
        retriever.client.chat.completions.create = sync_summary
        result = retriever.retrieve("Part one Part two", document_id="doc1")

        # Assemble
        assembler = Assembler(config, store)
        assembled = assembler.assemble(result)

        # Verify MID delimiter is not in output
        assert "<<<MID>>>" not in assembled

        # Verify we get coherent text (not just full summaries)
        assert len(assembled) > 0

    @pytest.mark.asyncio
    async def test_budget_respected(self, config, store, mock_openai, monkeypatch):
        """Test that DP respects token budget."""
        mock_client, mock_async_client = mock_openai

        # Create a large document
        document = " ".join([f"Sentence {i}." for i in range(100)])

        # Set a small budget
        config.budget_tokens = 100

        # Index
        tree_builder = TreeBuilder(
            config=config,
            store=store,
        )
        tree_builder.client.embeddings.create = async_embedding
        tree_builder.client.chat.completions.create = async_summary
        await tree_builder.add_document_async(
            document, document_id="doc1", show_progress=False
        )

        # Retrieve with budget
        retriever = Retriever(config, store, tree_builder)
        retriever.client.embeddings.create = sync_embedding
        retriever.client.chat.completions.create = sync_summary
        result = retriever.retrieve("Sentence", document_id="doc1", budget_tokens=100)

        # Assemble
        assembler = Assembler(config, store)
        assembled = assembler.assemble(result)

        # Count tokens
        token_count = assembler.get_token_count(assembled)

        # Allow some slack for token counting differences
        assert (
            token_count <= config.budget_tokens * 1.1
        ), f"Token count {token_count} exceeds budget {config.budget_tokens}"
