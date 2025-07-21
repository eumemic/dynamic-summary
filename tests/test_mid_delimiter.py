"""Tests for <<<MID>>> delimiter functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import RetrievalResult
from ragzoom.store import Store


class TestMidDelimiter:
    """Test <<<MID>>> delimiter functionality in tree building and assembly."""

    @pytest.fixture
    def config(self):
        """Mock config with test settings."""
        config = MagicMock(spec=RagZoomConfig)
        config.openai_api_key = "test-key"
        config.summary_model = "gpt-4o"
        config.summary_temperature = 0.3
        config.slope_cap = True
        config.leaf_tokens = 200
        config.adjacent_context_tokens = 75
        config.embedding_model = "text-embedding-3-small"
        config.embedding_dimensions = None
        return config

    @pytest.fixture
    def store(self):
        """Mock store."""
        return MagicMock(spec=Store)

    @pytest.fixture
    def tree_builder(self, config, store):
        """Tree builder with mocked dependencies."""
        builder = TreeBuilder(config, store, max_concurrent=1)
        builder.client = AsyncMock()
        builder.splitter = MagicMock()
        builder.splitter.tokenizer.encode.return_value = list(range(100))  # Mock tokens
        return builder

    @pytest.fixture
    def assembler(self, config, store):
        """Assembler with mocked dependencies."""
        return Assembler(config, store)

    @pytest.mark.asyncio
    async def test_summarize_text_with_mid_delimiter(self, tree_builder):
        """Test that _summarize_text includes <<<MID>>> delimiter."""
        # Mock LLM response with <<<MID>>>
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "Chapter 1 content <<<MID>>> Chapter 2 content"
        )
        tree_builder.client.chat.completions.create.return_value = mock_response

        # Call method
        summary, mid_offset = await tree_builder._summarize_text(
            "Chapter 1 text", "Chapter 2 text", 100
        )

        # Check results
        assert summary == "Chapter 1 content <<<MID>>> Chapter 2 content"
        assert mid_offset == 18  # Position of <<<MID>>>

        # Check prompt structure
        call_args = tree_builder.client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_prompt = messages[1]["content"]

        assert "[FIRST HALF]" in user_prompt
        assert "[SECOND HALF]" in user_prompt
        assert "<<<MID>>>" in user_prompt
        assert "Chapter 1 text" in user_prompt
        assert "Chapter 2 text" in user_prompt

    @pytest.mark.asyncio
    async def test_summarize_text_no_mid_delimiter(self, tree_builder):
        """Test handling when LLM doesn't include <<<MID>>> - should retry and eventually fail."""
        # Mock LLM response without <<<MID>>> (always fails)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Summary without delimiter"
        tree_builder.client.chat.completions.create.return_value = mock_response

        # Should raise ValueError after max attempts
        with pytest.raises(
            ValueError, match="LLM consistently failing to include required delimiter"
        ):
            await tree_builder._summarize_text("Left text", "Right text", 100)

    @pytest.mark.asyncio
    async def test_summarize_text_retry_success(self, tree_builder):
        """Test successful retry when LLM initially fails to include <<<MID>>>."""
        # Mock LLM responses: first fails, second succeeds
        responses = [
            MagicMock(),  # First attempt - no delimiter
            MagicMock(),  # Second attempt - with delimiter
        ]
        responses[0].choices[0].message.content = "Summary without delimiter"
        responses[1].choices[0].message.content = "First half <<<MID>>> Second half"

        tree_builder.client.chat.completions.create.side_effect = responses

        summary, mid_offset = await tree_builder._summarize_text(
            "Left text", "Right text", 100
        )

        assert summary == "First half <<<MID>>> Second half"
        assert mid_offset == 11  # Position of <<<MID>>>
        assert tree_builder.client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_process_node_pair_stores_mid_offset(self, tree_builder):
        """Test that node creation includes mid_offset."""
        # Mock dependencies
        tree_builder.client.chat.completions.create.return_value = MagicMock()
        tree_builder.client.chat.completions.create.return_value.choices[
            0
        ].message.content = "Left <<<MID>>> Right"
        tree_builder.client.embeddings.create.return_value = MagicMock()
        tree_builder.client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3])
        ]

        # Mock store methods
        left_node = MagicMock()
        left_node.span_start = 0
        left_node.span_end = 50
        right_node = MagicMock()
        right_node.span_start = 50
        right_node.span_end = 100
        tree_builder.store.get_node.side_effect = [left_node, right_node]

        # Mock the _update_parent_reference method to avoid DB complexity
        tree_builder._update_parent_reference = MagicMock()

        # Call method
        await tree_builder._process_node_pair(
            "left_id", "Left text", "right_id", "Right text", None, None, "doc_id"
        )

        # Check that add_node was called with mid_offset
        tree_builder.store.add_node.assert_called_once()
        call_kwargs = tree_builder.store.add_node.call_args.kwargs
        assert "mid_offset" in call_kwargs
        assert (
            call_kwargs["mid_offset"] == 5
        )  # Position of <<<MID>>> in "Left <<<MID>>> Right"

    @pytest.mark.skip(
        reason="Legacy assembler test, will be removed with DP implementation"
    )
    def test_assembly_with_exact_span_deduplication(self, assembler):
        """Test that the assembler correctly handles frontiers that would have overlaps
        if not for the exact span deduplication logic."""
        # Create mock retrieval result
        retrieval_result = MagicMock(spec=RetrievalResult)
        # Note: frontier_nodes field no longer exists, test uses legacy assembly
        retrieval_result.coverage_map = set()

        # Mock nodes - node1 and node2 have EXACT same span
        node1 = MagicMock()
        node1.span_start = 0
        node1.span_end = 100
        node1.mid_offset = None
        node1.text = "First text"
        node1.depth = 0

        node2 = MagicMock()  # Exact same span as node1
        node2.span_start = 0
        node2.span_end = 100
        node2.mid_offset = None
        node2.text = "Duplicate span text"
        node2.depth = 0

        node3 = MagicMock()  # Different span
        node3.span_start = 200
        node3.span_end = 300
        node3.mid_offset = None
        node3.text = "Third text"
        node3.depth = 0

        # Mock get_node to return the correct nodes for sorting and processing
        def mock_get_node(node_id):
            if node_id == "node1":
                return node1
            elif node_id == "node2":
                return node2
            elif node_id == "node3":
                return node3
            return None

        assembler.store.get_node.side_effect = mock_get_node
        # Mock get_children to return no children (these are leaf nodes)
        assembler.store.get_children.return_value = (None, None)
        assembler.config.slope_cap = False
        assembler.config.smoothing_pass_enabled = False

        result = assembler.assemble(retrieval_result)

        # Should only include node1 and node3 (node2 has exact same span as node1)
        assert result == "First text\n\nThird text"
