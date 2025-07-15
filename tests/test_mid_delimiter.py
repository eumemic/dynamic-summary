"""Tests for <<<MID>>> delimiter functionality."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ragzoom.index import TreeBuilder
from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.store import Store, TreeNode
from ragzoom.retrieve import RetrievalResult


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
        config.leaf_overlap_tokens = 20
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
        mock_response.choices[0].message.content = "Chapter 1 content <<<MID>>> Chapter 2 content"
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
        messages = call_args.kwargs['messages']
        user_prompt = messages[1]['content']
        
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
        with pytest.raises(ValueError, match="LLM consistently failing to include required delimiter"):
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
        tree_builder.client.chat.completions.create.return_value.choices[0].message.content = "Left <<<MID>>> Right"
        tree_builder.client.embeddings.create.return_value = MagicMock()
        tree_builder.client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        
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
            "left_id", "Left text", "right_id", "Right text", 
            None, None, 1, "doc_id"
        )
        
        # Check that add_node was called with mid_offset
        tree_builder.store.add_node.assert_called_once()
        call_kwargs = tree_builder.store.add_node.call_args.kwargs
        assert 'mid_offset' in call_kwargs
        assert call_kwargs['mid_offset'] == 5  # Position of <<<MID>>> in "Left <<<MID>>> Right"

    def test_extract_node_text_no_mid_offset(self, assembler):
        """Test text extraction for node without mid_offset."""
        node = MagicMock()
        node.mid_offset = None
        node.text = "Simple text <<<MID>>> content"
        
        result = assembler._extract_node_text(node, set())
        assert result == "Simple text  content"  # <<<MID>>> removed

    def test_extract_node_text_left_child_covered(self, assembler):
        """Test text extraction when left child is covered."""
        node = MagicMock()
        node.mid_offset = 12  # Position of <<<MID>>> in "Parent left <<<MID>>> parent right"
        node.text = "Parent left <<<MID>>> parent right"
        node.id = "parent_id"
        
        left_child = MagicMock()
        left_child.id = "left_id"
        left_child.text = "Detailed left content"
        
        right_child = MagicMock() 
        right_child.id = "right_id"
        
        assembler.store.get_children.return_value = (left_child, right_child)
        coverage_map = {"left_id"}  # Only left child covered
        
        result = assembler._extract_node_text(node, coverage_map)
        expected = "Detailed left content\n\n parent right"  # Left child + parent right half
        assert result == expected

    def test_extract_node_text_right_child_covered(self, assembler):
        """Test text extraction when right child is covered."""
        node = MagicMock()
        node.mid_offset = 12
        node.text = "Parent left <<<MID>>> parent right"
        
        left_child = MagicMock()
        left_child.id = "left_id"
        
        right_child = MagicMock()
        right_child.id = "right_id" 
        right_child.text = "Detailed right content"
        
        assembler.store.get_children.return_value = (left_child, right_child)
        coverage_map = {"right_id"}  # Only right child covered
        
        result = assembler._extract_node_text(node, coverage_map)
        expected = "Parent left \n\nDetailed right content"  # Parent left half + right child
        assert result == expected

    def test_extract_node_text_both_children_covered(self, assembler):
        """Test text extraction when both children are covered."""
        node = MagicMock()
        node.mid_offset = 10
        node.text = "Summary with <<<MID>>> delimiter"
        
        left_child = MagicMock()
        left_child.id = "left_id"
        
        right_child = MagicMock()
        right_child.id = "right_id"
        
        assembler.store.get_children.return_value = (left_child, right_child)
        coverage_map = {"left_id", "right_id"}  # Both children covered
        
        result = assembler._extract_node_text(node, coverage_map)
        assert result == "Summary with  delimiter"  # Full parent, <<<MID>>> removed

    def test_has_span_overlap(self, assembler):
        """Test span overlap detection."""
        seen_items = {(0, 100, 1, 'node1'), (200, 300, 2, 'node2')}
        
        # Test overlapping spans
        assert assembler._has_span_overlap_detailed((50, 150), seen_items) == True  # Overlaps (0,100)
        assert assembler._has_span_overlap_detailed((250, 350), seen_items) == True  # Overlaps (200,300)
        assert assembler._has_span_overlap_detailed((90, 210), seen_items) == True  # Overlaps both
        
        # Test non-overlapping spans
        assert assembler._has_span_overlap_detailed((100, 200), seen_items) == False  # Between
        assert assembler._has_span_overlap_detailed((300, 400), seen_items) == False  # After
        assert assembler._has_span_overlap_detailed((400, 500), seen_items) == False  # Far after

    def test_sort_nodes_chronologically(self, assembler):
        """Test chronological sorting of nodes."""
        # Mock nodes with different span_start values
        node1 = MagicMock()
        node1.span_start = 100
        node1.depth = 0
        node2 = MagicMock()
        node2.span_start = 50
        node2.depth = 0
        node3 = MagicMock()
        node3.span_start = 200
        node3.depth = 0
        
        # Need to set up get_node to be called multiple times
        def mock_get_node(node_id):
            if node_id == "id1":
                return node1
            elif node_id == "id2":
                return node2
            elif node_id == "id3":
                return node3
            return None
        
        assembler.store.get_node.side_effect = mock_get_node
        
        # Sort nodes
        sorted_ids = assembler._sort_nodes_chronologically(["id1", "id2", "id3"])
        
        # Should be sorted by span_start: id2 (50), id1 (100), id3 (200)
        assert sorted_ids == ["id2", "id1", "id3"]

    def test_assembly_with_exact_span_deduplication(self, assembler):
        """Test assembly deduplicates nodes with exact same spans."""
        # Create mock retrieval result
        retrieval_result = MagicMock(spec=RetrievalResult)
        retrieval_result.frontier_nodes = ["node1", "node2", "node3"]
        retrieval_result.coverage_map = set()
        
        # Mock nodes - node1 and node2 have EXACT same span
        node1 = MagicMock()
        node1.span_start = 0
        node1.span_end = 100
        node1.mid_offset = None
        node1.text = "First text"
        node1.depth = 0  # Add depth for sorting
        
        node2 = MagicMock()  # Exact same span as node1
        node2.span_start = 0
        node2.span_end = 100
        node2.mid_offset = None
        node2.text = "Duplicate span text"
        node2.depth = 0  # Add depth for sorting
        
        node3 = MagicMock()  # Different span
        node3.span_start = 200
        node3.span_end = 300
        node3.mid_offset = None
        node3.text = "Third text"
        node3.depth = 0  # Add depth for sorting
        
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
        assembler.config.slope_cap = False
        assembler.config.smoothing_pass_enabled = False
        
        result = assembler.assemble(retrieval_result)
        
        # Should only include node1 and node3 (node2 has exact same span as node1)
        assert result == "First text\n\nThird text"

    def test_left_half_plus_right_child(self, assembler):
        """Test dedicated case: parent left half + right child (no <<<MID>>> in final output)."""
        # Create a mock tree: parent with two children, only right child covered
        
        # Mock parent node with <<<MID>>> delimiter
        parent_node = MagicMock()
        parent_node.id = "parent_id"
        parent_node.mid_offset = 17  # Position of <<<MID>>> in parent text
        parent_node.text = "Parent left half <<<MID>>> parent right half"
        parent_node.span_start = 0
        parent_node.span_end = 300
        parent_node.depth = 1
        
        # Mock left child (not covered)
        left_child = MagicMock()
        left_child.id = "left_id"
        left_child.text = "Detailed left content"
        
        # Mock right child (covered)
        right_child = MagicMock()
        right_child.id = "right_id"
        right_child.text = "Detailed right child content with specifics"
        
        # Set up store mocks
        assembler.store.get_children.return_value = (left_child, right_child)
        
        def mock_get_node(node_id):
            if node_id == "parent_id":
                return parent_node
            elif node_id == "left_id":
                return left_child  
            elif node_id == "right_id":
                return right_child
            return None
        
        assembler.store.get_node.side_effect = mock_get_node
        
        # Ensure parent has no parent to avoid infinite loop
        parent_node.parent_id = None
        
        # Coverage map includes only the right child
        coverage_map = {"right_id"}
        
        # Mock retrieval result  
        retrieval_result = MagicMock()
        retrieval_result.frontier_nodes = ["parent_id"]
        retrieval_result.coverage_map = coverage_map
        
        # Mock config
        assembler.config.slope_cap = False
        assembler.config.smoothing_pass_enabled = False
        
        # Test the _extract_node_text method directly with the right coverage
        text = assembler._extract_node_text(parent_node, coverage_map)
        
        # Should start with parent left half and end with right child text
        expected = "Parent left half \n\nDetailed right child content with specifics"
        assert text == expected
        
        # Ensure no <<<MID>>> delimiter in final output
        assert "<<<MID>>>" not in text