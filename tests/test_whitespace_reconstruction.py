"""Test whitespace gap reconstruction in text splitter."""

import tempfile
from unittest.mock import Mock, patch

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.index import TreeBuilder
from ragzoom.splitter import TextSplitter
from ragzoom.store import Store


class TestWhitespaceReconstruction:
    """Test that whitespace gaps are properly reconstructed after chunking."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI API calls."""
        with patch("ragzoom.index.AsyncOpenAI") as mock_async:
            # Mock embedding responses
            async def mock_embeddings_create(*args, **kwargs):
                input_data = kwargs.get("input", args[0] if args else "")
                if isinstance(input_data, list):
                    return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in input_data])
                else:
                    return Mock(data=[Mock(embedding=[0.1] * 1536)])

            async def mock_chat_create(*args, **kwargs):
                return Mock(
                    choices=[
                        Mock(message=Mock(content="Summary of left and right content"))
                    ]
                )

            instance = Mock()
            instance.embeddings = Mock()
            instance.embeddings.create = Mock(side_effect=mock_embeddings_create)
            instance.chat = Mock()
            instance.chat.completions = Mock()
            instance.chat.completions.create = Mock(side_effect=mock_chat_create)
            mock_async.return_value = instance

            yield

    @pytest.fixture
    def setup(self, mock_openai):
        """Setup test environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create separate configs
            index_config = IndexConfig(
                target_chunk_tokens=50,  # Reasonable chunk size
                preceding_context_tokens=10,
            )
            query_config = QueryConfig(budget_tokens=1000)
            operational_config = OperationalConfig(
                openai_api_key="test-key",
                chroma_persist_directory=temp_dir,
                sqlite_database_url="sqlite:///:memory:",
            )

            store = Store(
                operational_config, embedding_model=index_config.embedding_model
            )
            tree_builder = TreeBuilder(
                index_config, store, api_key=operational_config.openai_api_key
            )

            # Create a config wrapper for TextSplitter backward compatibility
            from tests.conftest import BackwardCompatibilityConfig

            config = BackwardCompatibilityConfig(
                index_config, query_config, operational_config
            )
            splitter = TextSplitter(config)

            yield config, store, tree_builder, splitter

            store.close()

    def test_whitespace_gap_reconstruction(self, setup):
        """Test that whitespace gaps between chunks are properly reconstructed."""
        config, store, tree_builder, splitter = setup

        # Test text with various whitespace patterns
        test_text = "First paragraph.\n\nSecond paragraph with more text.\n\n\nThird paragraph after double newline.\n\n    Fourth paragraph with leading spaces.\n\nFinal paragraph."

        # Split the text
        chunks = splitter.split_text(test_text)

        # Verify complete coverage by reconstructing
        reconstructed = "".join(chunks)
        assert (
            reconstructed == test_text
        ), f"Reconstructed text doesn't match original.\nExpected: {repr(test_text)}\nGot: {repr(reconstructed)}"

        # Verify no character was lost
        assert len(reconstructed) == len(
            test_text
        ), f"Length mismatch: expected {len(test_text)}, got {len(reconstructed)}"

    def test_newline_preservation(self, setup):
        """Test that newlines are preserved in chunks."""
        config, store, tree_builder, splitter = setup

        # Text with various newline patterns
        test_text = "Line 1\nLine 2\n\nParagraph 2\n\n\nParagraph 3 with triple newline"

        chunks = splitter.split_text(test_text)

        # Verify complete coverage
        reconstructed = "".join(chunks)
        assert reconstructed == test_text

        # Verify specific newline patterns are preserved
        assert "\n\n" in reconstructed
        assert "\n\n\n" in reconstructed

    def test_mixed_whitespace_preservation(self, setup):
        """Test preservation of mixed whitespace (spaces, tabs, newlines)."""
        config, store, tree_builder, splitter = setup

        # Text with mixed whitespace
        test_text = "First line.\n\t\nSecond line with tab.\n    \nThird line with spaces.\n\n\nFinal line."

        chunks = splitter.split_text(test_text)

        # Verify complete coverage
        reconstructed = "".join(chunks)
        assert reconstructed == test_text

        # Verify specific whitespace patterns
        assert "\n\t\n" in reconstructed
        assert "\n    \n" in reconstructed

    def test_indexing_with_whitespace_gaps(self, setup):
        """Test that indexing works correctly with whitespace gap reconstruction."""
        config, store, tree_builder, splitter = setup

        # Test text that would create gaps without reconstruction
        test_text = "Paragraph one with content.\n\nParagraph two with more content.\n\n\nParagraph three after gaps."

        # Index the document
        doc_id = tree_builder.add_document(test_text, document_id="test-doc")

        # Verify all nodes have valid spans
        with store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            nodes = session.query(TreeNode).filter_by(document_id=doc_id).all()

            # Check leaf nodes for complete coverage
            leaf_nodes = [
                n for n in nodes if n.left_child_id is None and n.right_child_id is None
            ]
            leaf_nodes.sort(key=lambda x: x.span_start)

            # First node should start at 0
            assert leaf_nodes[0].span_start == 0

            # Last node should end at document length
            assert leaf_nodes[-1].span_end == len(test_text)

            # Check for gaps between consecutive nodes
            for i in range(len(leaf_nodes) - 1):
                current_end = leaf_nodes[i].span_end
                next_start = leaf_nodes[i + 1].span_start

                # With whitespace reconstruction, there should be no gaps
                assert (
                    current_end == next_start
                ), f"Gap found between nodes: {current_end} to {next_start}"

    def test_single_chunk_no_reconstruction(self, setup):
        """Test that single chunks don't get modified."""
        config, store, tree_builder, splitter = setup

        # Short text that fits in one chunk
        test_text = "Short text."

        chunks = splitter.split_text(test_text)

        # Should have one chunk
        assert len(chunks) == 1
        assert chunks[0] == test_text

    def test_edge_case_only_whitespace_gaps(self, setup):
        """Test edge case where gaps are only whitespace."""
        config, store, tree_builder, splitter = setup

        # Text designed to create whitespace-only gaps
        test_text = "Word1\n\nWord2\n\n\nWord3"

        chunks = splitter.split_text(test_text)

        # Verify complete coverage
        reconstructed = "".join(chunks)
        assert reconstructed == test_text

        # Verify no meaningful content was lost
        assert "Word1" in reconstructed
        assert "Word2" in reconstructed
        assert "Word3" in reconstructed

    def test_validation_passes_with_reconstruction(self, setup):
        """Test that validation passes when whitespace gaps are reconstructed."""
        config, store, tree_builder, splitter = setup

        # Enable validation
        from ragzoom.validate import set_validation_enabled

        set_validation_enabled(True)

        try:
            # Text with potential whitespace gaps
            test_text = (
                "Section 1 content.\n\nSection 2 content.\n\n\nSection 3 content."
            )

            # This should not raise any validation errors
            doc_id = tree_builder.add_document(test_text, document_id="test-doc")

            # Verify document was indexed successfully
            assert doc_id == "test-doc"

        finally:
            # Disable validation
            set_validation_enabled(False)
