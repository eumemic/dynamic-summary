"""Test whitespace gap reconstruction in text splitter."""

from collections.abc import Generator
from typing import Any

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.index import TreeBuilder
from ragzoom.splitter import TextSplitter
from tests.utils import mock_openai_context


class TestWhitespaceReconstruction:
    """Test that whitespace gaps are properly reconstructed after chunking."""

    @pytest.fixture
    def mock_openai(self) -> Generator[None, None, None]:
        """Mock OpenAI API calls using centralized utilities."""
        with mock_openai_context():
            yield

    @pytest.fixture
    def setup(
        self, mock_openai: Any, store: Any
    ) -> Generator[tuple[Any, Any, Any, Any], None, None]:
        """Setup test environment."""
        # Create separate configs
        index_config = IndexConfig.load(
            target_chunk_tokens=50,  # Reasonable chunk size
            preceding_context_tokens=10,
        )
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test-key"),
        )

        # Create document-scoped store
        doc_store = store.for_document("test-doc")
        tree_builder = TreeBuilder(
            index_config,
            doc_store,
            api_key=operational_config.openai_api_key.get_secret_value(),
        )

        # Create a config wrapper for TextSplitter backward compatibility
        from tests.conftest import BackwardCompatibilityConfig

        config = BackwardCompatibilityConfig(
            index_config, query_config, operational_config
        )
        splitter = TextSplitter(index_config)

        yield config, store, tree_builder, splitter

    def test_whitespace_gap_reconstruction(self, setup: Any) -> None:
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

    def test_newline_preservation(self, setup: Any) -> None:
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

    def test_mixed_whitespace_preservation(self, setup: Any) -> None:
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

    def test_indexing_with_whitespace_gaps(self, setup: Any) -> None:
        """Test that indexing works correctly with whitespace gap reconstruction."""
        config, store, tree_builder, splitter = setup

        # Test text that would create gaps without reconstruction
        test_text = "Paragraph one with content.\n\nParagraph two with more content.\n\n\nParagraph three after gaps."

        # Index the document
        doc_id = tree_builder.add_document(test_text)

        # Verify all nodes have valid spans
        with store.SessionLocal() as session:
            from ragzoom.models import TreeNode

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

    def test_single_chunk_no_reconstruction(self, setup: Any) -> None:
        """Test that single chunks don't get modified."""
        config, store, tree_builder, splitter = setup

        # Short text that fits in one chunk
        test_text = "Short text."

        chunks = splitter.split_text(test_text)

        # Should have one chunk
        assert len(chunks) == 1
        assert chunks[0] == test_text

    def test_edge_case_only_whitespace_gaps(self, setup: Any) -> None:
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

    def test_validation_passes_with_reconstruction(self, setup: Any) -> None:
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
            doc_id = tree_builder.add_document(test_text)

            # Verify document was indexed successfully
            assert doc_id == "test-doc"

        finally:
            # Disable validation
            set_validation_enabled(False)
