"""SQLite-based tests for whitespace gap reconstruction in text splitter.

These tests ensure that whitespace gaps are properly reconstructed after chunking,
using the real in-memory SQLite backend.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ragzoom.index import TreeBuilder
    from ragzoom.splitter import TextSplitter
    from tests.conftest import BackwardCompatibilityConfig

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol
from ragzoom.document_store import DocumentStore
from ragzoom.index import TreeBuilder
from ragzoom.splitter import TextSplitter
from tests.utils import mock_openai_context


@pytest.mark.usefixtures("sqlite_backend")
class TestWhitespaceReconstructionSQLite:
    """Test that whitespace gaps are properly reconstructed after chunking."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("test-doc")

    @pytest.fixture
    def mock_openai(self) -> Generator[None, None, None]:
        """Mock OpenAI API calls using centralized utilities."""
        with mock_openai_context():
            yield

    @pytest.fixture
    def setup(
        self,
        mock_openai: None,
        doc_store: DocumentStore,
        vector_index: _VectorIndexProtocol,
    ) -> Generator[
        tuple[
            BackwardCompatibilityConfig,
            DocumentStore,
            TreeBuilder,
            TextSplitter,
        ],
        None,
        None,
    ]:
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

        tree_builder = TreeBuilder(
            index_config,
            doc_store,
            vector_index,
            api_key=operational_config.openai_api_key.get_secret_value(),
        )

        # Create a config wrapper for TextSplitter backward compatibility
        from tests.conftest import BackwardCompatibilityConfig

        config = BackwardCompatibilityConfig(
            index_config, query_config, operational_config
        )
        splitter = TextSplitter(index_config)

        yield config, doc_store, tree_builder, splitter

    def test_whitespace_gap_reconstruction(
        self,
        setup: tuple[
            BackwardCompatibilityConfig,
            DocumentStore,
            TreeBuilder,
            TextSplitter,
        ],
    ) -> None:
        """Test that whitespace gaps between chunks are properly reconstructed."""
        config, doc_store, tree_builder, splitter = setup

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

    def test_newline_preservation(
        self,
        setup: tuple[
            BackwardCompatibilityConfig,
            DocumentStore,
            TreeBuilder,
            TextSplitter,
        ],
    ) -> None:
        """Test that newlines are preserved in chunks."""
        config, doc_store, tree_builder, splitter = setup

        # Text with various newline patterns
        test_text = "Line 1\nLine 2\n\nParagraph 2\n\n\nParagraph 3 with triple newline"

        chunks = splitter.split_text(test_text)

        # Verify complete coverage
        reconstructed = "".join(chunks)
        assert reconstructed == test_text

        # Verify specific newline patterns are preserved
        assert "\n\n" in reconstructed
        assert "\n\n\n" in reconstructed

    def test_mixed_whitespace_preservation(
        self,
        setup: tuple[
            BackwardCompatibilityConfig,
            DocumentStore,
            TreeBuilder,
            TextSplitter,
        ],
    ) -> None:
        """Test preservation of mixed whitespace (spaces, tabs, newlines)."""
        config, doc_store, tree_builder, splitter = setup

        # Text with mixed whitespace
        test_text = "First line.\n\t\nSecond line with tab.\n    \nThird line with spaces.\n\n\nFinal line."

        chunks = splitter.split_text(test_text)

        # Verify complete coverage
        reconstructed = "".join(chunks)
        assert reconstructed == test_text

        # Verify specific whitespace patterns
        assert "\n\t\n" in reconstructed
        assert "\n    \n" in reconstructed

    def test_indexing_with_whitespace_gaps(
        self,
        setup: tuple[
            BackwardCompatibilityConfig,
            DocumentStore,
            TreeBuilder,
            TextSplitter,
        ],
    ) -> None:
        """Test that indexing works correctly with whitespace gap reconstruction."""
        config, doc_store, tree_builder, splitter = setup

        # Test text that would create gaps without reconstruction
        test_text = "Paragraph one with content.\n\nParagraph two with more content.\n\n\nParagraph three after gaps."

        # Index the document
        tree_builder.add_document(test_text)

        # Verify all nodes have valid spans
        nodes = doc_store.nodes.get_all()

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

    def test_single_chunk_no_reconstruction(
        self,
        setup: tuple[
            BackwardCompatibilityConfig,
            DocumentStore,
            TreeBuilder,
            TextSplitter,
        ],
    ) -> None:
        """Test that single chunks don't get modified."""
        config, doc_store, tree_builder, splitter = setup

        # Short text that fits in one chunk
        test_text = "Short text."

        chunks = splitter.split_text(test_text)

        # Should have one chunk
        assert len(chunks) == 1
        assert chunks[0] == test_text

    def test_edge_case_only_whitespace_gaps(
        self,
        setup: tuple[
            BackwardCompatibilityConfig,
            DocumentStore,
            TreeBuilder,
            TextSplitter,
        ],
    ) -> None:
        """Test edge case where gaps are only whitespace."""
        config, doc_store, tree_builder, splitter = setup

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

    def test_validation_passes_with_reconstruction(
        self,
        setup: tuple[
            BackwardCompatibilityConfig,
            DocumentStore,
            TreeBuilder,
            TextSplitter,
        ],
    ) -> None:
        """Test that validation passes when whitespace gaps are reconstructed."""
        config, doc_store, tree_builder, splitter = setup

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

    def test_manual_tree_structure_whitespace_handling(
        self, doc_store: DocumentStore
    ) -> None:
        """Test whitespace handling with manually constructed tree structure."""
        # Build a simple tree structure to test reconstruction
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "leaf1",
                "text": "First chunk",
                "embedding": np.array([0.1, 0.2, 0.3]),
                "span_start": 0,
                "span_end": 11,
                "document_id": "test-doc",
                "token_count": 2,
                "height": 0,
                "parent_id": "parent",
                "path": "00",
            },
            {
                "node_id": "leaf2",
                "text": "\n\nSecond chunk",
                "embedding": np.array([0.4, 0.5, 0.6]),
                "span_start": 11,
                "span_end": 25,
                "document_id": "test-doc",
                "token_count": 3,
                "height": 0,
                "parent_id": "parent",
                "path": "01",
            },
            {
                "node_id": "parent",
                "text": "First chunk\n\nSecond chunk",
                "embedding": np.array([0.2, 0.3, 0.4]),
                "span_start": 0,
                "span_end": 25,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 1,
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
                "path": "0",
            },
        ]

        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("leaf1", "parent"), ("leaf2", "parent")]
        )

        # Verify the tree structure preserves whitespace
        parent_node = doc_store.nodes.get_node("parent")
        assert parent_node is not None
        assert "\n\n" in parent_node.text

        leaf1 = doc_store.nodes.get_node("leaf1")
        leaf2 = doc_store.nodes.get_node("leaf2")
        assert leaf1 is not None
        assert leaf2 is not None

        # Verify spans are contiguous (no gaps)
        assert leaf1.span_end == leaf2.span_start

        # Verify reconstruction
        reconstructed = leaf1.text + leaf2.text
        assert reconstructed == parent_node.text
