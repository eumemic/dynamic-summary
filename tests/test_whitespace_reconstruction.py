"""Tests for whitespace gap reconstruction in text splitter."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.splitter import TextSplitter
from tests.conftest import BackwardCompatibilityConfig, IndexerRuntimeHarness
from tests.utils import mock_openai_context


class TestWhitespaceReconstruction:
    """Test that whitespace gaps are properly reconstructed after chunking."""

    @pytest.fixture
    def mock_openai(self) -> Generator[None, None, None]:
        """Mock OpenAI API calls using centralized utilities."""
        with mock_openai_context():
            yield

    @pytest.fixture
    def splitter_fixture(
        self,
        mock_openai: None,
    ) -> Generator[tuple[BackwardCompatibilityConfig, TextSplitter], None, None]:
        """Provide a splitter with deterministic configuration."""
        index_config = IndexConfig.load(
            target_chunk_tokens=50,
            preceding_context_tokens=10,
        )
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test-key"),
        )
        config = BackwardCompatibilityConfig(
            index_config, query_config, operational_config
        )
        yield config, TextSplitter(index_config)

    def test_whitespace_gap_reconstruction(
        self,
        splitter_fixture: tuple[BackwardCompatibilityConfig, TextSplitter],
    ) -> None:
        """Test that whitespace gaps between chunks are properly reconstructed."""
        _, splitter = splitter_fixture

        test_text = (
            "First paragraph.\n\nSecond paragraph with more text.\n\n\n"
            "Third paragraph after double newline.\n\n    Fourth paragraph with leading spaces.\n\nFinal paragraph."
        )

        chunks = splitter.split_text(test_text)
        reconstructed = "".join(chunks)

        assert reconstructed == test_text
        assert len(reconstructed) == len(test_text)

    def test_newline_preservation(
        self,
        splitter_fixture: tuple[BackwardCompatibilityConfig, TextSplitter],
    ) -> None:
        """Test that newlines are preserved in chunks."""
        _, splitter = splitter_fixture

        test_text = "Line 1\nLine 2\n\nParagraph 2\n\n\nParagraph 3 with triple newline"

        chunks = splitter.split_text(test_text)
        reconstructed = "".join(chunks)

        assert reconstructed == test_text
        assert "\n\n" in reconstructed
        assert "\n\n\n" in reconstructed

    def test_mixed_whitespace_preservation(
        self,
        splitter_fixture: tuple[BackwardCompatibilityConfig, TextSplitter],
    ) -> None:
        """Test preservation of mixed whitespace (spaces, tabs, newlines)."""
        _, splitter = splitter_fixture

        test_text = (
            "First line.\n\t\nSecond line with tab.\n    \n"
            "Third line with spaces.\n\n\nFinal line."
        )

        chunks = splitter.split_text(test_text)
        reconstructed = "".join(chunks)

        assert reconstructed == test_text
        assert "\n\t\n" in reconstructed
        assert "\n    \n" in reconstructed

    @pytest.mark.asyncio
    async def test_indexing_with_whitespace_gaps(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that indexing works correctly with whitespace gap reconstruction."""
        document_id = "test-doc-indexing"
        storage_backend.clear_document(document_id)
        await indexer_runtime_harness.clear(document_id)

        test_text = (
            "Paragraph one with content.\n\nParagraph two with more content.\n\n\n"
            "Paragraph three after gaps."
        )

        await indexer_runtime_harness.append(
            document_id,
            test_text,
            replace_existing=True,
            file_path="indexing_whitespace_test.txt",
        )

        doc_store = storage_backend.for_document(document_id)
        nodes = doc_store.nodes.get_all()
        assert nodes, "Expected nodes to be created during indexing"
        for node in nodes:
            assert node.span_end >= node.span_start
