"""Tests for text splitter functionality."""

import os

import pytest

from ragzoom.config import IndexConfig
from ragzoom.splitter import TextSplitter

# Set required env var for tests
os.environ["OPENAI_API_KEY"] = "test-key"


class TestTextSplitter:
    """Test the TextSplitter class."""

    def test_split_basic_text(self) -> None:
        """Test basic text splitting."""
        index_config = IndexConfig.load(target_chunk_tokens=50)
        splitter = TextSplitter(index_config)

        text = "This is a test. " * 50  # ~200 tokens
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_split_respects_boundaries(self) -> None:
        """Test that splitter respects sentence boundaries."""
        index_config = IndexConfig.load(target_chunk_tokens=50)
        splitter = TextSplitter(index_config)

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = splitter.split_text(text)

        # Should split on sentence boundaries
        for chunk in chunks:
            # Each chunk should end with punctuation or be the last chunk
            assert chunk.strip().endswith(".") or chunk == chunks[-1]

    def test_adjacent_context(self) -> None:
        """Test getting adjacent context for chunks."""
        index_config = IndexConfig.load(target_chunk_tokens=50)
        splitter = TextSplitter(index_config)

        chunks = ["First chunk text.", "Second chunk text.", "Third chunk text."]

        # Test middle chunk
        prev_ctx, next_ctx = splitter.get_adjacent_context(chunks, 1)
        assert prev_ctx is not None
        assert next_ctx is not None
        assert "First" in prev_ctx
        assert "Third" in next_ctx

        # Test first chunk
        prev_ctx, next_ctx = splitter.get_adjacent_context(chunks, 0)
        assert prev_ctx is None
        assert next_ctx is not None

        # Test last chunk
        prev_ctx, next_ctx = splitter.get_adjacent_context(chunks, 2)
        assert prev_ctx is not None
        assert next_ctx is None

    def test_token_counting(self) -> None:
        """Test token counting accuracy."""
        index_config = IndexConfig.load(target_chunk_tokens=200)
        splitter = TextSplitter(index_config)

        text = "Hello world"
        token_count = splitter._token_length(text)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_empty_text(self) -> None:
        """Test handling of empty text."""
        index_config = IndexConfig.load(target_chunk_tokens=200)
        splitter = TextSplitter(index_config)

        # Our splitter now raises an error for empty text (correct-by-construction)
        with pytest.raises(ValueError, match="produced no valid chunks"):
            splitter.split_text("")

    def test_single_chunk_retains_trailing_whitespace(self) -> None:
        """Single chunk rebuild must preserve trailing whitespace and padding."""

        index_config = IndexConfig.load(target_chunk_tokens=512)
        splitter = TextSplitter(index_config)

        text = "Hello world" + " " * 10
        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_sequential_chunks(self) -> None:
        """Test that chunks are sequential without overlap."""
        index_config = IndexConfig.load(target_chunk_tokens=50)
        splitter = TextSplitter(index_config)

        text = " ".join([f"Word{i}" for i in range(200)])  # Long text
        chunks = splitter.split_text(text)

        # Chunks should be sequential without overlap

        # Main test: verify we got multiple chunks from long text
        assert len(chunks) > 1

        # Verify complete coverage
        reconstructed = "".join(chunks)
        assert len(reconstructed) == len(text), "Should have complete coverage"

    def test_text_splitter_handles_none_target(self) -> None:
        """Test that TextSplitter handles target_chunk_tokens=None.

        Spec: specs/client-managed-chunking.md § Append Operations
        Success: When target_chunk_tokens is None, splitter doesn't initialize RecursiveCharacterTextSplitter
        """
        index_config = IndexConfig.load(target_chunk_tokens=None)
        splitter = TextSplitter(index_config)

        # When target_chunk_tokens is None, the splitter should not have initialized
        # the RecursiveCharacterTextSplitter
        assert splitter.config.target_chunk_tokens is None
        # The splitter attribute should not exist or be None
        assert not hasattr(splitter, "splitter") or splitter.splitter is None

    def test_split_text_passthrough_when_none(self) -> None:
        """Test that split_text returns [text] unchanged when target_chunk_tokens is None.

        Spec: specs/client-managed-chunking.md § Append Operations
        Success: split_text("foo") returns ["foo"] when target_chunk_tokens is None
        """
        index_config = IndexConfig.load(target_chunk_tokens=None)
        splitter = TextSplitter(index_config)

        text = "This is a whole conversation turn that should not be split."
        chunks = splitter.split_text(text)

        # Should return exactly one chunk with the original text
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_text_backward_compatible(self) -> None:
        """Test that split_text preserves existing behavior when target_chunk_tokens is int.

        Spec: specs/client-managed-chunking.md § Append Operations
        Success: Existing behavior preserved when int
        """
        index_config = IndexConfig.load(target_chunk_tokens=50)
        splitter = TextSplitter(index_config)

        # Long text that should be split into multiple chunks
        text = "This is a test. " * 50  # ~200 tokens

        chunks = splitter.split_text(text)

        # Should split into multiple chunks
        assert len(chunks) > 1
        # All chunks should be non-empty
        assert all(chunk and chunk.strip() for chunk in chunks)
