"""Tests for utility functions."""

from ragzoom.utils import batch_process, format_token_count


class TestUtils:
    """Test utility functions."""

    def test_format_token_count_small(self):
        """Test formatting small token counts."""
        assert format_token_count(0) == "0 tokens"
        assert format_token_count(1) == "1 tokens"
        assert format_token_count(999) == "999 tokens"

    def test_format_token_count_large(self):
        """Test formatting large token counts."""
        assert format_token_count(1000) == "1.0k tokens"
        assert format_token_count(1500) == "1.5k tokens"
        assert format_token_count(2750) == "2.8k tokens"
        assert format_token_count(10000) == "10.0k tokens"

    def test_batch_process(self):
        """Test batch processing."""
        items = list(range(10))

        # Batch size 3
        batches = list(batch_process(items, 3))
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]

        # Batch size equal to list size
        batches = list(batch_process(items, 10))
        assert len(batches) == 1
        assert batches[0] == items

        # Batch size larger than list
        batches = list(batch_process(items, 20))
        assert len(batches) == 1
        assert batches[0] == items

    def test_batch_process_empty(self):
        """Test batch processing with empty list."""
        batches = list(batch_process([], 5))
        assert len(batches) == 0
