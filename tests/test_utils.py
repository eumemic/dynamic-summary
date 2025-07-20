"""Tests for utility functions."""

from ragzoom.utils import format_token_count


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
