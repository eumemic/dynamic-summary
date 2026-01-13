"""Tests for get_summary_target() function in indexing_engine."""

from ragzoom.server.indexing_engine import get_summary_target


class TestGetSummaryTarget:
    """Test dynamic summary target calculation."""

    def test_get_summary_target_dynamic_calculation(self) -> None:
        """Test that target scales by 2^height for various heights."""
        chars_per_token = 4.0
        node_span_chars = 8000  # 2000 tokens at 4 chars/token

        # Height 1: target = 2000 / 2^1 = 1000 tokens
        assert get_summary_target(node_span_chars, 1, chars_per_token) == 1000

        # Height 2: target = 2000 / 2^2 = 500 tokens
        assert get_summary_target(node_span_chars, 2, chars_per_token) == 500

        # Height 3: target = 2000 / 2^3 = 250 tokens
        assert get_summary_target(node_span_chars, 3, chars_per_token) == 250

        # Height 4: target = 2000 / 2^4 = 125 tokens
        assert get_summary_target(node_span_chars, 4, chars_per_token) == 125

    def test_get_summary_target_floor(self) -> None:
        """Test that targets below 50 tokens return 0 (passthrough signal)."""
        chars_per_token = 4.0

        # Small span: 100 chars = 25 tokens
        # Height 1: 25 / 2 = 12.5 tokens → below floor, return 0
        assert get_summary_target(100, 1, chars_per_token) == 0

        # 300 chars = 75 tokens at height 1 → 37.5 tokens → below floor
        assert get_summary_target(300, 1, chars_per_token) == 0

        # 400 chars = 100 tokens at height 1 → 50 tokens → exactly at floor
        # Spec says "below 50", so 50 should still compress
        assert get_summary_target(400, 1, chars_per_token) == 50

        # 396 chars = 99 tokens at height 1 → 49.5 tokens → below floor
        assert get_summary_target(396, 1, chars_per_token) == 0

    def test_get_summary_target_various_chars_per_token(self) -> None:
        """Test that function works correctly with different chars_per_token ratios."""
        # Different language/tokenization patterns
        node_span_chars = 1000

        # English-like: 4 chars/token → 250 tokens
        # Height 2: 250 / 4 = 62.5 → 62
        assert get_summary_target(node_span_chars, 2, 4.0) == 62

        # Dense text: 3 chars/token → 333 tokens
        # Height 2: 333 / 4 = 83.25 → 83
        assert get_summary_target(node_span_chars, 2, 3.0) == 83

        # Sparse text: 5 chars/token → 200 tokens
        # Height 2: 200 / 4 = 50 → 50
        assert get_summary_target(node_span_chars, 2, 5.0) == 50

    def test_get_summary_target_large_span(self) -> None:
        """Test behavior with very large spans (e.g., whole conversation turns)."""
        chars_per_token = 4.0
        # Large turn: 50k chars = 12,500 tokens
        node_span_chars = 50000

        # Height 1: 12500 / 2 = 6250 tokens
        assert get_summary_target(node_span_chars, 1, chars_per_token) == 6250

        # Height 5: 12500 / 32 = 390.625 → 390 tokens
        assert get_summary_target(node_span_chars, 5, chars_per_token) == 390

        # Height 8: 12500 / 256 = 48.828 → below floor → 0
        assert get_summary_target(node_span_chars, 8, chars_per_token) == 0

    def test_get_summary_target_height_zero(self) -> None:
        """Test that height 0 (leaves) don't get summarized."""
        chars_per_token = 4.0
        node_span_chars = 1000  # 250 tokens

        # Height 0: 250 / 2^0 = 250 / 1 = 250 tokens
        # This should return the full span size, no compression
        assert get_summary_target(node_span_chars, 0, chars_per_token) == 250

    def test_get_summary_target_edge_case_tiny_span(self) -> None:
        """Test behavior with very small spans."""
        chars_per_token = 4.0

        # 1 character = 0.25 tokens → at any height > 0, below floor
        assert get_summary_target(1, 1, chars_per_token) == 0
        assert get_summary_target(1, 5, chars_per_token) == 0

        # 200 chars = 50 tokens → height 1: 25 tokens → below floor
        assert get_summary_target(200, 1, chars_per_token) == 0

    def test_get_summary_target_rounding(self) -> None:
        """Test that fractional tokens are truncated (int conversion)."""
        chars_per_token = 4.0

        # 410 chars = 102.5 tokens
        # Height 1: 102.5 / 2 = 51.25 → int(51.25) = 51
        assert get_summary_target(410, 1, chars_per_token) == 51

        # 409 chars = 102.25 tokens
        # Height 1: 102.25 / 2 = 51.125 → int(51.125) = 51
        assert get_summary_target(409, 1, chars_per_token) == 51

        # 408 chars = 102 tokens
        # Height 1: 102 / 2 = 51 → int(51) = 51
        assert get_summary_target(408, 1, chars_per_token) == 51
