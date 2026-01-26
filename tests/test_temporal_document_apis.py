"""Tests for Temporal Document APIs.

Tests document-status and truncate_from_time APIs as specified in
specs/temporal-document-apis.md.
"""

from __future__ import annotations

from ragzoom.server.servicers import complete_forest_size


class TestCompleteForestSize:
    """Tests for the complete_forest_size helper function.

    The formula is: 2N - popcount(N)
    where popcount(N) is the number of 1-bits in N's binary representation.

    This represents the total nodes (leaves + inner) in a complete binary forest.
    """

    def test_complete_forest_size_zero(self) -> None:
        """Zero leaves means zero nodes."""
        assert complete_forest_size(0) == 0

    def test_complete_forest_size_negative(self) -> None:
        """Negative leaf count should return 0."""
        assert complete_forest_size(-1) == 0
        assert complete_forest_size(-100) == 0

    def test_complete_forest_size_powers_of_two(self) -> None:
        """Powers of two have popcount=1, so 2N - 1.

        These form a single perfect binary tree.
        """
        # 1 leaf (0b1): 2*1 - 1 = 1
        assert complete_forest_size(1) == 1
        # 2 leaves (0b10): 2*2 - 1 = 3 (2 leaves + 1 root)
        assert complete_forest_size(2) == 3
        # 4 leaves (0b100): 2*4 - 1 = 7 (perfect tree of depth 2)
        assert complete_forest_size(4) == 7
        # 8 leaves (0b1000): 2*8 - 1 = 15
        assert complete_forest_size(8) == 15
        # 16 leaves (0b10000): 2*16 - 1 = 31
        assert complete_forest_size(16) == 31

    def test_complete_forest_size_mixed(self) -> None:
        """Non-power-of-two counts have popcount > 1.

        These form a forest of multiple perfect binary trees.
        """
        # 3 leaves (0b11): popcount=2, 2*3 - 2 = 4
        # (tree of 2 + single leaf = 3 nodes + the root... wait, no)
        # Actually: 3 leaves, 1 inner node (pairs 2 leaves), so 3+1=4
        assert complete_forest_size(3) == 4

        # 5 leaves (0b101): popcount=2, 2*5 - 2 = 8
        # Tree of 4 (7 nodes) + 1 leaf = 8 nodes
        assert complete_forest_size(5) == 8

        # 6 leaves (0b110): popcount=2, 2*6 - 2 = 10
        # Tree of 4 + tree of 2 = 7 + 3 = 10
        assert complete_forest_size(6) == 10

        # 7 leaves (0b111): popcount=3, 2*7 - 3 = 11
        # Tree of 4 (7) + tree of 2 (3) + leaf (1) = 11
        assert complete_forest_size(7) == 11

        # 100 leaves (0b1100100): popcount=3, 2*100 - 3 = 197
        assert complete_forest_size(100) == 197

    def test_complete_forest_size_formula_correctness(self) -> None:
        """Verify formula against explicit calculation for small values."""
        # For N leaves, a complete binary forest has:
        # - N leaves
        # - N - popcount(N) inner nodes
        # Total = 2N - popcount(N)

        for n in range(1, 100):
            popcount = bin(n).count("1")
            expected = 2 * n - popcount
            actual = complete_forest_size(n)
            assert (
                actual == expected
            ), f"Failed for n={n}: expected {expected}, got {actual}"
