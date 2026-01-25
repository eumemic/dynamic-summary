"""Tests for Reciprocal Rank Fusion (RRF) function.

Spec: specs/bm25-hybrid-search.md § Architecture > Reciprocal Rank Fusion
"""

import pytest

from ragzoom.bm25 import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion() combining ranked lists."""

    def test_basic_fusion_two_identical_rankings(self) -> None:
        """Test RRF with two identical rankings.

        Spec: specs/bm25-hybrid-search.md § Architecture > Reciprocal Rank Fusion
        Success: Identical rankings produce expected fused scores
        """
        vector_ranking = ["a", "b", "c"]
        bm25_ranking = ["a", "b", "c"]

        result = reciprocal_rank_fusion(vector_ranking, bm25_ranking)

        # Items appearing in both lists get double the RRF score
        # Score for rank r = 1/(k+r+1), with k=60 by default
        # a: 1/61 + 1/61 = 2/61
        # b: 1/62 + 1/62 = 2/62
        # c: 1/63 + 1/63 = 2/63
        assert len(result) == 3
        assert [node_id for node_id, _ in result] == ["a", "b", "c"]

        # Check scores are in descending order
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_basic_fusion_different_rankings(self) -> None:
        """Test RRF with completely different rankings.

        Spec: specs/bm25-hybrid-search.md § Architecture > Reciprocal Rank Fusion
        Success: Different rankings are merged correctly
        """
        vector_ranking = ["a", "b", "c"]
        bm25_ranking = ["d", "e", "f"]

        result = reciprocal_rank_fusion(vector_ranking, bm25_ranking)

        # All 6 items should appear
        node_ids = {node_id for node_id, _ in result}
        assert node_ids == {"a", "b", "c", "d", "e", "f"}

        # First items from each list should tie (same rank contribution)
        # a gets 1/61 from vector, d gets 1/61 from bm25
        # Order among ties is implementation-dependent, but they should
        # have the same score
        a_score = next(s for nid, s in result if nid == "a")
        d_score = next(s for nid, s in result if nid == "d")
        assert a_score == d_score

    def test_overlapping_rankings(self) -> None:
        """Test RRF with partially overlapping rankings.

        Spec: specs/bm25-hybrid-search.md § Architecture > Reciprocal Rank Fusion
        Success: Overlapping items get combined scores
        """
        vector_ranking = ["a", "b", "c"]
        bm25_ranking = ["b", "a", "d"]

        result = reciprocal_rank_fusion(vector_ranking, bm25_ranking)

        # a: rank 0 in vector (1/61), rank 1 in bm25 (1/62) = 1/61 + 1/62
        # b: rank 1 in vector (1/62), rank 0 in bm25 (1/61) = 1/62 + 1/61
        # c: rank 2 in vector (1/63), not in bm25 = 1/63
        # d: not in vector, rank 2 in bm25 (1/63) = 1/63

        # a and b should have same score (1/61 + 1/62)
        a_score = next(s for nid, s in result if nid == "a")
        b_score = next(s for nid, s in result if nid == "b")
        assert a_score == pytest.approx(b_score)

        # c and d should have same score (1/63)
        c_score = next(s for nid, s in result if nid == "c")
        d_score = next(s for nid, s in result if nid == "d")
        assert c_score == pytest.approx(d_score)

        # a/b should score higher than c/d
        assert a_score > c_score

    def test_custom_k_parameter(self) -> None:
        """Test RRF with custom k constant.

        Spec: specs/bm25-hybrid-search.md § Architecture > Reciprocal Rank Fusion
        Success: k parameter affects score magnitudes
        """
        ranking = ["a", "b"]

        # With k=60 (default)
        result_k60 = reciprocal_rank_fusion(ranking, ranking, k=60)
        # With k=1 (more emphasis on top ranks)
        result_k1 = reciprocal_rank_fusion(ranking, ranking, k=1)

        # k=1: a score = 2 * 1/(1+0+1) = 2/2 = 1.0
        # k=60: a score = 2 * 1/(60+0+1) = 2/61 ≈ 0.0328
        a_score_k60 = next(s for nid, s in result_k60 if nid == "a")
        a_score_k1 = next(s for nid, s in result_k1 if nid == "a")

        # With smaller k, scores are larger
        assert a_score_k1 > a_score_k60

        # Verify exact values
        assert a_score_k1 == pytest.approx(2 / 2)  # 1.0
        assert a_score_k60 == pytest.approx(2 / 61)

    def test_empty_vector_ranking(self) -> None:
        """Test RRF with empty vector ranking.

        Spec: N/A (edge case handling)
        Success: Returns bm25 ranking scores only
        """
        vector_ranking: list[str] = []
        bm25_ranking = ["a", "b", "c"]

        result = reciprocal_rank_fusion(vector_ranking, bm25_ranking)

        assert len(result) == 3
        # Only bm25 contributes
        a_score = next(s for nid, s in result if nid == "a")
        assert a_score == pytest.approx(1 / 61)  # k=60, rank 0

    def test_empty_bm25_ranking(self) -> None:
        """Test RRF with empty BM25 ranking.

        Spec: N/A (edge case handling)
        Success: Returns vector ranking scores only
        """
        vector_ranking = ["a", "b", "c"]
        bm25_ranking: list[str] = []

        result = reciprocal_rank_fusion(vector_ranking, bm25_ranking)

        assert len(result) == 3
        # Only vector contributes
        a_score = next(s for nid, s in result if nid == "a")
        assert a_score == pytest.approx(1 / 61)

    def test_both_rankings_empty(self) -> None:
        """Test RRF with both rankings empty.

        Spec: N/A (edge case handling)
        Success: Returns empty result
        """
        result = reciprocal_rank_fusion([], [])

        assert result == []

    def test_single_item_each(self) -> None:
        """Test RRF with single item in each ranking.

        Spec: N/A (edge case handling)
        Success: Single items fused correctly
        """
        result = reciprocal_rank_fusion(["a"], ["b"])

        assert len(result) == 2
        # Both should have same score (1/61 each from rank 0)
        a_score = next(s for nid, s in result if nid == "a")
        b_score = next(s for nid, s in result if nid == "b")
        assert a_score == pytest.approx(b_score)
        assert a_score == pytest.approx(1 / 61)

    def test_same_item_different_positions(self) -> None:
        """Test RRF correctly accumulates scores across rankings.

        Spec: specs/bm25-hybrid-search.md § Architecture > Reciprocal Rank Fusion
        Success: Combines rankings using RRF formula
        """
        # "a" is first in vector but third in bm25
        # "c" is third in vector but first in bm25
        # "b" is second in both
        vector_ranking = ["a", "b", "c"]
        bm25_ranking = ["c", "b", "a"]

        result = reciprocal_rank_fusion(vector_ranking, bm25_ranking)

        # a: 1/61 + 1/63 (ranks 0 and 2)
        # b: 1/62 + 1/62 (rank 1 in both)
        # c: 1/63 + 1/61 (ranks 2 and 0)

        a_score = next(s for nid, s in result if nid == "a")
        b_score = next(s for nid, s in result if nid == "b")
        c_score = next(s for nid, s in result if nid == "c")

        # a and c should have same score (symmetric positions)
        assert a_score == pytest.approx(c_score)

        # b's score: 2/62 = 1/31 ≈ 0.0323
        # a/c's score: 1/61 + 1/63 ≈ 0.0164 + 0.0159 = 0.0323
        # These should be very close!
        expected_b = 2 / 62
        expected_a = 1 / 61 + 1 / 63
        assert b_score == pytest.approx(expected_b)
        assert a_score == pytest.approx(expected_a)

    def test_result_order_is_stable(self) -> None:
        """Test that result ordering is deterministic.

        Spec: N/A (reliability requirement)
        Success: Same inputs produce same output order
        """
        vector_ranking = ["a", "b", "c"]
        bm25_ranking = ["d", "a", "e"]

        result1 = reciprocal_rank_fusion(vector_ranking, bm25_ranking)
        result2 = reciprocal_rank_fusion(vector_ranking, bm25_ranking)

        assert result1 == result2

    def test_returns_list_of_tuples(self) -> None:
        """Test that result is list of (node_id, score) tuples.

        Spec: specs/bm25-hybrid-search.md § Architecture > Reciprocal Rank Fusion
        Success: Returns (node_id, rrf_score) pairs
        """
        result = reciprocal_rank_fusion(["a"], ["a"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2
        assert isinstance(result[0][0], str)
        assert isinstance(result[0][1], float)

    def test_many_items(self) -> None:
        """Test RRF with many items.

        Spec: N/A (scalability check)
        Success: Works correctly with longer rankings
        """
        # 50 items in each ranking
        vector_ranking = [f"v{i}" for i in range(50)]
        bm25_ranking = [f"b{i}" for i in range(50)]

        result = reciprocal_rank_fusion(vector_ranking, bm25_ranking)

        # Should have 100 unique items
        assert len(result) == 100

        # All items should be present
        node_ids = {nid for nid, _ in result}
        expected = {f"v{i}" for i in range(50)} | {f"b{i}" for i in range(50)}
        assert node_ids == expected

        # Top items should be from rank 0 of each list
        top_scores = {nid: s for nid, s in result[:2]}
        assert "v0" in top_scores
        assert "b0" in top_scores
        # Both should have score 1/61
        assert top_scores["v0"] == pytest.approx(1 / 61)
        assert top_scores["b0"] == pytest.approx(1 / 61)
