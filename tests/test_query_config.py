"""Tests for QueryConfig BM25 hybrid search configuration."""

from ragzoom.config import QueryConfig


def test_query_config_has_use_bm25() -> None:
    """Test that QueryConfig has use_bm25 field defaulting to True.

    Spec: specs/bm25-hybrid-search.md § Configuration > QueryConfig Field
    Success: QueryConfig(use_bm25=True) is default; False disables BM25
    """
    # Default should be True (BM25 enabled by default)
    config = QueryConfig()
    assert config.use_bm25 is True

    # Can explicitly set to False to disable
    config_disabled = QueryConfig(use_bm25=False)
    assert config_disabled.use_bm25 is False


def test_query_config_use_bm25_with_other_fields() -> None:
    """Test that use_bm25 works alongside other QueryConfig fields.

    Spec: specs/bm25-hybrid-search.md § Configuration > QueryConfig Field
    """
    config = QueryConfig(
        budget_tokens=1000,
        mmr_lambda=0.5,
        use_bm25=False,
    )

    assert config.budget_tokens == 1000
    assert config.mmr_lambda == 0.5
    assert config.use_bm25 is False


def test_query_config_replace_use_bm25() -> None:
    """Test that QueryConfig.replace can update use_bm25.

    Spec: specs/bm25-hybrid-search.md § Configuration > QueryConfig Field
    """
    config = QueryConfig(use_bm25=True)
    assert config.use_bm25 is True

    new_config = config.replace(use_bm25=False)

    assert new_config.use_bm25 is False
    # Original config unchanged
    assert config.use_bm25 is True


def test_query_config_has_bm25_weight() -> None:
    """Test that QueryConfig has bm25_weight field defaulting to 1.0.

    Spec: specs/bm25-hybrid-search.md § Configuration > QueryConfig Field
    Success: QueryConfig(bm25_weight=1.0) controls BM25 weight in RRF
    """
    # Default should be 1.0 (equal weight with vector search)
    config = QueryConfig()
    assert config.bm25_weight == 1.0

    # Can explicitly set to other values
    config_weighted = QueryConfig(bm25_weight=0.5)
    assert config_weighted.bm25_weight == 0.5

    # Higher weight emphasizes BM25 more
    config_high = QueryConfig(bm25_weight=2.0)
    assert config_high.bm25_weight == 2.0


def test_query_config_bm25_weight_validation() -> None:
    """Test that bm25_weight validates to positive values.

    Spec: specs/bm25-hybrid-search.md § Configuration > QueryConfig Field
    """
    import pytest

    # Zero weight should be invalid (would make BM25 contribute nothing)
    with pytest.raises(ValueError, match="bm25_weight must be positive"):
        QueryConfig(bm25_weight=0.0)

    # Negative weight should be invalid
    with pytest.raises(ValueError, match="bm25_weight must be positive"):
        QueryConfig(bm25_weight=-1.0)


def test_query_config_bm25_weight_with_other_fields() -> None:
    """Test that bm25_weight works alongside other QueryConfig fields.

    Spec: specs/bm25-hybrid-search.md § Configuration > QueryConfig Field
    """
    config = QueryConfig(
        budget_tokens=1000,
        mmr_lambda=0.5,
        use_bm25=True,
        bm25_weight=1.5,
    )

    assert config.budget_tokens == 1000
    assert config.mmr_lambda == 0.5
    assert config.use_bm25 is True
    assert config.bm25_weight == 1.5


def test_query_config_replace_bm25_weight() -> None:
    """Test that QueryConfig.replace can update bm25_weight.

    Spec: specs/bm25-hybrid-search.md § Configuration > QueryConfig Field
    """
    config = QueryConfig(bm25_weight=1.0)
    assert config.bm25_weight == 1.0

    new_config = config.replace(bm25_weight=2.0)

    assert new_config.bm25_weight == 2.0
    # Original config unchanged
    assert config.bm25_weight == 1.0
