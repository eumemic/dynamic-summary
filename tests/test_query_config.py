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
