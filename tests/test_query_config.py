"""Tests for QueryConfig BM25 hybrid search configuration."""

import os

import pytest

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


def test_query_config_retrieval_mode_defaults_to_coverage() -> None:
    """retrieval_mode defaults to "coverage" for backwards compatibility."""
    config = QueryConfig()
    assert config.retrieval_mode == "coverage"


def test_query_config_retrieval_mode_accepts_concentrate() -> None:
    """retrieval_mode can be explicitly set to "concentrate"."""
    config = QueryConfig(retrieval_mode="concentrate")
    assert config.retrieval_mode == "concentrate"


def test_query_config_retrieval_mode_invalid_fails_hard() -> None:
    """An unknown retrieval_mode raises ValueError (no silent fallback)."""
    with pytest.raises(ValueError, match="retrieval_mode must be one of"):
        QueryConfig(retrieval_mode="topk")


def test_query_config_replace_retrieval_mode() -> None:
    """QueryConfig.replace can update retrieval_mode without mutating the original."""
    config = QueryConfig(retrieval_mode="coverage")
    new_config = config.replace(retrieval_mode="concentrate")

    assert new_config.retrieval_mode == "concentrate"
    # Original config unchanged
    assert config.retrieval_mode == "coverage"


def test_query_config_retrieval_mode_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAGZOOM_RETRIEVAL_MODE env var flows into a default-constructed config.

    The benchmark server constructs QueryConfig() with no args, so the env
    var must be honoured at construction time for the experiment switch to
    reach the retriever's tiling decision.
    """
    monkeypatch.setenv("RAGZOOM_RETRIEVAL_MODE", "concentrate")
    config = QueryConfig()
    assert config.retrieval_mode == "concentrate"


def test_query_config_explicit_arg_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit retrieval_mode argument wins over the env var."""
    monkeypatch.setenv("RAGZOOM_RETRIEVAL_MODE", "concentrate")
    config = QueryConfig(retrieval_mode="coverage")
    assert config.retrieval_mode == "coverage"


def test_query_config_retrieval_mode_invalid_env_fails_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid RAGZOOM_RETRIEVAL_MODE env value fails hard at construction."""
    monkeypatch.setenv("RAGZOOM_RETRIEVAL_MODE", "bogus")
    with pytest.raises(ValueError, match="retrieval_mode must be one of"):
        QueryConfig()


def test_query_config_no_env_keeps_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent the env var, retrieval_mode stays at the "coverage" default."""
    monkeypatch.delenv("RAGZOOM_RETRIEVAL_MODE", raising=False)
    assert os.environ.get("RAGZOOM_RETRIEVAL_MODE") is None
    config = QueryConfig()
    assert config.retrieval_mode == "coverage"
