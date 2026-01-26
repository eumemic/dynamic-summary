"""Tests for IndexConfig configuration management."""

import json
from pathlib import Path

import pytest

from ragzoom.config import IndexConfig


def test_config_accepts_none_target_chunk_tokens() -> None:
    """Test that IndexConfig accepts None for target_chunk_tokens.

    Spec: specs/client-managed-chunking.md § Activation
    Success: IndexConfig(target_chunk_tokens=None, ...) is valid
    """
    # Create a config with target_chunk_tokens=None
    config = IndexConfig(
        target_chunk_tokens=None,
        target_embedding_tokens=200,
        max_parallelism=4,
        summary_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        retry_threshold=0.5,
        max_retries=3,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
    )

    # Verify that the config was created successfully
    assert config.target_chunk_tokens is None


def test_config_backward_compatible() -> None:
    """Test that existing configs with int target_chunk_tokens still work.

    Spec: specs/client-managed-chunking.md § Activation
    Success: Existing configs with int values still work
    """
    # Create a config with an integer target_chunk_tokens (current behavior)
    config = IndexConfig(
        target_chunk_tokens=200,
        target_embedding_tokens=200,
        max_parallelism=4,
        summary_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        retry_threshold=0.5,
        max_retries=3,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
    )

    # Verify that the config was created successfully
    assert config.target_chunk_tokens == 200
    assert isinstance(config.target_chunk_tokens, int)


def test_config_validates_positive_chunk_tokens() -> None:
    """Test that IndexConfig validates target_chunk_tokens > 0 when not None.

    Spec: N/A (defensive coding)
    Success: IndexConfig(target_chunk_tokens=0, ...) raises ValueError; None is allowed
    """
    # Test that zero raises ValueError
    with pytest.raises(ValueError, match="target_chunk_tokens must be positive"):
        IndexConfig(
            target_chunk_tokens=0,
            target_embedding_tokens=200,
            max_parallelism=4,
            summary_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            retry_threshold=0.5,
            max_retries=3,
            embedding_batch_size=100,
            use_anti_verbatim_vaccine=True,
            processing_strategy="bottom_to_top",
        )

    # Test that negative raises ValueError
    with pytest.raises(ValueError, match="target_chunk_tokens must be positive"):
        IndexConfig(
            target_chunk_tokens=-1,
            target_embedding_tokens=200,
            max_parallelism=4,
            summary_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            retry_threshold=0.5,
            max_retries=3,
            embedding_batch_size=100,
            use_anti_verbatim_vaccine=True,
            processing_strategy="bottom_to_top",
        )

    # Test that None is allowed (no exception)
    config = IndexConfig(
        target_chunk_tokens=None,
        target_embedding_tokens=200,
        max_parallelism=4,
        summary_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        retry_threshold=0.5,
        max_retries=3,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
    )
    assert config.target_chunk_tokens is None


def test_default_config_has_target_embedding_tokens() -> None:
    """Test that default_config.json contains target_embedding_tokens field.

    Spec: specs/embedding-text-optimization.md § default_config.json
    Success: default_config.json contains "target_embedding_tokens": 500
    """
    module_dir = Path(__file__).parent.parent / "ragzoom"
    default_config_path = module_dir / "default_config.json"

    with open(default_config_path) as f:
        config = json.load(f)

    assert "target_embedding_tokens" in config
    assert config["target_embedding_tokens"] == 500
