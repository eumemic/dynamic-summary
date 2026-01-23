"""Tests for IndexConfig custom prompt configuration.

Tests for the summary_system_prompt field in IndexConfig.
"""

from ragzoom.config import IndexConfig


def test_index_config_accepts_summary_system_prompt() -> None:
    """Test that IndexConfig accepts optional summary_system_prompt string.

    Spec: specs/custom-prompt-config.md § Configuration > IndexConfig Field
    Success: IndexConfig(summary_system_prompt="...") accepts optional string
    """
    custom_prompt = "You are a legal document summarizer."

    config = IndexConfig(
        target_chunk_tokens=200,
        target_embedding_context_tokens=200,
        max_parallelism=4,
        summary_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        retry_threshold=0.5,
        max_retries=3,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
        summary_system_prompt=custom_prompt,
    )

    assert config.summary_system_prompt == custom_prompt


def test_index_config_summary_system_prompt_defaults_to_none() -> None:
    """Test that summary_system_prompt defaults to None when not provided.

    Spec: specs/custom-prompt-config.md § Configuration > IndexConfig Field
    Success: IndexConfig without summary_system_prompt has None value
    """
    config = IndexConfig(
        target_chunk_tokens=200,
        target_embedding_context_tokens=200,
        max_parallelism=4,
        summary_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        retry_threshold=0.5,
        max_retries=3,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
    )

    assert config.summary_system_prompt is None


def test_index_config_load_with_summary_system_prompt() -> None:
    """Test that IndexConfig.load accepts summary_system_prompt as CLI option.

    Spec: specs/custom-prompt-config.md § Configuration > IndexConfig Field
    Success: IndexConfig.load with summary_system_prompt kwarg creates config with prompt
    """
    config = IndexConfig.load(
        summary_system_prompt="You are a medical note summarizer."
    )

    assert config.summary_system_prompt == "You are a medical note summarizer."


def test_index_config_replace_with_summary_system_prompt() -> None:
    """Test that IndexConfig.replace can update summary_system_prompt.

    Spec: specs/custom-prompt-config.md § Configuration > IndexConfig Field
    Success: config.replace(summary_system_prompt="...") returns new config with prompt
    """
    config = IndexConfig(
        target_chunk_tokens=200,
        target_embedding_context_tokens=200,
        max_parallelism=4,
        summary_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        retry_threshold=0.5,
        max_retries=3,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
    )

    assert config.summary_system_prompt is None

    new_config = config.replace(summary_system_prompt="Custom prompt")

    assert new_config.summary_system_prompt == "Custom prompt"
    # Original config unchanged
    assert config.summary_system_prompt is None
