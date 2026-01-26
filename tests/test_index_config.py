"""Tests for IndexConfig custom prompt configuration.

Tests for the summarization_guidance field in IndexConfig.
Note: The old field name 'summary_system_prompt' is deprecated but still works.
"""

import warnings

from ragzoom.config import IndexConfig


def test_summarization_guidance_field() -> None:
    """Test that IndexConfig accepts summarization_guidance field.

    Spec: specs/custom-prompt-config.md § IndexConfig Field
    Success: IndexConfig(summarization_guidance="...") works
    """
    guidance = "This is legal documentation. Preserve terminology exactly."

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
        summarization_guidance=guidance,
    )

    assert config.summarization_guidance == guidance


def test_summarization_guidance_defaults_to_none() -> None:
    """Test that summarization_guidance defaults to None when not provided.

    Spec: specs/custom-prompt-config.md § IndexConfig Field
    Success: IndexConfig without summarization_guidance has None value
    """
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

    assert config.summarization_guidance is None


def test_index_config_load_with_summarization_guidance() -> None:
    """Test that IndexConfig.load accepts summarization_guidance as CLI option.

    Spec: specs/custom-prompt-config.md § CLI Override
    Success: IndexConfig.load with summarization_guidance kwarg creates config with guidance
    """
    config = IndexConfig.load(
        summarization_guidance="You are a medical note summarizer."
    )

    assert config.summarization_guidance == "You are a medical note summarizer."


def test_index_config_replace_with_summarization_guidance() -> None:
    """Test that IndexConfig.replace can update summarization_guidance.

    Spec: specs/custom-prompt-config.md § IndexConfig Field
    Success: config.replace(summarization_guidance="...") returns new config with guidance
    """
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

    assert config.summarization_guidance is None

    new_config = config.replace(summarization_guidance="Custom guidance")

    assert new_config.summarization_guidance == "Custom guidance"
    # Original config unchanged
    assert config.summarization_guidance is None


def test_deprecated_summary_system_prompt_property_get() -> None:
    """Test that accessing summary_system_prompt logs deprecation warning.

    Spec: specs/custom-prompt-config.md § Migration > Field Rename
    Success: old name logs deprecation warning but returns correct value
    """
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
        summarization_guidance="Test guidance",
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        value = config.summary_system_prompt
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "summary_system_prompt" in str(w[0].message)
        assert "summarization_guidance" in str(w[0].message)

    assert value == "Test guidance"


def test_deprecated_summary_system_prompt_property_set() -> None:
    """Test that setting summary_system_prompt logs deprecation warning.

    Spec: specs/custom-prompt-config.md § Migration > Field Rename
    Success: old name setter logs deprecation warning but sets value correctly
    """
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

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config.summary_system_prompt = "New value via old name"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    # Value should be accessible via new name
    assert config.summarization_guidance == "New value via old name"


def test_from_dict_accepts_summarization_guidance() -> None:
    """Test that from_dict accepts the new field name.

    Spec: specs/custom-prompt-config.md § Config File
    Success: Config dict with summarization_guidance creates config with correct value
    """
    from ragzoom.config import ConfigValue

    config_dict: dict[str, ConfigValue] = {
        "target_chunk_tokens": 200,
        "target_embedding_tokens": 500,
        "max_parallelism": 4,
        "summary_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "retry_threshold": 0.5,
        "max_retries": 3,
        "embedding_batch_size": 100,
        "use_anti_verbatim_vaccine": True,
        "processing_strategy": "bottom_to_top",
        "preceding_context": {"leaf": {}, "inner": {}},  # type: ignore[dict-item]
        "summarization_guidance": "Config file guidance",
    }

    config = IndexConfig.from_dict(config_dict)

    assert config.summarization_guidance == "Config file guidance"


def test_from_dict_accepts_old_name_with_warning() -> None:
    """Test that from_dict accepts old field name with deprecation warning.

    Spec: specs/custom-prompt-config.md § Migration > Backward Compatibility
    Success: Config dict with summary_system_prompt creates config with warning
    """
    from ragzoom.config import ConfigValue

    config_dict: dict[str, ConfigValue] = {
        "target_chunk_tokens": 200,
        "target_embedding_tokens": 500,
        "max_parallelism": 4,
        "summary_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "retry_threshold": 0.5,
        "max_retries": 3,
        "embedding_batch_size": 100,
        "use_anti_verbatim_vaccine": True,
        "processing_strategy": "bottom_to_top",
        "preceding_context": {"leaf": {}, "inner": {}},  # type: ignore[dict-item]
        "summary_system_prompt": "Old config file prompt",
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = IndexConfig.from_dict(config_dict)
        # Check for deprecation warning
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 1
        assert "summary_system_prompt" in str(deprecation_warnings[0].message)

    assert config.summarization_guidance == "Old config file prompt"


def test_from_dict_new_name_takes_precedence() -> None:
    """Test that new field name takes precedence when both provided.

    Spec: specs/custom-prompt-config.md § Migration > Backward Compatibility
    Success: When both names provided, new name wins
    """
    from ragzoom.config import ConfigValue

    config_dict: dict[str, ConfigValue] = {
        "target_chunk_tokens": 200,
        "target_embedding_tokens": 500,
        "max_parallelism": 4,
        "summary_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "retry_threshold": 0.5,
        "max_retries": 3,
        "embedding_batch_size": 100,
        "use_anti_verbatim_vaccine": True,
        "processing_strategy": "bottom_to_top",
        "preceding_context": {"leaf": {}, "inner": {}},  # type: ignore[dict-item]
        "summary_system_prompt": "Old name value",
        "summarization_guidance": "New name value",
    }

    config = IndexConfig.from_dict(config_dict)

    assert config.summarization_guidance == "New name value"


# Tests for target_embedding_tokens field (embedding text optimization)


def test_target_embedding_tokens_field() -> None:
    """Test that IndexConfig accepts target_embedding_tokens field.

    Spec: specs/embedding-text-optimization.md § Configuration > New Parameter
    Success: IndexConfig(target_embedding_tokens=500) instantiates without error
    """
    config = IndexConfig(
        target_chunk_tokens=200,
        target_embedding_tokens=500,
        max_parallelism=4,
        summary_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        retry_threshold=0.5,
        max_retries=3,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
    )

    assert config.target_embedding_tokens == 500


def test_target_embedding_tokens_default_value() -> None:
    """Test that target_embedding_tokens defaults to 500.

    Spec: specs/embedding-text-optimization.md § Configuration > New Parameter
    Success: IndexConfig without target_embedding_tokens has default value of 500
    """
    config = IndexConfig(
        target_chunk_tokens=200,
        max_parallelism=4,
        summary_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        retry_threshold=0.5,
        max_retries=3,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
    )

    assert config.target_embedding_tokens == 500


def test_target_embedding_tokens_validation() -> None:
    """Test that target_embedding_tokens rejects invalid values.

    Spec: specs/embedding-text-optimization.md § Configuration > New Parameter
    Success: ValueError raised for non-positive values
    """
    import pytest

    # Zero is invalid - must be positive
    with pytest.raises(ValueError, match="target_embedding_tokens must be positive"):
        IndexConfig(
            target_chunk_tokens=200,
            target_embedding_tokens=0,
            max_parallelism=4,
            summary_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            retry_threshold=0.5,
            max_retries=3,
            embedding_batch_size=100,
            use_anti_verbatim_vaccine=True,
            processing_strategy="bottom_to_top",
        )

    # Negative is invalid
    with pytest.raises(ValueError, match="target_embedding_tokens must be positive"):
        IndexConfig(
            target_chunk_tokens=200,
            target_embedding_tokens=-100,
            max_parallelism=4,
            summary_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            retry_threshold=0.5,
            max_retries=3,
            embedding_batch_size=100,
            use_anti_verbatim_vaccine=True,
            processing_strategy="bottom_to_top",
        )


def test_from_dict_with_target_embedding_tokens() -> None:
    """Test that from_dict parses target_embedding_tokens field.

    Spec: specs/embedding-text-optimization.md § Configuration > IndexConfig Changes
    Success: IndexConfig.from_dict({"target_embedding_tokens": 500, ...}) works
    """
    from ragzoom.config import ConfigValue

    config_dict: dict[str, ConfigValue] = {
        "target_chunk_tokens": 200,
        "target_embedding_tokens": 600,
        "max_parallelism": 4,
        "summary_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "retry_threshold": 0.5,
        "max_retries": 3,
        "embedding_batch_size": 100,
        "use_anti_verbatim_vaccine": True,
        "processing_strategy": "bottom_to_top",
        "preceding_context": {"leaf": {}, "inner": {}},  # type: ignore[dict-item]
    }

    config = IndexConfig.from_dict(config_dict)

    assert config.target_embedding_tokens == 600


def test_from_dict_uses_default_target_embedding_tokens() -> None:
    """Test that from_dict uses default value when field not provided.

    Spec: specs/embedding-text-optimization.md § Configuration > IndexConfig Changes
    Success: from_dict without target_embedding_tokens uses default of 500
    """
    from ragzoom.config import ConfigValue

    config_dict: dict[str, ConfigValue] = {
        "target_chunk_tokens": 200,
        "max_parallelism": 4,
        "summary_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "retry_threshold": 0.5,
        "max_retries": 3,
        "embedding_batch_size": 100,
        "use_anti_verbatim_vaccine": True,
        "processing_strategy": "bottom_to_top",
        "preceding_context": {"leaf": {}, "inner": {}},  # type: ignore[dict-item]
    }

    config = IndexConfig.from_dict(config_dict)

    assert config.target_embedding_tokens == 500


def test_replace_target_embedding_tokens() -> None:
    """Test that IndexConfig.replace can update target_embedding_tokens.

    Spec: specs/embedding-text-optimization.md § Configuration > IndexConfig Changes
    Success: config.replace(target_embedding_tokens=600) returns new config
    """
    config = IndexConfig(
        target_chunk_tokens=200,
        target_embedding_tokens=500,
        max_parallelism=4,
        summary_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        retry_threshold=0.5,
        max_retries=3,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
    )

    assert config.target_embedding_tokens == 500

    new_config = config.replace(target_embedding_tokens=600)

    assert new_config.target_embedding_tokens == 600
    # Original config unchanged
    assert config.target_embedding_tokens == 500


def test_deprecated_target_embedding_context_tokens_error() -> None:
    """Test that from_dict raises clear error for deprecated config field.

    Spec: specs/embedding-text-optimization.md § Migration
    Success: Config with target_embedding_context_tokens raises ValueError with helpful message
    """
    import pytest

    config_dict = {
        "target_chunk_tokens": 200,
        "target_embedding_context_tokens": 200,  # Deprecated field
        "max_parallelism": 4,
        "summary_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "retry_threshold": 0.5,
        "max_retries": 3,
        "embedding_batch_size": 100,
        "use_anti_verbatim_vaccine": True,
        "processing_strategy": "bottom_to_top",
        "preceding_context": {"leaf": {}, "inner": {}},
    }

    with pytest.raises(ValueError) as exc_info:
        IndexConfig.from_dict(config_dict)  # type: ignore[arg-type]

    error_message = str(exc_info.value)
    # Error should mention the deprecated field
    assert "target_embedding_context_tokens" in error_message
    # Error should mention what to use instead
    assert "target_embedding_tokens" in error_message
    # Error should be helpful - explain it was removed
    assert "removed" in error_message.lower() or "deprecated" in error_message.lower()


def test_no_target_embedding_context_tokens_field() -> None:
    """Test that IndexConfig no longer has target_embedding_context_tokens attribute.

    Spec: specs/embedding-text-optimization.md § Configuration > Removed Parameter
    Success: IndexConfig no longer has target_embedding_context_tokens attribute
    """
    config = IndexConfig(
        target_chunk_tokens=200,
        max_parallelism=4,
        summary_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        retry_threshold=0.5,
        max_retries=3,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
    )

    # The old field should not exist on the config object
    assert not hasattr(config, "target_embedding_context_tokens")
    # But the new field should exist
    assert hasattr(config, "target_embedding_tokens")
