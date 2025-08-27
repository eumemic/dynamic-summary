"""Configuration management for RagZoom."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SecretStr(str):
    """String type that automatically redacts its value in string representations.

    Prevents accidental exposure of sensitive values like API keys in logs,
    stack traces, and error messages while preserving the actual value for
    legitimate usage.

    Usage:
        api_key = SecretStr("sk-1234567890...")
        print(f"Using key: {api_key}")  # "Using key: ***REDACTED***"

        # When you need the actual value (e.g., for API calls):
        actual_key = api_key.get_secret_value()  # "sk-1234567890..."
        client = OpenAI(api_key=actual_key)
    """

    def __repr__(self) -> str:
        return "***REDACTED***"

    def __str__(self) -> str:
        return "***REDACTED***"

    def get_secret_value(self) -> str:
        """Get the actual secret value.

        Returns:
            The unredacted secret string value
        """
        return super().__str__()


def ensure_secret_str(api_key: str | SecretStr, service_name: str = "Service") -> str:
    """Convert API key to SecretStr if needed and extract the actual value.

    Args:
        api_key: API key as string or SecretStr
        service_name: Name of the service (for error messages)

    Returns:
        The actual API key value for use with OpenAI client

    Raises:
        ValueError: If no valid API key is available and not in test environment
    """
    import os

    # Convert to SecretStr if needed
    if isinstance(api_key, str) and not isinstance(api_key, SecretStr):
        api_key = SecretStr(api_key or os.environ.get("OPENAI_API_KEY", ""))

    # Extract the actual value
    if hasattr(api_key, "get_secret_value"):
        actual_key = api_key.get_secret_value()
    else:
        # Fallback - should not happen but prevents issues
        actual_key = str(api_key)

    # Validate we have a key
    if not actual_key:
        # In test environments, allow empty API key (will be mocked)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            actual_key = "test-key-placeholder"
        else:
            raise ValueError(f"OpenAI API key required for {service_name}")

    return actual_key


def get_embedding_cost(model: str) -> float:
    """Get embedding cost per 1K tokens using ModelInfo."""
    from ragzoom.model_info import ModelInfo

    model_info = ModelInfo()
    try:
        return model_info.get_embedding_cost(model)
    except (KeyError, ValueError):
        return 0.0


def get_llm_costs(model: str) -> tuple[float, float]:
    """Get LLM input and output costs per 1K tokens using ModelInfo."""
    from ragzoom.model_info import ModelInfo

    model_info = ModelInfo()
    try:
        return model_info.get_llm_costs(model)
    except (KeyError, ValueError):
        return 0.0, 0.0


def get_cache_discount(model: str) -> float:
    """Get cache discount multiplier for LLM using ModelInfo."""
    from ragzoom.model_info import ModelInfo

    model_info = ModelInfo()
    try:
        return model_info.get_cache_discount(model)
    except (KeyError, ValueError):
        return 1.0


def is_gpt5_model(model: str) -> bool:
    """Check if model is a GPT-5 variant."""
    return model.startswith("gpt-5")


@dataclass
class IndexConfig:
    """Configuration for document indexing.

    These parameters control how documents are chunked, summarized, and embedded.
    This is the configuration that gets saved to and loaded from config files.

    Note: Always use IndexConfig.load() to create instances. Do not instantiate directly.
    """

    target_chunk_tokens: int
    preceding_context_tokens: int
    summary_model: str
    embedding_model: str
    retry_threshold: float
    max_retries: int
    embedding_batch_size: int
    use_anti_verbatim_vaccine: bool
    processing_strategy: str

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.retry_threshold <= 1.0:
            raise ValueError(
                f"retry_threshold must be between 0.0 and 1.0, got {self.retry_threshold}"
            )
        if self.max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {self.max_retries}")
        if self.embedding_batch_size <= 0:
            raise ValueError(
                f"embedding_batch_size must be positive, got {self.embedding_batch_size}"
            )

        # Validate processing strategy
        valid_strategies = {"bottom_to_top", "left_to_right"}
        if self.processing_strategy not in valid_strategies:
            raise ValueError(
                f"processing_strategy must be one of {valid_strategies}, got '{self.processing_strategy}'"
            )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "IndexConfig":
        """Create IndexConfig from a dictionary (e.g., from telemetry JSON).

        Args:
            config_dict: Dictionary with config fields

        Returns:
            IndexConfig instance
        """
        # Extract only the fields that IndexConfig expects
        index_config_fields = {
            "target_chunk_tokens": config_dict["target_chunk_tokens"],
            "preceding_context_tokens": config_dict["preceding_context_tokens"],
            "summary_model": config_dict["summary_model"],
            "embedding_model": config_dict["embedding_model"],
            "retry_threshold": config_dict["retry_threshold"],
            "max_retries": config_dict["max_retries"],
            "embedding_batch_size": config_dict["embedding_batch_size"],
            "use_anti_verbatim_vaccine": config_dict.get(
                "use_anti_verbatim_vaccine", True
            ),
            "processing_strategy": config_dict.get(
                "processing_strategy", "bottom_to_top"
            ),
        }

        return cls(**index_config_fields)

    @classmethod
    def load(cls, config_path: Path | None = None, **cli_options: Any) -> "IndexConfig":
        """Load IndexConfig from file with CLI overrides.

        This is the primary way to create IndexConfig instances.

        Args:
            config_path: Optional path to user config file. If not specified,
                        loads from internal defaults.
            **cli_options: CLI options that override config file

        Returns:
            IndexConfig instance
        """
        # Load the raw config dictionary (handles defaults internally)
        config_dict = _load_index_config(config_path, **cli_options)

        # Use from_dict to create the instance
        return cls.from_dict(config_dict)

    def replace(self, **changes: Any) -> "IndexConfig":
        """Create a new IndexConfig with some fields changed."""
        from dataclasses import replace

        return replace(self, **changes)


@dataclass
class QueryConfig:
    """Configuration for query/retrieval operations.

    These parameters control how queries are processed and results are retrieved.
    """

    budget_tokens: int = 8000
    mmr_lambda: float = 0.7
    mmr_k_multiplier: float = 2.0
    embedding_model: str = "text-embedding-3-small"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(
                f"mmr_lambda must be between 0.0 and 1.0, got {self.mmr_lambda}"
            )
        if self.budget_tokens <= 0:
            raise ValueError(
                f"budget_tokens must be positive, got {self.budget_tokens}"
            )
        if self.mmr_k_multiplier <= 0:
            raise ValueError(
                f"mmr_k_multiplier must be positive, got {self.mmr_k_multiplier}"
            )

    def replace(self, **changes: Any) -> "QueryConfig":
        """Create a new QueryConfig with some fields changed."""
        from dataclasses import replace

        return replace(self, **changes)


@dataclass
class OperationalConfig:
    """Operational configuration for runtime environment.

    These parameters are environment-specific and are never saved to config files.
    They include storage paths, API keys, and other runtime settings.
    """

    openai_api_key: SecretStr = SecretStr("")
    database_url: str = "postgresql+psycopg://localhost/ragzoom"
    cache_size: int = 1000
    log_level: str = "INFO"
    validate_pipeline: bool = False

    def __post_init__(self) -> None:
        """Load API key from environment if not set."""
        if not self.openai_api_key.get_secret_value():
            self.openai_api_key = SecretStr(os.environ.get("OPENAI_API_KEY", ""))

        # Allow environment overrides for storage path (for testing)
        if os.environ.get("RAGZOOM_DATABASE_URL"):
            self.database_url = os.environ["RAGZOOM_DATABASE_URL"]
        else:
            # Apply worktree-specific database isolation
            from ragzoom.worktree_utils import get_worktree_database_url

            self.database_url = get_worktree_database_url(self.database_url)

    def replace(self, **changes: Any) -> "OperationalConfig":
        """Create a new OperationalConfig with some fields changed."""
        from dataclasses import replace

        return replace(self, **changes)


def _load_index_config(
    config_path: Path | None = None, **cli_options: Any
) -> dict[str, Any]:
    """Load indexing configuration with proper precedence.

    Private function - use IndexConfig.load() instead.

    Args:
        config_path: Optional path to user config file
        **cli_options: CLI options that override config file

    Returns:
        Dictionary of configuration values
    """
    # Start with default config
    module_dir = Path(__file__).parent
    default_config_path = module_dir / "default_config.json"

    with open(default_config_path) as f:
        config = json.load(f)

    # Override with user config file if provided
    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = json.load(f)
            config.update(user_config)

    # Override with CLI options (filter out None values)
    for key, value in cli_options.items():
        if value is not None and key in config:
            config[key] = value

    return dict(config)
