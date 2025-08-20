"""Configuration management for RagZoom."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

    openai_api_key: str = ""
    database_url: str = "postgresql+psycopg://localhost/ragzoom"
    cache_size: int = 1000
    log_level: str = "INFO"
    validate_pipeline: bool = False

    def __post_init__(self) -> None:
        """Load API key from environment if not set."""
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

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
