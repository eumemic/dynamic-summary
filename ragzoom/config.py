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
    """

    target_chunk_tokens: int = 200
    prev_context_tokens: int = 75
    summary_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    retry_threshold: float = 0.2
    max_retries: int = 0
    embedding_batch_size: int = 100

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
    chroma_persist_directory: str = "./chroma_db"
    sqlite_database_url: str = "sqlite:///./ragzoom.db"
    cache_size: int = 1000
    log_level: str = "INFO"
    validate_pipeline: bool = False

    def __post_init__(self) -> None:
        """Load API key from environment if not set."""
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

        # Allow environment overrides for storage paths (for testing)
        if os.environ.get("RAGZOOM_CHROMA_PERSIST_DIRECTORY"):
            self.chroma_persist_directory = os.environ[
                "RAGZOOM_CHROMA_PERSIST_DIRECTORY"
            ]
        if os.environ.get("RAGZOOM_SQLITE_DATABASE_URL"):
            self.sqlite_database_url = os.environ["RAGZOOM_SQLITE_DATABASE_URL"]

    def replace(self, **changes: Any) -> "OperationalConfig":
        """Create a new OperationalConfig with some fields changed."""
        from dataclasses import replace

        return replace(self, **changes)


def load_indexing_config(
    config_path: Path | None = None, **cli_options: Any
) -> dict[str, Any]:
    """Load indexing configuration with proper precedence.

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
