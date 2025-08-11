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


# DEPRECATED: Backward compatibility class for telemetry modules
# Use IndexConfig, QueryConfig, and OperationalConfig directly for new code
class RagZoomConfig:
    """DEPRECATED: Backward compatibility wrapper for telemetry modules.

    This class combines the three separate config classes to maintain
    compatibility with existing telemetry code that expects a single config.

    For new code, use IndexConfig, QueryConfig, and OperationalConfig directly.
    """

    def __init__(
        self,
        # Index parameters
        target_chunk_tokens: int = 200,
        prev_context_tokens: int = 75,
        summary_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        retry_threshold: float = 0.2,
        max_retries: int = 0,
        embedding_batch_size: int = 100,
        # Query parameters
        budget_tokens: int = 4000,
        mmr_lambda: float = 0.7,
        mmr_k_multiplier: float = 2.0,
        # Operational parameters
        openai_api_key: str = "",
        sqlite_database_url: str = "sqlite:///./ragzoom.db",
        chroma_persist_directory: str = "./chroma_db",
        cache_size: int = 1000,
    ):
        # Create constituent configs
        self.index_config = IndexConfig(
            target_chunk_tokens=target_chunk_tokens,
            prev_context_tokens=prev_context_tokens,
            summary_model=summary_model,
            embedding_model=embedding_model,
            retry_threshold=retry_threshold,
            max_retries=max_retries,
            embedding_batch_size=embedding_batch_size,
        )

        self.query_config = QueryConfig(
            budget_tokens=budget_tokens,
            mmr_lambda=mmr_lambda,
            mmr_k_multiplier=mmr_k_multiplier,
        )

        self.operational_config = OperationalConfig(
            openai_api_key=openai_api_key,
            sqlite_database_url=sqlite_database_url,
            chroma_persist_directory=chroma_persist_directory,
            cache_size=cache_size,
        )

    # Properties for backward compatibility
    @property
    def target_chunk_tokens(self) -> int:
        return self.index_config.target_chunk_tokens

    @property
    def prev_context_tokens(self) -> int:
        return self.index_config.prev_context_tokens

    @property
    def summary_model(self) -> str:
        return self.index_config.summary_model

    @property
    def embedding_model(self) -> str:
        return self.index_config.embedding_model

    @property
    def retry_threshold(self) -> float:
        return self.index_config.retry_threshold

    @property
    def max_retries(self) -> int:
        return self.index_config.max_retries

    @property
    def embedding_batch_size(self) -> int:
        return self.index_config.embedding_batch_size

    @property
    def budget_tokens(self) -> int:
        return self.query_config.budget_tokens

    @property
    def mmr_lambda(self) -> float:
        return self.query_config.mmr_lambda

    @property
    def mmr_k_multiplier(self) -> float:
        return self.query_config.mmr_k_multiplier

    @property
    def openai_api_key(self) -> str:
        return self.operational_config.openai_api_key

    @property
    def sqlite_database_url(self) -> str:
        return self.operational_config.sqlite_database_url

    @property
    def chroma_persist_directory(self) -> str:
        return self.operational_config.chroma_persist_directory

    @property
    def cache_size(self) -> int:
        return self.operational_config.cache_size

    # Legacy property aliases
    @property
    def leaf_tokens(self) -> int:
        """DEPRECATED: Use target_chunk_tokens instead."""
        return self.target_chunk_tokens

    # Cost properties for telemetry compatibility
    @property
    def embedding_cost_per_1k(self) -> float:
        """Get embedding cost from model pricing constants."""
        return get_embedding_cost(self.embedding_model)

    @property
    def summary_input_cost_per_1k(self) -> float:
        """Get summary input cost from model pricing constants."""
        input_cost, _ = get_llm_costs(self.summary_model)
        return input_cost

    @property
    def summary_output_cost_per_1k(self) -> float:
        """Get summary output cost from model pricing constants."""
        _, output_cost = get_llm_costs(self.summary_model)
        return output_cost


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
