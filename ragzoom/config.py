"""Configuration management for RagZoom."""

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import TypedDict


class IndexConfigDict(TypedDict):
    """Type definition for IndexConfig dictionary representation."""

    target_chunk_tokens: int
    preceding_summary_budget_tokens: int
    summary_model: str
    embedding_model: str
    retry_threshold: float
    max_retries: int
    embedding_batch_size: int
    use_anti_verbatim_vaccine: bool
    processing_strategy: str
    preceding_context_verbatim_tokens: int
    preceding_context_max_extraneous_detail: int
    preceding_context_num_seeds: int | None


# Type for configuration values that can be primitives
ConfigValue = str | int | float | bool
# Type that includes None for CLI parameters
ConfigValueOrNone = str | int | float | bool | None


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
    preceding_summary_budget_tokens: int
    summary_model: str
    embedding_model: str
    retry_threshold: float
    max_retries: int
    embedding_batch_size: int
    use_anti_verbatim_vaccine: bool
    processing_strategy: str
    preceding_context_verbatim_tokens: int
    preceding_context_max_extraneous_detail: int
    preceding_context_num_seeds: int | None

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

        # Validate max extraneous detail
        if self.preceding_context_max_extraneous_detail < 0:
            raise ValueError(
                f"preceding_context_max_extraneous_detail must be >= 0, "
                f"got {self.preceding_context_max_extraneous_detail}"
            )

        # Validate num_seeds if provided
        if self.preceding_context_num_seeds is not None:
            if self.preceding_context_num_seeds < 1:
                raise ValueError(
                    f"preceding_context_num_seeds must be >= 1, "
                    f"got {self.preceding_context_num_seeds}"
                )

    @classmethod
    def from_dict(cls, config_dict: dict[str, ConfigValue]) -> "IndexConfig":
        """Create IndexConfig from a dictionary (e.g., from telemetry JSON).

        Args:
            config_dict: Dictionary with config fields

        Returns:
            IndexConfig instance
        """
        # Handle num_seeds which can be None or int
        raw_num_seeds = config_dict.get("preceding_context_num_seeds")
        num_seeds: int | None = (
            int(raw_num_seeds) if raw_num_seeds is not None else None
        )

        # Type-safe construction with proper field types
        return cls(
            target_chunk_tokens=int(config_dict["target_chunk_tokens"]),
            preceding_summary_budget_tokens=int(
                config_dict["preceding_summary_budget_tokens"]
            ),
            summary_model=str(config_dict["summary_model"]),
            embedding_model=str(config_dict["embedding_model"]),
            retry_threshold=float(config_dict["retry_threshold"]),
            max_retries=int(config_dict["max_retries"]),
            embedding_batch_size=int(config_dict["embedding_batch_size"]),
            use_anti_verbatim_vaccine=bool(
                config_dict.get("use_anti_verbatim_vaccine", True)
            ),
            processing_strategy=str(
                config_dict.get("processing_strategy", "bottom_to_top")
            ),
            preceding_context_verbatim_tokens=int(
                config_dict.get("preceding_context_verbatim_tokens", 0)
            ),
            preceding_context_max_extraneous_detail=int(
                config_dict.get("preceding_context_max_extraneous_detail", 5)
            ),
            preceding_context_num_seeds=num_seeds,
        )

    @classmethod
    def load(
        cls, config_path: Path | None = None, **cli_options: ConfigValueOrNone
    ) -> "IndexConfig":
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

    # Sentinel for "not provided" in replace() - distinct from None which is valid
    _NOT_PROVIDED: int = -999999

    def replace(
        self,
        target_chunk_tokens: int | None = None,
        preceding_summary_budget_tokens: int | None = None,
        summary_model: str | None = None,
        embedding_model: str | None = None,
        retry_threshold: float | None = None,
        max_retries: int | None = None,
        embedding_batch_size: int | None = None,
        use_anti_verbatim_vaccine: bool | None = None,
        processing_strategy: str | None = None,
        preceding_context_verbatim_tokens: int | None = None,
        preceding_context_max_extraneous_detail: int | None = None,
        preceding_context_num_seeds: int | None = _NOT_PROVIDED,
    ) -> "IndexConfig":
        """Create a new IndexConfig with some fields changed."""
        from dataclasses import replace

        return replace(
            self,
            target_chunk_tokens=(
                target_chunk_tokens
                if target_chunk_tokens is not None
                else self.target_chunk_tokens
            ),
            preceding_summary_budget_tokens=(
                preceding_summary_budget_tokens
                if preceding_summary_budget_tokens is not None
                else self.preceding_summary_budget_tokens
            ),
            summary_model=(
                summary_model if summary_model is not None else self.summary_model
            ),
            embedding_model=(
                embedding_model if embedding_model is not None else self.embedding_model
            ),
            retry_threshold=(
                retry_threshold if retry_threshold is not None else self.retry_threshold
            ),
            max_retries=max_retries if max_retries is not None else self.max_retries,
            embedding_batch_size=(
                embedding_batch_size
                if embedding_batch_size is not None
                else self.embedding_batch_size
            ),
            use_anti_verbatim_vaccine=(
                use_anti_verbatim_vaccine
                if use_anti_verbatim_vaccine is not None
                else self.use_anti_verbatim_vaccine
            ),
            processing_strategy=(
                processing_strategy
                if processing_strategy is not None
                else self.processing_strategy
            ),
            preceding_context_verbatim_tokens=(
                preceding_context_verbatim_tokens
                if preceding_context_verbatim_tokens is not None
                else self.preceding_context_verbatim_tokens
            ),
            preceding_context_max_extraneous_detail=(
                preceding_context_max_extraneous_detail
                if preceding_context_max_extraneous_detail is not None
                else self.preceding_context_max_extraneous_detail
            ),
            preceding_context_num_seeds=(
                preceding_context_num_seeds
                if preceding_context_num_seeds != self._NOT_PROVIDED
                else self.preceding_context_num_seeds
            ),
        )


@dataclass
class QueryConfig:
    """Configuration for query/retrieval operations.

    These parameters control how queries are processed and results are retrieved.
    """

    budget_tokens: int = 8000
    mmr_lambda: float = 0.7
    mmr_k_multiplier: float = 2.0
    embedding_model: str = "text-embedding-3-small"
    tiling_strategy: str = "greedy"  # "dp" or "greedy"

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
        if self.tiling_strategy not in ("dp", "greedy"):
            raise ValueError(
                f"tiling_strategy must be 'dp' or 'greedy', got {self.tiling_strategy}"
            )

    def replace(
        self,
        budget_tokens: int | None = None,
        mmr_lambda: float | None = None,
        mmr_k_multiplier: float | None = None,
        embedding_model: str | None = None,
        tiling_strategy: str | None = None,
    ) -> "QueryConfig":
        """Create a new QueryConfig with some fields changed."""
        from dataclasses import replace

        return replace(
            self,
            budget_tokens=(
                budget_tokens if budget_tokens is not None else self.budget_tokens
            ),
            mmr_lambda=mmr_lambda if mmr_lambda is not None else self.mmr_lambda,
            mmr_k_multiplier=(
                mmr_k_multiplier
                if mmr_k_multiplier is not None
                else self.mmr_k_multiplier
            ),
            embedding_model=(
                embedding_model if embedding_model is not None else self.embedding_model
            ),
            tiling_strategy=(
                tiling_strategy if tiling_strategy is not None else self.tiling_strategy
            ),
        )


@dataclass
class OperationalConfig:
    """Operational configuration for runtime environment.

    These parameters are environment-specific and are never saved to config files.
    They include storage paths, API keys, and other runtime settings.
    """

    openai_api_key: SecretStr = SecretStr("")
    # Durable backend selection and storage location
    backend: str = "sqlite"  # "sqlite" | "postgres"
    database_url: str = ""
    # Vector index backend for non-PostgreSQL setups. Tests override via env to use python.
    vector_backend: str = "chroma"  # "python" | "chroma"
    cache_size: int = 1000
    log_level: str = "INFO"
    validate_pipeline: bool = False

    def __post_init__(self) -> None:
        """Load API key from environment if not set."""
        if not self.openai_api_key.get_secret_value():
            self.openai_api_key = SecretStr(os.environ.get("OPENAI_API_KEY", ""))

        # Backend selection via env (overrides default)
        if os.environ.get("RAGZOOM_BACKEND"):
            self.backend = os.environ["RAGZOOM_BACKEND"].strip().lower()

        # Vector backend via env
        if os.environ.get("RAGZOOM_VECTOR_BACKEND"):
            self.vector_backend = os.environ["RAGZOOM_VECTOR_BACKEND"].strip().lower()

        # Data dir override for SQLite/vector persistence
        data_dir = os.environ.get("RAGZOOM_DATA_DIR")

        # Database URL resolution
        env_db = os.environ.get("RAGZOOM_DATABASE_URL")
        if env_db:
            self.database_url = env_db
        else:
            # If user provided an explicit URL at construction, respect it; otherwise infer from backend
            if not self.database_url:
                if self.backend == "postgres":
                    from ragzoom.worktree_utils import get_default_database_url

                    self.database_url = get_default_database_url()
                else:
                    from pathlib import Path

                    from ragzoom.worktree_utils import get_default_sqlite_url

                    base = Path(data_dir) if data_dir else None
                    self.database_url = get_default_sqlite_url(base)

        # If URL implies backend, update backend to stay consistent
        url = self.database_url.strip().lower()
        if url.startswith("sqlite"):
            self.backend = "sqlite"
        elif url.startswith("postgresql") or url.startswith("postgres"):
            self.backend = "postgres"

        # Apply worktree isolation when using PostgreSQL default name
        if self.backend == "postgres":
            from ragzoom.worktree_utils import get_worktree_database_url

            self.database_url = get_worktree_database_url(self.database_url)

        # Require chroma only when using the SQLite backend with chroma selected
        if self.backend == "sqlite" and self.vector_backend == "chroma":
            if importlib.util.find_spec("chromadb") is None:
                raise ImportError(
                    "chromadb is not installed but RAGZOOM_VECTOR_BACKEND=chroma was selected. "
                    "Install with `pip install chromadb` or set RAGZOOM_VECTOR_BACKEND=python."
                )

    def replace(
        self,
        openai_api_key: SecretStr | None = None,
        backend: str | None = None,
        database_url: str | None = None,
        vector_backend: str | None = None,
        cache_size: int | None = None,
        log_level: str | None = None,
        validate_pipeline: bool | None = None,
    ) -> "OperationalConfig":
        """Create a new OperationalConfig with some fields changed."""
        from dataclasses import replace

        return replace(
            self,
            openai_api_key=(
                openai_api_key if openai_api_key is not None else self.openai_api_key
            ),
            backend=backend if backend is not None else self.backend,
            database_url=(
                database_url if database_url is not None else self.database_url
            ),
            vector_backend=(
                vector_backend if vector_backend is not None else self.vector_backend
            ),
            cache_size=cache_size if cache_size is not None else self.cache_size,
            log_level=log_level if log_level is not None else self.log_level,
            validate_pipeline=(
                validate_pipeline
                if validate_pipeline is not None
                else self.validate_pipeline
            ),
        )


def _load_index_config(
    config_path: Path | None = None, **cli_options: ConfigValueOrNone
) -> dict[str, ConfigValue]:
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
