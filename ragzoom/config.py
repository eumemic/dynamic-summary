"""Configuration management for RagZoom."""

import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import TypedDict


class PrecedingContextConfigDict(TypedDict, total=False):
    """Type definition for per-node-type preceding context config."""

    num_seeds: int | None
    verbatim_tokens: int
    min_forest_completeness: float
    max_forest_height_differential: int | None
    token_cap: int | None


class PrecedingContextSettingsDict(TypedDict, total=False):
    """Type definition for nested preceding context settings."""

    leaf: PrecedingContextConfigDict
    inner: PrecedingContextConfigDict


class IndexConfigDict(TypedDict, total=False):
    """Type definition for IndexConfig dictionary representation."""

    target_chunk_tokens: int | None
    target_embedding_tokens: int
    max_parallelism: int
    summary_model: str
    summary_api_base: str | None
    summary_api_key: str | None
    embedding_model: str
    retry_threshold: float
    max_retries: int
    embedding_batch_size: int
    use_anti_verbatim_vaccine: bool
    processing_strategy: str
    preceding_context: PrecedingContextSettingsDict
    summary_reasoning_level: str | None
    summarization_guidance: str | None
    # Deprecated alias for summarization_guidance
    summary_system_prompt: str | None


# Sentinel value to distinguish "not provided" from "explicitly None"
class _NotProvided:
    """Sentinel value to indicate a parameter was not provided."""

    pass


_NOT_PROVIDED = _NotProvided()

# Type for configuration values that can be primitives
ConfigValue = str | int | float | bool
# Type that includes None for CLI parameters
ConfigValueOrNone = str | int | float | bool | None | _NotProvided

# Environment variables that override saved IndexConfig fields. Read in
# _load_index_config with precedence env > config file > default; CLI options
# still win over env. RAGZOOM_SUMMARY_MODEL was previously dead documentation.
_SUMMARY_ENV_OVERRIDES: dict[str, str] = {
    "RAGZOOM_SUMMARY_MODEL": "summary_model",
    "RAGZOOM_SUMMARY_API_BASE": "summary_api_base",
    "RAGZOOM_SUMMARY_API_KEY": "summary_api_key",
}


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
class PrecedingContextConfig:
    """Configuration for preceding context retrieval (per node type).

    Controls how preceding context is retrieved for leaf nodes (during embedding)
    or inner nodes (during summarization).

    The retrieval algorithm always produces a complete tiling - a sequence of
    adjacent, non-overlapping nodes that cover the document from start to the
    current position. The `token_cap` parameter selects the rightmost portion
    of this tiling.

    Attributes:
        num_seeds: Number of semantically similar nodes to use as seeds for
            retrieval. Must be 0 for inner nodes (they don't store embeddings).
        verbatim_tokens: Token budget for verbatim (leaf) content at the end
            of the tiling. The retrieval ensures the rightmost nodes are leaves
            up to this budget.
        min_forest_completeness: Minimum completeness ratio (0.0-1.0) for the
            preceding forest before a job becomes eligible.
        max_forest_height_differential: Maximum allowed height difference between
            the current node and the minimum height in the preceding forest.
        token_cap: If set, select the smallest suffix of the tiling with total
            tokens >= this value. Rounds up to whole nodes. None means use the
            full tiling. Example: token_cap=400 takes the rightmost ~400 tokens.
    """

    num_seeds: int | None = None
    verbatim_tokens: int = 2000
    min_forest_completeness: float = 0.0
    max_forest_height_differential: int | None = None
    token_cap: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.num_seeds is not None and self.num_seeds < 0:
            raise ValueError(f"num_seeds must be >= 0, got {self.num_seeds}")
        if self.verbatim_tokens < 0:
            raise ValueError(
                f"verbatim_tokens must be >= 0, got {self.verbatim_tokens}"
            )
        if not 0.0 <= self.min_forest_completeness <= 1.0:
            raise ValueError(
                f"min_forest_completeness must be between 0.0 and 1.0, "
                f"got {self.min_forest_completeness}"
            )
        if (
            self.max_forest_height_differential is not None
            and self.max_forest_height_differential < 0
        ):
            raise ValueError(
                f"max_forest_height_differential must be >= 0, "
                f"got {self.max_forest_height_differential}"
            )
        if self.token_cap is not None and self.token_cap < 0:
            raise ValueError(f"token_cap must be >= 0, got {self.token_cap}")

    @classmethod
    def from_dict(cls, d: PrecedingContextConfigDict) -> "PrecedingContextConfig":
        """Create from dictionary."""
        return cls(
            num_seeds=d.get("num_seeds"),
            verbatim_tokens=d.get("verbatim_tokens", 2000),
            min_forest_completeness=d.get("min_forest_completeness", 0.0),
            max_forest_height_differential=d.get("max_forest_height_differential"),
            token_cap=d.get("token_cap"),
        )


@dataclass
class PrecedingContextSettings:
    """Container for leaf and inner node preceding context configs.

    Allows different retrieval strategies for embedding (leaf) vs summarization (inner).

    Note: inner.num_seeds must be 0. Inner nodes don't store embeddings (to enable
    parallel summarization), so semantic retrieval is not supported.
    """

    leaf: PrecedingContextConfig = field(default_factory=PrecedingContextConfig)
    inner: PrecedingContextConfig = field(
        default_factory=lambda: PrecedingContextConfig(token_cap=400)
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Inner nodes don't store embeddings, so num_seeds must be 0.
        if (self.inner.num_seeds or 0) > 0:
            raise ValueError(
                "inner.num_seeds must be 0. Inner nodes don't store embeddings, "
                "so semantic retrieval is not supported for inner node preceding "
                "context."
            )

    @classmethod
    def from_dict(cls, d: PrecedingContextSettingsDict) -> "PrecedingContextSettings":
        """Create from dictionary."""
        leaf_dict = d.get("leaf", {})
        inner_dict = d.get("inner", {})
        # Inner defaults to token_cap=400 (cap output to rightmost ~400 tokens)
        inner_dict_with_default: PrecedingContextConfigDict = {
            "token_cap": 400,
            **inner_dict,
        }
        return cls(
            leaf=PrecedingContextConfig.from_dict(leaf_dict),
            inner=PrecedingContextConfig.from_dict(inner_dict_with_default),
        )


@dataclass
class IndexConfig:
    """Configuration for document indexing.

    These parameters control how documents are chunked, summarized, and embedded.
    This is the configuration that gets saved to and loaded from config files.

    Note: Always use IndexConfig.load() to create instances. Do not instantiate directly.
    """

    target_chunk_tokens: int | None
    max_parallelism: int
    summary_model: str
    embedding_model: str
    retry_threshold: float
    max_retries: int
    embedding_batch_size: int
    use_anti_verbatim_vaccine: bool
    processing_strategy: str
    preceding_context: PrecedingContextSettings = field(
        default_factory=PrecedingContextSettings
    )
    summary_reasoning_level: str | None = None
    summary_api_base: str | None = None
    """Optional endpoint override for the summary model (e.g. a proxy URL).

    When set, it is forwarded to the LiteLLM summary adapter as ``api_base``.
    Recorded per-index so the summarizer endpoint is reproducible.
    """
    summary_api_key: SecretStr | None = None
    """Optional API key for the summary model endpoint.

    Stored as SecretStr so it redacts in logs. Forwarded to the LiteLLM summary
    adapter as ``api_key``. ``None`` lets LiteLLM resolve credentials itself.
    """
    summarization_guidance: str | None = None
    """Additional guidance for summary generation.

    If provided, this guidance is appended to the default system prompt
    under a "# Summarization Guidance" section. The default prompt's
    essential instructions (output only compressed text) are preserved.

    Use this to provide domain context that improves summary quality
    for specific content types (legal, medical, code, etc.).
    """
    target_embedding_tokens: int = 500
    """Target token count for text sent to embedding.

    When combined input (preceding context + leaf text) exceeds this target,
    an LLM generates retrieval-optimized text to fit within the budget.
    When input fits within target, it passes through unchanged.

    Default: 500 tokens (optimal for focused, high-quality embeddings).
    """

    @property
    def summary_system_prompt(self) -> str | None:
        """Deprecated alias for summarization_guidance.

        Use summarization_guidance instead. This property will be removed
        in a future version.
        """
        import warnings

        warnings.warn(
            "summary_system_prompt is deprecated, use summarization_guidance instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.summarization_guidance

    @summary_system_prompt.setter
    def summary_system_prompt(self, value: str | None) -> None:
        """Deprecated setter for summarization_guidance."""
        import warnings

        warnings.warn(
            "summary_system_prompt is deprecated, use summarization_guidance instead",
            DeprecationWarning,
            stacklevel=2,
        )
        object.__setattr__(self, "summarization_guidance", value)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.target_chunk_tokens is not None and self.target_chunk_tokens <= 0:
            raise ValueError(
                f"target_chunk_tokens must be positive when set, got {self.target_chunk_tokens}"
            )
        if self.target_embedding_tokens <= 0:
            raise ValueError(
                f"target_embedding_tokens must be positive, got {self.target_embedding_tokens}"
            )
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

        valid_strategies = {"bottom_to_top", "left_to_right"}
        if self.processing_strategy not in valid_strategies:
            raise ValueError(
                f"processing_strategy must be one of {valid_strategies}, got '{self.processing_strategy}'"
            )

    @classmethod
    def from_dict(cls, config_dict: dict[str, ConfigValue]) -> "IndexConfig":
        """Create IndexConfig from a dictionary (e.g., from config JSON).

        Args:
            config_dict: Dictionary with config fields. Must include
                        preceding_context with leaf and inner sub-configs.

        Returns:
            IndexConfig instance
        """
        raw_nested = config_dict.get("preceding_context")
        if not isinstance(raw_nested, dict):
            raise ValueError(
                "preceding_context must be a dict with 'leaf' and 'inner' keys"
            )
        nested_dict: PrecedingContextSettingsDict = raw_nested
        preceding_context = PrecedingContextSettings.from_dict(nested_dict)

        # Reject deprecated field with helpful error
        if "target_embedding_context_tokens" in config_dict:
            raise ValueError(
                "target_embedding_context_tokens has been removed. "
                "Use target_embedding_tokens instead. "
                "See specs/embedding-text-optimization.md for migration details."
            )

        # Get optional summary_reasoning_level (may be str or None)
        raw_reasoning = config_dict.get("summary_reasoning_level")
        summary_reasoning_level: str | None = (
            str(raw_reasoning) if raw_reasoning is not None else None
        )

        # Get optional summary endpoint overrides (may be str or None)
        raw_summary_api_base = config_dict.get("summary_api_base")
        summary_api_base: str | None = (
            str(raw_summary_api_base) if raw_summary_api_base is not None else None
        )
        raw_summary_api_key = config_dict.get("summary_api_key")
        summary_api_key: SecretStr | None = (
            SecretStr(str(raw_summary_api_key))
            if raw_summary_api_key is not None
            else None
        )

        # Get optional summarization_guidance (may be str or None)
        # Support both new name and deprecated old name (summary_system_prompt)
        import warnings

        raw_guidance = config_dict.get("summarization_guidance")
        raw_old_prompt = config_dict.get("summary_system_prompt")

        summarization_guidance: str | None = None
        if raw_guidance is not None:
            # New name takes precedence
            summarization_guidance = str(raw_guidance)
        elif raw_old_prompt is not None:
            # Fall back to deprecated name with warning
            warnings.warn(
                "summary_system_prompt in config is deprecated, "
                "use summarization_guidance instead",
                DeprecationWarning,
                stacklevel=2,
            )
            summarization_guidance = str(raw_old_prompt)

        # Handle target_chunk_tokens which can be int or None
        raw_target_chunk = config_dict.get("target_chunk_tokens")
        target_chunk_tokens_value: int | None = (
            int(raw_target_chunk) if raw_target_chunk is not None else None
        )

        return cls(
            target_chunk_tokens=target_chunk_tokens_value,
            max_parallelism=int(config_dict.get("max_parallelism", 30)),
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
            preceding_context=preceding_context,
            summary_reasoning_level=summary_reasoning_level,
            summary_api_base=summary_api_base,
            summary_api_key=summary_api_key,
            summarization_guidance=summarization_guidance,
            target_embedding_tokens=int(
                config_dict.get("target_embedding_tokens", 500)
            ),
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
            **cli_options: CLI options that override config file. Use None to explicitly
                          set a field to None (e.g., target_chunk_tokens=None).

        Returns:
            IndexConfig instance
        """
        config_dict = _load_index_config(config_path, **cli_options)
        return cls.from_dict(config_dict)

    def replace(
        self,
        target_chunk_tokens: int | None = None,
        max_parallelism: int | None = None,
        summary_model: str | None = None,
        embedding_model: str | None = None,
        retry_threshold: float | None = None,
        max_retries: int | None = None,
        embedding_batch_size: int | None = None,
        use_anti_verbatim_vaccine: bool | None = None,
        processing_strategy: str | None = None,
        preceding_context: PrecedingContextSettings | None = None,
        summary_reasoning_level: str | None = None,
        summary_api_base: str | None = None,
        summary_api_key: SecretStr | None = None,
        summarization_guidance: str | None = None,
        target_embedding_tokens: int | None = None,
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
            max_parallelism=(
                max_parallelism if max_parallelism is not None else self.max_parallelism
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
            preceding_context=(
                preceding_context
                if preceding_context is not None
                else self.preceding_context
            ),
            summary_reasoning_level=(
                summary_reasoning_level
                if summary_reasoning_level is not None
                else self.summary_reasoning_level
            ),
            summary_api_base=(
                summary_api_base
                if summary_api_base is not None
                else self.summary_api_base
            ),
            summary_api_key=(
                summary_api_key if summary_api_key is not None else self.summary_api_key
            ),
            summarization_guidance=(
                summarization_guidance
                if summarization_guidance is not None
                else self.summarization_guidance
            ),
            target_embedding_tokens=(
                target_embedding_tokens
                if target_embedding_tokens is not None
                else self.target_embedding_tokens
            ),
        )

    # Maximum budget for preceding context retrieval.
    # Capped to prevent excessive seed calculation (budget // chunk_tokens = num_seeds).
    # 32K is a reasonable limit that allows substantial context while keeping
    # retrieval performant.
    _MAX_PRECEDING_CONTEXT_BUDGET = 32000

    @property
    def preceding_context_budget(self) -> int:
        """Get the preceding context budget derived from the summary model's context window.

        The budget is calculated as the model's context window minus overhead for:
        - The chunk being summarized (~target_chunk_tokens or target_embedding_context_tokens)
        - The output summary (~target_chunk_tokens or target_embedding_context_tokens)
        - System prompt and formatting (~1000 tokens)

        When target_chunk_tokens is None (client-managed chunking), uses
        target_embedding_tokens as the basis for overhead calculation.

        Capped at _MAX_PRECEDING_CONTEXT_BUDGET to prevent performance issues
        when num_seeds is calculated from budget (budget // chunk_tokens).
        """
        from ragzoom.model_info import ModelInfo

        context_window = ModelInfo().get_context_window(self.summary_model)
        # Use target_embedding_tokens as fallback when target_chunk_tokens is None
        chunk_size = (
            self.target_chunk_tokens
            if self.target_chunk_tokens is not None
            else self.target_embedding_tokens
        )
        overhead = chunk_size * 2 + 1000
        uncapped = max(context_window - overhead, chunk_size)
        return min(uncapped, self._MAX_PRECEDING_CONTEXT_BUDGET)


@dataclass
class QueryConfig:
    """Configuration for query/retrieval operations.

    These parameters control how queries are processed and results are retrieved.
    """

    budget_tokens: int | None = None
    mmr_lambda: float = 0.7
    mmr_k_multiplier: float = 2.0
    embedding_model: str = "text-embedding-3-small"
    use_bm25: bool = True
    """Enable BM25 hybrid search. Default True."""
    bm25_weight: float = 1.0
    """Weight for BM25 in RRF. 1.0 = equal weight with vector."""
    retrieval_mode: str | None = None
    """Final-tiling strategy: "coverage" (default) or "concentrate".

    "coverage" spreads the token budget across the whole timeline by rolling up
    the least-relevant sibling pairs into their summaries. "concentrate" is
    top-k over the tree's verbatim leaves: it admits the highest query-relevance
    leaves until the budget is hit, with no roll-up and no whole-range coverage
    guarantee.

    The constructor argument defaults to ``None``, meaning "unspecified": in
    that case the RAGZOOM_RETRIEVAL_MODE env var is consulted, falling back to
    "coverage". An explicit argument always wins over the env var. After
    ``__post_init__`` the field always holds a concrete mode string, never None.
    The server constructs ``QueryConfig()`` with no argument, so the env var
    reaches the retriever's tiling decision through the environment.
    """

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Resolve retrieval_mode: explicit argument wins; otherwise consult the
        # env var (the server's env-driven switch); otherwise default coverage.
        if self.retrieval_mode is None:
            env_mode = os.environ.get("RAGZOOM_RETRIEVAL_MODE")
            self.retrieval_mode = (
                env_mode.strip().lower() if env_mode is not None else "coverage"
            )

        valid_modes = {"coverage", "concentrate"}
        if self.retrieval_mode not in valid_modes:
            raise ValueError(
                f"retrieval_mode must be one of {sorted(valid_modes)}, "
                f"got '{self.retrieval_mode}'"
            )

        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(
                f"mmr_lambda must be between 0.0 and 1.0, got {self.mmr_lambda}"
            )
        if self.budget_tokens is not None and self.budget_tokens <= 0:
            raise ValueError(
                f"budget_tokens must be positive, got {self.budget_tokens}"
            )
        if self.mmr_k_multiplier <= 0:
            raise ValueError(
                f"mmr_k_multiplier must be positive, got {self.mmr_k_multiplier}"
            )
        if self.bm25_weight <= 0:
            raise ValueError(f"bm25_weight must be positive, got {self.bm25_weight}")

    def replace(
        self,
        budget_tokens: int | None = None,
        mmr_lambda: float | None = None,
        mmr_k_multiplier: float | None = None,
        embedding_model: str | None = None,
        use_bm25: bool | None = None,
        bm25_weight: float | None = None,
        retrieval_mode: str | None = None,
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
            use_bm25=use_bm25 if use_bm25 is not None else self.use_bm25,
            bm25_weight=bm25_weight if bm25_weight is not None else self.bm25_weight,
            retrieval_mode=(
                retrieval_mode if retrieval_mode is not None else self.retrieval_mode
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
    # OpenAI client timeout in seconds (default 10min is too long for interactive use)
    openai_timeout: float = 120.0

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

        # Database URL resolution (support both RAGZOOM_DATABASE_URL and DATABASE_URL for Railway)
        env_db = os.environ.get("RAGZOOM_DATABASE_URL") or os.environ.get(
            "DATABASE_URL"
        )
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
            # Convert postgresql:// to postgresql+psycopg:// for psycopg3 compatibility
            # Some providers use postgresql:// which defaults to psycopg2
            if self.database_url.startswith("postgresql://"):
                self.database_url = self.database_url.replace(
                    "postgresql://", "postgresql+psycopg://", 1
                )

        # Apply worktree isolation when using PostgreSQL default name
        # Skip if RAGZOOM_SKIP_WORKTREE_ISOLATION is set (for admin tools connecting to production)
        skip_isolation = os.environ.get(
            "RAGZOOM_SKIP_WORKTREE_ISOLATION", ""
        ).lower() in (
            "1",
            "true",
            "yes",
        )
        if self.backend == "postgres" and not skip_isolation:
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

    # Override with environment variables (precedence: env > file > default).
    # These knobs select the summary model and route it at an optional endpoint.
    for env_name, config_key in _SUMMARY_ENV_OVERRIDES.items():
        env_value = os.environ.get(env_name)
        if env_value is not None:
            config[config_key] = env_value

    # Override with CLI options (highest precedence)
    # Note: We filter out _NOT_PROVIDED sentinel but allow None (which is valid for target_chunk_tokens)
    for key, value in cli_options.items():
        if not isinstance(value, _NotProvided) and key in config:
            config[key] = value

    return dict(config)
