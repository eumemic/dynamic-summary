"""Configuration management for RagZoom."""

import json
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagZoomConfig(BaseSettings):
    """Main configuration for RagZoom system."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RAGZOOM_", case_sensitive=False
    )

    # Core parameters
    budget_tokens: int = Field(
        default=8000, description="Hard budget for stitched summary in tokens"
    )
    leaf_tokens: int = Field(
        default=200, description="Target size for leaf chunks in tokens"
    )

    # Retrieval parameters
    mmr_lambda: float = Field(
        default=0.7,
        description="MMR relevance vs diversity (0-1, higher=more relevant)",
    )
    mmr_k_multiplier: float = Field(
        default=2.0, description="Retrieve k_multiplier * N_max candidates for MMR"
    )

    # Slope cap and smoothing
    slope_cap: bool = Field(
        default=True, description="Forbid depth jumps > 1 level in tiling"
    )
    slope_cap_size: int = Field(
        default=1,
        description="Maximum depth difference allowed between adjacent tiling nodes",
    )
    adjacent_context_tokens: int = Field(
        default=75, description="Tokens from prev/next chunks for summarization context"
    )
    smoothing_pass_enabled: bool = Field(
        default=False, description="Enable smoothing pass for tiling joins"
    )

    # Summary correction parameters
    summary_deviation_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Max deviation from target before retry (0.2 = 20%)",
    )
    summary_max_retries: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Maximum retries for summary correction",
    )

    # Validation
    validate_pipeline: bool = Field(
        default=False, description="Enable validation checks for tiling invariants"
    )
    smoothing_model: str = Field(
        default="gpt-3.5-turbo", description="Model for smoothing pass"
    )
    smoothing_max_tokens: int = Field(
        default=150, description="Max tokens per smoothing operation"
    )

    # Storage configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    chroma_persist_directory: str = Field(
        default="./chroma_db", description="Directory for Chroma persistence"
    )
    sqlite_database_url: str = Field(
        default="sqlite:///./ragzoom.db", description="SQLite database URL"
    )

    # Embedding configuration
    embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    embedding_dimensions: int | None = Field(
        default=None, description="Embedding dimensions (None=use model default)"
    )

    # Summarization configuration
    summary_model: str = Field(
        default="gpt-4o", description="Model for node summarization"
    )
    summary_temperature: float = Field(
        default=0.3, description="Temperature for summarization"
    )

    # Operational settings
    log_level: str = Field(default="INFO", description="Logging level")
    cache_size: int = Field(
        default=1000, description="Maximum number of nodes in LRU cache"
    )
    embedding_batch_size: int = Field(
        default=100, description="Batch size for embedding API calls"
    )
    pin_depth_max: int = Field(
        default=2, description="Deepest level a node may be permanently pinned"
    )

    # Pricing configuration
    pricing_file: str | None = Field(
        default=None,
        description="Path to JSON file with LLM pricing (defaults to ragzoom/pricing.json)",
    )

    # Cost estimation settings (per 1K tokens) - populated from pricing file
    embedding_cost_per_1k: float = Field(
        default=0.0,  # Will be set by model_validator
        description="Cost per 1K tokens for embeddings",
    )
    summary_input_cost_per_1k: float = Field(
        default=0.0,  # Will be set by model_validator
        description="Cost per 1K input tokens for summary model",
    )
    summary_output_cost_per_1k: float = Field(
        default=0.0,  # Will be set by model_validator
        description="Cost per 1K output tokens for summary model",
    )

    @field_validator("mmr_lambda")
    @classmethod
    def validate_mmr_lambda(cls, v: float) -> float:
        """Ensure MMR lambda is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("mmr_lambda must be between 0 and 1")
        return v

    @field_validator("adjacent_context_tokens")
    @classmethod
    def validate_adjacent_context(cls, v: int, info: Any) -> int:
        """Ensure adjacent context doesn't exceed leaf size."""
        leaf_tokens = info.data.get("leaf_tokens", 200)
        if v > leaf_tokens:
            raise ValueError("adjacent_context_tokens cannot exceed leaf_tokens")
        return v

    @model_validator(mode="after")
    def load_pricing_from_file(self) -> "RagZoomConfig":
        """Load pricing from JSON file."""
        # Load pricing data
        pricing_data = self._load_pricing_data()

        # Get embedding price
        if self.embedding_model not in pricing_data["embeddings"]:
            available_models = list(pricing_data["embeddings"].keys())
            pricing_file_path = self.pricing_file or "ragzoom/pricing.json"
            raise ValueError(
                f"Embedding model '{self.embedding_model}' not found in pricing file. "
                f"Available models: {available_models}\n"
                f"To add support for '{self.embedding_model}', update the pricing file at: {pricing_file_path}"
            )
        self.embedding_cost_per_1k = pricing_data["embeddings"][self.embedding_model]

        # Get LLM prices
        if self.summary_model not in pricing_data["llms"]:
            available_models = list(pricing_data["llms"].keys())
            pricing_file_path = self.pricing_file or "ragzoom/pricing.json"
            raise ValueError(
                f"Summary model '{self.summary_model}' not found in pricing file. "
                f"Available models: {available_models}\n"
                f"To add support for '{self.summary_model}', update the pricing file at: {pricing_file_path}"
            )
        llm_pricing = pricing_data["llms"][self.summary_model]
        self.summary_input_cost_per_1k = llm_pricing["input"]
        self.summary_output_cost_per_1k = llm_pricing["output"]

        return self

    def _load_pricing_data(self) -> dict[str, Any]:
        """Load pricing data from JSON file."""
        if self.pricing_file:
            # Use explicitly provided file
            pricing_path = Path(self.pricing_file)
        else:
            # Default to ragzoom/pricing.json
            module_dir = Path(__file__).parent
            pricing_path = module_dir / "pricing.json"

        if not pricing_path.exists():
            raise FileNotFoundError(
                f"Pricing file not found at {pricing_path}. "
                "Please ensure ragzoom/pricing.json exists or specify a custom path "
                "using the RAGZOOM_PRICING_FILE environment variable or pricing_file parameter."
            )

        try:
            with open(pricing_path) as f:
                data: dict[str, Any] = json.load(f)
                return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in pricing file {pricing_path}: {e}")
        except OSError as e:
            raise OSError(f"Error reading pricing file {pricing_path}: {e}")

    @property
    def n_max(self) -> int:
        """Calculate maximum number of nodes based on budget."""
        # Increased from budget/2*leaf to budget/leaf for better coverage
        return self.budget_tokens // self.leaf_tokens
