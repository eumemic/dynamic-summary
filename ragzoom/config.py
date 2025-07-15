"""Configuration management for RagZoom."""

from typing import Optional

from pydantic import Field, field_validator
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
    budget_strategy: str = Field(
        default="drop", description="Budget enforcement strategy: 'drop' or 'truncate'"
    )
    leaf_tokens: int = Field(
        default=200, description="Target size for leaf chunks in tokens"
    )
    leaf_overlap_tokens: int = Field(
        default=20, description="Overlap between consecutive leaf chunks"
    )

    # Retrieval parameters
    mmr_lambda: float = Field(
        default=0.7, description="MMR relevance vs diversity (0-1, higher=more relevant)"
    )
    mmr_k_multiplier: float = Field(
        default=2.0, description="Retrieve k_multiplier * N_max candidates for MMR"
    )

    # Slope cap and smoothing
    slope_cap: bool = Field(
        default=True, description="Forbid depth jumps > 1 level in frontier"
    )
    slope_cap_size: int = Field(
        default=1, description="Maximum depth difference allowed between adjacent frontier nodes"
    )
    adjacent_context_tokens: int = Field(
        default=75, description="Tokens from prev/next chunks for summarization context"
    )
    smoothing_pass_enabled: bool = Field(
        default=False, description="Enable smoothing pass for frontier joins"
    )
    smoothing_model: str = Field(
        default="gpt-3.5-turbo", description="Model for smoothing pass"
    )
    smoothing_max_tokens: int = Field(
        default=150, description="Max tokens per smoothing operation"
    )

    # Pinning and eviction
    pin_depth_max: int = Field(
        default=2, description="Deepest level a node may be permanently pinned"
    )
    ttl_turns: int = Field(
        default=0, description="Working-set eviction TTL (0=disabled, use score queue)"
    )
    freshness_decay: float = Field(
        default=0.9, description="Decay factor for freshness in priority calculation"
    )

    # Storage configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
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
    embedding_dimensions: Optional[int] = Field(
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

    @field_validator("mmr_lambda")
    @classmethod
    def validate_mmr_lambda(cls, v: float) -> float:
        """Ensure MMR lambda is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("mmr_lambda must be between 0 and 1")
        return v

    @field_validator("freshness_decay")
    @classmethod
    def validate_freshness_decay(cls, v: float) -> float:
        """Ensure freshness decay is between 0 and 1."""
        if not 0 < v <= 1:
            raise ValueError("freshness_decay must be between 0 and 1")
        return v

    @field_validator("budget_strategy")
    @classmethod
    def validate_budget_strategy(cls, v: str) -> str:
        """Ensure budget strategy is valid."""
        if v not in ["drop", "truncate"]:
            raise ValueError("budget_strategy must be 'drop' or 'truncate'")
        return v

    @field_validator("adjacent_context_tokens")
    @classmethod
    def validate_adjacent_context(cls, v: int, info) -> int:
        """Ensure adjacent context doesn't exceed leaf size."""
        leaf_tokens = info.data.get("leaf_tokens", 200)
        if v > leaf_tokens:
            raise ValueError("adjacent_context_tokens cannot exceed leaf_tokens")
        return v

    @property
    def n_max(self) -> int:
        """Calculate maximum number of nodes based on budget."""
        # Increased from budget/2*leaf to budget/leaf for better coverage
        return self.budget_tokens // self.leaf_tokens
