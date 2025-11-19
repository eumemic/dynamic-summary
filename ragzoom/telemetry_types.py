"""Type definitions for telemetry data structures.

This module provides TypedDict definitions for all telemetry-related data structures,
enabling type-safe access to telemetry data and preventing bugs from accessing
non-existent fields.

Supports telemetry format v4.3. All legacy formats have been removed.
"""

from typing_extensions import NotRequired, TypedDict


# Embedding telemetry types
class EmbeddingTelemetryDict(TypedDict):
    """Type definition for embedding telemetry data."""

    text_tokens: int
    batch_size: int
    batch_position: int
    model: str
    start_time: float
    end_time: float


# Summary attempt types
class SummaryAttemptDict(TypedDict):
    """Type definition for a single summary attempt."""

    # Required fields
    target_tokens: int
    prompt_tokens: int
    completion_tokens: int
    actual_tokens: int
    model: str
    start_time: float
    end_time: float

    # Optional fields
    cached_tokens: NotRequired[
        int
    ]  # Number of cached prompt tokens (for prompt caching)
    prompt_tokens_details: NotRequired[
        "PromptTokensDetailsDict"
    ]  # Full OpenAI prompt token details


# Node telemetry types
class NodeTelemetryDict(TypedDict):
    """Type definition for node telemetry data."""

    # Required fields
    node_id: str
    height: int
    created_at: float

    # Optional fields
    span: NotRequired[
        tuple[int, int]
    ]  # Document character positions (start, end) - reintroduced in v4.2
    embedding: NotRequired[EmbeddingTelemetryDict]
    summary_attempts: NotRequired[list[SummaryAttemptDict]]
    accepted_attempt: NotRequired[int]  # Index of the accepted attempt
    input_text_tokens: NotRequired[
        int
    ]  # Combined tokens from children (for non-leaf nodes)
    fidelity: NotRequired[float]


# Document metadata types
class DocumentMetadataDict(TypedDict):
    """Type definition for document metadata."""

    source_document_tokens: int
    chunk_size: NotRequired[int]
    indexed_at: NotRequired[float]
    leaf_tokens: NotRequired[int]  # v1 name for chunk_size


# Document structure
class DocumentDict(TypedDict):
    """Type definition for a document in telemetry data."""

    metadata: DocumentMetadataDict
    nodes: list[NodeTelemetryDict]


# OpenAI API prompt token details structure
class PromptTokensDetailsDict(TypedDict):
    """Type definition for OpenAI API prompt token details.

    This represents the detailed token breakdown provided by OpenAI's API,
    typically including cached and audio tokens.
    """

    cached_tokens: NotRequired[int]
    audio_tokens: NotRequired[int]


# Model metadata structures
class EmbeddingModelMetadataDict(TypedDict):
    """Type definition for embedding model metadata."""

    model: str
    dimensions: NotRequired[int]
    cost_per_1k: NotRequired[float]
    error: NotRequired[str]  # Error message if metadata collection failed


class SummaryModelMetadataDict(TypedDict):
    """Type definition for summary model metadata."""

    model: str
    input_cost_per_1k: NotRequired[float]
    output_cost_per_1k: NotRequired[float]
    supports_temperature: NotRequired[bool]
    is_gpt5: NotRequired[bool]
    cache_discount: NotRequired[float]
    error: NotRequired[str]  # Error message if metadata collection failed


class ModelMetadataDict(TypedDict):
    """Type definition for complete model metadata in telemetry data.

    This structure captures model capabilities, costs, and configuration
    for reproducibility of indexing operations.
    """

    embedding: NotRequired[EmbeddingModelMetadataDict]
    summary: NotRequired[SummaryModelMetadataDict]
    models_last_updated: NotRequired[str]  # Last update timestamp from models.json
    error: NotRequired[str]  # Error message if metadata collection failed


class RuntimeInfoDict(TypedDict):
    """Type definition for runtime environment information in telemetry data.

    This structure captures environment details for reproducibility
    of indexing operations.
    """

    python_version: str
    platform: str
    ragzoom_version: str
    tiktoken_version: NotRequired[str]
    openai_version: NotRequired[str]


# Configuration dictionary for v3.1+
class ConfigDict(TypedDict):
    """Type definition for configuration object in v3.1+ formats."""

    target_chunk_tokens: int
    summary_model: str
    embedding_model: str
    # Additional config fields may be present
    budget_tokens: NotRequired[int]


# Top-level telemetry structure for v4.2 (current format)
class ChunkSplitTelemetryDict(TypedDict, total=False):
    """Details captured during chunk splitting."""

    start_time: float
    end_time: float
    duration: float
    chunk_count: int
    total_tokens: int
    new_text_chars: int
    existing_tail_chars: int
    combined_chars: int


class TelemetryDataDict(TypedDict):
    """Type definition for v4.3 telemetry data structure (current format)."""

    format_version: str
    document_id: str
    source_document_tokens: int
    indexed_at: float
    config: ConfigDict
    model_metadata: ModelMetadataDict
    system_prompts: dict[str, str]
    runtime_info: RuntimeInfoDict
    nodes: list[NodeTelemetryDict]
    # Optional document path
    document_path: NotRequired[str]
    append_metadata: NotRequired[dict[str, object]]
    chunk_split: NotRequired[ChunkSplitTelemetryDict]


# Analysis result types
# Note: TypedDict doesn't support non-identifier keys, so we use a regular dict[str, int]
# for retry_distribution in the RetryAnalysisDict below


class RetryAnalysisDict(TypedDict):
    """Type definition for retry analysis results."""

    retry_rate: float
    total_attempts: int
    successful_attempts: int
    retry_attempts: int
    retry_success_rate: float
    rejection_reasons: dict[str, int]
    nodes_with_retries: int
    total_nodes_with_summaries: int
    retry_distribution: dict[str, int]  # Keys are "0", "1", "2", "3+"
    avg_retries_per_node: float
    max_retries: int
    retry_time_seconds: float
    avg_time_per_retry: float
    time_wasted_on_rejections: float


class BatchEfficiencyDict(TypedDict):
    """Type definition for batch efficiency results."""

    total_embeddings: int
    batched_embeddings: int
    single_embeddings: int
    avg_batch_size: float
    batch_utilization: float
    max_batch_size: int
    batch_size_distribution: dict[int, int]
    # Legacy fields used by visualization
    batch_sizes: NotRequired[list[int]]
    total_batches: NotRequired[int]


class SummaryQualityDict(TypedDict):
    """Type definition for summary quality metrics."""

    total_summaries: int
    accepted_summaries: int
    rejection_distribution: dict[str, int]
    avg_attempts_per_summary: float
    first_attempt_success_rate: float


class TimingMetricsDict(TypedDict):
    """Type definition for timing metrics."""

    total_duration_seconds: float
    embedding_time_seconds: float
    summary_time_seconds: float
    embedding_time_percentage: float
    summary_time_percentage: float
    avg_embedding_time_ms: float
    avg_summary_time_ms: float
