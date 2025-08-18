"""Type definitions for telemetry data structures.

This module provides TypedDict definitions for all telemetry-related data structures,
enabling type-safe access to telemetry data and preventing bugs from accessing
non-existent fields.

Supports v3.0, v3.1, and v4.1 telemetry formats with appropriate NotRequired fields.
Legacy v1.0 and v2.0 formats are no longer supported.
"""

from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired


# Embedding telemetry types
class EmbeddingTelemetryDict(TypedDict):
    """Type definition for embedding telemetry data."""

    text_tokens: int
    batch_size: int
    batch_position: int
    model: str
    start_time: float
    end_time: float
    # v1 compatibility
    timestamp: NotRequired[float]


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

    # Optional fields (including backwards compatibility)
    cached_tokens: NotRequired[
        int
    ]  # Number of cached prompt tokens (for prompt caching)
    prompt_tokens_details: NotRequired[
        dict[str, Any]
    ]  # Full OpenAI prompt token details
    # Backwards compatibility for old telemetry files
    status: NotRequired[Literal["accepted", "rejected_over", "rejected_under", "error"]]
    rejection_reason: NotRequired[str | None]

    # v1 compatibility - removed in v2
    is_retry: NotRequired[bool]
    timestamp: NotRequired[float]


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

    # v1 compatibility fields
    node_type: NotRequired[Literal["leaf", "summary"]]
    level: NotRequired[int]  # renamed to height in v2


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


# Configuration dictionary for v3.1+
class ConfigDict(TypedDict):
    """Type definition for configuration object in v3.1+ formats."""

    target_chunk_tokens: int
    summary_model: str
    embedding_model: str
    # Additional config fields may be present
    budget_tokens: NotRequired[int]
    leaf_tokens: NotRequired[int]  # Legacy alias for target_chunk_tokens


# Top-level telemetry structure for v1.0/v2.0
class TelemetryDataDictV2(TypedDict):
    """Type definition for v1.0/v2.0 telemetry data structure."""

    format_version: str
    documents: dict[str, DocumentDict]


# Top-level telemetry structure for v3.0/v3.1
class TelemetryDataDictV3(TypedDict):
    """Type definition for v3.0/v3.1 telemetry data structure (flat).

    Note: v3.0 format may have legacy top-level chunk_size and models fields,
    but v3.1+ stores this information in the config object.
    """

    format_version: str
    document_id: str
    source_document_tokens: int
    indexed_at: float
    config: ConfigDict
    nodes: list[NodeTelemetryDict]
    # Optional document path
    document_path: NotRequired[str]

    # Legacy fields for v3.0 backward compatibility (removed in current implementation)
    chunk_size: NotRequired[int]
    models: NotRequired[dict[str, str]]


# Top-level telemetry structure for v4.1 (current)
class TelemetryDataDictV4(TypedDict):
    """Type definition for v4.1 telemetry data structure (current format).

    This is the current telemetry format with full model metadata,
    system prompts, and runtime information for reproducibility.
    """

    format_version: str
    document_id: str
    source_document_tokens: int
    indexed_at: float
    config: ConfigDict
    model_metadata: dict[str, Any]
    system_prompts: dict[str, str]
    runtime_info: dict[str, Any]
    nodes: list[NodeTelemetryDict]
    # Optional document path
    document_path: NotRequired[str]


# Union type for all supported telemetry formats
# For parse_telemetry_format, we normalize to the input format
TelemetryDataDict = TelemetryDataDictV3 | TelemetryDataDictV4


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
