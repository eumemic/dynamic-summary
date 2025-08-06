"""Type definitions for telemetry data structures.

This module provides TypedDict definitions for all telemetry-related data structures,
enabling type-safe access to telemetry data and preventing bugs from accessing
non-existent fields.

Supports v1.0, v2.0, and v3.0 telemetry formats with appropriate NotRequired fields.
"""

from typing import Literal, TypedDict

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
    input_text_tokens: int
    prompt_tokens: int
    completion_tokens: int
    actual_tokens: int
    status: Literal["accepted", "rejected_over", "rejected_under", "error"]
    model: str
    start_time: float
    end_time: float

    # Optional fields
    rejection_reason: NotRequired[str | None]
    prompt_hash: NotRequired[str | None]
    cached_tokens: NotRequired[
        int
    ]  # Number of cached prompt tokens (for prompt caching)
    prompt_tokens_details: NotRequired[dict]  # Full OpenAI prompt token details

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
    embedding: NotRequired[EmbeddingTelemetryDict]
    summary_attempts: NotRequired[list[SummaryAttemptDict]]

    # v1 compatibility fields
    node_type: NotRequired[Literal["leaf", "summary"]]
    level: NotRequired[int]  # renamed to height in v2
    span: NotRequired[list[int]]  # removed in v2


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


# Models configuration (v3.0)
class ModelsDict(TypedDict):
    """Type definition for models configuration in v3.0."""

    summary: str
    embedding: str


# Top-level telemetry structure for v1.0/v2.0
class TelemetryDataDictV2(TypedDict):
    """Type definition for v1.0/v2.0 telemetry data structure."""

    format_version: str
    documents: dict[str, DocumentDict]


# Top-level telemetry structure for v3.0
class TelemetryDataDictV3(TypedDict):
    """Type definition for v3.0 telemetry data structure (flat)."""

    format_version: str
    document_id: str
    source_document_tokens: int
    chunk_size: int
    indexed_at: float
    models: ModelsDict
    nodes: list[NodeTelemetryDict]


# Union type for all telemetry formats
# For parse_telemetry_format, we always return v3.0 format
TelemetryDataDict = TelemetryDataDictV3


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
