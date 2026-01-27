from __future__ import annotations

# ruff: noqa

from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar

from google.protobuf.descriptor import Descriptor

class Timestamp:
    """Timestamp for temporal metadata in client-controlled chunking."""

    time_start: str
    time_end: str

    def __init__(
        self,
        *,
        time_start: str = ...,
        time_end: str = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...

class AppendUnit:
    """Self-contained unit for batch_append with bundled text and timestamps."""

    DESCRIPTOR: ClassVar[Descriptor]
    content: bytes
    time_start: str
    time_end: str

    def __init__(
        self,
        *,
        content: bytes = ...,
        time_start: str = ...,
        time_end: str = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...

class DocumentStats:
    document_id: str
    chunks_created: int
    mutated_nodes: int
    resummarized_nodes: int
    new_leaves: int
    total_leaves: int
    tree_depth: int
    telemetry_json: str

    def __init__(
        self,
        *,
        document_id: str,
        chunks_created: int,
        mutated_nodes: int,
        resummarized_nodes: int,
        new_leaves: int,
        total_leaves: int,
        telemetry_json: str,
        tree_depth: int,
    ) -> None: ...

class IndexDocumentRequest:
    document_id: str
    content: bytes
    file_path: str
    collect_telemetry: bool

    def __init__(
        self,
        *,
        document_id: str = ...,
        content: bytes = ...,
        file_path: str = ...,
        collect_telemetry: bool = ...,
    ) -> None: ...
    def WhichOneof(self, name: str) -> str | None: ...

class IndexDocumentResponse:
    stats: DocumentStats

    def __init__(self, *, stats: DocumentStats) -> None: ...

class AppendTextRequest:
    DESCRIPTOR: ClassVar[Descriptor]
    document_id: str
    content: bytes
    collect_telemetry: bool
    replace_existing: bool
    timestamp: Timestamp
    summarization_guidance: str

    def __init__(
        self,
        *,
        document_id: str,
        content: bytes,
        collect_telemetry: bool = ...,
        replace_existing: bool = ...,
        timestamp: Timestamp = ...,
        summarization_guidance: str = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...

class AppendTextResponse:
    stats: DocumentStats
    telemetry_run_id: str
    span_start: int
    span_end: int

    def __init__(
        self,
        *,
        stats: DocumentStats,
        telemetry_run_id: str = ...,
        span_start: int = ...,
        span_end: int = ...,
    ) -> None: ...

class BatchAppendTextRequest:
    DESCRIPTOR: ClassVar[Descriptor]
    document_id: str
    units: Sequence[AppendUnit]
    collect_telemetry: bool
    summarization_guidance: str

    def __init__(
        self,
        *,
        document_id: str,
        units: Iterable[AppendUnit] = ...,
        collect_telemetry: bool = ...,
        summarization_guidance: str = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...

class BatchAppendTextResponse:
    stats: DocumentStats
    telemetry_run_id: str
    span_start: int
    span_end: int

    def __init__(
        self,
        *,
        stats: DocumentStats,
        telemetry_run_id: str = ...,
        span_start: int = ...,
        span_end: int = ...,
    ) -> None: ...

class RetrieveRequest:
    query: str
    document_id: str
    budget_tokens: int
    num_seeds: int

    def __init__(
        self,
        *,
        query: str = ...,
        document_id: str = ...,
        budget_tokens: int = ...,
        num_seeds: int = ...,
    ) -> None: ...

class Node:
    DESCRIPTOR: ClassVar[Descriptor]
    node_id: str
    text: str
    token_count: int
    span_start: int
    span_end: int
    parent_id: str
    left_child_id: str
    right_child_id: str
    height: int
    time_start: str
    time_end: str

    def __init__(
        self,
        *,
        node_id: str = ...,
        text: str = ...,
        token_count: int = ...,
        span_start: int = ...,
        span_end: int = ...,
        parent_id: str = ...,
        left_child_id: str = ...,
        right_child_id: str = ...,
        height: int = ...,
        time_start: str = ...,
        time_end: str = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...

class RetrieveResponse:
    selected_ids: Sequence[str]
    tiling_ids: Sequence[str]
    scores: Mapping[str, float]
    coverage_map: Mapping[str, bool]
    nodes: Mapping[str, Node]

    def __init__(
        self,
        *,
        selected_ids: Iterable[str],
        tiling_ids: Iterable[str],
        scores: Mapping[str, float],
        coverage_map: Mapping[str, bool],
        nodes: Mapping[str, Node],
    ) -> None: ...

class ExecuteQueryRequest:
    query: str
    document_id: str
    budget_tokens: int
    num_seeds: int
    embedding_model: str
    debug: bool
    viz_width: int
    use_token_coords: bool
    tiling_strategy: str
    recent_verbatim_token_budget: int
    profile: bool
    span_start: int
    span_end: int
    time_start: str
    time_end: str

    def __init__(
        self,
        *,
        query: str = ...,
        document_id: str = ...,
        budget_tokens: int = ...,
        num_seeds: int = ...,
        embedding_model: str = ...,
        debug: bool = ...,
        viz_width: int = ...,
        use_token_coords: bool = ...,
        tiling_strategy: str = ...,
        recent_verbatim_token_budget: int = ...,
        profile: bool = ...,
        span_start: int = ...,
        span_end: int = ...,
        time_start: str = ...,
        time_end: str = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...

class QueryTelemetry:
    embedding_ms: float
    search_ms: float
    mmr_ms: float
    coverage_map_ms: float
    scoring_ms: float
    tiling_ms: float
    assembly_ms: float
    total_ms: float
    seeds_requested: int
    seeds_found: int
    candidates_retrieved: int
    candidates_filtered: int
    coverage_size: int
    tiling_size: int
    output_tokens: int
    embedding_model: str

    def __init__(
        self,
        *,
        embedding_ms: float = ...,
        search_ms: float = ...,
        mmr_ms: float = ...,
        coverage_map_ms: float = ...,
        scoring_ms: float = ...,
        tiling_ms: float = ...,
        assembly_ms: float = ...,
        total_ms: float = ...,
        seeds_requested: int = ...,
        seeds_found: int = ...,
        candidates_retrieved: int = ...,
        candidates_filtered: int = ...,
        coverage_size: int = ...,
        tiling_size: int = ...,
        output_tokens: int = ...,
        embedding_model: str = ...,
    ) -> None: ...

class ExecuteQueryResponse:
    summary: str
    token_count: int
    nodes_retrieved: int
    tiling_size: int
    retrieval: RetrieveResponse
    visualization: str
    validation_warning: str
    query_id: str
    seed_count: int
    verbatim_count: int
    telemetry: QueryTelemetry
    actual_start: int
    actual_end: int

    def __init__(
        self,
        *,
        summary: str,
        token_count: int,
        nodes_retrieved: int,
        tiling_size: int,
        retrieval: RetrieveResponse,
        visualization: str,
        validation_warning: str,
        query_id: str,
        seed_count: int = ...,
        verbatim_count: int = ...,
        telemetry: QueryTelemetry = ...,
        actual_start: int = ...,
        actual_end: int = ...,
    ) -> None: ...

class RunWorkersRequest:
    mode: int

    def __init__(self, *, mode: int) -> None: ...

class WorkerDocumentProgress:
    document_id: str
    pending: int
    inflight: int
    completed: int
    total: int

    def __init__(
        self,
        *,
        document_id: str = ...,
        pending: int = ...,
        inflight: int = ...,
        completed: int = ...,
        total: int = ...,
    ) -> None: ...

class RunWorkersResponse:
    message: str
    idle: bool
    queue_depth: int
    inflight: int
    documents: Sequence[WorkerDocumentProgress]

    def __init__(
        self,
        *,
        message: str,
        idle: bool,
        queue_depth: int,
        inflight: int,
        documents: Iterable[WorkerDocumentProgress],
    ) -> None: ...

class GetDocumentRequest:
    document_id: str

    def __init__(self, *, document_id: str) -> None: ...

class DocumentStatus:
    DESCRIPTOR: ClassVar[Descriptor]
    document_id: str
    leaf_count: int
    has_pending_work: bool
    tree_depth: int
    is_temporal: bool

    def __init__(
        self,
        *,
        document_id: str = ...,
        leaf_count: int = ...,
        has_pending_work: bool = ...,
        tree_depth: int = ...,
        is_temporal: bool = ...,
    ) -> None: ...

class GetDocumentResponse:
    status: DocumentStatus

    def __init__(self, *, status: DocumentStatus) -> None: ...

class DocumentStatusRequest:
    """Request for document status with completion metrics."""

    document_id: str

    def __init__(self, *, document_id: str) -> None: ...

class DocumentStatusResponse:
    """Response with document completion metrics and temporal range."""

    DESCRIPTOR: ClassVar[Descriptor]
    document_id: str
    exists: bool
    is_temporal: bool
    leaf_count: int
    node_count: int
    complete_forest_size: int
    completion_pct: float
    time_start: str
    time_end: str

    def __init__(
        self,
        *,
        document_id: str = ...,
        exists: bool = ...,
        is_temporal: bool = ...,
        leaf_count: int = ...,
        node_count: int = ...,
        complete_forest_size: int = ...,
        completion_pct: float = ...,
        time_start: str = ...,
        time_end: str = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...

# List documents messages
class ListDocumentsRequest:
    """Request to list all indexed documents."""

    def __init__(self) -> None: ...

class DocumentInfo:
    """Information about a single indexed document."""

    DESCRIPTOR: ClassVar[Descriptor]
    document_id: str
    leaf_count: int
    node_count: int
    is_temporal: bool
    time_start: str
    time_end: str
    completion_pct: float

    def __init__(
        self,
        *,
        document_id: str = ...,
        leaf_count: int = ...,
        node_count: int = ...,
        is_temporal: bool = ...,
        time_start: str = ...,
        time_end: str = ...,
        completion_pct: float = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...

class ListDocumentsResponse:
    """Response containing list of all indexed documents."""

    documents: Sequence[DocumentInfo]

    def __init__(self, *, documents: Iterable[DocumentInfo] = ...) -> None: ...

# ValidateDocument messages
class ValidateDocumentRequest:
    """Request to validate document tree invariants."""

    DESCRIPTOR: ClassVar[Descriptor]
    document_id: str

    def __init__(self, *, document_id: str = ...) -> None: ...

class ValidateDocumentResponse:
    """Response containing validation result."""

    DESCRIPTOR: ClassVar[Descriptor]
    valid: bool
    errors: Sequence[str]

    def __init__(self, *, valid: bool = ..., errors: Iterable[str] = ...) -> None: ...


# GetSystemStatus messages
class GetSystemStatusRequest:
    """Request for system-wide status (empty message)."""

    DESCRIPTOR: ClassVar[Descriptor]

    def __init__(self) -> None: ...


class GetSystemStatusResponse:
    """Response containing aggregated system status across all documents."""

    DESCRIPTOR: ClassVar[Descriptor]
    total_nodes: int
    leaf_nodes: int
    tree_depth: int

    def __init__(
        self, *, total_nodes: int = ..., leaf_nodes: int = ..., tree_depth: int = ...
    ) -> None: ...


WORKER_RUN_MODE_UNTIL_IDLE: int

class GetTelemetryRequest:
    document_id: str
    run_id: str
    wait: bool

    def __init__(self, *, document_id: str, run_id: str, wait: bool = ...) -> None: ...

class GetTelemetryResponse:
    complete: bool
    telemetry_json: str
    error: str

    def __init__(
        self,
        *,
        complete: bool = ...,
        telemetry_json: str = ...,
        error: str = ...,
    ) -> None: ...

class ClearDocumentRequest:
    document_ids: Sequence[str]
    clear_all: bool

    def __init__(
        self, *, document_ids: Iterable[str] = ..., clear_all: bool = ...
    ) -> None: ...

class ClearDocumentResult:
    document_id: str
    deleted_nodes: int
    document_existed: bool

    def __init__(
        self,
        *,
        document_id: str = ...,
        deleted_nodes: int = ...,
        document_existed: bool = ...,
    ) -> None: ...

class ClearDocumentResponse:
    results: Sequence[ClearDocumentResult]

    def __init__(self, *, results: Iterable[ClearDocumentResult]) -> None: ...

class TruncateDocumentRequest:
    document_id: str
    span_start: int

    def __init__(self, *, document_id: str, span_start: int) -> None: ...

class TruncateDocumentResponse:
    document_id: str
    deleted_node_ids: Sequence[str]
    span_start: int

    def __init__(
        self,
        *,
        document_id: str,
        deleted_node_ids: Iterable[str],
        span_start: int,
    ) -> None: ...

class TruncateFromTimeRequest:
    """Request for time-based truncation of temporal documents."""

    document_id: str
    cutoff_time: str  # ISO 8601 timestamp

    def __init__(self, *, document_id: str, cutoff_time: str) -> None: ...

class TruncateFromTimeResponse:
    """Response from time-based truncation."""

    document_id: str
    deleted_node_ids: Sequence[str]
    cutoff_time: str

    def __init__(
        self,
        *,
        document_id: str,
        deleted_node_ids: Iterable[str],
        cutoff_time: str,
    ) -> None: ...

class ExportTelemetryRequest:
    document_id: str

    def __init__(self, *, document_id: str) -> None: ...

class ExportTelemetryResponse:
    telemetry_json: str
    error: str

    def __init__(
        self,
        *,
        telemetry_json: str = ...,
        error: str = ...,
    ) -> None: ...

# Session ingestion messages
class GetSessionCursorRequest:
    session_id: str

    def __init__(self, *, session_id: str) -> None: ...

class GetSessionCursorResponse:
    byte_offset: int

    def __init__(self, *, byte_offset: int = ...) -> None: ...

class IngestSessionRequest:
    session_id: str
    jsonl_delta: bytes

    def __init__(self, *, session_id: str, jsonl_delta: bytes = ...) -> None: ...

class IngestSessionResponse:
    new_byte_offset: int
    messages_processed: int
    truncated: bool
    truncate_span: int

    def __init__(
        self,
        *,
        new_byte_offset: int = ...,
        messages_processed: int = ...,
        truncated: bool = ...,
        truncate_span: int = ...,
    ) -> None: ...

class GetCompactionBoundaryRequest:
    session_id: str

    def __init__(self, *, session_id: str) -> None: ...

class GetCompactionBoundaryResponse:
    has_boundary: bool
    span_end: int

    def __init__(self, *, has_boundary: bool = ..., span_end: int = ...) -> None: ...

class ResetSessionCursorRequest:
    session_id: str

    def __init__(self, *, session_id: str) -> None: ...

class ResetSessionCursorResponse:
    success: bool
    message: str

    def __init__(self, *, success: bool = ..., message: str = ...) -> None: ...
