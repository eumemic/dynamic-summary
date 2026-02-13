from __future__ import annotations

# ruff: noqa
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class WorkerRunMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKER_RUN_MODE_UNSPECIFIED: _ClassVar[WorkerRunMode]
    WORKER_RUN_MODE_UNTIL_IDLE: _ClassVar[WorkerRunMode]
    WORKER_RUN_MODE_CONTINUOUS: _ClassVar[WorkerRunMode]

WORKER_RUN_MODE_UNSPECIFIED: WorkerRunMode
WORKER_RUN_MODE_UNTIL_IDLE: WorkerRunMode
WORKER_RUN_MODE_CONTINUOUS: WorkerRunMode

class Timestamp(_message.Message):
    __slots__ = ("time_start", "time_end")
    TIME_START_FIELD_NUMBER: _ClassVar[int]
    TIME_END_FIELD_NUMBER: _ClassVar[int]
    time_start: str
    time_end: str
    def __init__(
        self, time_start: str | None = ..., time_end: str | None = ...
    ) -> None: ...

class AppendUnit(_message.Message):
    __slots__ = ("content", "time_start", "time_end")
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TIME_START_FIELD_NUMBER: _ClassVar[int]
    TIME_END_FIELD_NUMBER: _ClassVar[int]
    content: bytes
    time_start: str
    time_end: str
    def __init__(
        self,
        content: bytes | None = ...,
        time_start: str | None = ...,
        time_end: str | None = ...,
    ) -> None: ...

class DocumentStats(_message.Message):
    __slots__ = (
        "document_id",
        "chunks_created",
        "mutated_nodes",
        "resummarized_nodes",
        "new_leaves",
        "total_leaves",
        "telemetry_json",
        "tree_depth",
    )
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_CREATED_FIELD_NUMBER: _ClassVar[int]
    MUTATED_NODES_FIELD_NUMBER: _ClassVar[int]
    RESUMMARIZED_NODES_FIELD_NUMBER: _ClassVar[int]
    NEW_LEAVES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_LEAVES_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_JSON_FIELD_NUMBER: _ClassVar[int]
    TREE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    chunks_created: int
    mutated_nodes: int
    resummarized_nodes: int
    new_leaves: int
    total_leaves: int
    telemetry_json: str
    tree_depth: int
    def __init__(
        self,
        document_id: str | None = ...,
        chunks_created: int | None = ...,
        mutated_nodes: int | None = ...,
        resummarized_nodes: int | None = ...,
        new_leaves: int | None = ...,
        total_leaves: int | None = ...,
        telemetry_json: str | None = ...,
        tree_depth: int | None = ...,
    ) -> None: ...

class IndexDocumentRequest(_message.Message):
    __slots__ = ("document_id", "content", "file_path", "collect_telemetry")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    COLLECT_TELEMETRY_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    content: bytes
    file_path: str
    collect_telemetry: bool
    def __init__(
        self,
        document_id: str | None = ...,
        content: bytes | None = ...,
        file_path: str | None = ...,
        collect_telemetry: bool = ...,
    ) -> None: ...

class IndexDocumentResponse(_message.Message):
    __slots__ = ("stats",)
    STATS_FIELD_NUMBER: _ClassVar[int]
    stats: DocumentStats
    def __init__(self, stats: DocumentStats | _Mapping | None = ...) -> None: ...

class AppendTextRequest(_message.Message):
    __slots__ = (
        "document_id",
        "content",
        "collect_telemetry",
        "replace_existing",
        "timestamp",
        "summarization_guidance",
    )
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    COLLECT_TELEMETRY_FIELD_NUMBER: _ClassVar[int]
    REPLACE_EXISTING_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SUMMARIZATION_GUIDANCE_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    content: bytes
    collect_telemetry: bool
    replace_existing: bool
    timestamp: Timestamp
    summarization_guidance: str
    def __init__(
        self,
        document_id: str | None = ...,
        content: bytes | None = ...,
        collect_telemetry: bool = ...,
        replace_existing: bool = ...,
        timestamp: Timestamp | _Mapping | None = ...,
        summarization_guidance: str | None = ...,
    ) -> None: ...

class AppendTextResponse(_message.Message):
    __slots__ = ("stats", "telemetry_run_id", "span_start", "span_end")
    STATS_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SPAN_START_FIELD_NUMBER: _ClassVar[int]
    SPAN_END_FIELD_NUMBER: _ClassVar[int]
    stats: DocumentStats
    telemetry_run_id: str
    span_start: int
    span_end: int
    def __init__(
        self,
        stats: DocumentStats | _Mapping | None = ...,
        telemetry_run_id: str | None = ...,
        span_start: int | None = ...,
        span_end: int | None = ...,
    ) -> None: ...

class BatchAppendTextRequest(_message.Message):
    __slots__ = ("document_id", "units", "collect_telemetry", "summarization_guidance")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    UNITS_FIELD_NUMBER: _ClassVar[int]
    COLLECT_TELEMETRY_FIELD_NUMBER: _ClassVar[int]
    SUMMARIZATION_GUIDANCE_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    units: _containers.RepeatedCompositeFieldContainer[AppendUnit]
    collect_telemetry: bool
    summarization_guidance: str
    def __init__(
        self,
        document_id: str | None = ...,
        units: _Iterable[AppendUnit | _Mapping] | None = ...,
        collect_telemetry: bool = ...,
        summarization_guidance: str | None = ...,
    ) -> None: ...

class BatchAppendTextResponse(_message.Message):
    __slots__ = ("stats", "telemetry_run_id", "span_start", "span_end")
    STATS_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SPAN_START_FIELD_NUMBER: _ClassVar[int]
    SPAN_END_FIELD_NUMBER: _ClassVar[int]
    stats: DocumentStats
    telemetry_run_id: str
    span_start: int
    span_end: int
    def __init__(
        self,
        stats: DocumentStats | _Mapping | None = ...,
        telemetry_run_id: str | None = ...,
        span_start: int | None = ...,
        span_end: int | None = ...,
    ) -> None: ...

class RetrieveRequest(_message.Message):
    __slots__ = ("query", "document_id", "budget_tokens", "num_seeds")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    BUDGET_TOKENS_FIELD_NUMBER: _ClassVar[int]
    NUM_SEEDS_FIELD_NUMBER: _ClassVar[int]
    query: str
    document_id: str
    budget_tokens: int
    num_seeds: int
    def __init__(
        self,
        query: str | None = ...,
        document_id: str | None = ...,
        budget_tokens: int | None = ...,
        num_seeds: int | None = ...,
    ) -> None: ...

class Node(_message.Message):
    __slots__ = (
        "node_id",
        "text",
        "token_count",
        "span_start",
        "span_end",
        "parent_id",
        "left_child_id",
        "right_child_id",
        "height",
        "time_start",
        "time_end",
    )
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    SPAN_START_FIELD_NUMBER: _ClassVar[int]
    SPAN_END_FIELD_NUMBER: _ClassVar[int]
    PARENT_ID_FIELD_NUMBER: _ClassVar[int]
    LEFT_CHILD_ID_FIELD_NUMBER: _ClassVar[int]
    RIGHT_CHILD_ID_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    TIME_START_FIELD_NUMBER: _ClassVar[int]
    TIME_END_FIELD_NUMBER: _ClassVar[int]
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
        node_id: str | None = ...,
        text: str | None = ...,
        token_count: int | None = ...,
        span_start: int | None = ...,
        span_end: int | None = ...,
        parent_id: str | None = ...,
        left_child_id: str | None = ...,
        right_child_id: str | None = ...,
        height: int | None = ...,
        time_start: str | None = ...,
        time_end: str | None = ...,
    ) -> None: ...

class RetrieveResponse(_message.Message):
    __slots__ = ("selected_ids", "tiling_ids", "scores", "coverage_map", "nodes")

    class ScoresEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: str | None = ..., value: float | None = ...
        ) -> None: ...

    class CoverageMapEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bool
        def __init__(self, key: str | None = ..., value: bool = ...) -> None: ...

    class NodesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Node
        def __init__(
            self,
            key: str | None = ...,
            value: Node | _Mapping | None = ...,
        ) -> None: ...

    SELECTED_IDS_FIELD_NUMBER: _ClassVar[int]
    TILING_IDS_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    COVERAGE_MAP_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    selected_ids: _containers.RepeatedScalarFieldContainer[str]
    tiling_ids: _containers.RepeatedScalarFieldContainer[str]
    scores: _containers.ScalarMap[str, float]
    coverage_map: _containers.ScalarMap[str, bool]
    nodes: _containers.MessageMap[str, Node]
    def __init__(
        self,
        selected_ids: _Iterable[str] | None = ...,
        tiling_ids: _Iterable[str] | None = ...,
        scores: _Mapping[str, float] | None = ...,
        coverage_map: _Mapping[str, bool] | None = ...,
        nodes: _Mapping[str, Node] | None = ...,
    ) -> None: ...

class ExecuteQueryRequest(_message.Message):
    __slots__ = (
        "query",
        "document_id",
        "budget_tokens",
        "num_seeds",
        "embedding_model",
        "debug",
        "viz_width",
        "use_token_coords",
        "tiling_strategy",
        "recent_verbatim_token_budget",
        "profile",
        "span_start",
        "span_end",
        "time_start",
        "time_end",
    )
    QUERY_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    BUDGET_TOKENS_FIELD_NUMBER: _ClassVar[int]
    NUM_SEEDS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_MODEL_FIELD_NUMBER: _ClassVar[int]
    DEBUG_FIELD_NUMBER: _ClassVar[int]
    VIZ_WIDTH_FIELD_NUMBER: _ClassVar[int]
    USE_TOKEN_COORDS_FIELD_NUMBER: _ClassVar[int]
    TILING_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    RECENT_VERBATIM_TOKEN_BUDGET_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    SPAN_START_FIELD_NUMBER: _ClassVar[int]
    SPAN_END_FIELD_NUMBER: _ClassVar[int]
    TIME_START_FIELD_NUMBER: _ClassVar[int]
    TIME_END_FIELD_NUMBER: _ClassVar[int]
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
        query: str | None = ...,
        document_id: str | None = ...,
        budget_tokens: int | None = ...,
        num_seeds: int | None = ...,
        embedding_model: str | None = ...,
        debug: bool = ...,
        viz_width: int | None = ...,
        use_token_coords: bool = ...,
        tiling_strategy: str | None = ...,
        recent_verbatim_token_budget: int | None = ...,
        profile: bool = ...,
        span_start: int | None = ...,
        span_end: int | None = ...,
        time_start: str | None = ...,
        time_end: str | None = ...,
    ) -> None: ...

class QueryTelemetry(_message.Message):
    __slots__ = (
        "embedding_ms",
        "search_ms",
        "mmr_ms",
        "coverage_map_ms",
        "scoring_ms",
        "tiling_ms",
        "assembly_ms",
        "total_ms",
        "seeds_requested",
        "seeds_found",
        "candidates_retrieved",
        "candidates_filtered",
        "coverage_size",
        "tiling_size",
        "output_tokens",
        "embedding_model",
    )
    EMBEDDING_MS_FIELD_NUMBER: _ClassVar[int]
    SEARCH_MS_FIELD_NUMBER: _ClassVar[int]
    MMR_MS_FIELD_NUMBER: _ClassVar[int]
    COVERAGE_MAP_MS_FIELD_NUMBER: _ClassVar[int]
    SCORING_MS_FIELD_NUMBER: _ClassVar[int]
    TILING_MS_FIELD_NUMBER: _ClassVar[int]
    ASSEMBLY_MS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MS_FIELD_NUMBER: _ClassVar[int]
    SEEDS_REQUESTED_FIELD_NUMBER: _ClassVar[int]
    SEEDS_FOUND_FIELD_NUMBER: _ClassVar[int]
    CANDIDATES_RETRIEVED_FIELD_NUMBER: _ClassVar[int]
    CANDIDATES_FILTERED_FIELD_NUMBER: _ClassVar[int]
    COVERAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    TILING_SIZE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_MODEL_FIELD_NUMBER: _ClassVar[int]
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
        embedding_ms: float | None = ...,
        search_ms: float | None = ...,
        mmr_ms: float | None = ...,
        coverage_map_ms: float | None = ...,
        scoring_ms: float | None = ...,
        tiling_ms: float | None = ...,
        assembly_ms: float | None = ...,
        total_ms: float | None = ...,
        seeds_requested: int | None = ...,
        seeds_found: int | None = ...,
        candidates_retrieved: int | None = ...,
        candidates_filtered: int | None = ...,
        coverage_size: int | None = ...,
        tiling_size: int | None = ...,
        output_tokens: int | None = ...,
        embedding_model: str | None = ...,
    ) -> None: ...

class ExecuteQueryResponse(_message.Message):
    __slots__ = (
        "summary",
        "token_count",
        "nodes_retrieved",
        "tiling_size",
        "retrieval",
        "visualization",
        "validation_warning",
        "query_id",
        "seed_count",
        "verbatim_count",
        "telemetry",
        "actual_start",
        "actual_end",
    )
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    NODES_RETRIEVED_FIELD_NUMBER: _ClassVar[int]
    TILING_SIZE_FIELD_NUMBER: _ClassVar[int]
    RETRIEVAL_FIELD_NUMBER: _ClassVar[int]
    VISUALIZATION_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_WARNING_FIELD_NUMBER: _ClassVar[int]
    QUERY_ID_FIELD_NUMBER: _ClassVar[int]
    SEED_COUNT_FIELD_NUMBER: _ClassVar[int]
    VERBATIM_COUNT_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_START_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_END_FIELD_NUMBER: _ClassVar[int]
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
        summary: str | None = ...,
        token_count: int | None = ...,
        nodes_retrieved: int | None = ...,
        tiling_size: int | None = ...,
        retrieval: RetrieveResponse | _Mapping | None = ...,
        visualization: str | None = ...,
        validation_warning: str | None = ...,
        query_id: str | None = ...,
        seed_count: int | None = ...,
        verbatim_count: int | None = ...,
        telemetry: QueryTelemetry | _Mapping | None = ...,
        actual_start: int | None = ...,
        actual_end: int | None = ...,
    ) -> None: ...

class RunWorkersRequest(_message.Message):
    __slots__ = ("mode",)
    MODE_FIELD_NUMBER: _ClassVar[int]
    mode: WorkerRunMode
    def __init__(self, mode: WorkerRunMode | str | None = ...) -> None: ...

class WorkerDocumentProgress(_message.Message):
    __slots__ = ("document_id", "pending", "inflight", "completed", "total")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    PENDING_FIELD_NUMBER: _ClassVar[int]
    INFLIGHT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    pending: int
    inflight: int
    completed: int
    total: int
    def __init__(
        self,
        document_id: str | None = ...,
        pending: int | None = ...,
        inflight: int | None = ...,
        completed: int | None = ...,
        total: int | None = ...,
    ) -> None: ...

class RunWorkersResponse(_message.Message):
    __slots__ = ("message", "idle", "queue_depth", "inflight", "documents")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    IDLE_FIELD_NUMBER: _ClassVar[int]
    QUEUE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    INFLIGHT_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    message: str
    idle: bool
    queue_depth: int
    inflight: int
    documents: _containers.RepeatedCompositeFieldContainer[WorkerDocumentProgress]
    def __init__(
        self,
        message: str | None = ...,
        idle: bool = ...,
        queue_depth: int | None = ...,
        inflight: int | None = ...,
        documents: _Iterable[WorkerDocumentProgress | _Mapping] | None = ...,
    ) -> None: ...

class GetDocumentRequest(_message.Message):
    __slots__ = ("document_id",)
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    def __init__(self, document_id: str | None = ...) -> None: ...

class DocumentStatus(_message.Message):
    __slots__ = (
        "document_id",
        "leaf_count",
        "has_pending_work",
        "tree_depth",
        "is_temporal",
    )
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    LEAF_COUNT_FIELD_NUMBER: _ClassVar[int]
    HAS_PENDING_WORK_FIELD_NUMBER: _ClassVar[int]
    TREE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    IS_TEMPORAL_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    leaf_count: int
    has_pending_work: bool
    tree_depth: int
    is_temporal: bool
    def __init__(
        self,
        document_id: str | None = ...,
        leaf_count: int | None = ...,
        has_pending_work: bool = ...,
        tree_depth: int | None = ...,
        is_temporal: bool = ...,
    ) -> None: ...

class GetDocumentResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: DocumentStatus
    def __init__(self, status: DocumentStatus | _Mapping | None = ...) -> None: ...

class GetTelemetryRequest(_message.Message):
    __slots__ = ("document_id", "run_id", "wait")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    WAIT_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    run_id: str
    wait: bool
    def __init__(
        self,
        document_id: str | None = ...,
        run_id: str | None = ...,
        wait: bool = ...,
    ) -> None: ...

class GetTelemetryResponse(_message.Message):
    __slots__ = ("complete", "telemetry_json", "error")
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    complete: bool
    telemetry_json: str
    error: str
    def __init__(
        self,
        complete: bool = ...,
        telemetry_json: str | None = ...,
        error: str | None = ...,
    ) -> None: ...

class ClearDocumentRequest(_message.Message):
    __slots__ = ("document_ids", "clear_all")
    DOCUMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    CLEAR_ALL_FIELD_NUMBER: _ClassVar[int]
    document_ids: _containers.RepeatedScalarFieldContainer[str]
    clear_all: bool
    def __init__(
        self, document_ids: _Iterable[str] | None = ..., clear_all: bool = ...
    ) -> None: ...

class ClearDocumentResult(_message.Message):
    __slots__ = ("document_id", "deleted_nodes", "document_existed")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    DELETED_NODES_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_EXISTED_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    deleted_nodes: int
    document_existed: bool
    def __init__(
        self,
        document_id: str | None = ...,
        deleted_nodes: int | None = ...,
        document_existed: bool = ...,
    ) -> None: ...

class ClearDocumentResponse(_message.Message):
    __slots__ = ("results",)
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[ClearDocumentResult]
    def __init__(
        self, results: _Iterable[ClearDocumentResult | _Mapping] | None = ...
    ) -> None: ...

class TruncateDocumentRequest(_message.Message):
    __slots__ = ("document_id", "span_start")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SPAN_START_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    span_start: int
    def __init__(
        self, document_id: str | None = ..., span_start: int | None = ...
    ) -> None: ...

class TruncateDocumentResponse(_message.Message):
    __slots__ = ("document_id", "deleted_node_ids", "span_start")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    DELETED_NODE_IDS_FIELD_NUMBER: _ClassVar[int]
    SPAN_START_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    deleted_node_ids: _containers.RepeatedScalarFieldContainer[str]
    span_start: int
    def __init__(
        self,
        document_id: str | None = ...,
        deleted_node_ids: _Iterable[str] | None = ...,
        span_start: int | None = ...,
    ) -> None: ...

class DocumentStatusRequest(_message.Message):
    __slots__ = ("document_id",)
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    def __init__(self, document_id: str | None = ...) -> None: ...

class DocumentStatusResponse(_message.Message):
    __slots__ = (
        "document_id",
        "exists",
        "is_temporal",
        "leaf_count",
        "node_count",
        "complete_forest_size",
        "completion_pct",
        "time_start",
        "time_end",
    )
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    EXISTS_FIELD_NUMBER: _ClassVar[int]
    IS_TEMPORAL_FIELD_NUMBER: _ClassVar[int]
    LEAF_COUNT_FIELD_NUMBER: _ClassVar[int]
    NODE_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FOREST_SIZE_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_PCT_FIELD_NUMBER: _ClassVar[int]
    TIME_START_FIELD_NUMBER: _ClassVar[int]
    TIME_END_FIELD_NUMBER: _ClassVar[int]
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
        document_id: str | None = ...,
        exists: bool = ...,
        is_temporal: bool = ...,
        leaf_count: int | None = ...,
        node_count: int | None = ...,
        complete_forest_size: int | None = ...,
        completion_pct: float | None = ...,
        time_start: str | None = ...,
        time_end: str | None = ...,
    ) -> None: ...

class TruncateFromTimeRequest(_message.Message):
    __slots__ = ("document_id", "cutoff_time")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    CUTOFF_TIME_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    cutoff_time: str
    def __init__(
        self, document_id: str | None = ..., cutoff_time: str | None = ...
    ) -> None: ...

class TruncateFromTimeResponse(_message.Message):
    __slots__ = ("document_id", "deleted_node_ids", "cutoff_time")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    DELETED_NODE_IDS_FIELD_NUMBER: _ClassVar[int]
    CUTOFF_TIME_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    deleted_node_ids: _containers.RepeatedScalarFieldContainer[str]
    cutoff_time: str
    def __init__(
        self,
        document_id: str | None = ...,
        deleted_node_ids: _Iterable[str] | None = ...,
        cutoff_time: str | None = ...,
    ) -> None: ...

class ExportTelemetryRequest(_message.Message):
    __slots__ = ("document_id",)
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    def __init__(self, document_id: str | None = ...) -> None: ...

class ExportTelemetryResponse(_message.Message):
    __slots__ = ("telemetry_json", "error")
    TELEMETRY_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    telemetry_json: str
    error: str
    def __init__(
        self, telemetry_json: str | None = ..., error: str | None = ...
    ) -> None: ...

class ListDocumentsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListDocumentsResponse(_message.Message):
    __slots__ = ("documents",)
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedCompositeFieldContainer[DocumentInfo]
    def __init__(
        self, documents: _Iterable[DocumentInfo | _Mapping] | None = ...
    ) -> None: ...

class DocumentInfo(_message.Message):
    __slots__ = (
        "document_id",
        "leaf_count",
        "node_count",
        "is_temporal",
        "time_start",
        "time_end",
        "completion_pct",
    )
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    LEAF_COUNT_FIELD_NUMBER: _ClassVar[int]
    NODE_COUNT_FIELD_NUMBER: _ClassVar[int]
    IS_TEMPORAL_FIELD_NUMBER: _ClassVar[int]
    TIME_START_FIELD_NUMBER: _ClassVar[int]
    TIME_END_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_PCT_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    leaf_count: int
    node_count: int
    is_temporal: bool
    time_start: str
    time_end: str
    completion_pct: float
    def __init__(
        self,
        document_id: str | None = ...,
        leaf_count: int | None = ...,
        node_count: int | None = ...,
        is_temporal: bool = ...,
        time_start: str | None = ...,
        time_end: str | None = ...,
        completion_pct: float | None = ...,
    ) -> None: ...

class GetSystemStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetSystemStatusResponse(_message.Message):
    __slots__ = ("total_nodes", "leaf_nodes", "tree_depth")
    TOTAL_NODES_FIELD_NUMBER: _ClassVar[int]
    LEAF_NODES_FIELD_NUMBER: _ClassVar[int]
    TREE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    total_nodes: int
    leaf_nodes: int
    tree_depth: int
    def __init__(
        self,
        total_nodes: int | None = ...,
        leaf_nodes: int | None = ...,
        tree_depth: int | None = ...,
    ) -> None: ...

class GetCostStatsRequest(_message.Message):
    __slots__ = ("document_id",)
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    def __init__(self, document_id: str | None = ...) -> None: ...

class GetCostStatsResponse(_message.Message):
    __slots__ = ("documents",)
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedCompositeFieldContainer[DocumentCostStats]
    def __init__(
        self, documents: _Iterable[DocumentCostStats | _Mapping] | None = ...
    ) -> None: ...

class DocumentCostStats(_message.Message):
    __slots__ = (
        "document_id",
        "total_cost",
        "total_nodes",
        "leaf_nodes",
        "summary_nodes",
    )
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COST_FIELD_NUMBER: _ClassVar[int]
    TOTAL_NODES_FIELD_NUMBER: _ClassVar[int]
    LEAF_NODES_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_NODES_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    total_cost: float
    total_nodes: int
    leaf_nodes: int
    summary_nodes: int
    def __init__(
        self,
        document_id: str | None = ...,
        total_cost: float | None = ...,
        total_nodes: int | None = ...,
        leaf_nodes: int | None = ...,
        summary_nodes: int | None = ...,
    ) -> None: ...

class GetSessionCursorRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: str | None = ...) -> None: ...

class GetSessionCursorResponse(_message.Message):
    __slots__ = ("byte_offset",)
    BYTE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    byte_offset: int
    def __init__(self, byte_offset: int | None = ...) -> None: ...

class IngestSessionRequest(_message.Message):
    __slots__ = ("session_id", "jsonl_delta")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    JSONL_DELTA_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    jsonl_delta: bytes
    def __init__(
        self, session_id: str | None = ..., jsonl_delta: bytes | None = ...
    ) -> None: ...

class IngestSessionResponse(_message.Message):
    __slots__ = ("new_byte_offset", "messages_processed", "truncated", "truncate_span")
    NEW_BYTE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATE_SPAN_FIELD_NUMBER: _ClassVar[int]
    new_byte_offset: int
    messages_processed: int
    truncated: bool
    truncate_span: int
    def __init__(
        self,
        new_byte_offset: int | None = ...,
        messages_processed: int | None = ...,
        truncated: bool = ...,
        truncate_span: int | None = ...,
    ) -> None: ...

class GetCompactionBoundaryRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: str | None = ...) -> None: ...

class GetCompactionBoundaryResponse(_message.Message):
    __slots__ = ("has_boundary", "span_end")
    HAS_BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    SPAN_END_FIELD_NUMBER: _ClassVar[int]
    has_boundary: bool
    span_end: int
    def __init__(
        self, has_boundary: bool = ..., span_end: int | None = ...
    ) -> None: ...

class ResetSessionCursorRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: str | None = ...) -> None: ...

class ResetSessionCursorResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: str | None = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = (
        "question",
        "document_id",
        "session_id",
        "time_start",
        "time_end",
        "search_guidance",
    )
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TIME_START_FIELD_NUMBER: _ClassVar[int]
    TIME_END_FIELD_NUMBER: _ClassVar[int]
    SEARCH_GUIDANCE_FIELD_NUMBER: _ClassVar[int]
    question: str
    document_id: str
    session_id: str
    time_start: str
    time_end: str
    search_guidance: str
    def __init__(
        self,
        question: str | None = ...,
        document_id: str | None = ...,
        session_id: str | None = ...,
        time_start: str | None = ...,
        time_end: str | None = ...,
        search_guidance: str | None = ...,
    ) -> None: ...

class SearchIterationProto(_message.Message):
    __slots__ = (
        "query",
        "budget_tokens",
        "time_start",
        "time_end",
        "result_text",
        "result_token_count",
        "agent_reasoning",
    )
    QUERY_FIELD_NUMBER: _ClassVar[int]
    BUDGET_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TIME_START_FIELD_NUMBER: _ClassVar[int]
    TIME_END_FIELD_NUMBER: _ClassVar[int]
    RESULT_TEXT_FIELD_NUMBER: _ClassVar[int]
    RESULT_TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    AGENT_REASONING_FIELD_NUMBER: _ClassVar[int]
    query: str
    budget_tokens: int
    time_start: str
    time_end: str
    result_text: str
    result_token_count: int
    agent_reasoning: str
    def __init__(
        self,
        query: str | None = ...,
        budget_tokens: int | None = ...,
        time_start: str | None = ...,
        time_end: str | None = ...,
        result_text: str | None = ...,
        result_token_count: int | None = ...,
        agent_reasoning: str | None = ...,
    ) -> None: ...

class SearchProfileProto(_message.Message):
    __slots__ = (
        "iterations",
        "total_input_tokens",
        "total_output_tokens",
        "total_cost_usd",
        "duration_seconds",
        "retrospective",
        "transcript",
    )
    ITERATIONS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COST_USD_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    RETROSPECTIVE_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    iterations: _containers.RepeatedCompositeFieldContainer[SearchIterationProto]
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    duration_seconds: float
    retrospective: str
    transcript: str
    def __init__(
        self,
        iterations: _Iterable[SearchIterationProto | _Mapping] | None = ...,
        total_input_tokens: int | None = ...,
        total_output_tokens: int | None = ...,
        total_cost_usd: float | None = ...,
        duration_seconds: float | None = ...,
        retrospective: str | None = ...,
        transcript: str | None = ...,
    ) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("answer", "profile", "session_id")
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    answer: str
    profile: SearchProfileProto
    session_id: str
    def __init__(
        self,
        answer: str | None = ...,
        profile: SearchProfileProto | _Mapping | None = ...,
        session_id: str | None = ...,
    ) -> None: ...
