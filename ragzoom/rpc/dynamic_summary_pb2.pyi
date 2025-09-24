from __future__ import annotations

# ruff: noqa

from collections.abc import Iterable, Mapping, Sequence

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
    document_id: str
    content: bytes
    collect_telemetry: bool

    def __init__(
        self, *, document_id: str, content: bytes, collect_telemetry: bool
    ) -> None: ...

class AppendTextResponse:
    stats: DocumentStats

    def __init__(self, *, stats: DocumentStats) -> None: ...

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
    node_id: str
    text: str
    token_count: int
    span_start: int
    span_end: int
    parent_id: str
    left_child_id: str
    right_child_id: str
    height: int

    def __init__(
        self,
        *,
        node_id: str,
        text: str,
        token_count: int,
        span_start: int,
        span_end: int,
        parent_id: str,
        left_child_id: str,
        right_child_id: str,
        height: int,
    ) -> None: ...

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

    def __init__(
        self,
        *,
        query: str,
        document_id: str,
        budget_tokens: int,
        num_seeds: int,
        embedding_model: str,
        debug: bool,
        viz_width: int,
        use_token_coords: bool,
    ) -> None: ...

class ExecuteQueryResponse:
    summary: str
    token_count: int
    nodes_retrieved: int
    tiling_size: int
    retrieval: RetrieveResponse
    visualization: str
    validation_warning: str

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
    ) -> None: ...

class RunWorkersRequest:
    mode: int

    def __init__(self, *, mode: int) -> None: ...

class RunWorkersResponse:
    message: str
    idle: bool

    def __init__(self, *, message: str, idle: bool) -> None: ...

class GetDocumentRequest:
    document_id: str

    def __init__(self, *, document_id: str) -> None: ...

class DocumentStatus:
    document_id: str
    leaf_count: int
    has_pending_work: bool
    tree_depth: int

    def __init__(
        self,
        *,
        document_id: str,
        leaf_count: int,
        has_pending_work: bool,
        tree_depth: int,
    ) -> None: ...

class GetDocumentResponse:
    status: DocumentStatus

    def __init__(self, *, status: DocumentStatus) -> None: ...

WORKER_RUN_MODE_UNTIL_IDLE: int
