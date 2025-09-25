"""Thin client wrapper around the RagZoom gRPC services."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import cast

import grpc

from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.services.query_service import QueryResult
from ragzoom.telemetry_types import TelemetryDataDict


def _decode_telemetry(payload: str) -> TelemetryDataDict | None:
    if not payload:
        return None
    data = json.loads(payload)
    return cast(TelemetryDataDict, data)


def _document_stats_to_result(stats: pb2.DocumentStats) -> IndexingResult:
    telemetry = _decode_telemetry(stats.telemetry_json)
    return IndexingResult(
        document_id=stats.document_id,
        chunks_created=stats.chunks_created,
        tree_depth=stats.tree_depth,
        mutated_nodes=stats.mutated_nodes,
        resummarized_nodes=stats.resummarized_nodes,
        new_leaves=stats.new_leaves,
        telemetry=telemetry,
    )


def _map_rpc_error(error: object) -> RuntimeError:
    status = getattr(error, "code", lambda: None)()
    details = getattr(error, "details", lambda: "")() or ""
    name = getattr(status, "name", None)
    message = f"gRPC {name}: {details}" if name else details
    return RuntimeError(message)


@dataclass(slots=True)
class NodeSummary:
    node_id: str
    text: str
    token_count: int
    span_start: int
    span_end: int
    parent_id: str
    left_child_id: str
    right_child_id: str
    height: int


@dataclass(slots=True)
class RetrievalView:
    selected_ids: list[str]
    tiling_ids: list[str]
    scores: dict[str, float]
    coverage_map: dict[str, bool]
    nodes: dict[str, NodeSummary]


@dataclass(slots=True)
class ExecuteQueryOutput:
    query_result: QueryResult
    retrieval: RetrievalView
    visualization: str
    validation_warning: str


@dataclass(slots=True)
class WorkerRunSnapshot:
    message: str
    idle: bool
    queue_depth: int
    inflight: int
    documents: dict[str, tuple[int, int]]


class GrpcRagzoomClient:
    """Synchronous convenience wrapper for the RagZoom gRPC services."""

    def __init__(self, address: str, *, timeout: float | None = None) -> None:
        self._address = address
        self._timeout = timeout
        self._channel = grpc.insecure_channel(address)
        self._indexer: pb2_grpc.IndexerServiceStub = pb2_grpc.IndexerServiceStub(
            self._channel
        )
        self._retrieval: pb2_grpc.RetrievalServiceStub = pb2_grpc.RetrievalServiceStub(
            self._channel
        )
        self._workers: pb2_grpc.WorkerServiceStub = pb2_grpc.WorkerServiceStub(
            self._channel
        )

    def __enter__(self) -> GrpcRagzoomClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._channel.close()

    def index_document(
        self,
        *,
        document_id: str | None,
        content: bytes,
        collect_telemetry: bool,
    ) -> IndexingResult:
        request = pb2.IndexDocumentRequest(
            document_id=document_id or "",
            content=content,
            collect_telemetry=collect_telemetry,
        )
        try:
            response = self._indexer.IndexDocument(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover - network failures
            raise _map_rpc_error(error) from error
        return _document_stats_to_result(response.stats)

    def append_text(
        self,
        *,
        document_id: str,
        content: bytes,
        collect_telemetry: bool,
    ) -> IndexingResult:
        request = pb2.AppendTextRequest(
            document_id=document_id,
            content=content,
            collect_telemetry=collect_telemetry,
        )
        try:
            response = self._indexer.AppendText(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error
        return _document_stats_to_result(response.stats)

    def execute_query(
        self,
        *,
        query: str,
        document_id: str,
        budget_tokens: int | None,
        num_seeds: int | None,
        embedding_model: str | None,
        debug: bool,
        viz_width: int,
        use_token_coords: bool,
    ) -> ExecuteQueryOutput:
        request = pb2.ExecuteQueryRequest(
            query=query,
            document_id=document_id,
            budget_tokens=budget_tokens or 0,
            num_seeds=num_seeds or 0,
            embedding_model=embedding_model or "",
            debug=debug,
            viz_width=viz_width,
            use_token_coords=use_token_coords,
        )
        try:
            response = self._retrieval.ExecuteQuery(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        query_result = QueryResult(
            summary=response.summary,
            token_count=response.token_count,
            nodes_retrieved=response.nodes_retrieved,
            tiling_size=response.tiling_size,
        )

        nodes_payload: dict[str, NodeSummary] = {}
        for node_id, node in response.retrieval.nodes.items():
            nodes_payload[node_id] = NodeSummary(
                node_id=node.node_id,
                text=node.text,
                token_count=node.token_count,
                span_start=node.span_start,
                span_end=node.span_end,
                parent_id=node.parent_id,
                left_child_id=node.left_child_id,
                right_child_id=node.right_child_id,
                height=node.height,
            )

        retrieval_view = RetrievalView(
            selected_ids=list(response.retrieval.selected_ids),
            tiling_ids=list(response.retrieval.tiling_ids),
            scores=dict(response.retrieval.scores),
            coverage_map=dict(response.retrieval.coverage_map),
            nodes=nodes_payload,
        )

        return ExecuteQueryOutput(
            query_result=query_result,
            retrieval=retrieval_view,
            visualization=response.visualization,
            validation_warning=response.validation_warning,
        )

    def run_workers_once(self) -> list[WorkerRunSnapshot]:
        return list(self.iter_worker_snapshots())

    def iter_worker_snapshots(self) -> Iterator[WorkerRunSnapshot]:
        request = pb2.RunWorkersRequest(mode=pb2.WORKER_RUN_MODE_UNTIL_IDLE)
        try:
            responses = self._workers.RunWorkers(request, timeout=self._timeout)
            for resp in responses:
                documents = {
                    progress.document_id: (progress.pending, progress.inflight)
                    for progress in resp.documents
                    if progress.document_id
                }
                yield WorkerRunSnapshot(
                    message=resp.message,
                    idle=resp.idle,
                    queue_depth=resp.queue_depth,
                    inflight=resp.inflight,
                    documents=documents,
                )
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error
