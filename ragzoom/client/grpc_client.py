"""Thin client wrapper around the RagZoom gRPC services."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import cast

import grpc

from ragzoom.constants import (
    DEFAULT_GRPC_ADDRESS,
    DEFAULT_GRPC_STREAM_TIMEOUT,
    DEFAULT_GRPC_TIMEOUT,
)
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


def _extract_telemetry_and_error(
    response: object,
) -> tuple[TelemetryDataDict | None, str | None]:
    """Extract telemetry data and error message from a gRPC response."""
    telemetry_json = getattr(response, "telemetry_json", "")
    telemetry = _decode_telemetry(telemetry_json)
    error_message = getattr(response, "error", "") or ""
    return telemetry, error_message or None


def _document_stats_to_result(
    stats: pb2.DocumentStats,
    *,
    telemetry_run_id: str | None = None,
    span_start: int = 0,
    span_end: int = 0,
) -> IndexingResult:
    telemetry = _decode_telemetry(stats.telemetry_json)
    return IndexingResult(
        document_id=stats.document_id,
        chunks_created=stats.chunks_created,
        tree_depth=stats.tree_depth,
        span_start=span_start,
        span_end=span_end,
        mutated_nodes=stats.mutated_nodes,
        resummarized_nodes=stats.resummarized_nodes,
        new_leaves=stats.new_leaves,
        telemetry=telemetry,
        telemetry_run_id=telemetry_run_id,
    )


def _map_rpc_error(error: object) -> RuntimeError:
    status = getattr(error, "code", lambda: None)()
    details = getattr(error, "details", lambda: "")() or ""
    name = getattr(status, "name", None)
    message = f"gRPC {name}: {details}" if name else details
    return RuntimeError(message)


@dataclass
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


@dataclass
class RetrievalView:
    selected_ids: list[str]
    tiling_ids: list[str]
    scores: dict[str, float]
    coverage_map: dict[str, bool]
    nodes: dict[str, NodeSummary]


@dataclass
class QueryProfileResult:
    """Profiling telemetry from query execution."""

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


@dataclass
class ExecuteQueryOutput:
    query_result: QueryResult
    retrieval: RetrievalView
    visualization: str
    validation_warning: str
    profile: QueryProfileResult | None = None


@dataclass
class DocumentProgressSnapshot:
    pending: int
    inflight: int
    completed: int
    total: int


@dataclass
class WorkerRunSnapshot:
    message: str
    idle: bool
    queue_depth: int
    inflight: int
    documents: dict[str, DocumentProgressSnapshot]


@dataclass
class ClearedDocumentResult:
    document_id: str
    deleted_nodes: int
    document_existed: bool


@dataclass
class TelemetryFetchResult:
    complete: bool
    telemetry: TelemetryDataDict | None
    error: str | None


@dataclass
class TelemetryExportResult:
    telemetry: TelemetryDataDict | None
    error: str | None


@dataclass
class TruncateResult:
    """Result from truncating a document at a span position."""

    document_id: str
    deleted_node_ids: list[str]
    span_start: int


@dataclass
class DocumentStatusView:
    document_id: str
    leaf_count: int
    tree_depth: int
    has_pending_work: bool


class GrpcRagzoomClient:
    """Synchronous convenience wrapper for the RagZoom gRPC services."""

    def __init__(
        self,
        address: str = DEFAULT_GRPC_ADDRESS,
        *,
        timeout: float | None = None,
        stream_timeout: float | None = DEFAULT_GRPC_STREAM_TIMEOUT,
    ) -> None:
        self._address = address
        self._timeout = DEFAULT_GRPC_TIMEOUT if timeout is None else timeout
        self._stream_timeout = stream_timeout
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
        replace_existing: bool,
    ) -> IndexingResult:
        request = pb2.AppendTextRequest(
            document_id=document_id,
            content=content,
            collect_telemetry=collect_telemetry,
        )
        setattr(request, "replace_existing", replace_existing)
        try:
            response = self._indexer.AppendText(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error
        telemetry_run_id = getattr(response, "telemetry_run_id", "") or None
        return _document_stats_to_result(
            response.stats,
            telemetry_run_id=telemetry_run_id,
            span_start=response.span_start,
            span_end=response.span_end,
        )

    # jscpd:ignore-start - Parallel structure to append_text intentional (batch vs single)
    def batch_append_text(
        self,
        *,
        document_id: str,
        units: list[str],
        collect_telemetry: bool = False,
    ) -> IndexingResult:
        """Append multiple text units with forced split boundaries between them.

        Each unit in the batch creates a forced split boundary, meaning text is
        never merged across unit boundaries. This is semantically equivalent to
        calling append_text() for each unit sequentially, but executed in a single
        transaction for efficiency.

        Args:
            document_id: The document to append to
            units: List of text units, each creating a forced boundary
            collect_telemetry: Whether to collect telemetry data

        Returns:
            IndexingResult with combined stats for all appended units
        """
        # Encode each unit as bytes
        encoded_units = [u.encode("utf-8") for u in units]
        request = pb2.BatchAppendTextRequest(
            document_id=document_id,
            units=encoded_units,
            collect_telemetry=collect_telemetry,
        )
        try:
            response = self._indexer.BatchAppendText(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error
        telemetry_run_id = getattr(response, "telemetry_run_id", "") or None
        return _document_stats_to_result(
            response.stats,
            telemetry_run_id=telemetry_run_id,
            span_start=response.span_start,
            span_end=response.span_end,
        )

    # jscpd:ignore-end

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
        recent_verbatim_token_budget: int | None = None,
        profile: bool = False,
        span_start: int = 0,
        span_end: int | None = None,
    ) -> ExecuteQueryOutput:
        request = pb2.ExecuteQueryRequest(
            query=query,
            document_id=document_id,
            budget_tokens=budget_tokens or 0,
            num_seeds=num_seeds if num_seeds is not None else -1,
            embedding_model=embedding_model or "",
            debug=debug,
            viz_width=viz_width,
            use_token_coords=use_token_coords,
            recent_verbatim_token_budget=recent_verbatim_token_budget or 0,
            profile=profile,
            span_start=span_start,
        )
        if span_end is not None:
            request.span_end = span_end
        try:
            response = self._retrieval.ExecuteQuery(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        query_result = QueryResult(
            summary=response.summary,
            token_count=response.token_count,
            nodes_retrieved=response.nodes_retrieved,
            tiling_size=response.tiling_size,
            query_id=response.query_id,
            seed_count=response.seed_count,
            verbatim_count=response.verbatim_count,
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

        # Extract profile telemetry if available
        profile_result: QueryProfileResult | None = None
        t = response.telemetry
        # Check if telemetry has data (total_ms > 0 indicates profiling was enabled)
        if t.total_ms > 0:
            profile_result = QueryProfileResult(
                embedding_ms=t.embedding_ms,
                search_ms=t.search_ms,
                mmr_ms=t.mmr_ms,
                coverage_map_ms=t.coverage_map_ms,
                scoring_ms=t.scoring_ms,
                tiling_ms=t.tiling_ms,
                assembly_ms=t.assembly_ms,
                total_ms=t.total_ms,
                seeds_requested=t.seeds_requested,
                seeds_found=t.seeds_found,
                candidates_retrieved=t.candidates_retrieved,
                candidates_filtered=t.candidates_filtered,
                coverage_size=t.coverage_size,
                tiling_size=t.tiling_size,
                output_tokens=t.output_tokens,
                embedding_model=t.embedding_model,
            )

        return ExecuteQueryOutput(
            query_result=query_result,
            retrieval=retrieval_view,
            visualization=response.visualization,
            validation_warning=response.validation_warning,
            profile=profile_result,
        )

    def run_workers_once(self) -> list[WorkerRunSnapshot]:
        return list(self.iter_worker_snapshots())

    def iter_worker_snapshots(self) -> Iterator[WorkerRunSnapshot]:
        request = pb2.RunWorkersRequest(mode=pb2.WORKER_RUN_MODE_UNTIL_IDLE)
        try:
            responses = self._workers.RunWorkers(request, timeout=self._stream_timeout)
            for resp in responses:
                documents: dict[str, DocumentProgressSnapshot] = {}
                for progress in resp.documents:
                    doc_id = progress.document_id
                    if not doc_id:
                        continue
                    pending = progress.pending
                    inflight = progress.inflight
                    completed = getattr(progress, "completed", 0)
                    total = getattr(progress, "total", 0)
                    if total <= 0:
                        total = pending + inflight + completed
                    documents[doc_id] = DocumentProgressSnapshot(
                        pending=pending,
                        inflight=inflight,
                        completed=completed,
                        total=total,
                    )
                yield WorkerRunSnapshot(
                    message=resp.message,
                    idle=resp.idle,
                    queue_depth=resp.queue_depth,
                    inflight=resp.inflight,
                    documents=documents,
                )
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

    def get_document_status(self, document_id: str) -> DocumentStatusView:
        request = pb2.GetDocumentRequest(document_id=document_id)
        try:
            response = self._workers.GetDocument(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover - network failures
            raise _map_rpc_error(error) from error

        status = getattr(response, "status", None)
        if status is None or not getattr(status, "document_id", ""):
            raise RuntimeError("Worker service returned empty document status")

        return DocumentStatusView(
            document_id=status.document_id,
            leaf_count=status.leaf_count,
            tree_depth=status.tree_depth,
            has_pending_work=status.has_pending_work,
        )

    def clear_document(self, document_id: str) -> ClearedDocumentResult:
        results = self.clear_documents(document_ids=[document_id])
        if results:
            return results[0]
        return ClearedDocumentResult(
            document_id=document_id, deleted_nodes=0, document_existed=False
        )

    def clear_all_documents(self) -> list[ClearedDocumentResult]:
        return self.clear_documents(clear_all=True)

    def clear_documents(
        self,
        document_ids: list[str] | None = None,
        *,
        clear_all: bool = False,
    ) -> list[ClearedDocumentResult]:
        request_type = getattr(pb2, "ClearDocumentRequest")
        request = request_type(
            document_ids=document_ids or [],
            clear_all=clear_all,
        )
        try:
            clear_rpc = getattr(self._workers, "ClearDocument")
            response = clear_rpc(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        results: list[ClearedDocumentResult] = []
        for result in getattr(response, "results", []):
            results.append(
                ClearedDocumentResult(
                    document_id=getattr(result, "document_id", ""),
                    deleted_nodes=int(getattr(result, "deleted_nodes", 0)),
                    document_existed=bool(getattr(result, "document_existed", False)),
                )
            )
        return results

    def get_telemetry(
        self,
        *,
        document_id: str,
        run_id: str,
        wait: bool = False,
    ) -> TelemetryFetchResult:
        request_cls = getattr(pb2, "GetTelemetryRequest")
        request = request_cls(
            document_id=document_id,
            run_id=run_id,
            wait=wait,
        )
        try:
            get_telemetry = getattr(self._workers, "GetTelemetry")
            # Use stream timeout for this unary RPC because fidelity computation
            # requires embedding parent summaries on-demand, which can take
            # significant time for large trees (many API calls to embedding service)
            response = get_telemetry(request, timeout=self._stream_timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        telemetry, error_msg = _extract_telemetry_and_error(response)
        return TelemetryFetchResult(
            complete=bool(getattr(response, "complete", False)),
            telemetry=telemetry,
            error=error_msg,
        )

    def export_document_telemetry(
        self,
        *,
        document_id: str,
    ) -> TelemetryExportResult:
        request = pb2.ExportTelemetryRequest(document_id=document_id)
        try:
            export_rpc = getattr(self._workers, "ExportTelemetry")
            # Use stream timeout because fidelity computation requires embedding
            # parent summaries on-demand, which can take significant time for
            # large trees (many API calls to embedding service)
            response = export_rpc(request, timeout=self._stream_timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        telemetry, error_msg = _extract_telemetry_and_error(response)
        return TelemetryExportResult(
            telemetry=telemetry,
            error=error_msg,
        )

    def truncate_document(
        self,
        *,
        document_id: str,
        span_start: int,
    ) -> TruncateResult:
        """Truncate a document by deleting all nodes at or after span_start.

        Args:
            document_id: The document to truncate
            span_start: Delete all nodes with span_start >= this value

        Returns:
            TruncateResult with deleted node IDs and final span position
        """
        request = pb2.TruncateDocumentRequest(
            document_id=document_id,
            span_start=span_start,
        )
        try:
            response = self._indexer.TruncateDocument(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        return TruncateResult(
            document_id=response.document_id,
            deleted_node_ids=list(response.deleted_node_ids),
            span_start=response.span_start,
        )
