"""Thin client wrapper around the RagZoom gRPC services."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from ragzoom.wrapper import AppendUnit


def _decode_telemetry(payload: str) -> TelemetryDataDict | None:
    if not payload:
        return None
    data = json.loads(payload)
    return cast(TelemetryDataDict, data)


def _build_timestamp_proto(timestamp: str | tuple[str, str]) -> pb2.Timestamp:
    """Build a Timestamp proto from a string or (start, end) tuple.

    Args:
        timestamp: ISO 8601 string (used for both start and end) or tuple of (start, end).

    Returns:
        Timestamp proto message.
    """
    if isinstance(timestamp, tuple):
        time_start, time_end = timestamp
        return pb2.Timestamp(time_start=time_start, time_end=time_end)
    # Single string: time_start only, server will use it for both
    return pb2.Timestamp(time_start=timestamp)


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
    time_start: str | None = None
    time_end: str | None = None


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
class TruncateFromTimeResult:
    """Result from time-based truncation of a temporal document.

    Time-based truncation removes all nodes where time_end > cutoff_time.
    This is the temporal analog of span-based truncation (TruncateResult).

    Attributes:
        document_id: Document that was truncated.
        deleted_node_ids: IDs of nodes that were removed.
        cutoff_time: The cutoff timestamp used (echoed from request).
    """

    document_id: str
    deleted_node_ids: list[str]
    cutoff_time: str


@dataclass
class SessionCursor:
    """Cursor position for session ingestion."""

    byte_offset: int


@dataclass
class SessionIngestResult:
    """Result from ingesting a session delta."""

    new_byte_offset: int
    messages_processed: int
    truncated: bool
    truncate_span: int


@dataclass
class CompactionBoundaryResult:
    """Result from getting the compaction boundary for a session."""

    has_boundary: bool
    span_end: int


@dataclass
class DocumentWorkStatus:
    """Work queue status for a document (tree depth, pending work)."""

    document_id: str
    leaf_count: int
    tree_depth: int
    has_pending_work: bool


@dataclass
class DocumentInfoView:
    """Document info for list_documents responses.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods

    Attributes:
        document_id: Document identifier.
        leaf_count: Number of leaf nodes.
        node_count: Total nodes (leaves + inner summary nodes).
        is_temporal: Whether the document has temporal metadata.
        time_start: Earliest content timestamp, None if non-temporal or empty.
        time_end: Latest content timestamp, None if non-temporal or empty.
        completion_pct: Indexing completion (node_count / forest_size * 100).
    """

    document_id: str
    leaf_count: int
    node_count: int
    is_temporal: bool
    time_start: str | None = None
    time_end: str | None = None
    completion_pct: float | None = None


@dataclass
class SystemStatusView:
    """System-wide status aggregated across all documents.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods

    Attributes:
        total_nodes: Total node count across all documents.
        leaf_nodes: Leaf node count across all documents.
        tree_depth: Maximum tree depth across all documents.
    """

    total_nodes: int
    leaf_nodes: int
    tree_depth: int


@dataclass
class CostStatsView:
    """Cost statistics for a single document.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods

    Attributes:
        document_id: Document identifier.
        total_cost: Sum of all node costs (arbitrary units).
        total_nodes: Total node count.
        leaf_nodes: Leaf node count.
        summary_nodes: Summary node count (total - leaf).
    """

    document_id: str
    total_cost: float
    total_nodes: int
    leaf_nodes: int
    summary_nodes: int


@dataclass
class DocumentStatusView:
    """Document status with completion metrics for stateless sync workflows.

    Provides document completeness and temporal range information for sync algorithms,
    unlike DocumentWorkStatus which focuses on work queue state.

    Attributes:
        document_id: Document identifier.
        exists: Whether the document has any nodes.
        is_temporal: Whether the document has temporal metadata.
        leaf_count: Number of leaf nodes.
        node_count: Total nodes (leaves + inner summary nodes).
        complete_forest_size: Expected nodes when fully indexed: 2N - popcount(N).
        completion_pct: Indexing completion (node_count / forest_size * 100).
        time_start: Earliest content timestamp, None if non-temporal or empty.
        time_end: Latest content timestamp, None if non-temporal or empty.
    """

    document_id: str
    exists: bool
    is_temporal: bool
    leaf_count: int
    node_count: int
    complete_forest_size: int
    completion_pct: float
    time_start: str | None = None
    time_end: str | None = None


@dataclass
class SearchResultView:
    """Result from the agentic search endpoint."""

    answer: str


class GrpcRagzoomClient:
    """Synchronous convenience wrapper for the RagZoom gRPC services."""

    def __init__(
        self,
        address: str = DEFAULT_GRPC_ADDRESS,
        *,
        timeout: float | None = None,
        stream_timeout: float | None = DEFAULT_GRPC_STREAM_TIMEOUT,
        secure: bool | None = None,
    ) -> None:
        self._address = address
        self._timeout = DEFAULT_GRPC_TIMEOUT if timeout is None else timeout
        self._stream_timeout = stream_timeout

        # Auto-detect TLS: use secure channel for port 443 or explicit secure=True
        if secure is None:
            secure = address.endswith(":443")

        if secure:
            self._channel = grpc.secure_channel(address, grpc.ssl_channel_credentials())
        else:
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
        self._session: pb2_grpc.SessionIngestionServiceStub = (
            pb2_grpc.SessionIngestionServiceStub(self._channel)
        )
        self._search: pb2_grpc.SearchServiceStub = pb2_grpc.SearchServiceStub(
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
        timestamp: str | tuple[str, str] | None = None,
        summarization_guidance: str | None = None,
    ) -> IndexingResult:
        request = pb2.AppendTextRequest(
            document_id=document_id,
            content=content,
            collect_telemetry=collect_telemetry,
            replace_existing=replace_existing,
        )
        if timestamp is not None:
            ts_proto = _build_timestamp_proto(timestamp)
            request.timestamp.time_start = ts_proto.time_start
            if ts_proto.HasField("time_end"):
                request.timestamp.time_end = ts_proto.time_end
        if summarization_guidance is not None:
            request.summarization_guidance = summarization_guidance
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
        timestamps: list[str | tuple[str, str]] | None = None,
        summarization_guidance: str | None = None,
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
            timestamps: Optional list of timestamps parallel to units.
                Each entry can be a single ISO 8601 string or a (start, end) tuple.
            summarization_guidance: Optional guidance for summary generation,
                appended to the default prompt.

        Returns:
            IndexingResult with combined stats for all appended units
        """
        # Build AppendUnit protos with bundled content and optional timestamps
        append_units: list[pb2.AppendUnit] = []
        for i, text in enumerate(units):
            content = text.encode("utf-8")

            if timestamps is None or i >= len(timestamps):
                append_units.append(pb2.AppendUnit(content=content))
                continue

            ts = timestamps[i]
            if isinstance(ts, tuple):
                time_start, time_end = ts
            else:
                # Single timestamp: use for both start and end
                time_start = time_end = ts

            append_units.append(
                pb2.AppendUnit(
                    content=content, time_start=time_start, time_end=time_end
                )
            )

        request = pb2.BatchAppendTextRequest(
            document_id=document_id,
            units=append_units,
            collect_telemetry=collect_telemetry,
        )
        if summarization_guidance is not None:
            request.summarization_guidance = summarization_guidance
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

    def batch_append(
        self,
        document_id: str,
        units: list[AppendUnit],
        summarization_guidance: str | None = None,
    ) -> IndexingResult:
        """Append AppendUnit objects with forced split boundaries.

        Convenience wrapper matching the TranscriptSyncClient protocol.
        Extracts text and timestamps from AppendUnit objects and delegates
        to batch_append_text().
        """
        text_units: list[str] = []
        timestamps: list[str | tuple[str, str]] = []
        has_timestamps = False

        for unit in units:
            text_units.append(unit.text)
            if unit.time_start is not None and unit.time_end is not None:
                timestamps.append((unit.time_start, unit.time_end))
                has_timestamps = True
            else:
                timestamps.append(("", ""))

        return self.batch_append_text(
            document_id=document_id,
            units=text_units,
            timestamps=timestamps if has_timestamps else None,
            summarization_guidance=summarization_guidance,
        )

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
        time_start: str | None = None,
        time_end: str | None = None,
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
        if time_start is not None:
            request.time_start = time_start
        if time_end is not None:
            request.time_end = time_end
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
            # Extract optional temporal fields (None if not set in proto)
            time_start = node.time_start if node.HasField("time_start") else None
            time_end = node.time_end if node.HasField("time_end") else None
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
                time_start=time_start,
                time_end=time_end,
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

    def get_document_work_status(self, document_id: str) -> DocumentWorkStatus:
        request = pb2.GetDocumentRequest(document_id=document_id)
        try:
            response = self._workers.GetDocument(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover - network failures
            raise _map_rpc_error(error) from error

        status = getattr(response, "status", None)
        if status is None or not getattr(status, "document_id", ""):
            raise RuntimeError("Worker service returned empty document status")

        return DocumentWorkStatus(
            document_id=status.document_id,
            leaf_count=status.leaf_count,
            tree_depth=status.tree_depth,
            has_pending_work=status.has_pending_work,
        )

    def get_document_status(self, document_id: str) -> DocumentStatusView:
        """Get document status with completion metrics and temporal range.

        This method returns status information needed for stateless sync workflows,
        including whether the document exists, indexing completion percentage,
        and temporal range for documents with temporal metadata.

        Args:
            document_id: The document to get status for.

        Returns:
            DocumentStatusView with completion metrics and temporal info.
        """
        request = pb2.DocumentStatusRequest(document_id=document_id)
        try:
            response = self._workers.GetDocumentStatus(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover - network failures
            raise _map_rpc_error(error) from error

        return DocumentStatusView(
            document_id=response.document_id,
            exists=response.exists,
            is_temporal=response.is_temporal,
            leaf_count=response.leaf_count,
            node_count=response.node_count,
            complete_forest_size=response.complete_forest_size,
            completion_pct=response.completion_pct,
            time_start=response.time_start if response.HasField("time_start") else None,
            time_end=response.time_end if response.HasField("time_end") else None,
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

        Raises:
            RuntimeError: If document doesn't exist or span_start is invalid
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

    def truncate_from_time(
        self,
        *,
        document_id: str,
        cutoff_time: str,
    ) -> TruncateFromTimeResult:
        """Truncate a temporal document by deleting nodes where time_end > cutoff.

        Only works on temporal documents. Removes all nodes whose time_end exceeds
        the cutoff timestamp. Children of deleted nodes are orphaned by setting
        their parent_id to NULL.

        Args:
            document_id: The temporal document to truncate
            cutoff_time: ISO 8601 timestamp. Nodes with time_end > cutoff are deleted

        Returns:
            TruncateFromTimeResult with deleted node IDs and echoed cutoff time

        Raises:
            RuntimeError: If document doesn't exist, is not temporal, or cutoff is invalid
        """
        request = pb2.TruncateFromTimeRequest(
            document_id=document_id,
            cutoff_time=cutoff_time,
        )
        try:
            response = self._indexer.TruncateFromTime(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        return TruncateFromTimeResult(
            document_id=response.document_id,
            deleted_node_ids=list(response.deleted_node_ids),
            cutoff_time=response.cutoff_time,
        )

    def get_session_cursor(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> SessionCursor:
        """Get the current byte offset for a session.

        Args:
            session_id: The session to get cursor for
            user_id: User identifier for multi-tenant isolation

        Returns:
            SessionCursor with the current byte offset
        """
        request = pb2.GetSessionCursorRequest(session_id=session_id)
        metadata = [("user_id", user_id)]
        try:
            response = self._session.GetSessionCursor(
                request, timeout=self._timeout, metadata=metadata
            )
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        return SessionCursor(byte_offset=response.byte_offset)

    def ingest_session(
        self,
        *,
        session_id: str,
        user_id: str,
        jsonl_delta: bytes,
    ) -> SessionIngestResult:
        """Ingest a JSONL delta for a session.

        Args:
            session_id: The session to ingest into
            user_id: User identifier for multi-tenant isolation
            jsonl_delta: New JSONL content since the last cursor position

        Returns:
            SessionIngestResult with new cursor position and processing stats
        """
        request = pb2.IngestSessionRequest(
            session_id=session_id,
            jsonl_delta=jsonl_delta,
        )
        metadata = [("user_id", user_id)]
        try:
            response = self._session.IngestSession(
                request, timeout=self._timeout, metadata=metadata
            )
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        return SessionIngestResult(
            new_byte_offset=response.new_byte_offset,
            messages_processed=response.messages_processed,
            truncated=response.truncated,
            truncate_span=response.truncate_span,
        )

    def get_compaction_boundary(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> CompactionBoundaryResult:
        """Get the compaction boundary span_end for a session.

        The compaction boundary is the span_end just before post-compaction content.
        This allows queries to be limited to pre-compaction history only.

        Args:
            session_id: The session to get boundary for
            user_id: User identifier for multi-tenant isolation

        Returns:
            CompactionBoundaryResult with has_boundary=True and span_end if
            compaction has occurred, or has_boundary=False otherwise.
        """
        request = pb2.GetCompactionBoundaryRequest(session_id=session_id)
        metadata = [("user_id", user_id)]
        try:
            response = self._session.GetCompactionBoundary(
                request, timeout=self._timeout, metadata=metadata
            )
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        return CompactionBoundaryResult(
            has_boundary=response.has_boundary,
            span_end=response.span_end,
        )

    def reset_session_cursor(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> tuple[bool, str]:
        """Reset a session's cursor to force full re-sync.

        Args:
            session_id: The session to reset
            user_id: User identifier for multi-tenant isolation

        Returns:
            Tuple of (success, message)
        """
        request = pb2.ResetSessionCursorRequest(session_id=session_id)
        metadata = [("user_id", user_id)]
        try:
            response = self._session.ResetSessionCursor(
                request, timeout=self._timeout, metadata=metadata
            )
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        return (response.success, response.message)

    def list_documents(self) -> list[DocumentInfoView]:
        """List all indexed documents with metadata.

        Spec: specs/grpc-cli-architecture.md § New gRPC Methods

        Returns:
            List of DocumentInfoView with document metadata.
        """
        request = pb2.ListDocumentsRequest()
        try:
            list_docs_rpc = getattr(self._workers, "ListDocuments")
            response = list_docs_rpc(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        documents: list[DocumentInfoView] = []
        for doc in getattr(response, "documents", []):
            documents.append(
                DocumentInfoView(
                    document_id=doc.document_id,
                    leaf_count=doc.leaf_count,
                    node_count=doc.node_count,
                    is_temporal=doc.is_temporal,
                    time_start=doc.time_start if doc.HasField("time_start") else None,
                    time_end=doc.time_end if doc.HasField("time_end") else None,
                    completion_pct=(
                        doc.completion_pct if doc.HasField("completion_pct") else None
                    ),
                )
            )
        return documents

    def get_system_status(self) -> SystemStatusView:
        """Get system-wide status aggregated across all documents.

        Spec: specs/grpc-cli-architecture.md § New gRPC Methods

        Returns:
            SystemStatusView with total_nodes, leaf_nodes, tree_depth.
        """
        request = pb2.GetSystemStatusRequest()
        try:
            get_status_rpc = getattr(self._workers, "GetSystemStatus")
            response = get_status_rpc(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        return SystemStatusView(
            total_nodes=response.total_nodes,
            leaf_nodes=response.leaf_nodes,
            tree_depth=response.tree_depth,
        )

    def get_cost_stats(self, document_id: str | None = None) -> list[CostStatsView]:
        """Get cost statistics for documents.

        Spec: specs/grpc-cli-architecture.md § New gRPC Methods

        Args:
            document_id: If provided, returns stats for only this document.
                If omitted, returns stats for all documents.

        Returns:
            List of CostStatsView with cost statistics per document.
        """
        request = pb2.GetCostStatsRequest(document_id=document_id or "")
        try:
            get_cost_rpc = getattr(self._workers, "GetCostStats")
            response = get_cost_rpc(request, timeout=self._timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        stats_list: list[CostStatsView] = []
        for doc in getattr(response, "documents", []):
            stats_list.append(
                CostStatsView(
                    document_id=doc.document_id,
                    total_cost=doc.total_cost,
                    total_nodes=doc.total_nodes,
                    leaf_nodes=doc.leaf_nodes,
                    summary_nodes=doc.summary_nodes,
                )
            )
        return stats_list

    def search(
        self,
        *,
        question: str,
        document_id: str,
    ) -> SearchResultView:
        """Run agentic search: question in, answer out.

        Args:
            question: The question to answer.
            document_id: Document to search within.

        Returns:
            SearchResultView with the answer.
        """
        request = pb2.SearchRequest(
            question=question,
            document_id=document_id,
        )
        try:
            response = self._search.Search(request, timeout=self._stream_timeout)
        except grpc.RpcError as error:  # pragma: no cover
            raise _map_rpc_error(error) from error

        return SearchResultView(answer=response.answer)
