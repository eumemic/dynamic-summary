"""gRPC servicer implementations for RagZoom."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol, TypeVar, cast

import grpc

from ragzoom.assemble import Assembler
from ragzoom.document_store import DocumentStore
from ragzoom.progress_display import DocumentProgressTotals, WorkerProgressDisplay
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc
from ragzoom.server.indexing_engine import IndexingEngine, IndexingStatus
from ragzoom.server.query_executor import build_retriever, build_server_query_executor
from ragzoom.server.state import ServerState
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.telemetry_embeddings import annotate_telemetry_fidelity
from ragzoom.telemetry_export import (
    TelemetryExportError,
    export_document_telemetry,
)
from ragzoom.tree_viz import build_ascii_tree
from ragzoom.validate import validate_tiling

if TYPE_CHECKING:
    from typing import Protocol

    from grpc import StatusCode as GrpcStatusCode
    from grpc.aio import Server as GrpcServer

    class TelemetryRequestProto(Protocol):
        document_id: str
        run_id: str
        wait: bool

    class TelemetryResponseProto(Protocol):
        complete: bool
        telemetry_json: str
        error: str

    class ExportTelemetryRequestProto(Protocol):
        document_id: str

    class ExportTelemetryResponseProto(Protocol):
        telemetry_json: str
        error: str

else:  # pragma: no cover - typing aid only
    GrpcStatusCode = object  # type: ignore[assignment]
    GrpcServer = object  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_UNSPECIFIED_WORKER_MODE = 0


def complete_forest_size(leaf_count: int) -> int:
    """Calculate expected total nodes when a binary forest is fully indexed.

    RagZoom builds a forest of perfect binary trees over N leaves.
    The binary representation of N determines the forest structure:
    - N decomposes into popcount(N) perfect binary trees
    - Each tree with 2^k leaves has 2^k - 1 inner nodes
    - Total inner nodes = N - popcount(N)
    - Total nodes = N + (N - popcount(N)) = 2N - popcount(N)

    Args:
        leaf_count: Number of leaf nodes in the document.

    Returns:
        Expected total node count (leaves + inner nodes) for a complete forest.

    Examples:
        >>> complete_forest_size(8)   # 0b1000, popcount=1 → 15
        15
        >>> complete_forest_size(7)   # 0b111, popcount=3 → 11
        11
        >>> complete_forest_size(100) # 0b1100100, popcount=3 → 197
        197
    """
    if leaf_count <= 0:
        return 0
    popcount = bin(leaf_count).count("1")
    return 2 * leaf_count - popcount


_UNTIL_IDLE_WORKER_MODE = getattr(pb2, "WORKER_RUN_MODE_UNTIL_IDLE", 1)
_CONTINUOUS_WORKER_MODE = 2


class ServicerContextProto(Protocol):
    async def abort(self, code: object, details: str) -> NoReturn: ...


class NodeLike(Protocol):
    text: str | None
    token_count: int | None
    span_start: int | None
    span_end: int | None
    parent_id: str | None
    left_child_id: str | None
    right_child_id: str | None
    height: int | None


class GrpcServerProto(Protocol):
    def add_insecure_port(self, address: str) -> int: ...

    def start(self) -> Awaitable[object]: ...

    def stop(self, grace: object | None = ...) -> Awaitable[object]: ...

    def wait_for_termination(self) -> Awaitable[object]: ...


def _stats_to_proto(stats: IndexingResult) -> pb2.DocumentStats:
    if (
        stats.mutated_nodes is None
        or stats.resummarized_nodes is None
        or stats.new_leaves is None
    ):
        raise ValueError("IndexingResult is missing required mutation metadata")

    telemetry_json = "" if stats.telemetry is None else json.dumps(stats.telemetry)
    return pb2.DocumentStats(
        document_id=stats.document_id,
        chunks_created=stats.chunks_created,
        mutated_nodes=stats.mutated_nodes,
        resummarized_nodes=stats.resummarized_nodes,
        new_leaves=stats.new_leaves,
        total_leaves=stats.chunks_created,
        telemetry_json=telemetry_json,
        tree_depth=stats.tree_depth,
    )


async def _abort(
    context: ServicerContextProto, *, code: object, message: str
) -> NoReturn:
    await context.abort(code, message)
    raise AssertionError("context.abort should raise")


async def _decode_text(data: bytes, context: ServicerContextProto) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover
        await _abort(
            context,
            code=grpc.StatusCode.INVALID_ARGUMENT,
            message=f"Invalid UTF-8 payload: {exc}",
        )


def _extract_timestamp(
    ts_proto: pb2.Timestamp,
) -> str | tuple[str, str]:
    """Extract timestamp from proto message.

    Returns:
        Either a single ISO 8601 string (if time_end == time_start or not set),
        or a tuple of (time_start, time_end) strings.
    """
    time_start = ts_proto.time_start
    # When time_end is not set (HasField returns False), use time_start
    if not ts_proto.HasField("time_end") or ts_proto.time_end == "":
        return time_start
    return (time_start, ts_proto.time_end)


T = TypeVar("T")


def _unix_to_iso8601(ts: float) -> str:
    """Convert Unix timestamp (float seconds) to ISO 8601 string with Z suffix.

    Args:
        ts: Unix timestamp as float seconds since epoch.

    Returns:
        ISO 8601 formatted string with Z timezone (e.g., "2024-01-21T14:00:00Z").
    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    # Format as ISO 8601 with Z suffix (replace +00:00 with Z for consistency)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _require(value: T | None, *, field: str, node_id: str) -> T:
    if value is None:
        raise ValueError(f"Node '{node_id}' is missing required field `{field}`")
    return value


def _retrieval_to_proto(retrieval_result: Retrievable) -> pb2.RetrieveResponse:
    tiling_ids: Sequence[str]
    if retrieval_result.tiling is None:
        tiling_ids = []
    else:
        tiling_ids = list(retrieval_result.tiling)

    nodes: dict[str, pb2.Node] = {}
    for node_id, node in (retrieval_result.nodes or {}).items():
        text = str(_require(node.text, field="text", node_id=node_id))
        token_count = int(
            _require(node.token_count, field="token_count", node_id=node_id)
        )
        span_start = int(_require(node.span_start, field="span_start", node_id=node_id))
        span_end = int(_require(node.span_end, field="span_end", node_id=node_id))
        height = int(_require(node.height, field="height", node_id=node_id))

        # Populate temporal fields if present (Unix timestamp → ISO 8601)
        time_start_attr = getattr(node, "time_start", None)
        time_end_attr = getattr(node, "time_end", None)
        time_start_iso = (
            _unix_to_iso8601(time_start_attr) if time_start_attr is not None else None
        )
        time_end_iso = (
            _unix_to_iso8601(time_end_attr) if time_end_attr is not None else None
        )

        node_proto = pb2.Node(
            node_id=node_id,
            text=text,
            token_count=token_count,
            span_start=span_start,
            span_end=span_end,
            parent_id=node.parent_id or "",
            left_child_id=node.left_child_id or "",
            right_child_id=node.right_child_id or "",
            height=height,
        )
        if time_start_iso is not None:
            node_proto.time_start = time_start_iso
        if time_end_iso is not None:
            node_proto.time_end = time_end_iso

        nodes[node_id] = node_proto

    return pb2.RetrieveResponse(
        selected_ids=list(retrieval_result.node_ids),
        tiling_ids=tiling_ids,
        scores=dict(retrieval_result.scores),
        coverage_map=dict(retrieval_result.coverage_map or {}),
        nodes=nodes,
    )


class Retrievable:
    node_ids: Sequence[str]
    tiling: Sequence[str] | None
    scores: dict[str, float]
    coverage_map: dict[str, bool] | None
    nodes: dict[str, NodeLike] | None


class IndexerServicer(pb2_grpc.IndexerServiceServicer):
    def __init__(self, state: ServerState) -> None:
        self._state = state
        self._runtime = state.index_runtime

    async def IndexDocument(  # noqa: N802
        self,
        request: pb2.IndexDocumentRequest,
        context: ServicerContextProto,
    ) -> pb2.IndexDocumentResponse:
        source = request.WhichOneof("source")
        if source is None:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="IndexDocument requires either `content` or `file_path`.",
            )

        document_id = request.document_id or None
        file_path: str | None = None

        if source == "content":
            text = await _decode_text(request.content, context)
        else:
            file_path = request.file_path
            try:
                text = Path(file_path).read_text(encoding="utf-8")
            except OSError as exc:
                await _abort(
                    context,
                    code=grpc.StatusCode.INVALID_ARGUMENT,
                    message=f"Failed to read file '{file_path}': {exc}",
                )

        resolved_document_id = document_id
        if not resolved_document_id:
            if file_path:
                resolved_document_id = Path(file_path).name
            else:
                await _abort(
                    context,
                    code=grpc.StatusCode.INVALID_ARGUMENT,
                    message="IndexDocument requires `document_id` when no file path is provided.",
                )

        session = self._runtime.get_session(resolved_document_id, file_path=file_path)
        result = await session.append_text(
            text,
            replace_existing=True,
            collect_telemetry=request.collect_telemetry,
        )

        return pb2.IndexDocumentResponse(stats=_stats_to_proto(result))

    async def AppendText(  # noqa: N802
        self,
        request: pb2.AppendTextRequest,
        context: ServicerContextProto,
    ) -> pb2.AppendTextResponse:
        if not request.document_id:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="AppendText requires `document_id`.",
            )

        text = await _decode_text(request.content, context)

        # Extract timestamp from proto if present
        timestamp: str | tuple[str, str] | None = None
        if request.HasField("timestamp"):
            timestamp = _extract_timestamp(request.timestamp)

        # Extract custom guidance if present (see specs/custom-prompt-config.md)
        summarization_guidance = (
            request.summarization_guidance
            if request.HasField("summarization_guidance")
            else None
        )

        session = self._runtime.get_session(request.document_id)
        result = await session.append_text(
            text,
            replace_existing=bool(getattr(request, "replace_existing", False)),
            collect_telemetry=request.collect_telemetry,
            timestamp=timestamp,
            summarization_guidance=summarization_guidance,
        )

        response = pb2.AppendTextResponse(
            stats=_stats_to_proto(result),
            span_start=result.span_start,
            span_end=result.span_end,
        )
        setattr(response, "telemetry_run_id", result.telemetry_run_id or "")
        return response

    async def BatchAppendText(  # noqa: N802
        self,
        request: pb2.BatchAppendTextRequest,
        context: ServicerContextProto,
    ) -> pb2.BatchAppendTextResponse:
        if not request.document_id:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="BatchAppendText requires `document_id`.",
            )

        # Extract content and timestamps from AppendUnit protos
        units: list[str] = []
        timestamps: list[str | tuple[str, str]] = []
        has_any_timestamp = False

        for i, append_unit in enumerate(request.units):
            # Decode content from bytes to string
            try:
                units.append(append_unit.content.decode("utf-8"))
            except UnicodeDecodeError as exc:
                await _abort(
                    context,
                    code=grpc.StatusCode.INVALID_ARGUMENT,
                    message=f"Invalid UTF-8 in unit {i}: {exc}",
                )

            # Extract timestamps
            has_start = append_unit.HasField("time_start")
            has_end = append_unit.HasField("time_end")

            if has_end and not has_start:
                # Only time_end without time_start is invalid
                await _abort(
                    context,
                    code=grpc.StatusCode.INVALID_ARGUMENT,
                    message=f"Unit {i}: time_end provided without time_start",
                )

            if has_start:
                has_any_timestamp = True
                if has_end:
                    timestamps.append((append_unit.time_start, append_unit.time_end))
                else:
                    # Only time_start provided: use for both
                    timestamps.append(append_unit.time_start)
            else:
                # No timestamps on this unit
                timestamps.append(None)  # type: ignore[arg-type]

        summarization_guidance = (
            request.summarization_guidance
            if request.HasField("summarization_guidance")
            else None
        )

        session = self._runtime.get_session(request.document_id)
        result = await session.batch_append_text(
            units,
            collect_telemetry=request.collect_telemetry,
            timestamps=timestamps if has_any_timestamp else None,
            summarization_guidance=summarization_guidance,
        )

        response = pb2.BatchAppendTextResponse(
            stats=_stats_to_proto(result),
            span_start=result.span_start,
            span_end=result.span_end,
        )
        setattr(response, "telemetry_run_id", result.telemetry_run_id or "")
        return response

    async def TruncateDocument(  # noqa: N802
        self,
        request: pb2.TruncateDocumentRequest,
        context: ServicerContextProto,
    ) -> pb2.TruncateDocumentResponse:
        if not request.document_id:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="TruncateDocument requires `document_id`.",
            )

        session = self._runtime.get_session(request.document_id)
        result = await session.truncate_from_span(request.span_start)

        return pb2.TruncateDocumentResponse(
            document_id=result.document_id,
            deleted_node_ids=result.deleted_node_ids,
            span_start=result.span_start,
        )

    async def TruncateFromTime(  # noqa: N802
        self,
        request: object,
        context: ServicerContextProto,
    ) -> object:
        """Truncate a temporal document from a given cutoff time.

        Removes all nodes where time_end > cutoff_time, including both
        leaf nodes and inner (summary) nodes. Vectors for deleted nodes
        are also removed from the vector index.

        See specs/temporal-document-apis.md for full specification.
        """
        document_id = getattr(request, "document_id", "")
        cutoff_time_str = getattr(request, "cutoff_time", "")

        if not document_id:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="TruncateFromTime requires `document_id`.",
            )

        if not cutoff_time_str:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="TruncateFromTime requires `cutoff_time`.",
            )

        # Parse and validate the ISO 8601 timestamp
        try:
            cutoff_dt = datetime.fromisoformat(cutoff_time_str.replace("Z", "+00:00"))
            cutoff_unix = cutoff_dt.timestamp()
        except ValueError:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message=f"Invalid ISO 8601 timestamp: {cutoff_time_str}",
            )

        # Check if document exists
        document_store = self._state.store.for_document(document_id)
        node_count = document_store.get_node_count()
        if node_count == 0:
            await _abort(
                context,
                code=grpc.StatusCode.NOT_FOUND,
                message=f"Document '{document_id}' not found.",
            )

        # Check if document is temporal
        is_temporal_result = document_store._doc_repo.get_document_is_temporal(
            document_id
        )
        is_temporal = (
            bool(is_temporal_result) if is_temporal_result is not None else False
        )
        if not is_temporal:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message=f"Document '{document_id}' is not a temporal document.",
            )

        # Delegate to runtime for actual truncation
        session = self._runtime.get_session(document_id)
        result = await session.truncate_from_time(cutoff_unix)

        response_cls = getattr(pb2, "TruncateFromTimeResponse")
        return response_cls(
            document_id=result.document_id,
            deleted_node_ids=result.deleted_node_ids,
            cutoff_time=cutoff_time_str,
        )


class RetrievalServicer(pb2_grpc.RetrievalServiceServicer):
    def __init__(self, state: ServerState) -> None:
        self._state = state

    async def Retrieve(  # noqa: N802
        self,
        request: pb2.RetrieveRequest,
        context: ServicerContextProto,
    ) -> pb2.RetrieveResponse:
        if not request.document_id:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="Retrieve requires `document_id`.",
            )
        num_seeds = request.num_seeds if request.num_seeds >= 0 else None
        # Query required unless num_seeds=0 (minimal summary mode)
        if not request.query and num_seeds != 0:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="Retrieve requires `query` (unless num_seeds=0).",
            )

        doc_id = request.document_id
        budget = request.budget_tokens or self._state.query_config.budget_tokens

        retriever, document_store = build_retriever(self._state, document_id=doc_id)

        retrieval_result = await retriever.retrieve_async(
            request.query,
            num_seeds=num_seeds,
            budget_tokens=budget,
            document_id=doc_id,
        )

        return _retrieval_to_proto(cast(Retrievable, retrieval_result))

    async def ExecuteQuery(  # noqa: N802
        self,
        request: pb2.ExecuteQueryRequest,
        context: ServicerContextProto,
    ) -> pb2.ExecuteQueryResponse:
        if not request.document_id:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="ExecuteQuery requires `document_id`.",
            )
        num_seeds = request.num_seeds if request.num_seeds >= 0 else None
        # Query required unless num_seeds=0 (minimal summary mode)
        if not request.query and num_seeds != 0:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="ExecuteQuery requires `query` (unless num_seeds=0).",
            )

        budget_default = self._state.query_config.budget_tokens
        budget = request.budget_tokens or budget_default
        embedding_model = (
            request.embedding_model or self._state.query_config.embedding_model
        )

        retriever, document_store = build_retriever(
            self._state,
            document_id=request.document_id,
            embedding_model=embedding_model,
        )

        recent_verbatim_budget = (
            request.recent_verbatim_token_budget
            if request.recent_verbatim_token_budget > 0
            else None
        )

        # Extract window bounds from request (span_end=0 treated as unset)
        span_start = request.span_start
        span_end = request.span_end if request.span_end > 0 else None

        # Extract temporal window from request (empty string treated as unset)
        time_start = request.time_start if request.HasField("time_start") else None
        time_end = request.time_end if request.HasField("time_end") else None

        # Use telemetry-enabled retrieval if profiling requested
        query_telemetry = None
        if request.profile:
            retrieval_result, query_telemetry = await retriever.retrieve_with_telemetry(
                request.query,
                num_seeds=num_seeds,
                budget_tokens=budget,
                document_id=request.document_id,
                recent_verbatim_budget=recent_verbatim_budget,
                span_start=span_start,
                span_end=span_end,
                time_start=time_start,
                time_end=time_end,
            )
        else:
            retrieval_result = await retriever.retrieve_async(
                request.query,
                num_seeds=num_seeds,
                budget_tokens=budget,
                document_id=request.document_id,
                recent_verbatim_budget=recent_verbatim_budget,
                span_start=span_start,
                span_end=span_end,
                time_start=time_start,
                time_end=time_end,
            )

        assembler = Assembler(document_store)

        # Track assembly time if profiling
        if query_telemetry:
            import time

            assembly_start = time.perf_counter()
            summary_text = assembler.assemble(retrieval_result)
            query_telemetry.assembly_time = time.perf_counter() - assembly_start
            query_telemetry.end_time = time.perf_counter()
        else:
            summary_text = assembler.assemble(retrieval_result)

        token_count = assembler.get_token_count(summary_text)
        nodes_retrieved = len(retrieval_result.node_ids)
        tiling_size = len(retrieval_result.tiling or [])

        if not retrieval_result.tiling:
            raise ValueError("ExecuteQuery returned no tiling; cannot log query")

        query_id = self._state.query_log.record_query(
            document_id=request.document_id,
            query_text=request.query,
            budget_tokens=budget,
            num_seeds=num_seeds,
            tiling_ids=retrieval_result.tiling,
            scores=retrieval_result.scores,
            seed_ids=set(retrieval_result.node_ids),
        )

        visualization = ""
        validation_warning = ""
        if request.debug and retrieval_result.tiling:
            width = request.viz_width or 120
            try:
                visualization = build_ascii_tree(
                    retrieval_result.tiling,
                    document_store,
                    width=width,
                    coverage_map=dict(retrieval_result.coverage_map or {}),
                    seed_node_ids=set(retrieval_result.node_ids),
                    use_token_coords=request.use_token_coords,
                    preloaded_nodes=retrieval_result.nodes,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to build visualization: %s", exc)
                visualization = f"Visualization error: {exc}"

            try:
                # Only validate budget if one was specified
                if budget is not None:
                    # Total budget includes verbatim budget if specified
                    total_budget = budget + (recent_verbatim_budget or 0)
                    validation_error = validate_tiling(
                        retrieval_result.tiling,
                        document_store,
                        budget_tokens=total_budget,
                        preloaded_nodes=retrieval_result.nodes,
                    )
                    if validation_error:
                        validation_warning = validation_error
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Tiling validation failed: %s", exc)
                validation_warning = f"Validation error: {exc}"

        retrieval_proto = _retrieval_to_proto(cast(Retrievable, retrieval_result))

        # Build telemetry proto if profiling was enabled
        telemetry_proto = None
        if query_telemetry:
            telemetry_proto = pb2.QueryTelemetry(
                embedding_ms=query_telemetry.embedding_time * 1000,
                search_ms=query_telemetry.search_time * 1000,
                mmr_ms=query_telemetry.mmr_time * 1000,
                coverage_map_ms=query_telemetry.coverage_map_time * 1000,
                scoring_ms=query_telemetry.scoring_time * 1000,
                tiling_ms=query_telemetry.tiling_time * 1000,
                assembly_ms=query_telemetry.assembly_time * 1000,
                total_ms=query_telemetry.total_time * 1000,
                seeds_requested=query_telemetry.seeds_requested,
                seeds_found=query_telemetry.seeds_found,
                candidates_retrieved=query_telemetry.candidates_retrieved,
                candidates_filtered=query_telemetry.candidates_filtered,
                coverage_size=query_telemetry.coverage_size,
                tiling_size=query_telemetry.tiling_size,
                output_tokens=query_telemetry.output_tokens,
                embedding_model=query_telemetry.embedding_model,
            )

        if telemetry_proto is not None:
            return pb2.ExecuteQueryResponse(
                summary=summary_text,
                token_count=token_count,
                nodes_retrieved=nodes_retrieved,
                tiling_size=tiling_size,
                retrieval=retrieval_proto,
                visualization=visualization,
                validation_warning=validation_warning,
                query_id=query_id,
                seed_count=retrieval_result.seed_count,
                verbatim_count=retrieval_result.verbatim_count,
                telemetry=telemetry_proto,
                actual_start=retrieval_result.actual_start,
                actual_end=retrieval_result.actual_end or 0,
            )
        return pb2.ExecuteQueryResponse(
            summary=summary_text,
            token_count=token_count,
            nodes_retrieved=nodes_retrieved,
            tiling_size=tiling_size,
            retrieval=retrieval_proto,
            visualization=visualization,
            validation_warning=validation_warning,
            query_id=query_id,
            seed_count=retrieval_result.seed_count,
            verbatim_count=retrieval_result.verbatim_count,
            actual_start=retrieval_result.actual_start,
            actual_end=retrieval_result.actual_end or 0,
        )


class WorkerServicer(pb2_grpc.WorkerServiceServicer):
    def __init__(self, state: ServerState) -> None:
        self._state = state

    async def _annotate_fidelity(
        self, doc_store: DocumentStore, nodes: Sequence[object]
    ) -> None:
        """Compute semantic fidelity for parent nodes in telemetry."""
        token_limit = getattr(
            self._state.llm_service, "_embedding_batch_token_limit", 8000
        )
        max_items = getattr(
            self._state.llm_service, "_provider_max_embedding_batch_size", 1000
        )
        await annotate_telemetry_fidelity(
            document_store=doc_store,
            telemetry_nodes=nodes,
            embedder=self._state.llm_service,
            token_limit=token_limit,
            max_batch_items=max_items,
        )

    async def RunWorkers(  # noqa: N802
        self,
        request: pb2.RunWorkersRequest,
        context: ServicerContextProto,
    ) -> AsyncIterator[pb2.RunWorkersResponse]:
        mode: int = request.mode
        if mode == _UNSPECIFIED_WORKER_MODE:
            mode = _UNTIL_IDLE_WORKER_MODE
        if mode not in {_UNTIL_IDLE_WORKER_MODE, _CONTINUOUS_WORKER_MODE}:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message=f"Unsupported worker run mode: {request.mode}",
            )

        existing_documents = self._state.store.list_documents()
        for document in existing_documents:
            doc_id = getattr(document, "id", None)
            if not doc_id:
                continue
            await self._state.indexing_engine.trigger_work(doc_id)

        poll_interval = 0.5

        while True:
            status = await self._state.indexing_engine.status()
            message = self._format_status(status)
            idle = status.in_flight == 0
            doc_ids = set(status.in_flight_by_document)
            doc_ids.update(status.completed_by_document)
            doc_ids.update(status.expected_total_by_document)
            document_progress = []
            for doc_id in sorted(doc_ids):
                totals = DocumentProgressTotals.from_status_dicts(
                    doc_id,
                    status.in_flight_by_document,
                    status.completed_by_document,
                    status.expected_total_by_document,
                )
                document_progress.append(
                    pb2.WorkerDocumentProgress(
                        document_id=doc_id,
                        pending=0,  # No pending queue in new model
                        inflight=totals.inflight,
                        completed=totals.completed,
                        total=totals.total,
                    )
                )
            yield pb2.RunWorkersResponse(
                message=message,
                idle=idle,
                queue_depth=0,  # No pending queue in new model
                inflight=status.in_flight,
                documents=document_progress,
            )

            # Return AFTER yielding the final idle status
            if mode == _UNTIL_IDLE_WORKER_MODE and idle:
                return

            try:
                await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:  # pragma: no cover - cooperative shutdown
                return

    async def GetDocumentStatus(  # noqa: N802
        self,
        request: object,
        context: ServicerContextProto,
    ) -> object:
        """Return document status with completion metrics.

        Unlike GetDocument which focuses on work queue state, this method
        returns document completeness and temporal range information for
        stateless sync workflows.

        See specs/temporal-document-apis.md for full specification.
        """
        document_id = getattr(request, "document_id", "")
        if not document_id:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="GetDocumentStatus requires `document_id`.",
            )

        # Check if document exists
        document_store = self._state.store.for_document(document_id)
        leaf_count = document_store.nodes.leaf_count()
        node_count = document_store.get_node_count()

        # Document doesn't exist if it has no nodes
        exists = node_count > 0

        response_cls = getattr(pb2, "DocumentStatusResponse")

        if not exists:
            # Return empty response for non-existent documents
            return response_cls(
                document_id=document_id,
                exists=False,
                is_temporal=False,
                leaf_count=0,
                node_count=0,
                complete_forest_size=0,
                completion_pct=0.0,
            )

        # Get temporal status from document repository
        is_temporal_result = document_store._doc_repo.get_document_is_temporal(
            document_id
        )
        is_temporal = (
            bool(is_temporal_result) if is_temporal_result is not None else False
        )

        # Calculate completion metrics using binary forest formula
        forest_size = complete_forest_size(leaf_count)
        completion_pct = (node_count / forest_size * 100.0) if forest_size > 0 else 0.0

        # Get temporal range if document is temporal
        time_start_iso: str | None = None
        time_end_iso: str | None = None
        if is_temporal:
            time_start, time_end = document_store.get_temporal_range()
            if time_start is not None:
                time_start_iso = _unix_to_iso8601(time_start)
            if time_end is not None:
                time_end_iso = _unix_to_iso8601(time_end)

        response = response_cls(
            document_id=document_id,
            exists=True,
            is_temporal=is_temporal,
            leaf_count=leaf_count,
            node_count=node_count,
            complete_forest_size=forest_size,
            completion_pct=completion_pct,
        )
        # Set optional fields if present
        if time_start_iso is not None:
            response.time_start = time_start_iso
        if time_end_iso is not None:
            response.time_end = time_end_iso

        return response

    async def GetDocument(  # noqa: N802
        self,
        request: pb2.GetDocumentRequest,
        context: ServicerContextProto,
    ) -> pb2.GetDocumentResponse:
        if not request.document_id:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="GetDocument requires `document_id`.",
            )

        document_store = self._state.store.for_document(request.document_id)
        leaf_count = document_store.nodes.leaf_count()
        root = document_store.tree.get_root()
        tree_depth = int(getattr(root, "height", 0) or 0) if root else 0

        # Get temporal status from document repository
        is_temporal_result = document_store._doc_repo.get_document_is_temporal(
            request.document_id
        )
        is_temporal = (
            bool(is_temporal_result) if is_temporal_result is not None else False
        )

        indexing_status = await self._state.indexing_engine.status()
        inflight = indexing_status.in_flight_by_document.get(request.document_id, 0)
        has_pending_work = inflight > 0

        doc_status = pb2.DocumentStatus(
            document_id=request.document_id,
            leaf_count=leaf_count,
            has_pending_work=has_pending_work,
            tree_depth=tree_depth,
            is_temporal=is_temporal,
        )
        return pb2.GetDocumentResponse(status=doc_status)

    async def GetTelemetry(  # noqa: N802
        self,
        request: TelemetryRequestProto,
        context: ServicerContextProto,
    ) -> TelemetryResponseProto:
        if not request.document_id:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="GetTelemetry requires `document_id`.",
            )
        if not request.run_id:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="GetTelemetry requires `run_id`.",
            )

        run = await self._state.telemetry_run_manager.get_run(request.run_id)
        if run is None or run.document_id != request.document_id:
            await _abort(
                context,
                code=grpc.StatusCode.NOT_FOUND,
                message="Telemetry run not found.",
            )

        if request.wait and run.status == "in_progress":
            run = await self._state.telemetry_run_manager.wait_for_completion(run)

        response_cls = getattr(pb2, "GetTelemetryResponse")

        if run.status == "in_progress":
            response = response_cls(complete=False)
            return cast("TelemetryResponseProto", response)

        error_message = run.error or ""
        telemetry_payload = ""
        if not error_message and run.result is not None:
            telemetry = run.result
            # Compute fidelity for parent nodes
            nodes = telemetry.get("nodes")
            if isinstance(nodes, list) and nodes:
                doc_store = self._state.store.for_document(request.document_id)
                await self._annotate_fidelity(doc_store, nodes)
            telemetry_payload = json.dumps(telemetry)

        response = response_cls(
            complete=True,
            telemetry_json=telemetry_payload,
            error=error_message,
        )
        return cast("TelemetryResponseProto", response)

    async def ExportTelemetry(  # noqa: N802
        self,
        request: ExportTelemetryRequestProto,
        context: ServicerContextProto,
    ) -> ExportTelemetryResponseProto:
        if not request.document_id:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="ExportTelemetry requires `document_id`.",
            )

        telemetry_log = self._state.telemetry_log
        if telemetry_log is None:
            await _abort(
                context,
                code=grpc.StatusCode.FAILED_PRECONDITION,
                message="Server telemetry logging is disabled.",
            )

        doc_store = self._state.store.for_document(request.document_id)
        active_nodes: dict[str, dict[str, object]] = {}
        for node in doc_store.nodes.iter_all():
            active_nodes[node.id] = {
                "height": int(getattr(node, "height", 0)),
                "span": (
                    int(getattr(node, "span_start", 0)),
                    int(getattr(node, "span_end", 0)),
                ),
            }

        try:
            telemetry = export_document_telemetry(
                telemetry_log,
                request.document_id,
                active_nodes=active_nodes,
            )
        except TelemetryExportError as exc:
            response_cls = getattr(pb2, "ExportTelemetryResponse")
            response = response_cls(
                telemetry_json="",
                error=str(exc),
            )
            return cast("ExportTelemetryResponseProto", response)

        nodes = telemetry.get("nodes")
        if isinstance(nodes, list) and nodes:
            await self._annotate_fidelity(doc_store, nodes)

        response_cls = getattr(pb2, "ExportTelemetryResponse")
        response = response_cls(
            telemetry_json=json.dumps(telemetry),
            error="",
        )
        return cast("ExportTelemetryResponseProto", response)

    async def ClearDocument(  # noqa: N802
        self,
        request: object,
        context: ServicerContextProto,
    ) -> object:
        clear_all = bool(getattr(request, "clear_all", False))
        raw_ids = list(getattr(request, "document_ids", []))

        if clear_all and raw_ids:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="Specify either `clear_all` or explicit `document_ids`, not both.",
            )

        if not clear_all and not raw_ids:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="ClearDocument requires `document_ids` when `clear_all` is false.",
            )

        if clear_all:
            documents = self._state.store.list_documents()
            doc_ids = [getattr(doc, "id", "") for doc in documents]
        else:
            doc_ids = raw_ids

        normalized_ids = [doc_id for doc_id in doc_ids if doc_id]
        unique_ids = sorted(set(normalized_ids))

        results: list[object] = []
        for document_id in unique_ids:
            deleted_nodes, existed = await self._clear_document(document_id)
            result_message = getattr(pb2, "ClearDocumentResult")(
                document_id=document_id,
                deleted_nodes=deleted_nodes,
                document_existed=existed,
            )
            results.append(result_message)

        response_type = getattr(pb2, "ClearDocumentResponse")
        return response_type(results=results)

    async def _clear_document(self, document_id: str) -> tuple[int, bool]:
        session = self._state.index_runtime.get_session(document_id)
        result = await session.clear()
        return result.deleted_nodes, result.document_existed

    def _format_status(self, status: IndexingStatus) -> str:
        parts = [f"inflight={status.in_flight}"]
        if status.in_flight_by_document:
            inflight = ", ".join(
                f"{doc}:{count}"
                for doc, count in sorted(status.in_flight_by_document.items())
            )
            parts.append(f"active=[{inflight}]")
        return " ".join(parts)

    async def ListDocuments(  # noqa: N802
        self,
        request: pb2.ListDocumentsRequest,
        context: ServicerContextProto,
    ) -> pb2.ListDocumentsResponse:
        """List all indexed documents with metadata.

        Returns DocumentInfo for each document including leaf count, node count,
        temporal status, time range, and completion percentage.

        Spec: specs/grpc-cli-architecture.md § New gRPC Methods
        """
        documents: list[pb2.DocumentInfo] = []

        for doc in self._state.store.list_documents():
            doc_id = getattr(doc, "id", "")
            if not doc_id:
                continue

            doc_store = self._state.store.for_document(doc_id)
            leaf_count = doc_store.nodes.leaf_count()
            node_count = doc_store.get_node_count()

            # Get temporal status
            is_temporal_result = doc_store._doc_repo.get_document_is_temporal(doc_id)
            is_temporal = (
                bool(is_temporal_result) if is_temporal_result is not None else False
            )

            # Calculate completion percentage using binary forest formula
            forest_size = complete_forest_size(leaf_count)
            completion_pct = (
                (node_count / forest_size * 100.0) if forest_size > 0 else 0.0
            )

            # Build DocumentInfo proto
            doc_info = pb2.DocumentInfo(
                document_id=doc_id,
                leaf_count=leaf_count,
                node_count=node_count,
                is_temporal=is_temporal,
                completion_pct=completion_pct,
            )

            # Add temporal range if document is temporal
            if is_temporal:
                time_start, time_end = doc_store.get_temporal_range()
                if time_start is not None:
                    doc_info.time_start = _unix_to_iso8601(time_start)
                if time_end is not None:
                    doc_info.time_end = _unix_to_iso8601(time_end)

            documents.append(doc_info)

        return pb2.ListDocumentsResponse(documents=documents)

    async def GetSystemStatus(  # noqa: N802
        self,
        request: pb2.GetSystemStatusRequest,
        context: ServicerContextProto,
    ) -> pb2.GetSystemStatusResponse:
        """Get aggregated system status across all documents.

        Returns total_nodes, leaf_nodes, and tree_depth aggregated across
        all indexed documents in the system.

        Spec: specs/grpc-cli-architecture.md § New gRPC Methods
        """
        total_nodes = 0
        leaf_nodes = 0
        tree_depth = 0

        for doc in self._state.store.list_documents():
            doc_store = self._state.store.for_document(doc.id)
            total_nodes += doc_store.nodes.count()
            leaf_nodes += doc_store.nodes.leaf_count()
            tree_depth = max(tree_depth, doc_store.nodes.max_height())

        return pb2.GetSystemStatusResponse(
            total_nodes=total_nodes,
            leaf_nodes=leaf_nodes,
            tree_depth=tree_depth,
        )

    async def GetCostStats(  # noqa: N802
        self,
        request: pb2.GetCostStatsRequest,
        context: ServicerContextProto,
    ) -> pb2.GetCostStatsResponse:
        """Get cost statistics for documents.

        If document_id is provided, returns stats for only that document.
        Otherwise, returns stats for all documents.

        Spec: specs/grpc-cli-architecture.md § New gRPC Methods
        """
        cost_stats_list: list[pb2.DocumentCostStats] = []

        # If document_id is specified, filter to just that document
        if request.document_id:
            doc_store = self._state.store.for_document(request.document_id)
            total_cost, total_nodes, leaf_nodes, summary_nodes = (
                doc_store.nodes._repo.get_cost_stats(request.document_id)
            )
            cost_stats_list.append(
                pb2.DocumentCostStats(
                    document_id=request.document_id,
                    total_cost=total_cost,
                    total_nodes=total_nodes,
                    leaf_nodes=leaf_nodes,
                    summary_nodes=summary_nodes,
                )
            )
        else:
            # Return stats for all documents
            for doc in self._state.store.list_documents():
                doc_id = getattr(doc, "id", "")
                if not doc_id:
                    continue
                doc_store = self._state.store.for_document(doc_id)
                total_cost, total_nodes, leaf_nodes, summary_nodes = (
                    doc_store.nodes._repo.get_cost_stats(doc_id)
                )
                cost_stats_list.append(
                    pb2.DocumentCostStats(
                        document_id=doc_id,
                        total_cost=total_cost,
                        total_nodes=total_nodes,
                        leaf_nodes=leaf_nodes,
                        summary_nodes=summary_nodes,
                    )
                )

        return pb2.GetCostStatsResponse(documents=cost_stats_list)


class SearchServicer(pb2_grpc.SearchServiceServicer):
    def __init__(self, state: ServerState) -> None:
        self._state = state

    async def Search(  # noqa: N802
        self,
        request: pb2.SearchRequest,
        context: ServicerContextProto,
    ) -> pb2.SearchResponse:
        if not request.question:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="Search requires `question`.",
            )

        executor = build_server_query_executor(self._state)

        # Route: session continuation vs new search
        if request.HasField("session_id"):
            try:
                result = await self._state.search_agent.search_continue(
                    request.session_id,
                    request.question,
                    executor,
                )
            except KeyError:
                await _abort(
                    context,
                    code=grpc.StatusCode.NOT_FOUND,
                    message=f"Session '{request.session_id}' not found or expired.",
                )
        else:
            if not request.document_id:
                await _abort(
                    context,
                    code=grpc.StatusCode.INVALID_ARGUMENT,
                    message="Search requires `document_id`.",
                )
            time_start = request.time_start if request.HasField("time_start") else None
            time_end = request.time_end if request.HasField("time_end") else None
            result = await self._state.search_agent.search(
                request.question,
                request.document_id,
                executor,
                time_start=time_start,
                time_end=time_end,
            )

        profile_proto = None
        if result.profile is not None:
            iterations = [
                pb2.SearchIterationProto(
                    query=it.query,
                    budget_tokens=it.budget_tokens,
                    result_text=it.result_text,
                    result_token_count=it.result_token_count,
                    agent_reasoning=it.agent_reasoning,
                    **({"time_start": it.time_start} if it.time_start else {}),
                    **({"time_end": it.time_end} if it.time_end else {}),
                )
                for it in result.profile.iterations
            ]
            profile_proto = pb2.SearchProfileProto(
                iterations=iterations,
                total_input_tokens=result.profile.total_input_tokens,
                total_output_tokens=result.profile.total_output_tokens,
                duration_seconds=result.profile.duration_seconds,
                retrospective=result.profile.retrospective,
                transcript=result.profile.transcript,
            )
            if result.profile.total_cost_usd is not None:
                profile_proto.total_cost_usd = result.profile.total_cost_usd

        response = pb2.SearchResponse(answer=result.answer)
        if profile_proto is not None:
            response = pb2.SearchResponse(
                answer=result.answer,
                profile=profile_proto,
            )
        if result.session_id is not None:
            response.session_id = result.session_id
        return response


async def shutdown_gracefully(server: GrpcServerProto) -> None:
    await server.stop(grace=None)
    await server.wait_for_termination()


async def serve(
    state: ServerState, *, host: str, port: int, http_port: int | None = None
) -> None:
    # 100MB max message size for large transcript uploads
    max_message_size = 100 * 1024 * 1024
    server = cast(
        GrpcServerProto,
        grpc.aio.server(
            options=[
                ("grpc.max_receive_message_length", max_message_size),
                ("grpc.max_send_message_length", max_message_size),
                # Disable SO_REUSEPORT to prevent multiple daemons binding to the same port.
                # This ensures only one daemon can serve requests, complementing the
                # single-writer lease mechanism.
                ("grpc.so_reuseport", 0),
            ]
        ),
    )
    pb2_grpc.add_IndexerServiceServicer_to_server(IndexerServicer(state), server)
    pb2_grpc.add_RetrievalServiceServicer_to_server(RetrievalServicer(state), server)
    pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(state), server)
    pb2_grpc.add_SearchServiceServicer_to_server(SearchServicer(state), server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("Starting RagZoom gRPC server on %s", listen_addr)

    # Start HTTP API server for REST access (useful for sandboxed clients)
    http_runner = None
    if http_port is not None:
        from ragzoom.server.http_api import start_http_server

        http_runner = await start_http_server(state, host=host, port=http_port)

    # IndexingEngine doesn't require explicit start - it auto-discovers work
    progress_task = asyncio.create_task(
        _render_indexing_progress(state.indexing_engine)
    )
    try:
        try:
            documents = state.store.list_documents()
            for document in documents:
                doc_id = getattr(document, "id", None)
                if not doc_id:
                    continue
                await state.indexing_engine.trigger_work(doc_id)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to trigger work for existing documents at startup")

        await server.start()

        try:
            await server.wait_for_termination()
        except asyncio.CancelledError:
            logger.info("Received cancellation; shutting down gRPC server")
            await shutdown_gracefully(server)
            raise
    finally:
        progress_task.cancel()
        with suppress(asyncio.CancelledError):
            await progress_task
        if http_runner is not None:
            await http_runner.stop()
        await state.indexing_engine.shutdown()


async def _render_indexing_progress(engine: IndexingEngine) -> None:
    display = WorkerProgressDisplay(focus_documents=None, line_printer=logger.info)
    try:
        while True:
            status = await engine.status()
            # Only show documents with active inflight work
            active_doc_ids = {
                doc_id
                for doc_id, count in status.in_flight_by_document.items()
                if count > 0
            }

            if not active_doc_ids and status.in_flight == 0:
                display.finish()
                await asyncio.sleep(0.5)
                continue

            documents = {
                doc_id: DocumentProgressTotals.from_status_dicts(
                    doc_id,
                    status.in_flight_by_document,
                    status.completed_by_document,
                    status.expected_total_by_document,
                )
                for doc_id in sorted(active_doc_ids)
            }

            display.update(
                queue_depth=0,  # No pending queue in new model
                inflight=status.in_flight,
                documents=documents,
                message=None,
            )

            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        raise
    finally:
        display.finish()
