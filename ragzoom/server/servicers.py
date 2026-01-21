"""gRPC servicer implementations for RagZoom."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol, TypeVar, cast

import grpc
from openai import OpenAI

from ragzoom.assemble import Assembler
from ragzoom.document_store import DocumentStore
from ragzoom.progress_display import DocumentProgressTotals, WorkerProgressDisplay
from ragzoom.retrieval.budget_planner import BudgetPlanner
from ragzoom.retrieval.embedding_service import EmbeddingService
from ragzoom.retrieve import Retriever
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc
from ragzoom.server.indexing_engine import IndexingEngine, IndexingStatus
from ragzoom.server.state import ServerState
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.telemetry_embeddings import annotate_telemetry_fidelity
from ragzoom.telemetry_export import (
    TelemetryExportError,
    export_document_telemetry,
)
from ragzoom.tree_viz import build_ascii_tree
from ragzoom.validate import validate_tiling
from ragzoom.vector_factory import create_vector_index

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

        nodes[node_id] = pb2.Node(
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

    return pb2.RetrieveResponse(
        selected_ids=list(retrieval_result.node_ids),
        tiling_ids=tiling_ids,
        scores=dict(retrieval_result.scores),
        coverage_map=dict(retrieval_result.coverage_map or {}),
        nodes=nodes,
    )


def _build_retriever(
    state: ServerState,
    *,
    document_id: str,
    embedding_model: str | None = None,
) -> tuple[Retriever, DocumentStore]:
    resolved_embedding = embedding_model or state.query_config.embedding_model
    document_store = state.store.for_document(document_id)
    client = OpenAI(
        api_key=state.operational_config.openai_api_key.get_secret_value(),
        timeout=state.operational_config.openai_timeout,
    )
    embedding_service = EmbeddingService(client, document_store, resolved_embedding)
    # For retrieval operations, use target_embedding_context_tokens as fallback
    # when target_chunk_tokens is None (client-managed chunking mode)
    chunk_tokens = (
        state.index_config.target_chunk_tokens
        if state.index_config.target_chunk_tokens is not None
        else state.index_config.target_embedding_context_tokens
    )
    budget_planner = BudgetPlanner(document_store, chunk_tokens)
    vector_index = create_vector_index(
        state.operational_config.vector_backend,
        state.operational_config.database_url,
        resolved_embedding,
    )
    query_config = state.query_config
    if resolved_embedding != state.query_config.embedding_model:
        query_config = query_config.replace(embedding_model=resolved_embedding)
    retriever = Retriever(
        query_config,
        document_store,
        embedding_service,
        budget_planner,
        vector_index,
    )
    return retriever, document_store


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

        session = self._runtime.get_session(request.document_id)
        result = await session.append_text(
            text,
            replace_existing=bool(getattr(request, "replace_existing", False)),
            collect_telemetry=request.collect_telemetry,
            timestamp=timestamp,
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

        # Decode each unit from bytes to string
        units: list[str] = []
        for i, unit_bytes in enumerate(request.units):
            try:
                units.append(unit_bytes.decode("utf-8"))
            except UnicodeDecodeError as exc:
                await _abort(
                    context,
                    code=grpc.StatusCode.INVALID_ARGUMENT,
                    message=f"Invalid UTF-8 in unit {i}: {exc}",
                )

        # Extract timestamps from proto if present
        timestamps: list[str | tuple[str, str]] | None = None
        if request.timestamps:
            # Validate length matches units
            if len(request.timestamps) != len(units):
                raise ValueError(
                    f"timestamps length ({len(request.timestamps)}) must match "
                    f"units length ({len(units)})"
                )
            timestamps = [_extract_timestamp(ts) for ts in request.timestamps]

        session = self._runtime.get_session(request.document_id)
        result = await session.batch_append_text(
            units,
            collect_telemetry=request.collect_telemetry,
            timestamps=timestamps,
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

        retriever, document_store = _build_retriever(self._state, document_id=doc_id)

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

        retriever, document_store = _build_retriever(
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

        # Use telemetry-enabled retrieval if profiling requested
        query_telemetry = None
        if request.profile:
            retrieval_result, query_telemetry = await retriever.retrieve_with_telemetry(
                request.query,
                num_seeds=num_seeds,
                budget_tokens=budget,
                document_id=request.document_id,
                recent_verbatim_budget=recent_verbatim_budget,
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
        mode = request.mode
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

        indexing_status = await self._state.indexing_engine.status()
        inflight = indexing_status.in_flight_by_document.get(request.document_id, 0)
        has_pending_work = inflight > 0

        doc_status = pb2.DocumentStatus(
            document_id=request.document_id,
            leaf_count=leaf_count,
            has_pending_work=has_pending_work,
            tree_depth=tree_depth,
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


async def shutdown_gracefully(server: GrpcServerProto) -> None:
    await server.stop(grace=None)
    await server.wait_for_termination()


async def serve(state: ServerState, *, host: str, port: int) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker

    from memory_service.grpc_servicer import SessionIngestionServicer
    from ragzoom.wrapper import AsyncRagZoom

    # 100MB max message size for large transcript uploads
    max_message_size = 100 * 1024 * 1024
    server = cast(
        GrpcServerProto,
        grpc.aio.server(
            options=[
                ("grpc.max_receive_message_length", max_message_size),
                ("grpc.max_send_message_length", max_message_size),
            ]
        ),
    )
    pb2_grpc.add_IndexerServiceServicer_to_server(IndexerServicer(state), server)
    pb2_grpc.add_RetrievalServiceServicer_to_server(RetrievalServicer(state), server)
    pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(state), server)

    # Add session ingestion service for Claude Code memory sync
    database_url = state.operational_config.database_url
    if database_url:
        # Ensure psycopg3-compatible URL
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+psycopg://", 1
            )
        engine = create_engine(database_url, pool_pre_ping=True)
        db_session_factory: sessionmaker[Session] = sessionmaker(bind=engine)

        def get_db_session() -> Session:
            return db_session_factory()

        def get_async_ragzoom_client(user_id: str) -> AsyncRagZoom:
            # For now, all users share the same runtime (no per-user isolation)
            # Document IDs should include user_id prefix for isolation
            return AsyncRagZoom(runtime=state.index_runtime)

        session_servicer = SessionIngestionServicer(
            get_db_session=get_db_session,
            get_async_ragzoom_client=get_async_ragzoom_client,
        )
        pb2_grpc.add_SessionIngestionServiceServicer_to_server(session_servicer, server)
        logger.info("Session ingestion service enabled")

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("Starting RagZoom gRPC server on %s", listen_addr)

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
        await state.indexing_engine.shutdown()


async def _render_indexing_progress(engine: IndexingEngine) -> None:
    display = WorkerProgressDisplay(focus_documents=None)
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
