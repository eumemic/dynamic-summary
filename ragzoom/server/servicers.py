"""gRPC servicer implementations for RagZoom."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import NoReturn, Protocol, TypeVar, cast

import grpc
from openai import OpenAI

from ragzoom.assemble import Assembler
from ragzoom.document_store import DocumentStore
from ragzoom.retrieval.budget_planner import BudgetPlanner
from ragzoom.retrieval.embedding_service import EmbeddingService
from ragzoom.retrieve import Retriever
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc
from ragzoom.server.state import ServerState
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.tree_viz import build_ascii_tree
from ragzoom.validate import validate_tiling
from ragzoom.vector_factory import create_vector_index

logger = logging.getLogger(__name__)


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


def _stats_to_proto(stats: IndexingResult) -> pb2.DocumentStats:
    if (
        stats.mutated_nodes is None
        or stats.resummarized_nodes is None
        or stats.new_leaves is None
    ):
        raise ValueError("IndexingResult is missing required mutation metadata")

    telemetry_json = "" if stats.telemetry is None else json_dumps(stats.telemetry)
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


def json_dumps(data: object) -> str:
    import json

    return json.dumps(data)


async def _abort(
    context: ServicerContextProto, *, code: grpc.StatusCode, message: str
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
    client = OpenAI(api_key=state.operational_config.openai_api_key.get_secret_value())
    embedding_service = EmbeddingService(client, document_store, resolved_embedding)
    budget_planner = BudgetPlanner(
        document_store, state.index_config.target_chunk_tokens
    )
    vector_index = create_vector_index(
        state.operational_config.vector_backend,
        state.operational_config.database_url,
        resolved_embedding,
    )
    query_config = (
        state.query_config
        if resolved_embedding == state.query_config.embedding_model
        else state.query_config.replace(embedding_model=resolved_embedding)
    )
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

        stats = await self._state.indexing_service.index_document_async(
            text=text,
            document_id=document_id,
            file_path=file_path,
            show_progress=False,
            collect_telemetry=request.collect_telemetry,
        )

        return pb2.IndexDocumentResponse(stats=_stats_to_proto(stats))

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

        stats = await self._state.indexing_service.append_to_document_async(
            document_id=request.document_id,
            new_text=text,
            show_progress=False,
            collect_telemetry=request.collect_telemetry,
        )

        return pb2.AppendTextResponse(stats=_stats_to_proto(stats))


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
        if not request.query:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="Retrieve requires `query`.",
            )

        doc_id = request.document_id
        budget = request.budget_tokens or self._state.query_config.budget_tokens
        num_seeds = request.num_seeds if request.num_seeds > 0 else None

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
        if not request.query:
            await _abort(
                context,
                code=grpc.StatusCode.INVALID_ARGUMENT,
                message="ExecuteQuery requires `query`.",
            )

        budget_default = self._state.query_config.budget_tokens
        budget = request.budget_tokens or budget_default
        num_seeds = request.num_seeds if request.num_seeds > 0 else None
        embedding_model = (
            request.embedding_model or self._state.query_config.embedding_model
        )

        retriever, document_store = _build_retriever(
            self._state,
            document_id=request.document_id,
            embedding_model=embedding_model,
        )

        retrieval_result = await retriever.retrieve_async(
            request.query,
            num_seeds=num_seeds,
            budget_tokens=budget,
            document_id=request.document_id,
        )

        assembler = Assembler(document_store)
        summary_text = assembler.assemble(retrieval_result)
        token_count = assembler.get_token_count(summary_text)
        nodes_retrieved = len(retrieval_result.node_ids)
        tiling_size = len(retrieval_result.tiling or [])

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
                validation_error = validate_tiling(
                    retrieval_result.tiling,
                    document_store,
                    budget_tokens=budget,
                    preloaded_nodes=retrieval_result.nodes,
                )
                if validation_error:
                    validation_warning = validation_error
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Tiling validation failed: %s", exc)
                validation_warning = f"Validation error: {exc}"

        retrieval_proto = _retrieval_to_proto(cast(Retrievable, retrieval_result))

        return pb2.ExecuteQueryResponse(
            summary=summary_text,
            token_count=token_count,
            nodes_retrieved=nodes_retrieved,
            tiling_size=tiling_size,
            retrieval=retrieval_proto,
            visualization=visualization,
            validation_warning=validation_warning,
        )


class WorkerServicer(pb2_grpc.WorkerServiceServicer):
    def __init__(self, state: ServerState) -> None:
        self._state = state

    async def RunWorkers(  # noqa: N802
        self,
        request: pb2.RunWorkersRequest,
        context: ServicerContextProto,
    ) -> AsyncIterator[pb2.RunWorkersResponse]:
        del request
        del context
        message = "Background workers are not yet implemented; returning immediate idle state."
        logger.info(message)
        yield pb2.RunWorkersResponse(message=message, idle=True)

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
        leaves = document_store.nodes.get_leaves()
        root = document_store.tree.get_root()
        tree_depth = int(getattr(root, "height", 0) or 0) if root else 0

        status = pb2.DocumentStatus(
            document_id=request.document_id,
            leaf_count=len(leaves),
            has_pending_work=False,
            tree_depth=tree_depth,
        )
        return pb2.GetDocumentResponse(status=status)


async def shutdown_gracefully(server: grpc.aio.Server) -> None:  # type: ignore[attr-defined]
    await server.stop(grace=None)
    await server.wait_for_termination()


async def serve(state: ServerState, *, host: str, port: int) -> None:
    server = grpc.aio.server()  # type: ignore[attr-defined]
    pb2_grpc.add_IndexerServiceServicer_to_server(IndexerServicer(state), server)
    pb2_grpc.add_RetrievalServiceServicer_to_server(RetrievalServicer(state), server)
    pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(state), server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("Starting RagZoom gRPC server on %s", listen_addr)

    await server.start()

    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        logger.info("Received cancellation; shutting down gRPC server")
        await shutdown_gracefully(server)
        raise
