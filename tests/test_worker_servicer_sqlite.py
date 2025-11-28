import asyncio
from collections.abc import Callable, Sequence
from contextlib import suppress
from types import SimpleNamespace
from typing import NoReturn, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.contracts.vector_filter import VectorFilter
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.indexing import IndexerRuntime
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.server.append_executor import AppendExecutor
from ragzoom.server.run_manager import TelemetryRunManager
from ragzoom.server.servicers import WorkerServicer
from ragzoom.server.state import ServerState
from ragzoom.server.worker_coordinator import WorkerCoordinator
from ragzoom.vector_api import Vector

NodePayloadValue = str | int | float | bool | list[float] | NDArray[np.float64] | None
NodePayload = dict[str, NodePayloadValue]


def _leaf_payload(
    node_id: str,
    span_start: int,
    span_end: int,
    **kwargs: NodePayloadValue,
) -> NodePayload:
    payload: NodePayload = {
        "node_id": node_id,
        "text": f"text-{node_id}",
        "span_start": span_start,
        "span_end": span_end,
        "document_id": "doc",
        "token_count": span_end - span_start,
        "height": 0,
        "level_index": 0,
        "parent_id": None,
        "left_child_id": None,
        "right_child_id": None,
    }
    payload.update(kwargs)
    return payload


def _add_batch(store: DocumentStore, *payloads: NodePayload) -> None:
    store.nodes.add_batch(list(payloads))


class StubVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self.upserts: list[
            tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
        ] = []

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        filters: Sequence[VectorFilter] | None = None,
    ) -> list[Vector]:  # pragma: no cover - unused in tests
        raise NotImplementedError

    def get_vectors(self, ids: list[str]) -> list[Vector]:  # pragma: no cover
        return []

    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
    ) -> None:
        self.upserts.extend(items)

    def delete(
        self,
        filter: dict[str, object] | None = None,
        ids: list[str] | None = None,
    ) -> int:  # pragma: no cover
        return 0


class _NoopAppendExecutor:
    async def append(
        self,
        *,
        store: DocumentStore,
        vector_index: VectorIndex,
        document_id: str,
        new_text: str,
        reporter: object | None = None,
    ) -> object:
        raise AssertionError("append should not be invoked in worker tests")


def _make_runtime(
    store: SQLiteStorageBackend,
    index_config: IndexConfig,
    coordinator: WorkerCoordinator,
    telemetry_manager: TelemetryRunManager,
    *,
    vector_factory: Callable[[str], VectorIndex] | None = None,
) -> IndexerRuntime:
    factory = vector_factory or (lambda _model: StubVectorIndex())
    return IndexerRuntime(
        store=store,
        index_config=index_config,
        append_executor=cast(AppendExecutor, _NoopAppendExecutor()),
        worker_coordinator=coordinator,
        telemetry_manager=telemetry_manager,
        vector_index_factory=factory,
    )


class BlockingLLMService:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._release = asyncio.Event()

    async def _summarize_text(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: object | None = None,
        prev_context: str | None = None,
        left_token_count: int | None = None,
        right_token_count: int | None = None,
    ) -> tuple[str, int, int]:
        self.started.set()
        await self._release.wait()
        pid = parent_id or "parent"
        summary = f"summary-{pid[:8]}"
        return summary, 0, len(summary.split())

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def release(self) -> None:
        self._release.set()


class StubContext:
    async def abort(self, code: object, details: str) -> NoReturn:
        raise AssertionError(f"Unexpected abort ({code}): {details}")


class ImmediateLLMService:
    async def _summarize_text(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: object | None = None,
        prev_context: str | None = None,
        left_token_count: int | None = None,
        right_token_count: int | None = None,
    ) -> tuple[str, int, int]:
        summary = f"summary-{parent_id or 'parent'}"
        return summary, 0, len(summary.split())

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_run_workers_until_idle_streams_status() -> None:
    backend = SQLiteStorageBackend()
    coordinator: WorkerCoordinator | None = None
    try:
        backend.add_document(
            document_id="doc",
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-5-nano",
        )
        store = backend.for_document("doc")
        _add_batch(
            store,
            _leaf_payload("L", 0, 10, following_neighbor_id="R"),
            _leaf_payload("R", 10, 20, preceding_neighbor_id="L"),
        )

        llm = BlockingLLMService()
        vector_index = StubVectorIndex()
        index_config = IndexConfig.load()

        coordinator = WorkerCoordinator(
            store=backend,
            index_config=index_config,
            operational_config=OperationalConfig(
                openai_api_key=SecretStr("test"),
                vector_backend="python",
                database_url="sqlite:///:memory:",
            ),
            llm_service=llm,
            vector_index_factory=lambda _doc: vector_index,
            worker_count=1,
        )

        await coordinator.start()

        telemetry_manager = TelemetryRunManager(index_config)
        runtime = _make_runtime(
            backend,
            index_config,
            coordinator,
            telemetry_manager,
            vector_factory=lambda _model: vector_index,
        )

        state = SimpleNamespace(
            store=backend,
            worker_coordinator=coordinator,
            telemetry_run_manager=telemetry_manager,
            index_runtime=runtime,
        )
        servicer = WorkerServicer(cast(ServerState, state))

        async def _release_when_ready(service: BlockingLLMService) -> None:
            await service.started.wait()
            service.release()

        request = pb2.RunWorkersRequest(mode=pb2.WORKER_RUN_MODE_UNTIL_IDLE)
        release_task = asyncio.create_task(_release_when_ready(llm))
        responses: list[pb2.RunWorkersResponse] = []
        async for response in servicer.RunWorkers(request, StubContext()):
            responses.append(response)
            if response.idle:
                break

        assert responses, "Expected at least one streamed response"
        first_snapshot = responses[0]
        assert first_snapshot.queue_depth >= 0
        if first_snapshot.queue_depth > 0:
            assert first_snapshot.documents
            doc_progress = {
                doc.document_id: (
                    doc.pending,
                    doc.inflight,
                    getattr(doc, "completed", 0),
                    getattr(doc, "total", 0),
                )
                for doc in first_snapshot.documents
            }
            assert "doc" in doc_progress

        final_snapshot = responses[-1]
        assert final_snapshot.idle is True
        assert final_snapshot.queue_depth == 0
        assert final_snapshot.inflight == 0
        assert "queue=" in final_snapshot.message

        parents = [node for node in store.nodes.get_all() if node.height == 1]
        assert parents, "Worker run should build a parent node"
    finally:
        release_task.cancel()
        with suppress(asyncio.CancelledError):
            await release_task
        if coordinator is not None:
            await coordinator.shutdown()
        backend.close()


@pytest.mark.asyncio
async def test_clear_document_rpc_clears_existing_document() -> None:
    backend = SQLiteStorageBackend()
    coordinator: WorkerCoordinator | None = None
    try:
        index_config = IndexConfig.load()
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        )

        backend.add_document(
            document_id="doc",
            file_path=None,
            embedding_model=index_config.embedding_model,
            summary_model=index_config.summary_model,
        )
        store = backend.for_document("doc")
        _add_batch(
            store,
            _leaf_payload("L", 0, 10, following_neighbor_id="R"),
            _leaf_payload("R", 10, 20, preceding_neighbor_id="L"),
        )

        vector_index = StubVectorIndex()
        coordinator = WorkerCoordinator(
            store=backend,
            index_config=index_config,
            operational_config=operational_config,
            llm_service=ImmediateLLMService(),
            vector_index_factory=lambda _doc: vector_index,
            worker_count=1,
        )
        await coordinator.start()

        telemetry_manager = TelemetryRunManager(index_config)
        runtime = _make_runtime(
            backend,
            index_config,
            coordinator,
            telemetry_manager,
            vector_factory=lambda _model: vector_index,
        )
        state = SimpleNamespace(
            store=backend,
            worker_coordinator=coordinator,
            telemetry_run_manager=telemetry_manager,
            index_config=index_config,
            operational_config=operational_config,
            append_executor=None,
            index_runtime=runtime,
        )
        servicer = WorkerServicer(cast(ServerState, state))

        request = getattr(pb2, "ClearDocumentRequest")(
            document_ids=["doc"], clear_all=False
        )
        response = await servicer.ClearDocument(request, StubContext())

        results = list(getattr(response, "results", []))
        assert results
        result = results[0]
        assert result.document_id == "doc"
        assert result.document_existed is True
        assert result.deleted_nodes > 0
        assert backend.for_document("doc").nodes.leaf_count() == 0
    finally:
        if coordinator is not None:
            await coordinator.shutdown()
        backend.close()


@pytest.mark.asyncio
async def test_clear_document_rpc_handles_missing_document() -> None:
    backend = SQLiteStorageBackend()
    coordinator: WorkerCoordinator | None = None
    try:
        index_config = IndexConfig.load()
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        )

        coordinator = WorkerCoordinator(
            store=backend,
            index_config=index_config,
            operational_config=operational_config,
            llm_service=ImmediateLLMService(),
            vector_index_factory=lambda _doc: StubVectorIndex(),
            worker_count=1,
        )
        await coordinator.start()

        telemetry_manager = TelemetryRunManager(index_config)
        runtime = _make_runtime(
            backend,
            index_config,
            coordinator,
            telemetry_manager,
        )
        state = SimpleNamespace(
            store=backend,
            worker_coordinator=coordinator,
            telemetry_run_manager=telemetry_manager,
            index_config=index_config,
            operational_config=operational_config,
            append_executor=None,
            index_runtime=runtime,
        )
        servicer = WorkerServicer(cast(ServerState, state))

        request = getattr(pb2, "ClearDocumentRequest")(
            document_ids=["missing"], clear_all=False
        )
        response = await servicer.ClearDocument(request, StubContext())

        results = list(getattr(response, "results", []))
        assert results
        result = results[0]
        assert result.document_id == "missing"
        assert result.document_existed is False
        assert result.deleted_nodes == 0
    finally:
        if coordinator is not None:
            await coordinator.shutdown()
        backend.close()


@pytest.mark.asyncio
async def test_clear_document_rpc_cancels_inflight_work() -> None:
    backend = SQLiteStorageBackend()
    coordinator: WorkerCoordinator | None = None
    try:
        index_config = IndexConfig.load()
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        )

        backend.add_document(
            document_id="doc",
            file_path=None,
            embedding_model=index_config.embedding_model,
            summary_model=index_config.summary_model,
        )
        store = backend.for_document("doc")
        _add_batch(
            store,
            _leaf_payload("L", 0, 10, following_neighbor_id="R"),
            _leaf_payload("R", 10, 20, preceding_neighbor_id="L"),
        )

        llm = BlockingLLMService()
        vector_index = StubVectorIndex()
        coordinator = WorkerCoordinator(
            store=backend,
            index_config=index_config,
            operational_config=operational_config,
            llm_service=llm,
            vector_index_factory=lambda _doc: vector_index,
            worker_count=1,
        )
        await coordinator.start()

        telemetry_manager = TelemetryRunManager(index_config)
        runtime = _make_runtime(
            backend,
            index_config,
            coordinator,
            telemetry_manager,
            vector_factory=lambda _model: vector_index,
        )
        state = SimpleNamespace(
            store=backend,
            worker_coordinator=coordinator,
            telemetry_run_manager=telemetry_manager,
            index_config=index_config,
            operational_config=operational_config,
            append_executor=None,
            index_runtime=runtime,
        )
        servicer = WorkerServicer(cast(ServerState, state))

        await coordinator.enqueue_document("doc")
        await llm.started.wait()

        request = getattr(pb2, "ClearDocumentRequest")(
            document_ids=["doc"], clear_all=False
        )
        clear_task = asyncio.create_task(servicer.ClearDocument(request, StubContext()))

        llm.release()
        response = await clear_task

        results = list(getattr(response, "results", []))
        assert results
        result = results[0]
        assert result.document_id == "doc"
        assert result.document_existed is True
        assert backend.for_document("doc").nodes.leaf_count() == 0
        parents = [
            node
            for node in backend.for_document("doc").nodes.get_all()
            if int(node.height) > 0
        ]
        assert not parents
    finally:
        if coordinator is not None:
            await coordinator.shutdown()
        backend.close()


@pytest.mark.asyncio
async def test_get_document_reflects_pending_work() -> None:
    backend = SQLiteStorageBackend()
    coordinator: WorkerCoordinator | None = None
    try:
        backend.add_document(
            document_id="doc",
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-5-nano",
        )
        store = backend.for_document("doc")
        _add_batch(
            store,
            _leaf_payload("L", 0, 10, following_neighbor_id="R"),
            _leaf_payload("R", 10, 20, preceding_neighbor_id="L"),
        )

        llm = BlockingLLMService()
        vector_index = StubVectorIndex()
        index_config = IndexConfig.load()
        coordinator = WorkerCoordinator(
            store=backend,
            index_config=index_config,
            operational_config=OperationalConfig(
                openai_api_key=SecretStr("test"),
                vector_backend="python",
                database_url="sqlite:///:memory:",
            ),
            llm_service=llm,
            vector_index_factory=lambda _doc: vector_index,
            worker_count=1,
        )

        await coordinator.start()
        telemetry_manager = TelemetryRunManager(index_config)
        runtime = _make_runtime(
            backend,
            index_config,
            coordinator,
            telemetry_manager,
            vector_factory=lambda _model: vector_index,
        )
        state = SimpleNamespace(
            store=backend,
            worker_coordinator=coordinator,
            telemetry_run_manager=telemetry_manager,
            index_runtime=runtime,
        )
        servicer = WorkerServicer(cast(ServerState, state))

        await coordinator.enqueue_document("doc")
        pending = await servicer.GetDocument(
            pb2.GetDocumentRequest(document_id="doc"),
            StubContext(),
        )
        assert pending.status.has_pending_work is True

        llm.release()
        await coordinator.wait_until_idle("doc")

        resolved = await servicer.GetDocument(
            pb2.GetDocumentRequest(document_id="doc"),
            StubContext(),
        )
        assert resolved.status.has_pending_work is False
    finally:
        if coordinator is not None:
            await coordinator.shutdown()
        backend.close()
