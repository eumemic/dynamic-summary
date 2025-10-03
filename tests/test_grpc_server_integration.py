"""Integration tests for the gRPC server stack."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import cast

import grpc
import pytest

from ragzoom.client.grpc_client import GrpcRagzoomClient, WorkerRunSnapshot
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc
from ragzoom.server.servicers import (
    GrpcServerProto,
    IndexerServicer,
    RetrievalServicer,
    WorkerServicer,
    shutdown_gracefully,
)
from ragzoom.server.state import ServerState


@pytest.fixture()
async def grpc_test_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[str, ServerState]]:
    """Spin up a lightweight gRPC server backed by in-memory components."""

    database_path = tmp_path / "integration.db"
    os.environ["PYTEST_CURRENT_TEST"] = "grpc-server-integration"

    class _StubEmbeddings:
        def __init__(self) -> None:
            self._vector = [1.0] + [0.0] * 7

        def create(self, *, model: str, input: object, **_: object) -> object:
            if isinstance(input, str):
                texts = [input]
            else:
                texts = list(cast(Sequence[str], input))

            class _Item:
                def __init__(self, embedding: list[float]) -> None:
                    self.embedding = embedding

            class _Resp:
                def __init__(self, items: list[_Item]) -> None:
                    self.data = items

            return _Resp([_Item(list(self._vector)) for _ in texts])

    class _StubOpenAI:
        def __init__(self, **_: object) -> None:
            self.embeddings = _StubEmbeddings()

    monkeypatch.setattr("openai.OpenAI", _StubOpenAI, raising=False)
    monkeypatch.setattr("ragzoom.server.servicers.OpenAI", _StubOpenAI, raising=False)
    monkeypatch.setattr(
        "ragzoom.retrieval.embedding_service.OpenAI", _StubOpenAI, raising=False
    )

    operational_cfg = OperationalConfig(
        openai_api_key=SecretStr("test-key"),
        backend="sqlite",
        database_url=f"sqlite:///{database_path}",
        vector_backend="python",
    )
    state = ServerState.create(
        index_config=IndexConfig.load(),
        query_config=QueryConfig(),
        operational_config=operational_cfg,
    )

    server = grpc.aio.server()
    pb2_grpc.add_IndexerServiceServicer_to_server(IndexerServicer(state), server)
    pb2_grpc.add_RetrievalServiceServicer_to_server(RetrievalServicer(state), server)
    pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(state), server)

    port = server.add_insecure_port("127.0.0.1:0")
    await state.worker_coordinator.start()
    await server.start()

    address = f"127.0.0.1:{port}"

    try:
        yield address, state
    finally:
        try:
            await asyncio.wait_for(
                state.worker_coordinator.wait_until_idle(), timeout=5
            )
        except Exception:
            pass
        await shutdown_gracefully(cast(GrpcServerProto, server))
        await state.worker_coordinator.shutdown()
        state.store.close()


@pytest.mark.asyncio
async def test_grpc_server_lifecycle(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    address, state = grpc_test_environment
    client = GrpcRagzoomClient(address)
    try:
        await asyncio.to_thread(
            client.append_text,
            document_id="doc",
            content=b"Hello world",
            collect_telemetry=False,
            replace_existing=True,
        )
        snapshots = await asyncio.to_thread(client.run_workers_once)
        assert snapshots, "worker snapshots should not be empty"
        assert snapshots[-1].idle, "workers should reach idle state"

        result = await asyncio.to_thread(
            client.execute_query,
            query="Hello",
            document_id="doc",
            budget_tokens=256,
            num_seeds=5,
            embedding_model=None,
            debug=False,
            viz_width=80,
            use_token_coords=False,
        )
        assert result.query_result.summary
    finally:
        client.close()

    status = await state.worker_coordinator.status()
    assert status.queue_depth == 0


@pytest.mark.asyncio
async def test_grpc_server_handles_concurrent_clients(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    address, state = grpc_test_environment

    async def _append(doc_id: str, text: str) -> None:
        client = GrpcRagzoomClient(address)
        try:
            await asyncio.to_thread(
                client.append_text,
                document_id=doc_id,
                content=text.encode("utf-8"),
                collect_telemetry=False,
                replace_existing=True,
            )
        finally:
            client.close()

    await asyncio.gather(
        _append("alpha", "first document"),
        _append("beta", "second document"),
    )

    client = GrpcRagzoomClient(address)
    try:
        await asyncio.to_thread(client.run_workers_once)
    finally:
        client.close()

    final_status = await state.worker_coordinator.status()
    assert final_status.queue_depth == 0


@pytest.mark.asyncio
async def test_worker_service_streaming(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    address, _ = grpc_test_environment
    client = GrpcRagzoomClient(address, stream_timeout=None)
    try:
        await asyncio.to_thread(
            client.append_text,
            document_id="stream",
            content=b"stream document",
            collect_telemetry=False,
            replace_existing=True,
        )

        def _consume_first_snapshot() -> WorkerRunSnapshot | None:
            for snapshot in client.iter_worker_snapshots():
                return snapshot
            return None

        first_snapshot = await asyncio.to_thread(_consume_first_snapshot)
        assert first_snapshot is not None
        assert isinstance(first_snapshot.message, str)
    finally:
        client.close()
