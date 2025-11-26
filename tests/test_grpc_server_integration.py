"""Integration tests for the gRPC server stack."""

from __future__ import annotations

import asyncio

import pytest

from ragzoom.client.grpc_client import GrpcRagzoomClient, WorkerRunSnapshot
from ragzoom.server.state import ServerState


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
            tiling_strategy=None,
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
