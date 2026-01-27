"""Integration tests for the gRPC server stack."""

from __future__ import annotations

import asyncio

import grpc.aio
import pytest

from ragzoom.client.grpc_client import GrpcRagzoomClient, WorkerRunSnapshot
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc
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
        )
        assert result.query_result.summary
    finally:
        client.close()

    status = await state.indexing_engine.status()
    assert status.in_flight == 0


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

    final_status = await state.indexing_engine.status()
    assert final_status.in_flight == 0


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


@pytest.mark.asyncio
async def test_list_documents_servicer_returns_document_info(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """ListDocuments servicer returns DocumentInfo for indexed documents.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    Test: tests/test_grpc_server_integration.py::test_list_documents_servicer_returns_document_info
    """
    address, state = grpc_test_environment

    # Index a document first
    client = GrpcRagzoomClient(address)
    try:
        await asyncio.to_thread(
            client.append_text,
            document_id="list-test-doc",
            content=b"Hello world for list test",
            collect_telemetry=False,
            replace_existing=True,
        )
        await asyncio.to_thread(client.run_workers_once)
    finally:
        client.close()

    # Call ListDocuments via direct gRPC stub
    async with grpc.aio.insecure_channel(address) as channel:
        stub = pb2_grpc.WorkerServiceStub(channel)
        request = pb2.ListDocumentsRequest()
        response = await stub.ListDocuments(request)

    # Verify response contains our document
    assert len(response.documents) >= 1
    doc_ids = [d.document_id for d in response.documents]
    assert "list-test-doc" in doc_ids

    # Find our document and verify its fields
    doc_info = next(d for d in response.documents if d.document_id == "list-test-doc")
    assert doc_info.leaf_count >= 1
    assert doc_info.node_count >= doc_info.leaf_count
    assert doc_info.HasField("completion_pct")
    assert doc_info.completion_pct > 0.0


@pytest.mark.asyncio
async def test_list_documents_servicer_returns_empty_for_no_documents(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """ListDocuments servicer returns empty list when no documents exist.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    Test: tests/test_grpc_server_integration.py::test_list_documents_servicer_returns_empty_for_no_documents
    """
    address, state = grpc_test_environment

    # Clear all documents first to ensure clean state
    client = GrpcRagzoomClient(address)
    try:
        await asyncio.to_thread(client.clear_documents, clear_all=True)
    finally:
        client.close()

    # Call ListDocuments
    async with grpc.aio.insecure_channel(address) as channel:
        stub = pb2_grpc.WorkerServiceStub(channel)
        request = pb2.ListDocumentsRequest()
        response = await stub.ListDocuments(request)

    # Verify response is empty
    assert len(response.documents) == 0


@pytest.mark.asyncio
async def test_list_documents_servicer_includes_temporal_info(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """ListDocuments servicer includes temporal info for temporal documents.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    Test: tests/test_grpc_server_integration.py::test_list_documents_servicer_includes_temporal_info
    """
    address, state = grpc_test_environment

    # Index a temporal document
    client = GrpcRagzoomClient(address)
    try:
        await asyncio.to_thread(
            client.append_text,
            document_id="temporal-list-doc",
            content=b"Temporal content",
            collect_telemetry=False,
            replace_existing=True,
            timestamp=("2024-01-21T10:00:00Z", "2024-01-21T10:05:00Z"),
        )
        await asyncio.to_thread(client.run_workers_once)
    finally:
        client.close()

    # Call ListDocuments
    async with grpc.aio.insecure_channel(address) as channel:
        stub = pb2_grpc.WorkerServiceStub(channel)
        request = pb2.ListDocumentsRequest()
        response = await stub.ListDocuments(request)

    # Find our temporal document
    doc_info = next(
        (d for d in response.documents if d.document_id == "temporal-list-doc"), None
    )
    assert doc_info is not None
    assert doc_info.is_temporal is True
    # Temporal documents should have time_start and time_end set
    assert doc_info.HasField("time_start")
    assert doc_info.HasField("time_end")
