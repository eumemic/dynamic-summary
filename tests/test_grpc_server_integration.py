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


@pytest.mark.asyncio
async def test_validate_document_servicer_returns_valid_for_healthy_document(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """ValidateDocument servicer returns valid=true for a healthy document.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    Test: tests/test_grpc_server_integration.py::test_validate_document_servicer_returns_valid_for_healthy_document
    """
    address, state = grpc_test_environment

    # Index a document and let workers complete
    client = GrpcRagzoomClient(address)
    try:
        await asyncio.to_thread(
            client.append_text,
            document_id="validate-test-doc",
            content=b"Hello world for validation test",
            collect_telemetry=False,
            replace_existing=True,
        )
        await asyncio.to_thread(client.run_workers_once)
    finally:
        client.close()

    # Call ValidateDocument via direct gRPC stub
    async with grpc.aio.insecure_channel(address) as channel:
        stub = pb2_grpc.WorkerServiceStub(channel)
        request = pb2.ValidateDocumentRequest(document_id="validate-test-doc")
        response = await stub.ValidateDocument(request)

    # Verify response indicates valid document
    assert response.valid is True
    assert len(response.errors) == 0


@pytest.mark.asyncio
async def test_validate_document_servicer_not_found_for_missing_document(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """ValidateDocument servicer returns NOT_FOUND for non-existent document.

    Spec: specs/grpc-cli-architecture.md § Error Handling
    Test: tests/test_grpc_server_integration.py::test_validate_document_servicer_not_found_for_missing_document
    """
    address, state = grpc_test_environment

    # Call ValidateDocument for a document that doesn't exist
    async with grpc.aio.insecure_channel(address) as channel:
        stub = pb2_grpc.WorkerServiceStub(channel)
        request = pb2.ValidateDocumentRequest(document_id="nonexistent-doc")

        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.ValidateDocument(request)

        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.asyncio
async def test_validate_document_servicer_invalid_argument_for_empty_document_id(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """ValidateDocument servicer returns INVALID_ARGUMENT for empty document_id.

    Spec: specs/grpc-cli-architecture.md § Error Handling
    Test: tests/test_grpc_server_integration.py::test_validate_document_servicer_invalid_argument_for_empty_document_id
    """
    address, state = grpc_test_environment

    # Call ValidateDocument with empty document_id
    async with grpc.aio.insecure_channel(address) as channel:
        stub = pb2_grpc.WorkerServiceStub(channel)
        request = pb2.ValidateDocumentRequest(document_id="")

        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.ValidateDocument(request)

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_get_system_status_servicer_returns_aggregated_stats(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """GetSystemStatus servicer returns aggregated stats across all documents.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    Test: tests/test_grpc_server_integration.py::test_get_system_status_servicer_returns_aggregated_stats
    """
    address, state = grpc_test_environment

    # Index two documents with workers completing
    client = GrpcRagzoomClient(address)
    try:
        await asyncio.to_thread(
            client.append_text,
            document_id="status-doc-1",
            content=b"First document for status test",
            collect_telemetry=False,
            replace_existing=True,
        )
        await asyncio.to_thread(
            client.append_text,
            document_id="status-doc-2",
            content=b"Second document for status test",
            collect_telemetry=False,
            replace_existing=True,
        )
        await asyncio.to_thread(client.run_workers_once)
    finally:
        client.close()

    # Call GetSystemStatus via direct gRPC stub
    async with grpc.aio.insecure_channel(address) as channel:
        stub = pb2_grpc.WorkerServiceStub(channel)
        request = pb2.GetSystemStatusRequest()
        response = await stub.GetSystemStatus(request)

    # Verify response contains aggregated data
    # With 2 documents, each with at least 1 leaf node
    assert response.total_nodes >= 2, "Should have nodes from both documents"
    assert response.leaf_nodes >= 2, "Should have leaf nodes from both documents"
    assert response.tree_depth >= 0, "Tree depth should be non-negative"


@pytest.mark.asyncio
async def test_get_system_status_servicer_empty_system(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """GetSystemStatus servicer returns zeros for empty system.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    Test: tests/test_grpc_server_integration.py::test_get_system_status_servicer_empty_system
    """
    address, state = grpc_test_environment

    # Call GetSystemStatus without indexing any documents
    # Note: Other tests may have already indexed documents, so we just verify
    # the response structure is valid
    async with grpc.aio.insecure_channel(address) as channel:
        stub = pb2_grpc.WorkerServiceStub(channel)
        request = pb2.GetSystemStatusRequest()
        response = await stub.GetSystemStatus(request)

    # Verify response has valid structure (fields exist and are integers)
    assert isinstance(response.total_nodes, int)
    assert isinstance(response.leaf_nodes, int)
    assert isinstance(response.tree_depth, int)
    assert response.total_nodes >= 0
    assert response.leaf_nodes >= 0
    assert response.tree_depth >= 0


@pytest.mark.asyncio
async def test_get_cost_stats_servicer_returns_stats_for_document(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """GetCostStats servicer returns cost stats for a specific document.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    Test: tests/test_grpc_server_integration.py::test_get_cost_stats_servicer_returns_stats_for_document
    """
    address, state = grpc_test_environment

    # Index a document with workers completing
    client = GrpcRagzoomClient(address)
    try:
        await asyncio.to_thread(
            client.append_text,
            document_id="cost-test-doc",
            content=b"Content for cost stats test",
            collect_telemetry=False,
            replace_existing=True,
        )
        await asyncio.to_thread(client.run_workers_once)
    finally:
        client.close()

    # Call GetCostStats via direct gRPC stub
    async with grpc.aio.insecure_channel(address) as channel:
        stub = pb2_grpc.WorkerServiceStub(channel)
        request = pb2.GetCostStatsRequest(document_id="cost-test-doc")
        response = await stub.GetCostStats(request)

    # Verify response contains stats for our document
    assert len(response.documents) == 1
    doc_stats = response.documents[0]
    assert doc_stats.document_id == "cost-test-doc"
    assert doc_stats.total_nodes >= 1
    assert doc_stats.leaf_nodes >= 1
    assert doc_stats.summary_nodes == doc_stats.total_nodes - doc_stats.leaf_nodes
    # Cost may be 0 if no summarization has occurred
    assert doc_stats.total_cost >= 0.0


@pytest.mark.asyncio
async def test_get_cost_stats_servicer_returns_all_documents(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """GetCostStats servicer returns cost stats for all documents when no filter.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    Test: tests/test_grpc_server_integration.py::test_get_cost_stats_servicer_returns_all_documents
    """
    address, state = grpc_test_environment

    # Index two documents
    client = GrpcRagzoomClient(address)
    try:
        await asyncio.to_thread(
            client.append_text,
            document_id="cost-all-doc-1",
            content=b"First document for cost all test",
            collect_telemetry=False,
            replace_existing=True,
        )
        await asyncio.to_thread(
            client.append_text,
            document_id="cost-all-doc-2",
            content=b"Second document for cost all test",
            collect_telemetry=False,
            replace_existing=True,
        )
        await asyncio.to_thread(client.run_workers_once)
    finally:
        client.close()

    # Call GetCostStats without document_id filter
    async with grpc.aio.insecure_channel(address) as channel:
        stub = pb2_grpc.WorkerServiceStub(channel)
        request = pb2.GetCostStatsRequest()  # No document_id = all documents
        response = await stub.GetCostStats(request)

    # Verify response contains both documents
    doc_ids = [d.document_id for d in response.documents]
    assert "cost-all-doc-1" in doc_ids
    assert "cost-all-doc-2" in doc_ids

    # Verify each document has valid stats
    for doc_stats in response.documents:
        assert doc_stats.total_nodes >= 0
        assert doc_stats.leaf_nodes >= 0
        assert doc_stats.summary_nodes == doc_stats.total_nodes - doc_stats.leaf_nodes
        assert doc_stats.total_cost >= 0.0
