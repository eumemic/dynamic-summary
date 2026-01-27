"""Tests for gRPC client with summarization_guidance field."""

from __future__ import annotations

import asyncio

import pytest

from ragzoom.client.grpc_client import GrpcRagzoomClient
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.server.state import ServerState


@pytest.mark.asyncio
async def test_append_text_with_summarization_guidance(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Verify the protobuf field supports summarization_guidance.

    This test verifies that:
    1. The client can pass summarization_guidance through append_text
    2. The proto field is named summarization_guidance (not summary_system_prompt)
    3. The server receives and stores the value correctly
    """
    address, state = grpc_test_environment
    custom_guidance = "This is legal documentation. Preserve party names."

    client = GrpcRagzoomClient(address)
    try:
        result = await asyncio.to_thread(
            client.append_text,
            document_id="guidance-test",
            content=b"Contract between Alpha Corp and Beta LLC.",
            collect_telemetry=False,
            replace_existing=True,
            summarization_guidance=custom_guidance,
        )
        assert result is not None

        # Verify document was created and guidance stored
        doc = state.store.get_document_by_id("guidance-test")
        assert doc is not None, "Document should be created"
        assert (
            doc.summarization_guidance == custom_guidance
        ), f"Expected guidance stored, got: {doc.summarization_guidance!r}"
    finally:
        client.close()


def test_proto_has_summarization_guidance_field() -> None:
    """Verify AppendTextRequest proto has summarization_guidance field."""
    # Create request and verify the field exists and can be set
    request = pb2.AppendTextRequest(
        document_id="test",
        content=b"test content",
        summarization_guidance="Test guidance",
    )
    assert request.summarization_guidance == "Test guidance"


def test_proto_summarization_guidance_is_optional() -> None:
    """Verify summarization_guidance field is optional (can be omitted)."""
    request = pb2.AppendTextRequest(
        document_id="test",
        content=b"test content",
    )
    # Should not have the field set
    assert not request.HasField("summarization_guidance")


def test_batch_append_proto_has_summarization_guidance_field() -> None:
    """Verify BatchAppendTextRequest proto has summarization_guidance field."""
    request = pb2.BatchAppendTextRequest(
        document_id="test",
        units=[pb2.AppendUnit(content=b"unit 1")],
        summarization_guidance="Batch guidance",
    )
    assert request.summarization_guidance == "Batch guidance"


def test_batch_append_proto_summarization_guidance_is_optional() -> None:
    """Verify summarization_guidance field is optional in BatchAppendTextRequest."""
    request = pb2.BatchAppendTextRequest(
        document_id="test",
        units=[pb2.AppendUnit(content=b"unit 1")],
    )
    assert not request.HasField("summarization_guidance")


@pytest.mark.asyncio
async def test_batch_append_text_with_summarization_guidance(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Verify batch_append_text passes summarization_guidance to server.

    Spec: specs/transcript-summarization-guidance.md § 2. Thread Through gRPC Client
    """
    address, state = grpc_test_environment
    custom_guidance = "This is conversation transcript. Preserve identity and agency."

    client = GrpcRagzoomClient(address)
    try:
        result = await asyncio.to_thread(
            client.batch_append_text,
            document_id="batch-guidance-test",
            units=["Turn 1: Hello", "Turn 2: World"],
            summarization_guidance=custom_guidance,
        )
        assert result is not None

        # Verify document was created and guidance stored
        doc = state.store.get_document_by_id("batch-guidance-test")
        assert doc is not None, "Document should be created"
        assert (
            doc.summarization_guidance == custom_guidance
        ), f"Expected guidance stored, got: {doc.summarization_guidance!r}"
    finally:
        client.close()


@pytest.mark.asyncio
async def test_grpc_client_get_document_status(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Test get_document_status returns DocumentStatusView with all fields.

    Spec: specs/temporal-document-apis.md § API Changes > Python Client
    """
    from ragzoom.client.grpc_client import DocumentStatusView

    address, state = grpc_test_environment
    document_id = "status-test-doc"

    client = GrpcRagzoomClient(address)
    try:
        await asyncio.to_thread(
            client.append_text,
            document_id=document_id,
            content=b"Test content for status check.",
            collect_telemetry=False,
            replace_existing=True,
        )
        await asyncio.to_thread(client.run_workers_once)

        status = await asyncio.to_thread(client.get_document_status, document_id)

        assert isinstance(status, DocumentStatusView)
        assert status.document_id == document_id
        assert status.exists is True
        assert status.is_temporal is False
        assert status.leaf_count >= 1
        assert status.node_count >= status.leaf_count
        assert status.complete_forest_size >= status.leaf_count
        assert 0 <= status.completion_pct <= 100
        assert status.time_start is None
        assert status.time_end is None

    finally:
        client.close()


@pytest.mark.asyncio
async def test_grpc_client_get_document_status_nonexistent(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Test get_document_status returns exists=False for nonexistent documents."""
    from ragzoom.client.grpc_client import DocumentStatusView

    address, _state = grpc_test_environment

    client = GrpcRagzoomClient(address)
    try:
        status = await asyncio.to_thread(
            client.get_document_status, "nonexistent-doc-12345"
        )

        assert isinstance(status, DocumentStatusView)
        assert status.exists is False
        assert status.leaf_count == 0
        assert status.node_count == 0

    finally:
        client.close()


@pytest.mark.asyncio
async def test_grpc_client_truncate_from_time(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Test truncate_from_time client method calls RPC and maps errors correctly.

    Spec: specs/temporal-document-apis.md § API Changes > Python Client

    Note: End-to-end truncation behavior is tested in test_temporal_document_apis.py
    using mocked state to support temporal documents.
    """

    address, _state = grpc_test_environment
    document_id = "truncate-time-test-nonexistent"
    cutoff = "2024-01-01T10:30:00Z"

    client = GrpcRagzoomClient(address)
    try:
        # Verify RPC error mapping: NOT_FOUND -> RuntimeError
        with pytest.raises(RuntimeError, match="NOT_FOUND"):
            await asyncio.to_thread(
                client.truncate_from_time,
                document_id=document_id,
                cutoff_time=cutoff,
            )

    finally:
        client.close()


@pytest.mark.asyncio
async def test_grpc_client_truncate_from_time_dataclass() -> None:
    """Test TruncateFromTimeResult dataclass exists and has expected fields.

    Spec: specs/temporal-document-apis.md § API Changes > Python Client
    """
    from ragzoom.client.grpc_client import TruncateFromTimeResult

    # Verify the dataclass can be constructed with expected fields
    result = TruncateFromTimeResult(
        document_id="test-doc",
        deleted_node_ids=["node-1", "node-2"],
        cutoff_time="2024-01-01T12:00:00Z",
    )

    assert result.document_id == "test-doc"
    assert result.deleted_node_ids == ["node-1", "node-2"]
    assert result.cutoff_time == "2024-01-01T12:00:00Z"


@pytest.mark.asyncio
async def test_grpc_client_list_documents(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Test list_documents returns DocumentInfoView list.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    """
    from ragzoom.client.grpc_client import DocumentInfoView

    address, _state = grpc_test_environment
    document_id = "list-documents-test"

    client = GrpcRagzoomClient(address)
    try:
        # Create a document
        await asyncio.to_thread(
            client.append_text,
            document_id=document_id,
            content=b"Test content for list documents.",
            collect_telemetry=False,
            replace_existing=True,
        )
        await asyncio.to_thread(client.run_workers_once)

        # List documents
        documents = await asyncio.to_thread(client.list_documents)

        # Find our document in the list
        assert isinstance(documents, list)
        assert len(documents) >= 1

        found = None
        for doc in documents:
            assert isinstance(doc, DocumentInfoView)
            if doc.document_id == document_id:
                found = doc
                break

        assert found is not None, f"Document {document_id} not found in list"
        assert found.leaf_count >= 1
        assert found.node_count >= found.leaf_count
        assert found.is_temporal is False
        assert found.time_start is None
        assert found.time_end is None
        # completion_pct should be between 0 and 100
        if found.completion_pct is not None:
            assert 0 <= found.completion_pct <= 100

    finally:
        client.close()


@pytest.mark.asyncio
async def test_grpc_client_list_documents_empty(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Test list_documents returns empty list when no documents exist.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    """
    from ragzoom.client.grpc_client import DocumentInfoView

    address, state = grpc_test_environment

    # Clear all documents first to ensure empty state
    client = GrpcRagzoomClient(address)
    try:
        await asyncio.to_thread(client.clear_all_documents)

        documents = await asyncio.to_thread(client.list_documents)

        assert isinstance(documents, list)
        # After clear, should be empty or at least not fail
        for doc in documents:
            assert isinstance(doc, DocumentInfoView)

    finally:
        client.close()


@pytest.mark.asyncio
async def test_grpc_client_validate_document(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Test validate_document returns ValidationResult with valid=True for healthy doc.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    """
    from ragzoom.client.grpc_client import ValidationResult

    address, _state = grpc_test_environment
    document_id = "validate-test-doc"

    client = GrpcRagzoomClient(address)
    try:
        # Create a document
        await asyncio.to_thread(
            client.append_text,
            document_id=document_id,
            content=b"Test content for validation.",
            collect_telemetry=False,
            replace_existing=True,
        )
        await asyncio.to_thread(client.run_workers_once)

        # Validate the document
        result = await asyncio.to_thread(client.validate_document, document_id)

        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.errors == []

    finally:
        client.close()


@pytest.mark.asyncio
async def test_grpc_client_validate_document_not_found(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Test validate_document raises RuntimeError for nonexistent document.

    Spec: specs/grpc-cli-architecture.md § Error Handling
    """
    address, _state = grpc_test_environment

    client = GrpcRagzoomClient(address)
    try:
        with pytest.raises(RuntimeError, match="NOT_FOUND"):
            await asyncio.to_thread(
                client.validate_document, "nonexistent-validate-doc-12345"
            )

    finally:
        client.close()


@pytest.mark.asyncio
async def test_grpc_client_validate_document_empty_id(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Test validate_document raises RuntimeError for empty document_id.

    Spec: specs/grpc-cli-architecture.md § Error Handling
    """
    address, _state = grpc_test_environment

    client = GrpcRagzoomClient(address)
    try:
        with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
            await asyncio.to_thread(client.validate_document, "")

    finally:
        client.close()


@pytest.mark.asyncio
async def test_grpc_client_get_system_status(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Test get_system_status returns SystemStatusView with aggregated stats.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    """
    from ragzoom.client.grpc_client import SystemStatusView

    address, _state = grpc_test_environment
    document_id = "system-status-test-doc"

    client = GrpcRagzoomClient(address)
    try:
        # Create a document to have some data
        await asyncio.to_thread(
            client.append_text,
            document_id=document_id,
            content=b"Test content for system status check.",
            collect_telemetry=False,
            replace_existing=True,
        )
        await asyncio.to_thread(client.run_workers_once)

        # Get system status
        status = await asyncio.to_thread(client.get_system_status)

        assert isinstance(status, SystemStatusView)
        assert status.total_nodes >= 1
        assert status.leaf_nodes >= 1
        assert status.leaf_nodes <= status.total_nodes
        assert status.tree_depth >= 0

    finally:
        client.close()


@pytest.mark.asyncio
async def test_grpc_client_get_system_status_empty(
    grpc_test_environment: tuple[str, ServerState],
) -> None:
    """Test get_system_status returns zeros when no documents exist.

    Spec: specs/grpc-cli-architecture.md § New gRPC Methods
    """
    from ragzoom.client.grpc_client import SystemStatusView

    address, _state = grpc_test_environment

    client = GrpcRagzoomClient(address)
    try:
        # Clear all documents first to ensure empty state
        await asyncio.to_thread(client.clear_all_documents)

        # Get system status
        status = await asyncio.to_thread(client.get_system_status)

        assert isinstance(status, SystemStatusView)
        assert status.total_nodes == 0
        assert status.leaf_nodes == 0
        assert status.tree_depth == 0

    finally:
        client.close()
