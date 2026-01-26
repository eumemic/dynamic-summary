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
