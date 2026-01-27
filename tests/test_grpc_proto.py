"""Tests for gRPC proto definitions for CLI-to-server communication.

Verifies that the new proto definitions for ListDocuments, ValidateDocument,
GetSystemStatus, and GetCostStats are correctly generated and available.
"""

from __future__ import annotations

from ragzoom.rpc import dynamic_summary_pb2


class TestListDocumentsProto:
    """Test the ListDocuments RPC proto definitions."""

    def test_list_documents_request_exists(self) -> None:
        """ListDocumentsRequest message type should be importable."""
        assert hasattr(dynamic_summary_pb2, "ListDocumentsRequest")
        # Request is empty - verify it can be created
        req = dynamic_summary_pb2.ListDocumentsRequest()
        assert req is not None

    def test_list_documents_response_exists(self) -> None:
        """ListDocumentsResponse message type should be importable."""
        assert hasattr(dynamic_summary_pb2, "ListDocumentsResponse")

    def test_document_info_message_exists(self) -> None:
        """DocumentInfo message type should be importable."""
        assert hasattr(dynamic_summary_pb2, "DocumentInfo")

    def test_document_info_has_required_fields(self) -> None:
        """DocumentInfo should have all specified fields."""
        doc_info = dynamic_summary_pb2.DocumentInfo(
            document_id="test_doc",
            leaf_count=10,
            node_count=19,
            is_temporal=True,
            time_start="2024-01-21T10:00:00Z",
            time_end="2024-01-21T12:00:00Z",
            completion_pct=95.5,
        )
        assert doc_info.document_id == "test_doc"
        assert doc_info.leaf_count == 10
        assert doc_info.node_count == 19
        assert doc_info.is_temporal is True
        assert doc_info.time_start == "2024-01-21T10:00:00Z"
        assert doc_info.time_end == "2024-01-21T12:00:00Z"
        assert abs(doc_info.completion_pct - 95.5) < 0.01

    def test_document_info_optional_fields(self) -> None:
        """DocumentInfo optional fields should be unset by default."""
        doc_info = dynamic_summary_pb2.DocumentInfo(
            document_id="test_doc",
            leaf_count=5,
            node_count=9,
        )
        # Optional fields should not have values when unset
        assert doc_info.HasField("time_start") is False
        assert doc_info.HasField("time_end") is False
        assert doc_info.HasField("completion_pct") is False

    def test_list_documents_response_has_repeated_documents(self) -> None:
        """ListDocumentsResponse should have repeated DocumentInfo field."""
        doc1 = dynamic_summary_pb2.DocumentInfo(
            document_id="doc1", leaf_count=5, node_count=9
        )
        doc2 = dynamic_summary_pb2.DocumentInfo(
            document_id="doc2", leaf_count=10, node_count=19
        )
        response = dynamic_summary_pb2.ListDocumentsResponse(documents=[doc1, doc2])
        assert len(response.documents) == 2
        assert response.documents[0].document_id == "doc1"
        assert response.documents[1].document_id == "doc2"

    def test_document_info_field_numbers(self) -> None:
        """DocumentInfo fields should have correct field numbers per spec."""
        descriptor = dynamic_summary_pb2.DocumentInfo.DESCRIPTOR
        fields = {f.name: f.number for f in descriptor.fields}
        assert fields["document_id"] == 1
        assert fields["leaf_count"] == 2
        assert fields["node_count"] == 3
        assert fields["is_temporal"] == 4
        assert fields["time_start"] == 5
        assert fields["time_end"] == 6
        assert fields["completion_pct"] == 7


class TestValidateDocumentProto:
    """Test the ValidateDocument RPC proto definitions."""

    def test_validate_document_request_exists(self) -> None:
        """ValidateDocumentRequest message type should be importable."""
        assert hasattr(dynamic_summary_pb2, "ValidateDocumentRequest")
        req = dynamic_summary_pb2.ValidateDocumentRequest(document_id="test_doc")
        assert req.document_id == "test_doc"

    def test_validate_document_response_exists(self) -> None:
        """ValidateDocumentResponse message type should be importable."""
        assert hasattr(dynamic_summary_pb2, "ValidateDocumentResponse")

    def test_validate_document_response_has_required_fields(self) -> None:
        """ValidateDocumentResponse should have valid and errors fields."""
        response = dynamic_summary_pb2.ValidateDocumentResponse(
            valid=True,
            errors=[],
        )
        assert response.valid is True
        assert len(response.errors) == 0

    def test_validate_document_response_with_errors(self) -> None:
        """ValidateDocumentResponse should support error list."""
        response = dynamic_summary_pb2.ValidateDocumentResponse(
            valid=False,
            errors=["Missing root node", "Invalid parent reference"],
        )
        assert response.valid is False
        assert len(response.errors) == 2
        assert response.errors[0] == "Missing root node"
        assert response.errors[1] == "Invalid parent reference"

    def test_validate_document_field_numbers(self) -> None:
        """ValidateDocument messages should have correct field numbers."""
        req_desc = dynamic_summary_pb2.ValidateDocumentRequest.DESCRIPTOR
        req_fields = {f.name: f.number for f in req_desc.fields}
        assert req_fields["document_id"] == 1

        resp_desc = dynamic_summary_pb2.ValidateDocumentResponse.DESCRIPTOR
        resp_fields = {f.name: f.number for f in resp_desc.fields}
        assert resp_fields["valid"] == 1
        assert resp_fields["errors"] == 2
