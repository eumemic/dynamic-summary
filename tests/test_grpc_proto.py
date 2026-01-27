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


class TestGetSystemStatusProto:
    """Test the GetSystemStatus RPC proto definitions."""

    def test_get_system_status_request_exists(self) -> None:
        """GetSystemStatusRequest message type should be importable."""
        assert hasattr(dynamic_summary_pb2, "GetSystemStatusRequest")
        # Request is empty - verify it can be created
        req = dynamic_summary_pb2.GetSystemStatusRequest()
        assert req is not None

    def test_get_system_status_response_exists(self) -> None:
        """GetSystemStatusResponse message type should be importable."""
        assert hasattr(dynamic_summary_pb2, "GetSystemStatusResponse")

    def test_get_system_status_response_has_required_fields(self) -> None:
        """GetSystemStatusResponse should have total_nodes, leaf_nodes, tree_depth."""
        response = dynamic_summary_pb2.GetSystemStatusResponse(
            total_nodes=100,
            leaf_nodes=51,
            tree_depth=6,
        )
        assert response.total_nodes == 100
        assert response.leaf_nodes == 51
        assert response.tree_depth == 6

    def test_get_system_status_response_field_numbers(self) -> None:
        """GetSystemStatusResponse fields should have correct field numbers per spec."""
        descriptor = dynamic_summary_pb2.GetSystemStatusResponse.DESCRIPTOR
        fields = {f.name: f.number for f in descriptor.fields}
        assert fields["total_nodes"] == 1
        assert fields["leaf_nodes"] == 2
        assert fields["tree_depth"] == 3


class TestGetCostStatsProto:
    """Test the GetCostStats RPC proto definitions."""

    def test_get_cost_stats_request_exists(self) -> None:
        """GetCostStatsRequest message type should be importable."""
        assert hasattr(dynamic_summary_pb2, "GetCostStatsRequest")
        # Request has optional document_id - verify it can be created empty
        req = dynamic_summary_pb2.GetCostStatsRequest()
        assert req is not None

    def test_get_cost_stats_request_with_document_id(self) -> None:
        """GetCostStatsRequest should accept optional document_id."""
        req = dynamic_summary_pb2.GetCostStatsRequest(document_id="test_doc")
        assert req.document_id == "test_doc"

    def test_get_cost_stats_request_document_id_optional(self) -> None:
        """GetCostStatsRequest document_id should be optional."""
        req = dynamic_summary_pb2.GetCostStatsRequest()
        assert req.HasField("document_id") is False

    def test_get_cost_stats_response_exists(self) -> None:
        """GetCostStatsResponse message type should be importable."""
        assert hasattr(dynamic_summary_pb2, "GetCostStatsResponse")

    def test_document_cost_stats_message_exists(self) -> None:
        """DocumentCostStats message type should be importable."""
        assert hasattr(dynamic_summary_pb2, "DocumentCostStats")

    def test_document_cost_stats_has_required_fields(self) -> None:
        """DocumentCostStats should have all specified fields."""
        cost_stats = dynamic_summary_pb2.DocumentCostStats(
            document_id="test_doc",
            total_cost=1.234,
            total_nodes=100,
            leaf_nodes=51,
            summary_nodes=49,
        )
        assert cost_stats.document_id == "test_doc"
        assert abs(cost_stats.total_cost - 1.234) < 0.001
        assert cost_stats.total_nodes == 100
        assert cost_stats.leaf_nodes == 51
        assert cost_stats.summary_nodes == 49

    def test_get_cost_stats_response_has_repeated_documents(self) -> None:
        """GetCostStatsResponse should have repeated DocumentCostStats field."""
        doc1 = dynamic_summary_pb2.DocumentCostStats(
            document_id="doc1",
            total_cost=1.0,
            total_nodes=10,
            leaf_nodes=5,
            summary_nodes=5,
        )
        doc2 = dynamic_summary_pb2.DocumentCostStats(
            document_id="doc2",
            total_cost=2.0,
            total_nodes=20,
            leaf_nodes=10,
            summary_nodes=10,
        )
        response = dynamic_summary_pb2.GetCostStatsResponse(documents=[doc1, doc2])
        assert len(response.documents) == 2
        assert response.documents[0].document_id == "doc1"
        assert response.documents[1].document_id == "doc2"

    def test_get_cost_stats_request_field_numbers(self) -> None:
        """GetCostStatsRequest fields should have correct field numbers per spec."""
        descriptor = dynamic_summary_pb2.GetCostStatsRequest.DESCRIPTOR
        fields = {f.name: f.number for f in descriptor.fields}
        assert fields["document_id"] == 1

    def test_document_cost_stats_field_numbers(self) -> None:
        """DocumentCostStats fields should have correct field numbers per spec."""
        descriptor = dynamic_summary_pb2.DocumentCostStats.DESCRIPTOR
        fields = {f.name: f.number for f in descriptor.fields}
        assert fields["document_id"] == 1
        assert fields["total_cost"] == 2
        assert fields["total_nodes"] == 3
        assert fields["leaf_nodes"] == 4
        assert fields["summary_nodes"] == 5
