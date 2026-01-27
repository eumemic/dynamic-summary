"""Tests for transcript summarization guidance feature.

Acceptance tests verifying all criteria from specs/transcript-summarization-guidance.md.
"""

from __future__ import annotations

import inspect

from ragzoom.client.grpc_client import GrpcRagzoomClient
from ragzoom.rpc import dynamic_summary_pb2


class TestBatchAppendRequestHasGuidanceField:
    """Acceptance Criteria #1: BatchAppendTextRequest proto has summarization_guidance field."""

    def test_batch_append_request_has_summarization_guidance_field(self) -> None:
        """BatchAppendTextRequest should have a summarization_guidance field."""
        req = dynamic_summary_pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[],
        )
        # Field should exist
        assert hasattr(req, "summarization_guidance")
        # Optional field should be unset by default
        assert req.HasField("summarization_guidance") is False

    def test_batch_append_request_accepts_summarization_guidance(self) -> None:
        """BatchAppendTextRequest can accept a summarization_guidance value."""
        guidance = "Preserve identity and agency in summaries."
        req = dynamic_summary_pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[],
            summarization_guidance=guidance,
        )
        assert req.HasField("summarization_guidance") is True
        assert req.summarization_guidance == guidance

    def test_batch_append_request_guidance_is_string_type(self) -> None:
        """summarization_guidance field should be string type."""
        guidance = """This is a multiline guidance text.

It should preserve:
- Identity and agency
- Decisions and outcomes
"""
        req = dynamic_summary_pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[],
            summarization_guidance=guidance,
        )
        assert isinstance(req.summarization_guidance, str)
        assert req.summarization_guidance == guidance

    def test_batch_append_request_guidance_is_field_number_4(self) -> None:
        """summarization_guidance field should be field number 4 per the spec."""
        field = (
            dynamic_summary_pb2.BatchAppendTextRequest.DESCRIPTOR.fields_by_name.get(
                "summarization_guidance"
            )
        )
        assert field is not None
        assert field.number == 4

    def test_batch_append_request_guidance_defaults_unset(self) -> None:
        """When guidance is not provided, HasField returns False."""
        req = dynamic_summary_pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[],
        )
        # Default value is empty string but field is not set
        assert req.summarization_guidance == ""
        assert req.HasField("summarization_guidance") is False


class TestGrpcClientBatchAppendAcceptsGuidance:
    """Acceptance Criteria #2: GrpcRagzoomClient.batch_append_text() accepts summarization_guidance."""

    def test_grpc_client_batch_append_accepts_guidance(self) -> None:
        """batch_append_text method should accept summarization_guidance parameter.

        Spec: specs/transcript-summarization-guidance.md § Acceptance Criteria #2
        """
        # Verify the method signature includes summarization_guidance
        sig = inspect.signature(GrpcRagzoomClient.batch_append_text)
        assert "summarization_guidance" in sig.parameters

        # Verify the parameter has the correct type annotation and default
        param = sig.parameters["summarization_guidance"]
        assert param.default is None, "summarization_guidance should default to None"
        # Note: With `from __future__ import annotations`, annotations are strings
        assert "str" in str(param.annotation), "Should include str type"
        assert "None" in str(param.annotation), "Should include None type"

    def test_grpc_client_batch_append_guidance_is_keyword_only(self) -> None:
        """summarization_guidance should be a keyword-only parameter."""
        sig = inspect.signature(GrpcRagzoomClient.batch_append_text)
        param = sig.parameters["summarization_guidance"]
        assert (
            param.kind == inspect.Parameter.KEYWORD_ONLY
        ), "summarization_guidance should be keyword-only"

    def test_grpc_client_batch_append_guidance_sets_proto_field(self) -> None:
        """Verify the method sets summarization_guidance on the request proto.

        This verifies the implementation logic without making a network call.
        """
        # Check the source code sets the field (static analysis)
        import ragzoom.client.grpc_client as grpc_module

        source = inspect.getsource(grpc_module.GrpcRagzoomClient.batch_append_text)
        assert (
            "request.summarization_guidance" in source
        ), "Method should set request.summarization_guidance"
