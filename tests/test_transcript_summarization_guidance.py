"""Tests for transcript summarization guidance feature.

Acceptance tests verifying all criteria from specs/transcript-summarization-guidance.md.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING

import pytest

from ragzoom.client.grpc_client import GrpcRagzoomClient
from ragzoom.rpc import dynamic_summary_pb2
from ragzoom.wrapper import RagZoom

if TYPE_CHECKING:
    from ragzoom.server.state import ServerState


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


class TestGuidanceStoredOnDocument:
    """Acceptance Criteria #4: Guidance is threaded to the summarizer.

    Tests verify that summarization_guidance is stored on the document record
    after batch_append() is called with guidance.

    Spec: specs/transcript-summarization-guidance.md § Acceptance Criteria #4
    """

    @pytest.mark.asyncio
    async def test_guidance_stored_on_document_via_grpc(
        self,
        grpc_test_environment: tuple[str, ServerState],
    ) -> None:
        """Document record contains summarization_guidance after batch_append with guidance.

        This is an end-to-end test using the gRPC client path:
        GrpcRagzoomClient.batch_append_text() → BatchAppendText servicer → storage
        """
        address, state = grpc_test_environment
        document_id = "guidance-storage-test"
        custom_guidance = (
            "This is a conversation transcript.\n"
            "Preserve identity, agency, and decision outcomes."
        )

        client = GrpcRagzoomClient(address)
        try:
            await asyncio.to_thread(
                client.batch_append_text,
                document_id=document_id,
                units=["Turn 1: User asks a question", "Turn 2: Assistant responds"],
                summarization_guidance=custom_guidance,
            )

            # Verify document was created with guidance stored
            doc = state.store.get_document_by_id(document_id)
            assert doc is not None, "Document should be created"
            assert doc.summarization_guidance == custom_guidance
        finally:
            client.close()


class TestWrapperBatchAppendAcceptsGuidance:
    """Acceptance Criteria #3: RagZoom.batch_append() accepts summarization_guidance."""

    def test_wrapper_batch_append_accepts_guidance(self) -> None:
        """batch_append method should accept summarization_guidance parameter.

        Spec: specs/transcript-summarization-guidance.md § Acceptance Criteria #3
        """
        # Verify the method signature includes summarization_guidance
        sig = inspect.signature(RagZoom.batch_append)
        assert "summarization_guidance" in sig.parameters

        # Verify the parameter has the correct type annotation and default
        param = sig.parameters["summarization_guidance"]
        assert param.default is None, "summarization_guidance should default to None"
        # Note: With `from __future__ import annotations`, annotations are strings
        assert "str" in str(param.annotation), "Should include str type"
        assert "None" in str(param.annotation), "Should include None type"

    def test_wrapper_batch_append_guidance_is_keyword_only(self) -> None:
        """summarization_guidance should be a keyword-only parameter."""
        sig = inspect.signature(RagZoom.batch_append)
        param = sig.parameters["summarization_guidance"]
        assert (
            param.kind == inspect.Parameter.KEYWORD_ONLY
        ), "summarization_guidance should be keyword-only"

    def test_wrapper_batch_append_threads_to_runtime(self) -> None:
        """Verify batch_append passes summarization_guidance to runtime session.

        This verifies the implementation logic via static analysis.
        """
        import ragzoom.wrapper as wrapper_module

        source = inspect.getsource(wrapper_module.RagZoom.batch_append)
        # Should pass guidance to session.batch_append_text()
        assert (
            "summarization_guidance=summarization_guidance" in source
        ), "Method should pass summarization_guidance to session"

    def test_wrapper_batch_append_threads_to_grpc_client(self) -> None:
        """Verify batch_append passes summarization_guidance to gRPC client.

        This verifies the implementation logic via static analysis.
        """
        import ragzoom.wrapper as wrapper_module

        source = inspect.getsource(wrapper_module.RagZoom.batch_append)
        # Should pass guidance to client.batch_append_text()
        assert (
            "summarization_guidance=summarization_guidance" in source
        ), "Method should pass summarization_guidance to client"
