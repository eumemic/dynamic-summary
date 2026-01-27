"""Tests for transcript summarization guidance feature.

Acceptance tests verifying all criteria from specs/transcript-summarization-guidance.md.
"""

from __future__ import annotations

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
