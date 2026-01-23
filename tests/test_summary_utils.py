"""Tests for summary_utils module."""

from __future__ import annotations

from ragzoom.services.summary_utils import prepare_summary_inputs


class TestPrepareSummaryInputsSystemPrompt:
    """Tests for system_prompt parameter in prepare_summary_inputs."""

    def test_default_system_prompt_when_none(self) -> None:
        """Uses default system prompt when system_prompt is None."""
        result = prepare_summary_inputs(
            text="Some text to summarize",
            target_tokens=100,
        )

        # Find the system message
        system_message = next(
            (m for m in result.messages if m["role"] == "system"), None
        )
        assert system_message is not None, "Should have a system message"

        # Verify default prompt content
        expected_default = (
            "You are a text compressor. You compress sections of documents while "
            "preserving their meaning. You output ONLY the compressed text, nothing else."
        )
        assert system_message["content"] == expected_default

    def test_prepare_summary_inputs_uses_custom_system_prompt(self) -> None:
        """Custom system_prompt is used in messages when provided."""
        custom_prompt = (
            "You are a legal document summarizer. Preserve exact legal terminology. "
            "Output ONLY the compressed text, nothing else."
        )

        result = prepare_summary_inputs(
            text="Some legal text to summarize",
            target_tokens=100,
            system_prompt=custom_prompt,
        )

        # Find the system message
        system_message = next(
            (m for m in result.messages if m["role"] == "system"), None
        )
        assert system_message is not None, "Should have a system message"
        assert system_message["content"] == custom_prompt

    def test_custom_system_prompt_with_prev_context(self) -> None:
        """Custom system_prompt works correctly with prev_context."""
        custom_prompt = "You are a medical note summarizer."

        result = prepare_summary_inputs(
            text="Patient exhibited symptoms of...",
            target_tokens=50,
            prev_context="Previous medical history notes...",
            system_prompt=custom_prompt,
        )

        # Find the system message
        system_message = next(
            (m for m in result.messages if m["role"] == "system"), None
        )
        assert system_message is not None, "Should have a system message"
        assert system_message["content"] == custom_prompt

    def test_custom_system_prompt_with_anti_verbatim_vaccine(self) -> None:
        """Custom system_prompt is preserved when anti_verbatim_vaccine is enabled."""
        custom_prompt = "You are a code documentation summarizer."

        result = prepare_summary_inputs(
            text="def foo(): pass",
            target_tokens=50,
            use_anti_verbatim_vaccine=True,
            system_prompt=custom_prompt,
        )

        # System message should be first
        assert result.messages[0]["role"] == "system"
        assert result.messages[0]["content"] == custom_prompt

        # Anti-verbatim vaccine adds assistant and user messages
        assert (
            len(result.messages) == 4
        )  # system, user, assistant (vaccine), user (vaccine)

    def test_empty_string_system_prompt_uses_empty_string(self) -> None:
        """Empty string system_prompt is used as-is (not treated as None)."""
        result = prepare_summary_inputs(
            text="Some text",
            target_tokens=100,
            system_prompt="",
        )

        system_message = next(
            (m for m in result.messages if m["role"] == "system"), None
        )
        assert system_message is not None
        # Empty string should be used (falsy but explicit)
        assert system_message["content"] == ""
