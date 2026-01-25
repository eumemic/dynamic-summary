"""Tests for summary_utils module."""

from __future__ import annotations

from ragzoom.constants import DEFAULT_SUMMARY_SYSTEM_PROMPT
from ragzoom.services.summary_utils import SummaryWorkflowConfig, prepare_summary_inputs


class TestPrepareSummaryInputsSummarizationGuidance:
    """Tests for summarization_guidance parameter in prepare_summary_inputs.

    Per specs/custom-prompt-config.md: summarization_guidance is ADDITIVE,
    appended under "# Summarization Guidance" section, never replacing the
    default base prompt.
    """

    def test_default_system_prompt_when_none(self) -> None:
        """Uses default system prompt when summarization_guidance is None."""
        result = prepare_summary_inputs(
            text="Some text to summarize",
            target_tokens=100,
        )

        system_message = next(
            (m for m in result.messages if m["role"] == "system"), None
        )
        assert system_message is not None, "Should have a system message"

        assert system_message["content"] == DEFAULT_SUMMARY_SYSTEM_PROMPT

    def test_guidance_appends_to_default_prompt(self) -> None:
        """Custom guidance is APPENDED under '# Summarization Guidance' section.

        This is the core test for the spec requirement: guidance is additive,
        not replacement. The default base prompt must always be present.
        """
        custom_guidance = (
            "This document contains legal contracts. "
            "Preserve exact legal terminology."
        )

        result = prepare_summary_inputs(
            text="Some legal text to summarize",
            target_tokens=100,
            summarization_guidance=custom_guidance,
        )

        # Find the system message
        system_message = next(
            (m for m in result.messages if m["role"] == "system"), None
        )
        assert system_message is not None, "Should have a system message"

        content = system_message["content"]

        # Base prompt must be present at the beginning
        assert content.startswith(
            DEFAULT_SUMMARY_SYSTEM_PROMPT
        ), "Default prompt must be preserved at the start"

        # Guidance section header must be present
        assert (
            "# Summarization Guidance" in content
        ), "Guidance must be under '# Summarization Guidance' section"

        # Custom guidance must be present after the header
        assert custom_guidance in content, "Custom guidance must be included"

        # Verify structure: base prompt + newlines + header + guidance
        expected = (
            f"{DEFAULT_SUMMARY_SYSTEM_PROMPT}\n\n"
            f"# Summarization Guidance\n{custom_guidance}"
        )
        assert content == expected, f"Expected:\n{expected}\n\nGot:\n{content}"

    def test_guidance_with_prev_context(self) -> None:
        """Custom guidance works correctly with prev_context."""
        custom_guidance = "Preserve medical terminology exactly as written."

        result = prepare_summary_inputs(
            text="Patient exhibited symptoms of...",
            target_tokens=50,
            prev_context="Previous medical history notes...",
            summarization_guidance=custom_guidance,
        )

        # Find the system message
        system_message = next(
            (m for m in result.messages if m["role"] == "system"), None
        )
        assert system_message is not None, "Should have a system message"

        content = system_message["content"]

        # Both base prompt and custom guidance should be present
        assert content.startswith(DEFAULT_SUMMARY_SYSTEM_PROMPT)
        assert "# Summarization Guidance" in content
        assert custom_guidance in content

    def test_guidance_with_anti_verbatim_vaccine(self) -> None:
        """Custom guidance is preserved when anti_verbatim_vaccine is enabled."""
        custom_guidance = "Preserve code structure and function names."

        result = prepare_summary_inputs(
            text="def foo(): pass",
            target_tokens=50,
            use_anti_verbatim_vaccine=True,
            summarization_guidance=custom_guidance,
        )

        # System message should be first
        assert result.messages[0]["role"] == "system"

        content = result.messages[0]["content"]
        assert content.startswith(DEFAULT_SUMMARY_SYSTEM_PROMPT)
        assert custom_guidance in content

        # Anti-verbatim vaccine adds assistant and user messages
        assert (
            len(result.messages) == 4
        )  # system, user, assistant (vaccine), user (vaccine)

    def test_empty_string_guidance_uses_default_prompt_only(self) -> None:
        """Empty string guidance is treated as no guidance (falsy)."""
        result = prepare_summary_inputs(
            text="Some text",
            target_tokens=100,
            summarization_guidance="",
        )

        system_message = next(
            (m for m in result.messages if m["role"] == "system"), None
        )
        assert system_message is not None

        # Empty string is falsy, so only default prompt should be used
        assert system_message["content"] == DEFAULT_SUMMARY_SYSTEM_PROMPT

    def test_whitespace_only_guidance_uses_default_prompt_only(self) -> None:
        """Whitespace-only guidance is treated as no guidance."""
        result = prepare_summary_inputs(
            text="Some text",
            target_tokens=100,
            summarization_guidance="   \n\t  ",
        )

        system_message = next(
            (m for m in result.messages if m["role"] == "system"), None
        )
        assert system_message is not None

        # Whitespace-only is treated as empty
        assert system_message["content"] == DEFAULT_SUMMARY_SYSTEM_PROMPT

    def test_multiline_guidance(self) -> None:
        """Multiline guidance is preserved correctly."""
        custom_guidance = """This document contains legal contracts.
Preserve exact legal terminology.
Pay attention to dates and amounts."""

        result = prepare_summary_inputs(
            text="Contract text here",
            target_tokens=100,
            summarization_guidance=custom_guidance,
        )

        system_message = next(
            (m for m in result.messages if m["role"] == "system"), None
        )
        assert system_message is not None

        content = system_message["content"]
        assert custom_guidance in content
        assert content.startswith(DEFAULT_SUMMARY_SYSTEM_PROMPT)


class TestSummaryWorkflowConfigSummarizationGuidance:
    """Tests for summarization_guidance field in SummaryWorkflowConfig."""

    def test_workflow_config_has_summarization_guidance(self) -> None:
        """SummaryWorkflowConfig has summarization_guidance field with None default."""
        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=2,
            retry_threshold=0.2,
        )

        # Field exists and defaults to None
        assert hasattr(config, "summarization_guidance")
        assert config.summarization_guidance is None

    def test_workflow_config_accepts_summarization_guidance(self) -> None:
        """SummaryWorkflowConfig accepts custom summarization_guidance."""
        custom_guidance = "Preserve legal terminology exactly as written."

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=2,
            retry_threshold=0.2,
            summarization_guidance=custom_guidance,
        )

        assert config.summarization_guidance == custom_guidance

    def test_workflow_config_is_frozen(self) -> None:
        """SummaryWorkflowConfig is frozen (immutable)."""
        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=2,
            retry_threshold=0.2,
            summarization_guidance="test",
        )

        # Attempting to modify should raise
        try:
            config.summarization_guidance = "new value"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass  # Expected - frozen dataclass
