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


class TestPrepareEmbeddingTextInputs:
    """Tests for prepare_embedding_text_inputs function.

    Per specs/embedding-text-optimization.md: produces retrieval-optimized text
    for embedding. The prompt should preserve key terms, named entities, and
    concepts that users might query via cosine similarity search.
    """

    def test_prepare_embedding_text_inputs_returns_summary_preparation(self) -> None:
        """Returns a SummaryPreparation with messages for LLM."""
        from ragzoom.services.summary_utils import (
            SummaryPreparation,
            prepare_embedding_text_inputs,
        )

        result = prepare_embedding_text_inputs(
            preceding_context="User discussed authentication issues.",
            leaf_text="Then user asked about JWT tokens and OAuth.",
            target_tokens=200,
        )

        assert isinstance(result, SummaryPreparation)
        assert result.messages is not None
        assert len(result.messages) >= 2  # At least system and user messages

    def test_prepare_embedding_text_inputs_system_prompt_for_retrieval(self) -> None:
        """System prompt instructs LLM to optimize for semantic search."""
        from ragzoom.services.summary_utils import prepare_embedding_text_inputs

        result = prepare_embedding_text_inputs(
            preceding_context="Background context here.",
            leaf_text="Main content about authentication.",
            target_tokens=100,
        )

        system_message = next(
            (m for m in result.messages if m["role"] == "system"), None
        )
        assert system_message is not None

        content = system_message["content"].lower()
        # Prompt should mention semantic search / retrieval optimization
        assert "semantic search" in content or "retrieval" in content
        # Prompt should mention preserving key concepts
        assert "key" in content or "concept" in content or "term" in content

    def test_prepare_embedding_text_inputs_user_prompt_structure(self) -> None:
        """User prompt includes context, target text, and word limit."""
        from ragzoom.services.summary_utils import prepare_embedding_text_inputs

        result = prepare_embedding_text_inputs(
            preceding_context="User discussed Python programming.",
            leaf_text="Then user asked about decorators and generators.",
            target_tokens=100,
        )

        user_message = next((m for m in result.messages if m["role"] == "user"), None)
        assert user_message is not None

        content = user_message["content"]
        # Should contain both context and target text
        assert "Python programming" in content
        assert "decorators and generators" in content
        # Should have word limit instruction
        assert "word" in content.lower()

    def test_prepare_embedding_text_inputs_prioritizes_target_over_context(
        self,
    ) -> None:
        """Prompt instructs LLM to prioritize target (leaf) over context."""
        from ragzoom.services.summary_utils import prepare_embedding_text_inputs

        result = prepare_embedding_text_inputs(
            preceding_context="Some background context.",
            leaf_text="The actual important content.",
            target_tokens=100,
        )

        user_message = next((m for m in result.messages if m["role"] == "user"), None)
        assert user_message is not None

        content = user_message["content"].lower()
        # Should instruct prioritization of target/leaf content
        assert "prioritize" in content or "priority" in content or "target" in content

    def test_prepare_embedding_text_inputs_combined_tokens_count(self) -> None:
        """combined_tokens reflects the combined input size."""
        from ragzoom.services.summary_utils import prepare_embedding_text_inputs
        from ragzoom.utils.tokenization import tokenizer

        context = "Short context."
        leaf = "Short leaf text."

        result = prepare_embedding_text_inputs(
            preceding_context=context,
            leaf_text=leaf,
            target_tokens=100,
        )

        # Combined tokens should be approximately the sum
        expected_combined = tokenizer.count_tokens(f"{context}\n{leaf}")
        assert result.combined_tokens == expected_combined

    def test_prepare_embedding_text_inputs_no_context(self) -> None:
        """Works correctly when preceding_context is empty."""
        from ragzoom.services.summary_utils import prepare_embedding_text_inputs

        result = prepare_embedding_text_inputs(
            preceding_context="",
            leaf_text="Just the leaf content about API design.",
            target_tokens=100,
        )

        # Should still produce valid messages
        assert len(result.messages) >= 2

        user_message = next((m for m in result.messages if m["role"] == "user"), None)
        assert user_message is not None
        assert "API design" in user_message["content"]
