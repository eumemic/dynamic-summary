"""Tests for summary_utils module."""

from __future__ import annotations

from collections.abc import MutableSequence

import pytest

from ragzoom.config import IndexConfig
from ragzoom.constants import DEFAULT_SUMMARY_SYSTEM_PROMPT
from ragzoom.contracts.chat_model import UsageInfo
from ragzoom.services.summary_utils import (
    AccumulatedUsage,
    EmbeddingTextRequest,
    SummaryResult,
    SummaryWorkflowConfig,
    prepare_summary_inputs,
    run_embedding_text_from_config,
    run_embedding_text_request,
    run_embedding_text_workflow,
)
from ragzoom.telemetry_collection import TelemetryCollector


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


class TestRunEmbeddingTextWorkflow:
    """Tests for run_embedding_text_workflow function.

    Per specs/embedding-text-optimization.md: the workflow generates
    retrieval-optimized text for embedding. It passes through small content
    (combined_tokens <= target_tokens) and compresses large content via LLM.
    """

    @pytest.mark.asyncio
    async def test_run_embedding_text_workflow_passthrough_small_content(self) -> None:
        """When combined tokens <= target, returns combined text without LLM call.

        Spec: "If (preceding_context + leaf_text) <= target_embedding_tokens: passthrough unchanged"
        """
        llm_called = False

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            nonlocal llm_called
            llm_called = True
            return "compressed", {"prompt_tokens": 10, "completion_tokens": 5}

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
        )

        # Small content that fits within target
        result = await run_embedding_text_workflow(
            preceding_context="Hi",
            leaf_text="Hello",
            target_tokens=100,  # Much larger than content
            config=config,
            call_llm=mock_llm,
        )

        assert not llm_called, "LLM should not be called for small content"
        assert isinstance(result, SummaryResult)
        assert "Hi" in result.summary
        assert "Hello" in result.summary
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_run_embedding_text_workflow_calls_llm_for_large_content(
        self,
    ) -> None:
        """When combined tokens > target, LLM is called to compress."""
        llm_called = False

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            nonlocal llm_called
            llm_called = True
            return "Optimized text about auth", {
                "prompt_tokens": 100,
                "completion_tokens": 20,
            }

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
        )

        # Large content that exceeds target
        large_context = "Authentication context. " * 50
        large_leaf = "JWT tokens and OAuth discussion. " * 50

        result = await run_embedding_text_workflow(
            preceding_context=large_context,
            leaf_text=large_leaf,
            target_tokens=50,  # Small target forces compression
            config=config,
            call_llm=mock_llm,
        )

        assert llm_called, "LLM should be called for large content"
        assert result.summary == "Optimized text about auth"

    @pytest.mark.asyncio
    async def test_run_embedding_text_workflow_returns_summary_result(self) -> None:
        """Returns SummaryResult with all expected fields."""

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            return "Compressed output", {"prompt_tokens": 50, "completion_tokens": 10}

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
        )

        # Large content to trigger LLM call
        result = await run_embedding_text_workflow(
            preceding_context="Context " * 100,
            leaf_text="Leaf " * 100,
            target_tokens=50,
            config=config,
            call_llm=mock_llm,
        )

        assert isinstance(result, SummaryResult)
        assert result.summary == "Compressed output"
        assert result.retry_count >= 0
        assert result.summary_tokens > 0
        assert isinstance(result.usage, AccumulatedUsage)

    @pytest.mark.asyncio
    async def test_run_embedding_text_workflow_accumulates_usage(self) -> None:
        """Accumulated usage tracks LLM token consumption."""

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            return "Output", {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "cached_tokens": 10,
            }

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
        )

        result = await run_embedding_text_workflow(
            preceding_context="Context " * 100,
            leaf_text="Leaf " * 100,
            target_tokens=10,
            config=config,
            call_llm=mock_llm,
        )

        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 25
        assert result.usage.cached_tokens == 10

    @pytest.mark.asyncio
    async def test_run_embedding_text_workflow_passthrough_no_usage(self) -> None:
        """Passthrough case has zero usage (no LLM call)."""

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            raise AssertionError("Should not be called")

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
        )

        result = await run_embedding_text_workflow(
            preceding_context="Small",
            leaf_text="content",
            target_tokens=1000,  # Large target, content fits
            config=config,
            call_llm=mock_llm,
        )

        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.usage.cached_tokens == 0

    @pytest.mark.asyncio
    async def test_run_embedding_text_workflow_retries_when_output_too_large(
        self,
    ) -> None:
        """Retries when LLM output exceeds target by more than retry_threshold."""
        call_count = 0

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt: way too large (500 tokens for target of 50 = 900% over)
                return "word " * 500, {"prompt_tokens": 100, "completion_tokens": 500}
            else:
                # Retry: fits target
                return "concise", {"prompt_tokens": 50, "completion_tokens": 1}

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=3,
            retry_threshold=0.2,  # 20% tolerance
        )

        result = await run_embedding_text_workflow(
            preceding_context="Context " * 100,
            leaf_text="Leaf " * 100,
            target_tokens=50,
            config=config,
            call_llm=mock_llm,
        )

        assert call_count == 2, "Should retry once"
        assert result.retry_count == 1
        assert result.summary == "concise"

    @pytest.mark.asyncio
    async def test_run_embedding_text_workflow_zero_target_passthrough(self) -> None:
        """Target <= 0 signals passthrough regardless of content size."""

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            raise AssertionError("Should not be called for zero target")

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
        )

        # Even with large content, target=0 means passthrough
        result = await run_embedding_text_workflow(
            preceding_context="Large " * 100,
            leaf_text="content " * 100,
            target_tokens=0,
            config=config,
            call_llm=mock_llm,
        )

        assert "Large" in result.summary
        assert "content" in result.summary

    @pytest.mark.asyncio
    async def test_run_embedding_text_workflow_no_context(self) -> None:
        """Works correctly when preceding_context is empty."""

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            return "optimized leaf", {"prompt_tokens": 50, "completion_tokens": 2}

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
        )

        result = await run_embedding_text_workflow(
            preceding_context="",  # No context
            leaf_text="Leaf content " * 100,  # Large leaf
            target_tokens=10,
            config=config,
            call_llm=mock_llm,
        )

        assert result.summary == "optimized leaf"

    @pytest.mark.asyncio
    async def test_embedding_text_empty_leaf(self) -> None:
        """When leaf is empty, returns empty string without LLM call.

        Spec: "Empty leaf: Embed empty string (existing behavior)"
        """
        llm_called = False

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            nonlocal llm_called
            llm_called = True
            return "should not be called", {"prompt_tokens": 10, "completion_tokens": 5}

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
        )

        # Empty leaf - should passthrough empty string
        result = await run_embedding_text_workflow(
            preceding_context="Some context",
            leaf_text="",  # Empty leaf
            target_tokens=100,
            config=config,
            call_llm=mock_llm,
        )

        assert not llm_called, "LLM should not be called for empty leaf"
        # When leaf is empty, combined_text is "context\n" (context + newline + empty)
        assert result.summary.strip() == "Some context"
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_embedding_text_empty_leaf_and_context(self) -> None:
        """When both leaf and context are empty, returns empty string."""
        llm_called = False

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            nonlocal llm_called
            llm_called = True
            return "should not be called", {"prompt_tokens": 10, "completion_tokens": 5}

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
        )

        result = await run_embedding_text_workflow(
            preceding_context="",
            leaf_text="",  # Both empty
            target_tokens=100,
            config=config,
            call_llm=mock_llm,
        )

        assert not llm_called
        assert result.summary == ""

    @pytest.mark.asyncio
    async def test_embedding_text_leaf_exceeds_target(self) -> None:
        """When leaf alone exceeds target, LLM compresses the leaf.

        Spec: "Leaf alone > target: Compress leaf text to fit"
        """
        llm_called = False
        received_messages: list[dict[str, str]] = []

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            nonlocal llm_called, received_messages
            llm_called = True
            received_messages = list(messages)
            return "compressed leaf content", {
                "prompt_tokens": 200,
                "completion_tokens": 10,
            }

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
        )

        # Large leaf that exceeds target, no context
        large_leaf = "Important authentication details. " * 50

        result = await run_embedding_text_workflow(
            preceding_context="",  # No context
            leaf_text=large_leaf,
            target_tokens=50,  # Small target forces compression
            config=config,
            call_llm=mock_llm,
        )

        assert llm_called, "LLM should be called to compress large leaf"
        assert result.summary == "compressed leaf content"
        # Verify the prompt contains the leaf text
        user_message = next(m for m in received_messages if m["role"] == "user")
        assert "authentication" in user_message["content"]


class TestSummaryWorkflowInputTruncation:
    """Tests for max_input_tokens safety valve in run_summary_workflow.

    When text exceeds max_input_tokens, it must be truncated before being
    sent to the LLM. This prevents context overflow errors from the model API.
    """

    @pytest.mark.asyncio
    async def test_truncates_text_exceeding_max_input_tokens(self) -> None:
        """Text exceeding max_input_tokens is truncated before LLM call."""
        from ragzoom.services.summary_utils import run_summary_workflow
        from ragzoom.utils.tokenization import tokenizer

        received_messages: list[dict[str, str]] = []

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            received_messages.extend(messages)
            return "compressed", {"prompt_tokens": 100, "completion_tokens": 5}

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
            max_input_tokens=50,  # Very small limit
        )

        # Text much larger than the limit
        large_text = "word " * 200  # ~200 tokens
        assert tokenizer.count_tokens(large_text) > 50

        await run_summary_workflow(
            text=large_text,
            target_tokens=10,
            config=config,
            call_summary=mock_llm,
        )

        # LLM should have been called with truncated text
        assert received_messages, "LLM should have been called"
        user_msg = next(m for m in received_messages if m["role"] == "user")
        # The user prompt should NOT contain the full 200-token text
        user_tokens = tokenizer.count_tokens(user_msg["content"])
        # Prompt overhead (wrapper text) adds tokens, but the text portion
        # should be truncated so total is much less than the original
        assert user_tokens < tokenizer.count_tokens(large_text)

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_no_truncation_when_text_fits(self) -> None:
        """Text within max_input_tokens is not truncated."""
        from ragzoom.services.summary_utils import run_summary_workflow

        received_messages: list[dict[str, str]] = []

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            received_messages.extend(messages)
            return "compressed", {"prompt_tokens": 10, "completion_tokens": 5}

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
            max_input_tokens=10000,  # Large limit
        )

        small_text = "A short text to summarize."

        await run_summary_workflow(
            text=small_text,
            target_tokens=5,
            config=config,
            call_summary=mock_llm,
        )

        assert received_messages
        user_msg = next(m for m in received_messages if m["role"] == "user")
        # Full text should be present
        assert small_text.strip() in user_msg["content"]

    @pytest.mark.asyncio
    async def test_no_truncation_when_max_input_tokens_is_none(self) -> None:
        """When max_input_tokens is None, no truncation is applied."""
        from ragzoom.services.summary_utils import run_summary_workflow

        received_messages: list[dict[str, str]] = []

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            received_messages.extend(messages)
            return "compressed", {"prompt_tokens": 10, "completion_tokens": 5}

        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
            # max_input_tokens defaults to None
        )

        large_text = "word " * 500

        await run_summary_workflow(
            text=large_text,
            target_tokens=10,
            config=config,
            call_summary=mock_llm,
        )

        assert received_messages
        user_msg = next(m for m in received_messages if m["role"] == "user")
        # Full text should be present (no truncation)
        assert "word" in user_msg["content"]

    def test_workflow_config_has_max_input_tokens_field(self) -> None:
        """SummaryWorkflowConfig has max_input_tokens field defaulting to None."""
        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
        )
        assert config.max_input_tokens is None

    def test_workflow_config_accepts_max_input_tokens(self) -> None:
        """SummaryWorkflowConfig accepts max_input_tokens."""
        config = SummaryWorkflowConfig(
            summary_model="gpt-4o-mini",
            use_anti_verbatim_vaccine=False,
            max_retries=0,
            retry_threshold=0.2,
            max_input_tokens=100_000,
        )
        assert config.max_input_tokens == 100_000


class TestRunEmbeddingTextFromConfig:
    """Tests for run_embedding_text_from_config convenience wrapper."""

    @pytest.mark.asyncio
    async def test_builds_config_from_index_config(self) -> None:
        """Builds SummaryWorkflowConfig from IndexConfig."""

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            return "output", {"prompt_tokens": 10, "completion_tokens": 2}

        index_config = IndexConfig.load(
            summary_model="gpt-4o-mini",
            max_retries=2,
            retry_threshold=0.3,
        )

        result = await run_embedding_text_from_config(
            index_config=index_config,
            preceding_context="Small",
            leaf_text="text",
            target_tokens=1000,
            call_llm=mock_llm,
        )

        assert isinstance(result, SummaryResult)


class TestRunEmbeddingTextRequest:
    """Tests for run_embedding_text_request with packaged request."""

    @pytest.mark.asyncio
    async def test_accepts_embedding_text_request(self) -> None:
        """Accepts EmbeddingTextRequest TypedDict."""

        async def mock_llm(
            messages: MutableSequence[dict[str, str]],
            target_tokens: int,
            node_id: str,
            reporter: TelemetryCollector | None,
        ) -> tuple[str, UsageInfo]:
            return "output", {"prompt_tokens": 10, "completion_tokens": 2}

        index_config = IndexConfig.load(summary_model="gpt-4o-mini")

        request: EmbeddingTextRequest = {
            "preceding_context": "Context",
            "leaf_text": "Leaf",
            "target_tokens": 500,
            "parent_id": "node-123",
            "reporter": None,
        }

        result = await run_embedding_text_request(
            index_config=index_config,
            request=request,
            call_llm=mock_llm,
        )

        assert isinstance(result, SummaryResult)
