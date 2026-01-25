"""Integration tests for custom prompt threading through summary workflow.

These tests verify that the summarization_guidance flows from IndexConfig
all the way to the LLM API call, as specified in specs/custom-prompt-config.md.

KEY SEMANTIC: summarization_guidance is ADDITIVE - it gets appended under a
"# Summarization Guidance" section, never replacing the default base prompt.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableSequence
from unittest.mock import MagicMock

import pytest

from ragzoom.config import IndexConfig
from ragzoom.constants import DEFAULT_SUMMARY_SYSTEM_PROMPT
from ragzoom.contracts.chat_model import UsageInfo
from ragzoom.services.summary_utils import (
    SummaryWorkflowConfig,
    run_summary_from_config,
    run_summary_workflow,
)
from ragzoom.telemetry_collection import TelemetryCollector


def create_message_capture() -> tuple[
    list[dict[str, str]],
    Callable[
        [MutableSequence[dict[str, str]], int, str, TelemetryCollector | None],
        Awaitable[tuple[str, UsageInfo]],
    ],
]:
    """Create a message capture list and mock call_summary function.

    Returns:
        Tuple of (captured_messages, mock_call_summary) where mock_call_summary
        captures all messages sent to it in captured_messages.
    """
    captured_messages: list[dict[str, str]] = []

    async def mock_call_summary(
        messages: MutableSequence[dict[str, str]],
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None,
    ) -> tuple[str, UsageInfo]:
        captured_messages.extend(messages)
        return "Summarized text", UsageInfo(
            prompt_tokens=400, completion_tokens=80, cached_tokens=0
        )

    return captured_messages, mock_call_summary


def get_system_message(messages: list[dict[str, str]]) -> dict[str, str] | None:
    """Extract the system message from a list of messages."""
    return next((m for m in messages if m["role"] == "system"), None)


def assert_system_prompt(messages: list[dict[str, str]], expected_prompt: str) -> None:
    """Assert that messages contain a system message with the expected prompt."""
    assert len(messages) > 0, "Messages should have been sent to LLM"
    system_message = get_system_message(messages)
    assert system_message is not None, "Should have a system message"
    assert (
        system_message["content"] == expected_prompt
    ), f"Expected prompt: {expected_prompt!r}, Got: {system_message['content']!r}"


def assert_guidance_appended(messages: list[dict[str, str]], guidance: str) -> None:
    """Assert that guidance was appended under '# Summarization Guidance' section.

    Verifies the additive semantic: base prompt + header + guidance.
    """
    assert len(messages) > 0, "Messages should have been sent to LLM"
    system_message = get_system_message(messages)
    assert system_message is not None, "Should have a system message"

    content = system_message["content"]

    # Base prompt must be at the start
    assert content.startswith(DEFAULT_SUMMARY_SYSTEM_PROMPT), (
        f"System prompt must start with default base prompt.\n"
        f"Got: {content[:100]}..."
    )

    # Guidance header must be present
    assert (
        "# Summarization Guidance" in content
    ), "Custom guidance must be under '# Summarization Guidance' section"

    # Custom guidance must be present
    assert guidance in content, f"Custom guidance '{guidance}' must be in system prompt"

    # Verify exact structure
    expected = (
        f"{DEFAULT_SUMMARY_SYSTEM_PROMPT}\n\n" f"# Summarization Guidance\n{guidance}"
    )
    assert content == expected, f"Expected:\n{expected}\n\nGot:\n{content}"


@pytest.mark.asyncio
async def test_guidance_appended_to_base_prompt() -> None:
    """Custom guidance flows from IndexConfig and gets APPENDED to base prompt.

    This integration test verifies the complete flow:
    IndexConfig.summarization_guidance → SummaryWorkflowConfig
    → run_summary_workflow → prepare_summary_inputs → messages[0]["content"]
    → call_summary

    KEY: Guidance is ADDITIVE, not replacement.
    """
    custom_guidance = (
        "This document contains legal contracts. "
        "Preserve exact legal terminology and citations."
    )

    config = IndexConfig.load()
    object.__setattr__(config, "summarization_guidance", custom_guidance)

    text = "This is a legal document about contract law. " * 100
    captured_messages, mock_call_summary = create_message_capture()

    await run_summary_from_config(
        index_config=config,
        text=text,
        target_tokens=100,
        call_summary=mock_call_summary,
        reporter=MagicMock(),
        parent_id="test-node",
        prev_context=None,
        text_tokens=None,
    )

    assert_guidance_appended(captured_messages, custom_guidance)


@pytest.mark.asyncio
async def test_default_prompt_used_when_no_guidance() -> None:
    """Default system prompt used when IndexConfig.summarization_guidance is None."""
    config = IndexConfig.load()
    assert config.summarization_guidance is None

    text = "This is some text to summarize. " * 100
    captured_messages, mock_call_summary = create_message_capture()

    await run_summary_from_config(
        index_config=config,
        text=text,
        target_tokens=100,
        call_summary=mock_call_summary,
        reporter=MagicMock(),
        parent_id="test-node",
        prev_context=None,
        text_tokens=None,
    )

    assert_system_prompt(captured_messages, DEFAULT_SUMMARY_SYSTEM_PROMPT)


@pytest.mark.asyncio
async def test_guidance_via_workflow_config() -> None:
    """Custom guidance works when using SummaryWorkflowConfig directly."""
    custom_guidance = "Preserve clinical terminology exactly as written."

    config = SummaryWorkflowConfig(
        summary_model="gpt-4o-mini",
        use_anti_verbatim_vaccine=False,
        max_retries=2,
        retry_threshold=0.2,
        summarization_guidance=custom_guidance,
    )

    text = "Patient presents with symptoms of... " * 100
    captured_messages, mock_call_summary = create_message_capture()

    await run_summary_workflow(
        text=text,
        target_tokens=100,
        config=config,
        call_summary=mock_call_summary,
        reporter=MagicMock(),
        parent_id="test-node",
        prev_context=None,
        text_tokens=None,
    )

    assert_guidance_appended(captured_messages, custom_guidance)


@pytest.mark.asyncio
async def test_guidance_with_preceding_context() -> None:
    """Custom guidance is appended correctly when preceding context is provided."""
    custom_guidance = "Preserve function names and code structure."

    config = IndexConfig.load()
    object.__setattr__(config, "summarization_guidance", custom_guidance)

    text = "def calculate_sum(a, b): return a + b " * 50
    prev_context = "This module provides mathematical utilities."
    captured_messages, mock_call_summary = create_message_capture()

    await run_summary_from_config(
        index_config=config,
        text=text,
        target_tokens=80,
        call_summary=mock_call_summary,
        reporter=MagicMock(),
        parent_id="test-node",
        prev_context=prev_context,
        text_tokens=None,
    )

    assert_guidance_appended(captured_messages, custom_guidance)

    user_messages = [m for m in captured_messages if m["role"] == "user"]
    assert len(user_messages) > 0
    user_content = user_messages[0]["content"]
    assert (
        "Context from previous section" in user_content or prev_context in user_content
    )


@pytest.mark.asyncio
async def test_guidance_override_in_run_summary_request() -> None:
    """Custom guidance passed via summarization_guidance overrides IndexConfig.

    This tests the explicit override parameter that will be used when
    the document's stored custom guidance differs from IndexConfig.
    Spec: specs/custom-prompt-config.md § Implementation
    """
    from ragzoom.services.summary_utils import SummaryRequest, run_summary_request

    # IndexConfig has NO custom guidance set
    config = IndexConfig.load()
    assert config.summarization_guidance is None

    # But we pass an explicit override
    document_custom_guidance = (
        "This document contains legal contracts. " "Preserve exact legal terminology."
    )

    text = "This is legal contract text. " * 100
    captured_messages, mock_call_summary = create_message_capture()

    request: SummaryRequest = {
        "text": text,
        "target_tokens": 100,
        "prev_context": None,
        "text_tokens": None,
        "parent_id": "test-node",
        "reporter": MagicMock(),
    }

    await run_summary_request(
        index_config=config,
        request=request,
        call_summary=mock_call_summary,
        summarization_guidance=document_custom_guidance,  # Override
    )

    # Document's custom guidance should be appended to base prompt
    assert_guidance_appended(captured_messages, document_custom_guidance)
