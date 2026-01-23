"""Integration tests for custom prompt threading through summary workflow.

These tests verify that the custom system prompt flows from IndexConfig
all the way to the LLM API call, as specified in specs/custom-prompt-config.md.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableSequence
from unittest.mock import MagicMock

import pytest

from ragzoom.config import IndexConfig
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


@pytest.mark.asyncio
async def test_custom_prompt_reaches_llm_call() -> None:
    """Custom prompt flows from IndexConfig to the actual LLM API call.

    This integration test verifies the complete flow:
    IndexConfig.summary_system_prompt → SummaryWorkflowConfig
    → run_summary_workflow → prepare_summary_inputs → messages[0]["content"]
    → call_summary
    """
    custom_prompt = (
        "You are a legal document summarizer. Preserve exact legal terminology "
        "and citations. Output ONLY the compressed text, nothing else."
    )

    config = IndexConfig.load()
    object.__setattr__(config, "summary_system_prompt", custom_prompt)

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

    assert_system_prompt(captured_messages, custom_prompt)


@pytest.mark.asyncio
async def test_default_prompt_used_when_none() -> None:
    """Default system prompt used when IndexConfig.summary_system_prompt is None."""
    config = IndexConfig.load()
    assert config.summary_system_prompt is None

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

    expected_default = (
        "You are a text compressor. You compress sections of documents while "
        "preserving their meaning. You output ONLY the compressed text, nothing else."
    )
    assert_system_prompt(captured_messages, expected_default)


@pytest.mark.asyncio
async def test_custom_prompt_via_workflow_config() -> None:
    """Custom prompt works when using SummaryWorkflowConfig directly."""
    custom_prompt = "You are a medical note summarizer. Preserve clinical terminology."

    config = SummaryWorkflowConfig(
        summary_model="gpt-4o-mini",
        use_anti_verbatim_vaccine=False,
        max_retries=2,
        retry_threshold=0.2,
        summary_system_prompt=custom_prompt,
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

    assert_system_prompt(captured_messages, custom_prompt)


@pytest.mark.asyncio
async def test_custom_prompt_with_preceding_context() -> None:
    """Custom prompt is used correctly when preceding context is provided."""
    custom_prompt = "You are a code documentation summarizer."

    config = IndexConfig.load()
    object.__setattr__(config, "summary_system_prompt", custom_prompt)

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

    assert_system_prompt(captured_messages, custom_prompt)

    user_messages = [m for m in captured_messages if m["role"] == "user"]
    assert len(user_messages) > 0
    user_content = user_messages[0]["content"]
    assert (
        "Context from previous section" in user_content or prev_context in user_content
    )


@pytest.mark.asyncio
async def test_custom_prompt_override_in_run_summary_request() -> None:
    """Custom prompt passed via summary_system_prompt overrides IndexConfig.

    This tests the explicit override parameter that will be used when
    the document's stored custom prompt differs from IndexConfig.
    Spec: specs/custom-prompt-config.md § Implementation
    """
    from ragzoom.services.summary_utils import SummaryRequest, run_summary_request

    # IndexConfig has NO custom prompt set
    config = IndexConfig.load()
    assert config.summary_system_prompt is None

    # But we pass an explicit override
    document_custom_prompt = (
        "You are a legal document summarizer. Preserve exact legal terminology."
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
        summary_system_prompt=document_custom_prompt,  # Override
    )

    # Document's custom prompt should be used, not IndexConfig's default
    assert_system_prompt(captured_messages, document_custom_prompt)
