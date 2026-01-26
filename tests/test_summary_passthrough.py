"""Tests for summary passthrough behavior when target <= 0 tokens."""

from __future__ import annotations

from collections.abc import MutableSequence
from unittest.mock import MagicMock

import pytest

from ragzoom.config import IndexConfig
from ragzoom.contracts.chat_model import UsageInfo
from ragzoom.services.summary_utils import run_summary_from_config
from ragzoom.telemetry_collection import TelemetryCollector


@pytest.mark.asyncio
async def test_passthrough_when_target_is_zero() -> None:
    """Test that target_tokens=0 triggers passthrough without LLM call.

    When get_summary_target() returns 0 (below 50-token floor), text should
    pass through unsummarized. This is the signal from dynamic target calculation
    that the text is too short to benefit from summarization.
    """
    config = IndexConfig.load()
    text = "Short text that shouldn't be summarized"
    target_tokens = 0  # Signal: passthrough, don't summarize

    # Mock call_summary function - should NOT be called
    async def mock_call_summary(
        messages: MutableSequence[dict[str, str]],
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None,
    ) -> tuple[str, UsageInfo]:
        raise AssertionError("LLM should not be called when target_tokens=0")

    # Run summarization with target_tokens=0
    reporter = MagicMock()
    result = await run_summary_from_config(
        index_config=config,
        text=text,
        target_tokens=target_tokens,
        call_summary=mock_call_summary,
        reporter=reporter,
        parent_id="test-node",
        prev_context=None,
        text_tokens=None,
    )

    # Verify passthrough behavior
    assert result.summary == text, "Should return original text unchanged"
    assert result.retry_count == 0, "Should have zero retries"
    assert result.usage.prompt_tokens == 0, "Should have no LLM usage"
    assert result.usage.completion_tokens == 0, "Should have no LLM usage"

    # Success: LLM was not called (would have raised AssertionError)


@pytest.mark.asyncio
async def test_summary_uses_llm_when_target_above_zero() -> None:
    """Test that normal summarization still works when target > 0."""
    config = IndexConfig.load()
    text = "This is a longer text that needs summarization " * 50  # ~350 tokens
    target_tokens = 100  # Normal target

    # Track if LLM was called
    llm_called = False

    async def mock_call_summary(
        messages: MutableSequence[dict[str, str]],
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None,
    ) -> tuple[str, UsageInfo]:
        nonlocal llm_called
        llm_called = True
        return "Summarized text", UsageInfo(
            prompt_tokens=350, completion_tokens=80, cached_tokens=0
        )

    # Run summarization with normal target
    reporter = MagicMock()
    result = await run_summary_from_config(
        index_config=config,
        text=text,
        target_tokens=target_tokens,
        call_summary=mock_call_summary,
        reporter=reporter,
        parent_id="test-node",
        prev_context=None,
        text_tokens=None,
    )

    # Verify LLM was called
    assert llm_called, "LLM should be called for normal target"

    # Verify result
    assert result.summary == "Summarized text"
