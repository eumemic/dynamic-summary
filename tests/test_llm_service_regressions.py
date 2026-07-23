"""Tests validating LLMService retry/summarization behaviour."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import litellm
import pytest

from ragzoom.config import IndexConfig
from ragzoom.services import summary_utils
from ragzoom.services.llm_service import LLMService
from ragzoom.telemetry_collection import TelemetryCollector


class MockOpenAIResponse:
    """Mock OpenAI response with usage tracking."""

    def __init__(
        self,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details={"cached_tokens": cached_tokens},
        )


def _fake_encode(text: str) -> list[int]:
    return [0] * len(text)


def _fake_count(text: str) -> int:
    return len(text)


def _fake_decode(_tokens: object) -> str:
    return "decoded"


@contextmanager
def patched_tokenizers() -> Iterator[None]:
    """Temporarily make tokenizer behave deterministically for tests."""

    with ExitStack() as stack:
        for path in (
            "ragzoom.services.summary_utils.tokenizer.encode",
            "ragzoom.services.llm_service.tokenizer.encode",
        ):
            stack.enter_context(patch(path, side_effect=_fake_encode))

        for path in (
            "ragzoom.services.summary_utils.tokenizer.count_tokens",
            "ragzoom.services.llm_service.tokenizer.count_tokens",
        ):
            stack.enter_context(patch(path, side_effect=_fake_count))

        for path in (
            "ragzoom.services.summary_utils.tokenizer.decode",
            "ragzoom.services.llm_service.tokenizer.decode",
        ):
            stack.enter_context(patch(path, side_effect=_fake_decode))

        yield


@pytest.mark.asyncio
async def test_mark_accepted_attempt_is_called() -> None:
    """Test that mark_accepted_attempt is called after summarization completes."""
    config = IndexConfig.load(
        retry_threshold=0.2,  # 20% deviation triggers retry
        max_retries=2,
        target_chunk_tokens=100,
    )

    llm_service = LLMService(config, api_key="test-key")

    # Track API calls and telemetry calls
    api_calls = []
    mark_accepted_calls = []

    async def mock_create(**kwargs: object) -> MockOpenAIResponse:
        """Mock OpenAI API calls."""
        api_calls.append(kwargs)

        if len(api_calls) == 1:
            # First attempt: overshoot to trigger retry
            return MockOpenAIResponse(
                content="A" * 150,  # 150 tokens (50% over target)
                prompt_tokens=500,
                completion_tokens=150,
                cached_tokens=0,
            )
        else:
            # Retry: return acceptable summary
            return MockOpenAIResponse(
                content="B" * 105,  # 105 tokens (within threshold)
                prompt_tokens=600,
                completion_tokens=105,
                cached_tokens=100,
            )

    # Mock the telemetry collector
    reporter = TelemetryCollector(
        document_id="test_doc",
        source_tokens=1000,
        config=config,
    )

    # Track calls to mark_accepted_attempt
    original_mark_accepted = reporter.mark_accepted_attempt

    def track_mark_accepted(node_id: str, attempt_index: int) -> None:
        mark_accepted_calls.append((node_id, attempt_index))
        return original_mark_accepted(node_id, attempt_index)

    reporter.mark_accepted_attempt = track_mark_accepted  # type: ignore[method-assign]

    # Pre-track the test node
    reporter.track_node_created("test_node", height=1)

    with patch.object(litellm, "acompletion", new=mock_create):
        with patched_tokenizers():
            result = await llm_service._summarize_text(
                "Test left text " * 10 + " " + "Test right text " * 10,
                100,
                parent_id="test_node",
                reporter=reporter,
            )

    # Verify summarization worked
    assert result.summary == "B" * 105
    assert result.retry_count == 1  # One retry was made

    # CRITICAL: Verify mark_accepted_attempt was called
    assert len(mark_accepted_calls) == 1, "mark_accepted_attempt should be called once"
    node_id, attempt_index = mark_accepted_calls[0]
    assert node_id == "test_node"
    assert attempt_index == 1, "Retry attempt (index 1) should be marked as accepted"


def test_is_better_summary_logic() -> None:
    """Test that is_better_summary logic correctly prioritizes under-target summaries."""
    # This tests summary_utils.is_better_summary, which prioritizes:
    # 1. Under-target summaries over over-target summaries
    # 2. Among under-target summaries, prefer closer to target (higher tokens)
    # 3. Among over-target summaries, prefer closer to target (lower tokens)

    target = 100

    # Test case 1: New is under target and closer to target than current
    # Should prefer 95 tokens over 85 tokens when target is 100
    assert summary_utils.is_better_summary(
        new_tokens=95,
        current_best_tokens=85,
        target_tokens=target,
    )

    # Test case 2: New is under target, current is over target
    # Should prefer under-target (95) over over-target (110)
    assert summary_utils.is_better_summary(
        new_tokens=95,
        current_best_tokens=110,
        target_tokens=target,
    )

    # Test case 3: Both over target, new is closer (smaller)
    # Should prefer 110 tokens over 120 tokens
    assert summary_utils.is_better_summary(
        new_tokens=110,
        current_best_tokens=120,
        target_tokens=target,
    )

    # Test case 4: Both over target, new is not better (farther)
    # Should NOT prefer 125 tokens over 120 tokens
    assert not summary_utils.is_better_summary(
        new_tokens=125,
        current_best_tokens=120,
        target_tokens=target,
    )

    # Test case 5: New is over target, current is under target
    # Should NOT prefer over-target (110) over under-target (95)
    assert not summary_utils.is_better_summary(
        new_tokens=110,
        current_best_tokens=95,
        target_tokens=target,
    )

    # Test case 6: Both under target, new is farther from target
    # Should NOT prefer 80 tokens over 95 tokens
    assert not summary_utils.is_better_summary(
        new_tokens=80,
        current_best_tokens=95,
        target_tokens=target,
    )


@pytest.mark.asyncio
async def test_retry_selection_uses_proper_logic() -> None:
    """Test that retry selection uses the proper _is_better_summary logic."""
    config = IndexConfig.load(
        retry_threshold=0.2,
        max_retries=3,
        target_chunk_tokens=100,
    )

    llm_service = LLMService(config, api_key="test-key")

    api_calls = []

    async def mock_create(**kwargs: object) -> MockOpenAIResponse:
        """Mock OpenAI API calls with specific token counts."""
        api_calls.append(kwargs)

        if len(api_calls) == 1:
            # Initial: 130 tokens (30% over, triggers retry)
            content = "A" * 130
        elif len(api_calls) == 2:
            # Retry 1: 85 tokens (under target)
            content = "B" * 85
        elif len(api_calls) == 3:
            # Retry 2: 95 tokens (under target but closer)
            content = "C" * 95
        elif len(api_calls) == 4:
            # Retry 3: 120 tokens (over but less than initial)
            content = "D" * 120
        else:
            content = "E" * 100

        return MockOpenAIResponse(
            content=content,
            prompt_tokens=500,
            completion_tokens=len(content),
        )

    with patch.object(litellm, "acompletion", new=mock_create):
        with patched_tokenizers():
            result = await llm_service._summarize_text(
                "Test " * 50 + " " + "Text " * 50,
                100,
                parent_id="test_node",
            )

    # With proper _is_better_summary logic:
    # - Initial: 130 tokens (30% over target, triggers retry)
    # - Retry 1: 85 tokens (15% under target, within 20% threshold, stops here)
    # - No more retries because 85 is acceptable

    # The final summary should be "B" * 85 (first acceptable result)
    assert (
        result.summary == "B" * 85
    ), f"Should select first acceptable summary, got {result.summary[:10]}..."
    assert result.summary_tokens == 85
    assert result.retry_count == 1  # Only one retry needed to get acceptable result
