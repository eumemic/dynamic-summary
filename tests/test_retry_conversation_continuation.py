"""Test that retry mechanism maintains conversation context."""

from unittest.mock import MagicMock, patch

import pytest

from ragzoom.config import IndexConfig
from ragzoom.index import TreeBuilder
from ragzoom.telemetry_collection import TelemetryCollector


def create_test_reporter(config):
    """Create a test reporter with common test nodes pre-tracked."""
    # TelemetryCollector expects IndexConfig, so extract it if needed
    index_config = config.index_config if hasattr(config, "index_config") else config
    reporter = TelemetryCollector(
        document_id="test_doc", source_tokens=1000, config=index_config
    )
    # Pre-track common test nodes
    for node_id in ["test", "test_node"]:
        reporter.track_node_created(node_id, height=1)
    return reporter


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
            prompt_tokens_details={"cached_tokens": cached_tokens},
        )


@pytest.mark.asyncio
async def test_retry_maintains_conversation_history(mock_store):
    """Test that retries append to existing conversation instead of creating new ones."""
    config = IndexConfig.load(
        retry_threshold=0.2,  # 20% deviation
        max_retries=3,
        target_chunk_tokens=100,  # Target tokens
    )

    indexer = TreeBuilder(config, mock_store)

    # Track all API calls
    api_calls = []

    async def mock_create(**kwargs):
        """Capture API calls and return appropriate responses."""
        # Make a deep copy of kwargs to avoid mutation issues
        import copy

        api_calls.append(copy.deepcopy(kwargs))
        messages = kwargs.get("messages", [])

        # First call: return oversized summary (150 tokens, 50% over)
        if len(api_calls) == 1:
            return MockOpenAIResponse(
                content="A" * 150,  # Simulating 150 tokens
                prompt_tokens=1350,
                completion_tokens=150,
                cached_tokens=0,  # First call, nothing cached
            )

        # Second call (retry): should have full conversation
        elif len(api_calls) == 2:
            # Verify conversation continuity
            assert len(messages) == 4  # system, user, assistant, user
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
            assert messages[2]["content"] == "A" * 150  # Previous response
            assert messages[3]["role"] == "user"
            # Check that retry prompt contains expected content (word-based format)
            assert "words" in messages[3]["content"]
            assert "50%" in messages[3]["content"]  # Deviation percentage
            assert "larger" in messages[3]["content"]  # Direction

            return MockOpenAIResponse(
                content="B" * 95,  # Close to target
                prompt_tokens=1500,
                completion_tokens=95,
                cached_tokens=1200,  # Most of prompt is cached
            )

        return MockOpenAIResponse("", 0, 0)

    with patch.object(indexer.client.chat.completions, "create", new=mock_create):
        reporter = create_test_reporter(config)

        # Mock tokenizer to return length as token count
        with patch.object(
            indexer.splitter.tokenizer, "encode", side_effect=lambda x: [0] * len(x)
        ):
            summary, retry_count, token_count = await indexer._summarize_text(
                left_text="Left text content that is much longer to ensure we exceed the target"
                * 2,
                right_text="Right text content that is also much longer to trigger summarization"
                * 2,
                target_tokens=100,
                parent_id="test_node",
                reporter=reporter,
            )

    # Verify results
    assert len(api_calls) == 2, "Should have made initial call + 1 retry"
    assert retry_count == 1, "Should have performed 1 retry"
    assert summary == "B" * 95, "Should return the adjusted summary"

    # Verify first call structure - it starts with 2 messages
    first_call = api_calls[0]
    assert len(first_call["messages"]) == 2  # system + user

    # Verify second call has the full conversation (messages were appended)
    second_call = api_calls[1]
    assert len(second_call["messages"]) == 4  # system + user + assistant + user
    assert second_call["messages"][0]["role"] == "system"
    assert second_call["messages"][1]["role"] == "user"  # Original prompt
    assert second_call["messages"][2]["role"] == "assistant"  # First attempt response
    assert second_call["messages"][2]["content"] == "A" * 150  # Previous summary
    assert second_call["messages"][3]["role"] == "user"  # Retry instruction


@pytest.mark.asyncio
async def test_retry_preserves_original_context(mock_store):
    """Test that retry requests can still see the original text being summarized."""
    config = IndexConfig.load(
        retry_threshold=0.1,  # 10% deviation
        max_retries=1,  # Enable retries for this test
        target_chunk_tokens=100,
    )

    indexer = TreeBuilder(config, mock_store)
    api_calls = []

    original_text = (
        "This is the original text that needs to be summarized properly. " * 5
    )  # Make it long enough

    async def mock_create(**kwargs):
        import copy

        api_calls.append(copy.deepcopy(kwargs))
        messages = kwargs.get("messages", [])

        if len(api_calls) == 1:
            # Verify parts of original text are in the first prompt
            # (text is split between left and right)
            user_content = " ".join(
                msg["content"] for msg in messages if msg["role"] == "user"
            )
            # Check that at least part of the original text is there
            assert any(
                part in user_content
                for part in [original_text[:30], original_text[-30:]]
            )
            return MockOpenAIResponse("A" * 150, 150, 10, 0)  # 50% over

        elif len(api_calls) == 2:
            # Verify original text is STILL accessible in conversation history
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            assert len(user_messages) == 2  # Original + retry prompt

            # Original prompt should still contain parts of the text
            assert any(
                part in user_messages[0]["content"]
                for part in [original_text[:30], original_text[-30:]]
            )

            return MockOpenAIResponse("B" * 100, 100, 20, 800)

        return MockOpenAIResponse("", 0, 0)

    with patch.object(indexer.client.chat.completions, "create", new=mock_create):
        with patch.object(
            indexer.splitter.tokenizer, "encode", side_effect=lambda x: [0] * len(x)
        ):
            summary, _, _ = await indexer._summarize_text(
                left_text=original_text[: len(original_text) // 2],
                right_text=original_text[len(original_text) // 2 :],
                target_tokens=100,
                parent_id="test",
            )

    assert len(api_calls) == 2
    # Original context preserved throughout retry


@pytest.mark.asyncio
async def test_multiple_retries_build_conversation(mock_store):
    """Test that multiple retries continue building on the same conversation."""
    config = IndexConfig.load(
        retry_threshold=0.1,
        max_retries=3,
        target_chunk_tokens=100,
    )

    indexer = TreeBuilder(config, mock_store)
    api_calls = []

    async def mock_create(**kwargs):
        import copy

        api_calls.append(copy.deepcopy(kwargs))
        messages = kwargs.get("messages", [])
        call_num = len(api_calls)

        if call_num == 1:
            return MockOpenAIResponse("A" * 150, 1000, 150, 0)  # Too long

        elif call_num == 2:
            assert len(messages) == 4  # system, user, assistant, user
            return MockOpenAIResponse("B" * 130, 1200, 130, 1000)  # Still too long

        elif call_num == 3:
            assert len(messages) == 6  # Conversation continues to grow
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"  # Original
            assert messages[2]["role"] == "assistant"  # First attempt
            assert messages[3]["role"] == "user"  # First retry prompt
            assert messages[4]["role"] == "assistant"  # Second attempt
            assert messages[5]["role"] == "user"  # Second retry prompt
            return MockOpenAIResponse("C" * 105, 1400, 105, 1200)  # Acceptable

        return MockOpenAIResponse("", 0, 0)

    with patch.object(indexer.client.chat.completions, "create", new=mock_create):
        with patch.object(
            indexer.splitter.tokenizer, "encode", side_effect=lambda x: [0] * len(x)
        ):
            summary, retry_count, token_count = await indexer._summarize_text(
                left_text="Test content that is much longer to trigger summarization"
                * 2,
                right_text="More content that also needs to be long enough" * 2,
                target_tokens=100,
                parent_id="test",
            )

    assert len(api_calls) == 3
    assert retry_count == 2  # Two retries after initial
    assert summary == "C" * 105

    # Verify conversation grew correctly
    final_messages = api_calls[-1]["messages"]
    assert len(final_messages) == 6


@pytest.mark.asyncio
async def test_no_retry_when_within_threshold(mock_store):
    """Test that no retry occurs when initial summary is within threshold."""
    config = IndexConfig.load(
        retry_threshold=0.2,
        target_chunk_tokens=100,
    )

    indexer = TreeBuilder(config, mock_store)
    api_calls = []

    async def mock_create(**kwargs):
        import copy

        api_calls.append(copy.deepcopy(kwargs))
        # Return summary within threshold (105 tokens, 5% over - within 10% threshold)
        return MockOpenAIResponse("A" * 105, 1000, 105, 0)

    with patch.object(indexer.client.chat.completions, "create", new=mock_create):
        with patch.object(
            indexer.splitter.tokenizer, "encode", side_effect=lambda x: [0] * len(x)
        ):
            summary, retry_count, token_count = await indexer._summarize_text(
                left_text="Test content that is much longer to trigger summarization"
                * 2,
                right_text="More content that also needs to be long enough" * 2,
                target_tokens=100,
                parent_id="test",
            )

    assert len(api_calls) == 1, "Should only make initial call"
    assert retry_count == 0, "No retries needed"
    assert summary == "A" * 105


@pytest.mark.asyncio
async def test_accept_retry_within_threshold_immediately(mock_store):
    """Test that we accept a retry attempt immediately when it's within threshold.

    This tests the bug where attempts within the threshold were being ignored
    if they weren't 'better' than previous attempts.
    """
    config = IndexConfig.load(
        retry_threshold=0.2,  # 20% deviation threshold
        max_retries=3,
        target_chunk_tokens=100,
    )

    indexer = TreeBuilder(config, mock_store)
    api_calls = []

    async def mock_create(**kwargs):
        """Return different responses based on call number."""
        import copy

        api_calls.append(copy.deepcopy(kwargs))

        if len(api_calls) == 1:
            # First attempt: 130 tokens (30% over, outside threshold)
            return MockOpenAIResponse("A" * 130, 1000, 130, 0)
        elif len(api_calls) == 2:
            # Second attempt: 115 tokens (15% over, WITHIN threshold)
            # This should be accepted immediately
            return MockOpenAIResponse("B" * 115, 1200, 115, 1000)
        else:
            # We should never get here!
            pytest.fail(
                f"Should not make call #{len(api_calls)} - "
                "attempt 2 was within threshold"
            )

    with patch.object(indexer.client.chat.completions, "create", new=mock_create):
        with patch.object(
            indexer.splitter.tokenizer, "encode", side_effect=lambda x: [0] * len(x)
        ):
            summary, retry_count, token_count = await indexer._summarize_text(
                left_text="Test content that is much longer to trigger summarization"
                * 2,
                right_text="More content that also needs to be long enough" * 2,
                target_tokens=100,
                parent_id="test",
            )

    # With the BUGGY version: 115 is not "better" than 70, so it keeps 70 as best
    # and continues retrying, making a 3rd call
    # With the FIXED version: it should stop at attempt 2 and return 115

    # This assertion will FAIL with buggy version (will be 3 instead of 2)
    assert len(api_calls) == 2, "Should stop after second attempt (within threshold)"
    assert retry_count == 1, "Should have done exactly 1 retry"
    assert summary == "B" * 115, "Should return the second attempt's summary"
    assert token_count == 115, "Should return the second attempt's token count"


@pytest.mark.asyncio
async def test_passthrough_for_text_under_target(mock_store):
    """Test that text under target tokens is passed through without LLM call."""
    config = IndexConfig.load(target_chunk_tokens=100)
    indexer = TreeBuilder(config, mock_store)

    api_calls = []

    async def mock_create(**kwargs):
        import copy

        api_calls.append(copy.deepcopy(kwargs))
        pytest.fail("Should not call LLM for text under target")

    with patch.object(indexer.client.chat.completions, "create", new=mock_create):
        with patch.object(
            indexer.splitter.tokenizer,
            "encode",
            side_effect=lambda x: [0] * min(len(x), 50),  # Always under 100
        ):
            reporter = create_test_reporter(config)
            summary, retry_count, token_count = await indexer._summarize_text(
                left_text="Short",
                right_text="Text",
                target_tokens=100,
                parent_id="test",
                reporter=reporter,
            )

    assert len(api_calls) == 0, "Should not call LLM"
    assert retry_count == 0
    assert summary == "Short Text"

    # Verify telemetry - passthrough nodes now record attempts for visualization
    data = reporter.get_telemetry_data("test_doc", config.target_chunk_tokens)
    # Passthrough nodes should have summary_attempts with model="passthrough"
    assert "summary_attempts" in data["nodes"][0]
    assert len(data["nodes"][0]["summary_attempts"]) == 1
    assert data["nodes"][0]["summary_attempts"][0]["model"] == "passthrough"
