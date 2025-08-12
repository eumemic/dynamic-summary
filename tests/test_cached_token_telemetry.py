"""Test that cached tokens are properly tracked in telemetry."""

from unittest.mock import MagicMock, patch

import pytest

from ragzoom.config import IndexConfig, OperationalConfig
from ragzoom.index import TreeBuilder


def create_test_reporter(config):
    """Create a test reporter with common test nodes pre-tracked."""
    from ragzoom.telemetry_collection import TelemetryCollector

    # If config is a wrapper, extract the IndexConfig
    index_config = config.index_config if hasattr(config, "index_config") else config

    reporter = TelemetryCollector(
        document_id="test_doc", source_tokens=1000, config=index_config
    )
    # Pre-track common test nodes
    for node_id in ["test", "test_node"]:
        reporter.track_node_created(node_id, height=1)
    return reporter


class MockOpenAIResponseWithCache:
    """Mock OpenAI response with detailed usage including cached tokens."""

    def __init__(
        self,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        audio_tokens: int = 0,
    ):
        self.choices = [MagicMock(message=MagicMock(content=content))]

        # Mimic OpenAI's response structure
        self.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        # Add prompt_tokens_details if there are cached tokens
        if cached_tokens > 0:
            self.usage.prompt_tokens_details = {
                "cached_tokens": cached_tokens,
                "audio_tokens": audio_tokens,
            }
        else:
            # When no caching, this attribute might not exist
            self.usage.prompt_tokens_details = None


@pytest.mark.asyncio
async def test_cached_tokens_recorded_in_telemetry(mock_store):
    """Test that cached tokens from OpenAI response are properly recorded."""
    index_config = IndexConfig.load(
        retry_threshold=0.1,
        max_retries=1,  # Enable retries for this test
        target_chunk_tokens=100,
    )
    operational_config = OperationalConfig(
        openai_api_key="test-key",
    )

    # Create a config wrapper for backward compatibility with telemetry
    from tests.conftest import BackwardCompatibilityConfig

    config = BackwardCompatibilityConfig(index_config, None, operational_config)

    indexer = TreeBuilder(
        index_config, mock_store, api_key=operational_config.openai_api_key
    )
    reporter = create_test_reporter(config)

    api_calls = []

    async def mock_create(**kwargs):
        api_calls.append(kwargs)
        call_num = len(api_calls)

        if call_num == 1:
            # Initial call - no caching
            return MockOpenAIResponseWithCache(
                content="A" * 150,  # Too long
                prompt_tokens=1350,
                completion_tokens=150,
                cached_tokens=0,
            )
        elif call_num == 2:
            # Retry - with caching
            return MockOpenAIResponseWithCache(
                content="B" * 100,  # Just right
                prompt_tokens=1500,
                completion_tokens=100,
                cached_tokens=1200,  # 80% cached
            )

        return MockOpenAIResponseWithCache("", 0, 0)

    with patch.object(indexer.client.chat.completions, "create", new=mock_create):
        with patch.object(
            indexer.splitter.tokenizer, "encode", side_effect=lambda x: [0] * len(x)
        ):
            await indexer._summarize_text(
                left_text="Test content that needs to be long enough to trigger summarization"
                * 2,
                right_text="More content that also needs to be sufficiently long" * 2,
                target_tokens=100,
                parent_id="test_node",
                reporter=reporter,
            )

    # Get telemetry data
    data = reporter.get_telemetry_data("test_doc", index_config.target_chunk_tokens)
    nodes = data["nodes"]

    # Find the node with summary attempts
    test_node = next(n for n in nodes if n["node_id"] == "test_node")
    attempts = test_node["summary_attempts"]

    assert len(attempts) == 2, "Should have initial + 1 retry"

    # First attempt should have no cached tokens
    assert attempts[0]["prompt_tokens"] == 1350
    assert attempts[0].get("cached_tokens", 0) == 0

    # Second attempt should have cached tokens
    assert attempts[1]["prompt_tokens"] == 1500
    assert attempts[1]["cached_tokens"] == 1200


@pytest.mark.asyncio
async def test_backward_compatibility_without_cached_tokens(mock_store):
    """Test that telemetry works correctly when OpenAI doesn't return cached_tokens."""
    index_config = IndexConfig.load(target_chunk_tokens=100)
    operational_config = OperationalConfig(openai_api_key="test-key")

    # Create a config wrapper for backward compatibility with telemetry
    from tests.conftest import BackwardCompatibilityConfig

    config = BackwardCompatibilityConfig(index_config, None, operational_config)

    indexer = TreeBuilder(
        index_config, mock_store, api_key=operational_config.openai_api_key
    )
    reporter = create_test_reporter(config)

    # Mock response without prompt_tokens_details
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Summary"))]
    response.usage = MagicMock(
        prompt_tokens=1000,
        completion_tokens=100,
    )
    # Explicitly no prompt_tokens_details
    response.usage.prompt_tokens_details = None

    async def mock_create(**kwargs):
        return response

    with patch.object(indexer.client.chat.completions, "create", new=mock_create):
        # Mock encode to return appropriate lengths - combined text should exceed target
        def mock_encode(text):
            if "Test content" in text and "More content" in text:
                # Combined text
                return [0] * 200  # Greater than target of 100
            elif "Test content" in text:
                return [0] * 120  # Left text
            elif "More content" in text:
                return [0] * 120  # Right text
            else:
                return [0] * 100  # Default

        with patch.object(
            indexer.splitter.tokenizer, "encode", side_effect=mock_encode
        ):
            await indexer._summarize_text(
                left_text="Test content that needs to be long enough to trigger summarization"
                * 2,
                right_text="More content that also needs to be sufficiently long" * 2,
                target_tokens=100,
                parent_id="test_node",
                reporter=reporter,
            )

    data = reporter.get_telemetry_data("test_doc", index_config.target_chunk_tokens)
    nodes = data["nodes"]
    test_node = next(n for n in nodes if n["node_id"] == "test_node")
    attempts = test_node["summary_attempts"]

    assert len(attempts) == 1
    # Should handle missing cached_tokens gracefully
    assert "cached_tokens" not in attempts[0] or attempts[0]["cached_tokens"] == 0


@pytest.mark.asyncio
async def test_cached_tokens_across_multiple_retries(mock_store):
    """Test that cached tokens increase with each retry as conversation grows."""
    index_config = IndexConfig.load(
        retry_threshold=0.05,  # Very strict
        max_retries=3,
        target_chunk_tokens=100,
    )
    operational_config = OperationalConfig(openai_api_key="test-key")

    # Create a config wrapper for backward compatibility with telemetry
    from tests.conftest import BackwardCompatibilityConfig

    config = BackwardCompatibilityConfig(index_config, None, operational_config)

    indexer = TreeBuilder(
        index_config, mock_store, api_key=operational_config.openai_api_key
    )
    reporter = create_test_reporter(config)

    api_calls = []

    async def mock_create(**kwargs):
        api_calls.append(kwargs)
        call_num = len(api_calls)

        if call_num == 1:
            return MockOpenAIResponseWithCache(
                content="A" * 150,
                prompt_tokens=1350,
                completion_tokens=150,
                cached_tokens=0,  # Nothing cached initially
            )
        elif call_num == 2:
            return MockOpenAIResponseWithCache(
                content="B" * 130,
                prompt_tokens=1500,  # More tokens (includes previous response)
                completion_tokens=130,
                cached_tokens=1200,  # Most of original prompt cached
            )
        elif call_num == 3:
            return MockOpenAIResponseWithCache(
                content="C" * 102,
                prompt_tokens=1650,  # Even more tokens
                completion_tokens=102,
                cached_tokens=1400,  # Even more cached
            )

        return MockOpenAIResponseWithCache("", 0, 0)

    with patch.object(indexer.client.chat.completions, "create", new=mock_create):
        with patch.object(
            indexer.splitter.tokenizer, "encode", side_effect=lambda x: [0] * len(x)
        ):
            await indexer._summarize_text(
                left_text="Test content that needs to be long enough to trigger summarization"
                * 2,
                right_text="More content that also needs to be sufficiently long" * 2,
                target_tokens=100,
                parent_id="test_node",
                reporter=reporter,
            )

    data = reporter.get_telemetry_data("test_doc", index_config.target_chunk_tokens)
    nodes = data["nodes"]
    test_node = next(n for n in nodes if n["node_id"] == "test_node")
    attempts = test_node["summary_attempts"]

    assert len(attempts) == 3

    # Verify cached tokens increase
    assert attempts[0].get("cached_tokens", 0) == 0
    assert attempts[1]["cached_tokens"] == 1200
    assert attempts[2]["cached_tokens"] == 1400

    # Verify cache efficiency improves
    cache_rates = [
        a.get("cached_tokens", 0) / a["prompt_tokens"] if a["prompt_tokens"] > 0 else 0
        for a in attempts
    ]
    assert cache_rates[0] == 0  # No caching initially
    assert cache_rates[1] > 0.7  # High cache rate on first retry
    assert cache_rates[2] > cache_rates[1]  # Even better on second retry


@pytest.mark.asyncio
async def test_passthrough_summary_has_no_cached_tokens(mock_store):
    """Test that passthrough summaries correctly report 0 cached tokens."""
    index_config = IndexConfig.load(target_chunk_tokens=100)
    operational_config = OperationalConfig(openai_api_key="test-key")

    # Create a config wrapper for backward compatibility with telemetry
    from tests.conftest import BackwardCompatibilityConfig

    config = BackwardCompatibilityConfig(index_config, None, operational_config)

    indexer = TreeBuilder(
        index_config, mock_store, api_key=operational_config.openai_api_key
    )
    reporter = create_test_reporter(config)

    # No API calls should be made
    api_calls = []

    async def mock_create(**kwargs):
        api_calls.append(kwargs)
        pytest.fail("Should not call API for passthrough")

    with patch.object(indexer.client.chat.completions, "create", new=mock_create):
        with patch.object(
            indexer.splitter.tokenizer, "encode", return_value=[0] * 50  # Under target
        ):
            await indexer._summarize_text(
                left_text="Short",
                right_text="Text",
                target_tokens=100,
                parent_id="test_node",
                reporter=reporter,
            )

    data = reporter.get_telemetry_data("test_doc", index_config.target_chunk_tokens)
    nodes = data["nodes"]
    test_node = next(n for n in nodes if n["node_id"] == "test_node")

    # Passthrough nodes no longer record summary_attempts
    assert "summary_attempts" not in test_node or test_node["summary_attempts"] == []


@pytest.mark.asyncio
async def test_cached_tokens_with_high_cache_rate(mock_store):
    """Test scenario with very high cache hit rate (95%+)."""
    index_config = IndexConfig.load(
        retry_threshold=0.1,
        max_retries=1,  # Enable retries for this test
        target_chunk_tokens=100,
    )
    operational_config = OperationalConfig(openai_api_key="test-key")

    # Create a config wrapper for backward compatibility with telemetry
    from tests.conftest import BackwardCompatibilityConfig

    config = BackwardCompatibilityConfig(index_config, None, operational_config)

    indexer = TreeBuilder(
        index_config, mock_store, api_key=operational_config.openai_api_key
    )
    reporter = create_test_reporter(config)

    async def mock_create(**kwargs):
        messages = kwargs.get("messages", [])

        if len(messages) == 2:  # Initial call
            return MockOpenAIResponseWithCache(
                content="A" * 120,
                prompt_tokens=2000,
                completion_tokens=120,
                cached_tokens=0,
            )
        else:  # Retry with very high cache rate
            return MockOpenAIResponseWithCache(
                content="B" * 100,
                prompt_tokens=2200,
                completion_tokens=100,
                cached_tokens=2090,  # 95% cached
            )

    with patch.object(indexer.client.chat.completions, "create", new=mock_create):
        with patch.object(
            indexer.splitter.tokenizer, "encode", side_effect=lambda x: [0] * len(x)
        ):
            await indexer._summarize_text(
                left_text="Long content " * 50,
                right_text="More content " * 50,
                target_tokens=100,
                parent_id="test_node",
                reporter=reporter,
            )

    data = reporter.get_telemetry_data("test_doc", index_config.target_chunk_tokens)
    nodes = data["nodes"]
    test_node = next(n for n in nodes if n["node_id"] == "test_node")
    attempts = test_node["summary_attempts"]

    # Verify high cache rate on retry
    retry_attempt = attempts[1]
    cache_rate = retry_attempt["cached_tokens"] / retry_attempt["prompt_tokens"]
    assert cache_rate >= 0.95, f"Expected 95%+ cache rate, got {cache_rate:.1%}"
