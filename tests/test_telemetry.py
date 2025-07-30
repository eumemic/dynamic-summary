"""Test node-level telemetry collection."""

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.metrics import (
    IndexingMetricsReporter,
    NodeTelemetry,
    SummaryAttempt,
)


class TestTelemetryDataStructures:
    """Test telemetry data structures."""

    def test_node_telemetry_creation(self) -> None:
        """Test NodeTelemetry dataclass creation."""
        telemetry = NodeTelemetry(
            node_id="test-123",
            height=0,
        )

        assert telemetry.node_id == "test-123"
        assert telemetry.height == 0
        assert telemetry.embedding is None
        assert telemetry.summary_attempts == []
        assert telemetry.created_at > 0

    def test_summary_attempt_timing_fields(self) -> None:
        """Test SummaryAttempt has start_time and end_time fields."""
        attempt = SummaryAttempt(
            target_tokens=200,
            input_text_tokens=400,
            prompt_tokens=500,
            completion_tokens=180,
            actual_tokens=180,
            status="accepted",
            model="gpt-4o-mini",
            start_time=1234567890.0,
            end_time=1234567891.0,
        )

        # Verify timing fields exist
        assert attempt.start_time == 1234567890.0
        assert attempt.end_time == 1234567891.0
        assert attempt.status == "accepted"
        # Verify is_retry field removed
        assert not hasattr(attempt, "is_retry")


class TestTelemetryCollection:
    """Test telemetry collection during indexing."""

    @pytest.fixture
    def config(self) -> RagZoomConfig:
        """Create test config."""
        return RagZoomConfig(
            openai_api_key="test-key",
            leaf_tokens=100,
            adjacent_context_tokens=50,
            embedding_batch_size=2,
        )

    @pytest.fixture
    def reporter(self, config: RagZoomConfig) -> IndexingMetricsReporter:
        """Create test reporter."""
        return IndexingMetricsReporter(
            document_id="test-doc",
            source_tokens=1000,
            config=config,
        )

    def test_track_node_created(self, reporter: IndexingMetricsReporter) -> None:
        """Test node creation tracking."""
        reporter.track_node_created(
            node_id="leaf-1",
            height=0,
        )

        assert "leaf-1" in reporter.metrics.node_telemetry
        node = reporter.metrics.node_telemetry["leaf-1"]
        assert node.height == 0
        # Verify span fields removed
        assert not hasattr(node, "span_start")
        assert not hasattr(node, "span_end")
        # Verify node_type field removed
        assert not hasattr(node, "node_type")

    def test_record_embedding_v2(self, reporter: IndexingMetricsReporter) -> None:
        """Test v2 embedding recording with node-level detail."""
        # First track nodes
        reporter.track_node_created("node-1", 0)
        reporter.track_node_created("node-2", 0)

        # Record embedding batch
        start_time = time.time()
        reporter.record_embedding_call_v2(
            node_embeddings=[("node-1", 45), ("node-2", 48)],
            batch_size=2,
            model="text-embedding-3-small",
            start_time=start_time,
        )

        # Check aggregate metrics
        assert reporter.metrics.embedding_api_calls == 1
        assert reporter.metrics.total_embedding_tokens == 93

        # Check telemetry
        node1 = reporter.metrics.node_telemetry["node-1"]
        assert node1.embedding is not None
        assert node1.embedding.text_tokens == 45
        assert node1.embedding.batch_size == 2
        assert node1.embedding.batch_position == 0
        assert node1.embedding.model == "text-embedding-3-small"

        node2 = reporter.metrics.node_telemetry["node-2"]
        assert node2.embedding is not None
        assert node2.embedding.batch_position == 1

    def test_record_summary_attempt_v2(self, reporter: IndexingMetricsReporter) -> None:
        """Test v2 summary recording with telemetry."""
        # Track a summary node
        reporter.track_node_created("parent-1", 1)

        # Record a failed attempt
        start_time1 = time.time()
        reporter.record_summary_attempt_v2(
            node_id="parent-1",
            target_tokens=100,
            input_text_tokens=200,
            prompt_tokens=250,
            completion_tokens=130,
            actual_tokens=130,
            status="rejected_over",
            model="gpt-4o-mini",
            start_time=start_time1,
            rejection_reason="30% over target",
        )

        # Record a successful retry
        start_time2 = time.time()
        reporter.record_summary_attempt_v2(
            node_id="parent-1",
            target_tokens=100,
            input_text_tokens=200,
            prompt_tokens=250,
            completion_tokens=95,
            actual_tokens=95,
            status="accepted",
            model="gpt-4o-mini",
            start_time=start_time2,
        )

        # Check aggregate metrics
        assert reporter.metrics.summary_api_calls == 2
        assert reporter.metrics.total_summary_prompt_tokens == 500
        assert reporter.metrics.total_summary_completion_tokens == 225

        # Check telemetry
        node = reporter.metrics.node_telemetry["parent-1"]
        assert len(node.summary_attempts) == 2

        # First attempt (failed)
        attempt1 = node.summary_attempts[0]
        assert attempt1.status == "rejected_over"
        assert attempt1.rejection_reason == "30% over target"
        assert attempt1.completion_tokens == 130
        assert attempt1.start_time > 0
        assert attempt1.end_time > attempt1.start_time

        # Second attempt (successful)
        attempt2 = node.summary_attempts[1]
        assert attempt2.status == "accepted"
        assert attempt2.completion_tokens == 95
        assert attempt2.start_time > attempt1.end_time
        assert attempt2.end_time > attempt2.start_time

    def test_backward_compatibility(self, reporter: IndexingMetricsReporter) -> None:
        """Test that old methods still work without telemetry."""
        # Use old method
        reporter.record_embedding_call(
            batch_size=3,
            token_counts=[50, 45, 48],
        )

        # Check aggregate metrics work
        assert reporter.metrics.embedding_api_calls == 1
        assert reporter.metrics.total_embedding_tokens == 143

        # No telemetry should be created
        assert len(reporter.metrics.node_telemetry) == 0


class TestTelemetryIntegration:
    """Test telemetry integration with real indexing."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("store_type", ["mock", "real"])
    async def test_telemetry_captures_all_nodes(
        self, store_type: str, request: pytest.FixtureRequest
    ) -> None:
        """Test that telemetry captures all nodes during indexing."""
        store = request.getfixturevalue(f"{store_type}_store")
        config = RagZoomConfig(
            openai_api_key="test-key",
            leaf_tokens=100,
            adjacent_context_tokens=50,
            embedding_batch_size=2,
        )

        # Create test text that will generate multiple nodes
        # Each chunk should be ~100 tokens, so make each section larger
        test_text = (
            " ".join(["Word" + str(i) for i in range(150)])
            + " BREAK "
            + " ".join(["Text" + str(i) for i in range(150)])
            + " BREAK "
            + " ".join(["Data" + str(i) for i in range(150)])
        )

        # Create mock AsyncOpenAI client
        mock_async_client = MagicMock()

        # Mock embeddings - needs to be async
        async def mock_embeddings(*args: Any, **kwargs: Any) -> Any:
            input_texts = kwargs.get("input", [])
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            # Return appropriate number of embeddings based on store type
            embedding_value = [0.1] * 1536 if store_type == "real" else "mock-embedding"
            return MagicMock(
                data=[MagicMock(embedding=embedding_value) for _ in input_texts]
            )

        mock_async_client.embeddings.create = mock_embeddings

        # Mock summaries - needs to be async
        async def mock_chat_completion(*args: Any, **kwargs: Any) -> Any:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message = MagicMock()
            response.choices[0].message.content = "Summary of content"
            # Add usage data for telemetry
            response.usage = MagicMock()
            response.usage.prompt_tokens = 250
            response.usage.completion_tokens = 50
            return response

        mock_async_client.chat.completions.create = mock_chat_completion

        # Index with mocked client
        with patch("ragzoom.index.AsyncOpenAI", return_value=mock_async_client):
            builder = TreeBuilder(config, store)

            # Create reporter for metrics
            source_tokens = len(builder.splitter.tokenizer.encode(test_text))
            reporter = IndexingMetricsReporter(
                document_id="telemetry-test",
                source_tokens=source_tokens,
                config=config,
            )

            # Index document
            _ = await builder._add_document_impl(
                test_text,
                document_id="telemetry-test",
                file_path=None,
                show_progress=False,
                reporter=reporter,
            )

        # Get final metrics
        metrics = reporter.finalize()

        # Verify telemetry was collected
        assert len(metrics.node_telemetry) > 0

        # Count node types by height (height 0 = leaves)
        leaf_count = sum(1 for n in metrics.node_telemetry.values() if n.height == 0)
        summary_count = sum(1 for n in metrics.node_telemetry.values() if n.height > 0)

        # Should have multiple leaves from our test text
        assert leaf_count >= 3
        # Should have summary nodes since we have multiple leaves
        assert summary_count >= 1

        # Every node should have embedding telemetry
        for node_id, node_data in metrics.node_telemetry.items():
            assert node_data.embedding is not None, f"Node {node_id} missing embedding"
            assert node_data.embedding.model in [
                "text-embedding-3-small",
                "mock-embedding",
            ]

            # Summary nodes (height > 0) should have summary attempts
            if node_data.height > 0:
                assert len(node_data.summary_attempts) > 0
                # At least one attempt should be accepted
                assert any(a.status == "accepted" for a in node_data.summary_attempts)

    def test_telemetry_serialization(self, base_config: RagZoomConfig) -> None:
        """Test that telemetry can be serialized to JSON."""
        import json

        reporter = IndexingMetricsReporter("test", 1000, base_config)

        # Create some telemetry
        reporter.track_node_created("node-1", 0)
        start_time = time.time()
        reporter.record_embedding_call_v2(
            [("node-1", 50)], 1, "text-embedding-3-small", start_time
        )

        # Convert to dict format (like benchmark output)
        telemetry_dict = {}
        for node_id, node_data in reporter.metrics.node_telemetry.items():
            telemetry_dict[node_id] = node_data.to_telemetry_dict()

        # Should be JSON serializable
        json_str = json.dumps(telemetry_dict)
        assert json_str is not None

        # Can round-trip
        loaded = json.loads(json_str)
        assert loaded["node-1"]["height"] == 0
        assert "node_type" not in loaded["node-1"]  # Field removed
        assert "span" not in loaded["node-1"]  # Field removed
        assert loaded["node-1"]["embedding"]["text_tokens"] == 50
        assert loaded["node-1"]["embedding"]["start_time"] > 0
        assert (
            loaded["node-1"]["embedding"]["end_time"]
            > loaded["node-1"]["embedding"]["start_time"]
        )
