"""Test node-level telemetry collection."""

import time
from unittest.mock import MagicMock, patch

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol
from ragzoom.index import TreeBuilder
from ragzoom.telemetry_collection import (
    NodeTelemetry,
    SummaryAttempt,
    TelemetryCollector,
)
from tests.utils import create_telemetry_summary_mock


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
            target_tokens=100,
            prompt_tokens=500,
            completion_tokens=105,
            actual_tokens=105,
            model="gpt-4o-mini",
            start_time=1234567890.0,
            end_time=1234567891.0,
        )

        # Verify timing fields exist
        assert attempt.start_time == 1234567890.0
        assert attempt.end_time == 1234567891.0
        assert attempt.deviation_percent == 5.0  # 105/100 - 1 = 5%
        # Verify status field is removed
        assert not hasattr(attempt, "status")
        # Verify is_retry field removed
        assert not hasattr(attempt, "is_retry")


class TestTelemetryCollection:
    """Test telemetry collection during indexing."""

    @pytest.fixture
    def config(self) -> IndexConfig:
        """Create test config."""
        return IndexConfig.load(
            target_chunk_tokens=100,
            preceding_context_tokens=50,
            embedding_batch_size=2,
        )

    @pytest.fixture
    def reporter(self, config: IndexConfig) -> TelemetryCollector:
        """Create test reporter."""
        return TelemetryCollector(
            document_id="test-doc",
            source_tokens=1000,
            config=config,
        )

    def test_track_node_created(self, reporter: TelemetryCollector) -> None:
        """Test node creation tracking."""
        reporter.track_node_created(
            node_id="leaf-1",
            height=0,
        )

        assert "leaf-1" in reporter.node_telemetry
        node = reporter.node_telemetry["leaf-1"]
        assert node.height == 0
        # Verify node is tracked
        assert node.node_id == "leaf-1"
        # Verify node_type field removed
        assert not hasattr(node, "node_type")

    def test_record_embedding_v2(self, reporter: TelemetryCollector) -> None:
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
        assert reporter.embedding_api_calls == 1
        assert reporter.total_embedding_tokens == 93

        # Check telemetry
        node1 = reporter.node_telemetry["node-1"]
        assert node1.embedding is not None
        assert node1.embedding.text_tokens == 45
        assert node1.embedding.batch_size == 2
        assert node1.embedding.batch_position == 0
        assert node1.embedding.model == "text-embedding-3-small"

        node2 = reporter.node_telemetry["node-2"]
        assert node2.embedding is not None
        assert node2.embedding.batch_position == 1

    def test_record_summary_attempt_v2(self, reporter: TelemetryCollector) -> None:
        """Test v2 summary recording with telemetry."""
        # Track a summary node
        reporter.track_node_created("parent-1", 1)

        # Record first attempt (30% over target)
        start_time1 = time.time()
        reporter.record_summary_attempt_v2(
            node_id="parent-1",
            target_tokens=100,
            input_text_tokens=200,
            prompt_tokens=250,
            completion_tokens=130,
            actual_tokens=130,
            model="gpt-4o-mini",
            start_time=start_time1,
        )

        # Record a retry (5% under target - acceptable)
        start_time2 = time.time()
        reporter.record_summary_attempt_v2(
            node_id="parent-1",
            target_tokens=100,
            input_text_tokens=200,
            prompt_tokens=250,
            completion_tokens=95,
            actual_tokens=95,
            model="gpt-4o-mini",
            start_time=start_time2,
        )

        # Check aggregate metrics
        assert reporter.summary_api_calls == 2
        assert reporter.total_summary_prompt_tokens == 500
        assert reporter.total_summary_completion_tokens == 225

        # Check telemetry
        node = reporter.node_telemetry["parent-1"]
        assert len(node.summary_attempts) == 2

        # First attempt (30% over)
        attempt1 = node.summary_attempts[0]
        assert attempt1.deviation_percent == 30.0  # 130/100 - 1 = 30%
        assert attempt1.completion_tokens == 130
        assert attempt1.start_time > 0
        assert attempt1.end_time >= attempt1.start_time

        # Second attempt (5% under)
        attempt2 = node.summary_attempts[1]
        assert attempt2.deviation_percent == -5.0  # 95/100 - 1 = -5%
        assert attempt2.completion_tokens == 95
        assert attempt2.start_time >= attempt1.end_time
        assert attempt2.end_time >= attempt2.start_time

    def test_backward_compatibility(self, reporter: TelemetryCollector) -> None:
        """Test that old methods still work without telemetry."""
        # Use old method
        reporter.record_embedding_call(
            batch_size=3,
            token_counts=[50, 45, 48],
        )

        # Check aggregate metrics work
        assert reporter.embedding_api_calls == 1
        assert reporter.total_embedding_tokens == 143

        # No telemetry should be created
        assert len(reporter.node_telemetry) == 0


class TestTelemetryIntegration:
    """Test telemetry integration with real indexing."""

    @pytest.mark.asyncio
    async def test_telemetry_captures_all_nodes(
        self, storage_backend: StorageBackend, vector_index: _VectorIndexProtocol
    ) -> None:
        """Test that telemetry captures all nodes during indexing."""
        index_config = IndexConfig.load(
            target_chunk_tokens=100,
            preceding_context_tokens=50,
            embedding_batch_size=2,
        )
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test-key"),
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

        # Create mock with telemetry-specific behavior
        mock_async_client = MagicMock()

        # Mock embeddings with store-type specific behavior
        async def mock_embeddings(*args: object, **kwargs: object) -> object:
            from typing import cast

            input_texts = cast(list[str] | str, kwargs.get("input", []))
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            # Always return numeric embeddings for backend-agnostic runs
            embedding_value = [0.1] * 1536
            return MagicMock(
                data=[MagicMock(embedding=embedding_value) for _ in input_texts]
            )

        mock_async_client.embeddings.create = mock_embeddings

        # Use centralized telemetry summary mock
        _, telemetry_chat_async = create_telemetry_summary_mock()
        mock_async_client.chat.completions.create = telemetry_chat_async

        # Index with mocked client
        with patch(
            "ragzoom.services.llm_service.AsyncOpenAI", return_value=mock_async_client
        ):
            # Create document-scoped store and metadata
            doc_store = storage_backend.for_document("telemetry-test")
            doc_store.set_metadata(
                file_path=None,
                embedding_model="text-embedding-3-small",
                summary_model="gpt-4o-mini",
            )
            builder = TreeBuilder(
                index_config,
                doc_store,
                vector_index,
                operational_config.openai_api_key.get_secret_value(),
            )

            # Create reporter for metrics
            source_tokens = len(builder.splitter.tokenizer.encode(test_text))
            reporter = TelemetryCollector(
                document_id="telemetry-test",
                source_tokens=source_tokens,
                config=index_config,
            )

            # Index document (file_path parameter removed in refactoring)
            _ = await builder._add_document_impl(
                test_text,
                show_progress=False,
                reporter=reporter,
            )

        # Get final telemetry data
        telemetry_data = reporter.finalize()

        # Verify telemetry was collected (v4.2 format)
        assert telemetry_data["format_version"] == "4.2"
        assert telemetry_data["document_id"] == "telemetry-test"
        assert "nodes" in telemetry_data

        # Verify config field is present (new in v4.0)
        assert "config" in telemetry_data
        config = telemetry_data["config"]
        assert "target_chunk_tokens" in config
        assert "preceding_context_tokens" in config
        assert "summary_model" in config
        assert "embedding_model" in config

        # Verify new reproducibility fields (new in v4.1)
        assert "model_metadata" in telemetry_data
        assert "system_prompts" in telemetry_data
        assert "runtime_info" in telemetry_data

        # Verify model metadata includes necessary details
        model_metadata = telemetry_data["model_metadata"]
        assert "embedding" in model_metadata
        assert "summary" in model_metadata

        # Verify system prompts are captured
        system_prompts = telemetry_data["system_prompts"]
        assert "summary_system_prompt" in system_prompts

        # Verify runtime info is captured
        runtime_info = telemetry_data["runtime_info"]
        assert "python_version" in runtime_info
        assert "platform" in runtime_info
        assert "ragzoom_version" in runtime_info

        nodes = telemetry_data["nodes"]
        assert len(nodes) > 0

        # Count node types by height (height 0 = leaves)
        leaf_count = sum(1 for n in nodes if n["height"] == 0)
        summary_count = sum(1 for n in nodes if n["height"] > 0)

        # Should have multiple leaves from our test text
        assert leaf_count >= 3
        # Should have summary nodes since we have multiple leaves
        assert summary_count >= 1

        # Every node should have embedding telemetry
        for node_data in nodes:
            assert (
                "embedding" in node_data
            ), f"Node {node_data['node_id']} missing embedding"
            assert node_data["embedding"]["model"] == "text-embedding-3-small"

            # Summary nodes (height > 0) should have summary attempts
            # unless they are passthrough nodes (text was already short enough)
            if node_data["height"] > 0:
                # Only check for summary_attempts if the node actually performed a summary
                # Passthrough nodes won't have summary_attempts
                if "summary_attempts" in node_data:
                    assert len(node_data["summary_attempts"]) > 0
                    # The node should have marked which attempt was accepted
                    # (or we fall back to the last attempt for backward compatibility)
                    has_accepted = "accepted_attempt" in node_data
                    if not has_accepted:
                        # For backward compatibility: if no accepted_attempt field,
                        # the last attempt should be the one used
                        assert len(node_data["summary_attempts"]) > 0

    def test_telemetry_serialization(self, base_config: object) -> None:
        """Test that telemetry can be serialized to JSON."""
        import json
        from typing import cast

        from tests.conftest import BackwardCompatibilityConfig

        config = cast(BackwardCompatibilityConfig, base_config)
        reporter = TelemetryCollector("test", 1000, config.index_config)

        # Create some telemetry
        reporter.track_node_created("node-1", 0)
        start_time = time.time()
        reporter.record_embedding_call_v2(
            [("node-1", 50)], 1, "text-embedding-3-small", start_time
        )

        # Convert to dict format (like benchmark output)
        telemetry_dict = {}
        for node_id, node_data in reporter.node_telemetry.items():
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
