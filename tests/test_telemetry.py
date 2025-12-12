"""Test node-level telemetry collection."""

import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.telemetry_collection import (
    NodeTelemetry,
    SummaryAttempt,
    TelemetryCollector,
)
from ragzoom.telemetry_embeddings import (
    annotate_telemetry_fidelity,
    compute_fidelity_for_telemetry,
)
from tests.conftest import IndexerRuntimeHarness
from tests.utils import create_telemetry_summary_mock
from tests.vector_index_stubs import RecordingVectorIndex


def _configure_runtime(
    harness: IndexerRuntimeHarness,
    config: IndexConfig,
    vector_index: RecordingVectorIndex,
) -> None:
    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = (
        harness.runtime._append_executor._splitter.__class__(config)
    )
    harness.worker_coordinator._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config
    vector_factory = lambda _model: vector_index  # noqa: E731
    harness.runtime._vector_index_factory = vector_factory
    harness.worker_coordinator._vector_index_factory = vector_factory


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
            node_embeddings=[
                ("node-1", 45),
                ("node-2", 48),
            ],
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
        assert node2.embedding.model == "text-embedding-3-small"

    def test_record_node_fidelity(self, reporter: TelemetryCollector) -> None:
        """Verify node fidelity is recorded."""

        reporter.track_node_created("node-1", 1)
        reporter.record_node_fidelity("node-1", 0.85)

        node = reporter.node_telemetry["node-1"]
        assert node.fidelity == 0.85

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

    def test_record_chunk_split(self, reporter: TelemetryCollector) -> None:
        """Chunk split timings should be captured in telemetry output."""

        start = time.time()
        reporter.record_chunk_split_start(
            start_time=start,
            new_text_chars=120,
            existing_tail_chars=40,
            combined_chars=160,
        )
        reporter.record_chunk_split_end(
            end_time=start + 0.5,
            chunk_count=6,
            total_tokens=420,
        )

        telemetry = reporter.get_telemetry_data("test-doc", 100)
        assert "chunk_split" in telemetry
        chunk_split = telemetry["chunk_split"]
        assert chunk_split["chunk_count"] == 6
        assert chunk_split["total_tokens"] == 420
        assert chunk_split["new_text_chars"] == 120
        assert chunk_split["existing_tail_chars"] == 40
        assert chunk_split["combined_chars"] == 160
        assert chunk_split["duration"] == pytest.approx(0.5, rel=1e-6)

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

    @pytest.mark.asyncio
    async def test_compute_fidelity_reembeds_missing_parent_vectors(
        self, reporter: TelemetryCollector
    ) -> None:
        """Telemetry fidelity computation should re-embed when vectors are absent."""

        @dataclass
        class _Node:
            id: str
            text: str
            height: int
            document_id: str = "test-doc"
            left_child_id: str | None = None
            right_child_id: str | None = None
            span_start: int = 0
            span_end: int = 0

        class _Nodes:
            def __init__(self, nodes: list[_Node]) -> None:
                self._nodes = {node.id: node for node in nodes}

            def get_many(self, ids: list[str]) -> list[_Node]:
                return [
                    self._nodes[node_id] for node_id in ids if node_id in self._nodes
                ]

            def get(self, node_id: str) -> _Node | None:
                return self._nodes.get(node_id)

        class _Store:
            def __init__(self, nodes: list[_Node]) -> None:
                self.nodes = _Nodes(nodes)

        class _MissingVectorIndex:
            def get_vectors(self, ids: list[str]) -> list[object]:
                raise KeyError(f"Vector not found for id {ids[0]}")

        class _MappingEmbedder:
            def __init__(self, mapping: dict[str, list[float]]) -> None:
                self.mapping = mapping

            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [self.mapping[text] for text in texts]

        child_a = _Node(id="child-a", text="Bilbo meets Gandalf", height=0)
        child_b = _Node(id="child-b", text="They plan an adventure", height=0)
        parent = _Node(
            id="parent",
            text="Bilbo and Gandalf plan a grand adventure",
            height=1,
            left_child_id=child_a.id,
            right_child_id=child_b.id,
        )
        store = _Store([parent, child_a, child_b])

        combined_children = f"{child_a.text}\n{child_b.text}"
        embedder = _MappingEmbedder(
            {
                parent.text: [1.0, 0.0],
                combined_children: [0.6, 0.8],
            }
        )

        reporter.track_node_created(parent.id, parent.height)
        reporter.track_node_created(child_a.id, child_a.height)
        reporter.track_node_created(child_b.id, child_b.height)

        await compute_fidelity_for_telemetry(
            document_store=store,  # type: ignore[arg-type]
            collector=reporter,
            embedder=embedder,
            token_limit=2048,
            max_batch_items=16,
        )

        fidelity = reporter.node_telemetry[parent.id].fidelity
        assert fidelity is not None
        assert fidelity == pytest.approx(0.6, rel=1e-6)

    @pytest.mark.asyncio
    async def test_annotate_telemetry_fidelity_updates_nodes(self) -> None:
        """Exported telemetry payloads should be annotated with fidelity values."""

        @dataclass
        class _Node:
            id: str
            text: str
            height: int
            span_start: int
            span_end: int
            document_id: str = "doc"
            parent_id: str | None = None
            left_child_id: str | None = None
            right_child_id: str | None = None

        class _Nodes:
            def __init__(self, nodes: list[_Node]) -> None:
                self._nodes = {node.id: node for node in nodes}

            def get_many(self, ids: list[str]) -> list[_Node]:
                return [
                    self._nodes[node_id] for node_id in ids if node_id in self._nodes
                ]

            def get(self, node_id: str) -> _Node | None:
                return self._nodes.get(node_id)

        class _Store:
            def __init__(self, nodes: list[_Node]) -> None:
                self.nodes = _Nodes(nodes)

            def get_embedding_model(self) -> str:
                return "text-embedding-3-small"

        class _MappingEmbedder:
            def __init__(self, mapping: dict[str, list[float]]) -> None:
                self.mapping = mapping

            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [self.mapping[text] for text in texts]

        class _MissingVectorIndex:
            def get_vectors(self, ids: list[str]) -> list[object]:
                raise KeyError(f"Vector not found for id {ids[0]}")

        parent = _Node(
            id="parent",
            text="Summary",
            height=1,
            span_start=0,
            span_end=64,
        )
        child_a = _Node(
            id="child-a",
            text="Chunk A",
            height=0,
            span_start=0,
            span_end=32,
            parent_id=parent.id,
        )
        child_b = _Node(
            id="child-b",
            text="Chunk B",
            height=0,
            span_start=32,
            span_end=64,
            parent_id=parent.id,
        )
        parent.left_child_id = child_a.id
        parent.right_child_id = child_b.id

        store = _Store([parent, child_a, child_b])
        combined_children = f"{child_a.text}\n{child_b.text}"
        embedder = _MappingEmbedder(
            {parent.text: [1.0, 0.0], combined_children: [0.6, 0.8]}
        )

        telemetry_nodes = [
            {"node_id": parent.id, "height": 1},
            {"node_id": child_a.id, "height": 0},
            {"node_id": child_b.id, "height": 0},
        ]

        await annotate_telemetry_fidelity(
            document_store=store,  # type: ignore[arg-type]
            telemetry_nodes=telemetry_nodes,
            embedder=embedder,
            token_limit=2048,
            max_batch_items=16,
        )

        parent_entry = telemetry_nodes[0]
        fidelity = parent_entry.get("fidelity")
        assert fidelity is not None
        assert fidelity == pytest.approx(0.6, rel=1e-6)


class TestTelemetryIntegration:
    """Test telemetry integration with real indexing."""

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(3.0)
    async def test_telemetry_captures_all_nodes(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that telemetry captures all nodes during indexing."""
        index_config = IndexConfig.load(
            target_chunk_tokens=100,
            preceding_context_tokens=50,
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

        # Create mock with telemetry-specific behavior
        vector_index = RecordingVectorIndex()
        _configure_runtime(indexer_runtime_harness, index_config, vector_index)

        mock_async_client = AsyncMock()

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

        indexer_runtime_harness.llm_service.client = mock_async_client

        document_id = "telemetry-test"
        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        await indexer_runtime_harness.append(
            document_id,
            test_text,
            replace_existing=True,
            file_path="telemetry-test.txt",
            collect_telemetry=True,
        )
        await indexer_runtime_harness.wait_for_idle(document_id)

        run_context = (
            await indexer_runtime_harness.telemetry_manager.latest_for_document(
                document_id
            )
        )
        assert run_context is not None
        completed = await indexer_runtime_harness.telemetry_manager.wait_for_completion(
            run_context
        )
        telemetry_data = completed.result
        assert telemetry_data is not None

        # Verify telemetry was collected (current format)
        assert telemetry_data["format_version"] == "4.3"
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

        # Leaf nodes (height == 0) have embeddings generated asynchronously,
        # so embedding telemetry may not be present during append.
        # Summary nodes (height > 0) do not have embeddings - scores come from
        # bottom-up propagation from leaves.
        for node_data in nodes:
            if node_data["height"] == 0:
                # Leaf nodes may have embedding telemetry (if captured during async phase)
                # but it's not required since embedding is now async
                if "embedding" in node_data:
                    assert node_data["embedding"]["model"] == "text-embedding-3-small"
            else:
                # Summary nodes should NOT have embeddings
                assert (
                    "embedding" not in node_data
                ), f"Summary node {node_data['node_id']} should not have embedding"
                # Summary nodes should have summary attempts
                # unless they are passthrough nodes (text was already short enough)
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
        reporter.record_node_fidelity("node-1", 0.77)

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
        assert loaded["node-1"]["fidelity"] == 0.77
