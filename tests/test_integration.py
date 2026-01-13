"""End-to-end workflow tests for RagZoom (index → query → assemble).

These tests verify the complete pipeline from document indexing through
retrieval to final assembly, using mock OpenAI clients.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from unittest.mock import Mock

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.server.indexing_engine import get_summary_target
from tests.chunk_size_regression_harness import configure_runtime
from tests.conftest import IndexerRuntimeHarness
from tests.utils import mock_openai_context
from tests.vector_index_stubs import RecordingVectorIndex


@pytest.mark.integration
class TestIntegration:
    """Test complete workflow integration scenarios.

    Focus: End-to-end testing of index → retrieve → assemble pipeline
    with various document types and retrieval scenarios.
    """

    @pytest.fixture
    def mock_openai(self) -> Generator[tuple[Mock, Mock, Mock], None, None]:
        """Mock OpenAI API calls using centralized utilities."""
        with mock_openai_context() as mocks:
            yield mocks

    @staticmethod
    def _bind_vector_index(
        harness: IndexerRuntimeHarness, vector_index: RecordingVectorIndex
    ) -> None:
        """Route runtime vector lookups to the provided stub index."""
        vector_factory = lambda _model_id: vector_index  # noqa: E731
        harness.runtime._vector_index_factory = vector_factory
        harness.indexing_engine._vector_index_factory = vector_factory

    @staticmethod
    async def _index_document(
        harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
        document_id: str,
        text: str,
    ) -> None:
        await harness.append(
            document_id,
            text,
            replace_existing=True,
            file_path=f"{document_id}.txt",
        )
        await harness.wait_for_idle(document_id)

    @pytest.mark.asyncio
    async def test_index_and_query(
        self,
        mock_openai: tuple[Mock, Mock, Mock],
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test indexing a document and querying it."""

        index_config = IndexConfig.load(
            target_chunk_tokens=50,
        )
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        text = "The quick brown fox jumps over the lazy dog. " * 20
        await self._index_document(
            indexer_runtime_harness,
            storage_backend,
            document_id="test-doc",
            text=text,
        )

        doc_store = storage_backend.for_document("test-doc")
        leaf_nodes = doc_store.nodes.get_leaves()
        assert len(leaf_nodes) > 0

        root = doc_store.tree.get_root()
        assert root is not None

        from tests.utils import create_retriever

        _, mock_client, _ = mock_openai

        retriever = create_retriever(
            query_config,
            doc_store,
            client=mock_client,
            vector_index=vector_index,
        )
        assembler = Assembler(doc_store)

        result = await retriever.retrieve_async("Tell me about the fox")

        assert len(result.node_ids) > 0
        assert result.tiling is not None
        assert len(result.tiling) > 0

        summary = assembler.assemble(result)
        assert isinstance(summary, str)
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_multiple_documents(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test indexing multiple documents."""

        index_config = IndexConfig.load(
            target_chunk_tokens=50,
        )
        vector_index = RecordingVectorIndex()
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        text1 = "First document content. " * 10
        await self._index_document(
            indexer_runtime_harness,
            storage_backend,
            document_id="doc1",
            text=text1,
        )

        doc1_store = storage_backend.for_document("doc1")
        initial_leaf_count = len(doc1_store.nodes.get_leaves())

        text2 = "Second document content. " * 10
        await self._index_document(
            indexer_runtime_harness,
            storage_backend,
            document_id="doc2",
            text=text2,
        )

        doc2_store = storage_backend.for_document("doc2")
        doc1_leaf_count = len(doc1_store.nodes.get_leaves())
        doc2_leaf_count = len(doc2_store.nodes.get_leaves())
        assert doc1_leaf_count > 0
        assert doc2_leaf_count > 0
        assert doc1_leaf_count + doc2_leaf_count > initial_leaf_count

    @pytest.mark.asyncio
    async def test_mmr_diversity(
        self,
        mock_openai: tuple[Mock, Mock, Mock],
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that MMR returns diverse results."""

        index_config = IndexConfig.load(
            target_chunk_tokens=50,
        )
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        combined_text = """
        The cat sat on the mat. Cats are feline animals.
        Dogs are loyal pets. The dog barked loudly.
        Birds can fly. Eagles are large birds.
        Fish swim in water. Salmon swim upstream.
        Cats and dogs are common pets. Many people love cats.
        """

        await self._index_document(
            indexer_runtime_harness,
            storage_backend,
            document_id="doc-diverse",
            text=combined_text,
        )

        doc_store = storage_backend.for_document("doc-diverse")

        from tests.utils import create_retriever

        _, mock_client, _ = mock_openai
        retriever = create_retriever(
            query_config,
            doc_store,
            client=mock_client,
            vector_index=vector_index,
        )

        result = await retriever.retrieve_async("Tell me about cats", num_seeds=3)

        assert len(result.node_ids) <= 3
        assert len(set(result.node_ids)) == len(result.node_ids)

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(4.0)
    async def test_token_budget_enforcement(
        self,
        mock_openai: tuple[Mock, Mock, Mock],
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that assembly respects token budget."""

        index_config = IndexConfig.load(
            target_chunk_tokens=50,
        )
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        text = "This is a test sentence. " * 200
        await self._index_document(
            indexer_runtime_harness,
            storage_backend,
            document_id="budget-test",
            text=text,
        )

        doc_store = storage_backend.for_document("budget-test")

        from tests.utils import create_retriever

        _, mock_client, _ = mock_openai
        retriever = create_retriever(
            query_config,
            doc_store,
            client=mock_client,
            vector_index=vector_index,
        )
        assembler = Assembler(doc_store)

        result = await retriever.retrieve_async("test sentence", budget_tokens=100)
        summary = assembler.assemble(result)
        token_count = assembler.get_token_count(summary)

        assert token_count <= 110
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_node_pinning(
        self,
        mock_openai: tuple[Mock, Mock, Mock],
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that pinned nodes are always included."""

        index_config = IndexConfig.load(
            target_chunk_tokens=50,
        )
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        combined_text = (
            "Important content. " * 10 + "Other content. " * 10 + "More content. " * 10
        )
        await self._index_document(
            indexer_runtime_harness,
            storage_backend,
            document_id="doc-pinning",
            text=combined_text,
        )

        doc_store = storage_backend.for_document("doc-pinning")
        all_nodes = doc_store.nodes.get_leaves()
        if not all_nodes:
            pytest.skip("Document produced no leaves")

        important_node = all_nodes[0]
        if hasattr(storage_backend, "pin_node"):
            storage_backend.pin_node(important_node.id)
        else:
            pytest.skip("Node pinning not implemented for this backend")

        from tests.utils import create_retriever

        _, mock_client, _ = mock_openai
        retriever = create_retriever(
            query_config,
            doc_store,
            client=mock_client,
            vector_index=vector_index,
        )

        result = await retriever.retrieve_async("unrelated query")
        assert important_node.id in result.coverage_map

    @pytest.mark.asyncio
    async def test_client_managed_chunking_end_to_end(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that target_chunk_tokens=None creates one atomic leaf per input unit.

        Spec: specs/client-managed-chunking.md § Acceptance Criteria #1
        """
        index_config = IndexConfig.load(
            target_chunk_tokens=None,
            target_embedding_context_tokens=200,
        )
        vector_index = RecordingVectorIndex()
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        units = ["Turn A: First conversation turn", "Turn B: Second conversation turn"]

        session = indexer_runtime_harness.runtime.get_session(
            "client-managed-test", file_path="test.txt"
        )
        await session.batch_append_text(units, collect_telemetry=False)
        await indexer_runtime_harness.indexing_engine.wait_until_idle(
            "client-managed-test"
        )

        doc_store = storage_backend.for_document("client-managed-test")
        leaf_nodes = doc_store.nodes.get_leaves()

        assert len(leaf_nodes) == len(units)
        assert [node.text for node in leaf_nodes] == units
        assert all(node.height == 0 for node in leaf_nodes)

        # Verify contiguous span coverage
        expected_offset = 0
        for i, node in enumerate(leaf_nodes):
            assert node.span_start == expected_offset
            assert node.span_end == expected_offset + len(units[i])
            expected_offset = node.span_end

    @pytest.mark.asyncio
    async def test_dynamic_targets_scale_by_height(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that dynamic summary targets produce 2x compression per level.

        Spec: specs/client-managed-chunking.md § Acceptance Criteria #3

        When target_chunk_tokens=None, summary targets should scale dynamically:
        - Height 1: span_tokens / 2^1 = span_tokens / 2
        - Height 2: span_tokens / 2^2 = span_tokens / 4
        - Height 3: span_tokens / 2^3 = span_tokens / 8
        """
        index_config = IndexConfig.load(
            target_chunk_tokens=None,
            target_embedding_context_tokens=200,
        )
        vector_index = RecordingVectorIndex()
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        # Create enough units to build a tree with height >= 2
        # Each unit should be large enough to create a reasonable span
        # Using ~400 chars per unit to get ~100 tokens (at 4 chars/token)
        units = [
            f"Turn {i}: " + ("This is a substantial conversation turn. " * 10)
            for i in range(8)
        ]

        session = indexer_runtime_harness.runtime.get_session(
            "dynamic-target-test", file_path="test.txt"
        )
        await session.batch_append_text(units, collect_telemetry=False)
        await indexer_runtime_harness.indexing_engine.wait_until_idle(
            "dynamic-target-test"
        )

        doc_store = storage_backend.for_document("dynamic-target-test")

        # Verify tree was built with sufficient height
        max_height = doc_store.nodes.max_height()
        assert max_height >= 2, f"Tree should have height >= 2, got {max_height}"

        # Get chars_per_token ratio for this document

        chars_per_token = indexer_runtime_harness.indexing_engine.get_chars_per_token(
            "dynamic-target-test"
        )

        # Query all nodes to find height-2 nodes
        all_nodes = doc_store.nodes.get_all()
        height_2_nodes = [node for node in all_nodes if node.height == 2]

        assert len(height_2_nodes) > 0, "Should have at least one height-2 node"

        # Verify each height-2 node's summary target is ~1/4 of span tokens
        for node in height_2_nodes:
            span_chars = node.span_end - node.span_start
            span_tokens = span_chars / chars_per_token
            expected_target = get_summary_target(span_chars, 2, chars_per_token)

            # At height 2, target should be span_tokens / 4
            # (unless below 50-token floor, in which case target = 0)
            if span_tokens / 4 >= 50:
                # Above floor: should be approximately 1/4 of span
                assert expected_target > 0, "Target should not be passthrough"
                actual_compression = span_tokens / expected_target
                # Should be close to 4x compression (2^2)
                assert (
                    3.5 <= actual_compression <= 4.5
                ), f"Expected ~4x compression, got {actual_compression:.2f}x"
            else:
                # Below floor: should passthrough (target = 0)
                assert expected_target == 0, "Target should signal passthrough"

            # Verify the node was summarized (internal nodes have summaries as text)
            assert node.text is not None, "Height-2 node should have text content"
            assert len(node.text) > 0, "Height-2 node text should not be empty"

    @pytest.mark.asyncio
    async def test_passthrough_floor_integration(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that summaries targeting < 50 tokens pass through unchanged.

        Spec: specs/client-managed-chunking.md § Acceptance Criteria #4

        When dynamic targets fall below the 50-token floor, text should pass
        through unsummarized. This test creates small units that will trigger
        passthrough behavior at higher tree heights.
        """
        index_config = IndexConfig.load(
            target_chunk_tokens=None,
            target_embedding_context_tokens=200,
        )
        vector_index = RecordingVectorIndex()
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        # Create 8 very small units (each ~20 chars = ~5 tokens at 4 chars/token)
        # This ensures that at higher heights, the target will fall below 50 tokens
        units = [f"Turn {i}: Small text" for i in range(8)]

        session = indexer_runtime_harness.runtime.get_session(
            "passthrough-test", file_path="test.txt"
        )
        await session.batch_append_text(units, collect_telemetry=False)
        await indexer_runtime_harness.indexing_engine.wait_until_idle(
            "passthrough-test"
        )

        doc_store = storage_backend.for_document("passthrough-test")

        # Get chars_per_token ratio for this document
        chars_per_token = indexer_runtime_harness.indexing_engine.get_chars_per_token(
            "passthrough-test"
        )

        # Get all nodes to examine passthrough behavior
        all_nodes = doc_store.nodes.get_all()

        # Find nodes where the dynamic target would be below 50 tokens
        passthrough_nodes = []
        for node in all_nodes:
            if node.height > 0:  # Only check internal nodes
                span_chars = node.span_end - node.span_start
                target = get_summary_target(span_chars, node.height, chars_per_token)
                if target == 0:  # Signals passthrough
                    passthrough_nodes.append(node)

        # Should have at least one passthrough node due to small units
        assert (
            len(passthrough_nodes) > 0
        ), "Should have nodes that triggered passthrough floor"

        # Verify passthrough nodes: their text should be the concatenation of children
        # Note: Summarization joins child texts with a space, and passthrough returns
        # this prepared text unchanged (no LLM call, but space-separated)
        for node in passthrough_nodes:
            # Get child nodes
            left_child, right_child = doc_store.tree.get_children(node.id)
            assert left_child is not None, "Left child should exist"
            assert right_child is not None, "Right child should exist"

            # Expected text is space-separated concatenation (as per summary preparation)
            expected_text = f"{left_child.text} {right_child.text}"

            # Verify the node's text matches (passthrough = no summarization)
            assert (
                node.text == expected_text
            ), f"Passthrough node text should be space-separated concatenation of children (height={node.height})"

    @pytest.mark.asyncio
    async def test_backward_compatibility_fixed_chunking(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that existing configs with target_chunk_tokens=int work identically.

        Spec: specs/client-managed-chunking.md § Acceptance Criteria #5

        When target_chunk_tokens is set to an integer value (the original behavior),
        the system should:
        1. Split text into fixed-size chunks based on the token target
        2. Use fixed summary targets (not dynamic)
        3. Produce the same tree structure as before the feature was added
        """
        target_chunk_tokens = 200
        index_config = IndexConfig.load(
            target_chunk_tokens=target_chunk_tokens,
            target_embedding_context_tokens=target_chunk_tokens,
        )
        vector_index = RecordingVectorIndex()
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        # ~2500 chars = ~625 tokens, should create multiple leaves at 200 tokens each
        text = "This is a test sentence. " * 100

        await self._index_document(
            indexer_runtime_harness,
            storage_backend,
            document_id="backward-compat-test",
            text=text,
        )

        doc_store = storage_backend.for_document("backward-compat-test")
        leaf_nodes = doc_store.nodes.get_leaves()

        assert (
            len(leaf_nodes) > 1
        ), "Fixed chunking should split text into multiple leaves"

        chars_per_token = indexer_runtime_harness.indexing_engine.get_chars_per_token(
            "backward-compat-test"
        )
        target_chars = target_chunk_tokens * chars_per_token

        for node in leaf_nodes:
            assert len(node.text) <= target_chars * 1.5, (
                f"Leaf chunk exceeds target by >50%: "
                f"{len(node.text)} chars vs {target_chars:.0f} target"
            )

        full_text = "".join(node.text for node in leaf_nodes)
        assert full_text == text, "Reconstructed text should match original"

        expected_offset = 0
        for node in leaf_nodes:
            assert node.span_start == expected_offset
            expected_offset = node.span_end

    @pytest.mark.asyncio
    async def test_large_unit_truncation_integration(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that units > 50k chars are truncated with warning in client-managed mode.

        Spec: specs/client-managed-chunking.md § Acceptance Criteria #6

        When target_chunk_tokens=None (client-managed chunking mode) and a unit
        exceeds 50,000 characters:
        1. The unit should be truncated to exactly 50,000 characters
        2. A warning should be logged indicating truncation
        3. The leaf node should contain the truncated text
        4. The tree structure should remain valid
        """
        index_config = IndexConfig.load(
            target_chunk_tokens=None,
            target_embedding_context_tokens=200,
        )
        vector_index = RecordingVectorIndex()
        self._bind_vector_index(indexer_runtime_harness, vector_index)
        configure_runtime(indexer_runtime_harness, index_config)

        large_text = "A" * 60_000

        with caplog.at_level(logging.WARNING):
            await self._index_document(
                indexer_runtime_harness,
                storage_backend,
                document_id="large-unit-test",
                text=large_text,
            )

        assert any(
            "truncating" in record.message.lower() and "50000" in record.message
            for record in caplog.records
        ), "Expected warning about truncation to 50,000 characters"

        doc_store = storage_backend.for_document("large-unit-test")
        leaf_nodes = doc_store.nodes.get_leaves()
        assert (
            len(leaf_nodes) == 1
        ), "Client-managed mode should create exactly one leaf"

        leaf = leaf_nodes[0]
        assert len(leaf.text) == 50_000, f"Expected 50k chars, got {len(leaf.text)}"
        assert leaf.text == "A" * 50_000

        assert leaf.height == 0
        assert leaf.span_start == 0
        assert leaf.span_end == 50_000

        root = doc_store.tree.get_root()
        assert root is not None
        assert root.span_start == 0
        assert root.span_end == 50_000
