"""End-to-end workflow tests for RagZoom (index → query → assemble).

These tests verify the complete pipeline from document indexing through
retrieval to final assembly, using mock OpenAI clients.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import Mock

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
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
