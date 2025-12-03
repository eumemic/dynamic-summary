"""SQLite-based document isolation tests using the runtime harness."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from openai import AsyncOpenAI

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.splitter import TextSplitter
from tests.conftest import BackwardCompatibilityConfig, IndexerRuntimeHarness
from tests.utils import create_retriever, mock_openai_context
from tests.vector_index_stubs import RecordingVectorIndex


def _configure_runtime(
    harness: IndexerRuntimeHarness,
    config: IndexConfig,
    vector_index: RecordingVectorIndex,
) -> None:
    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = TextSplitter(config)
    harness.worker_coordinator._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config
    vector_factory = lambda _model: vector_index  # noqa: E731
    harness.runtime._vector_index_factory = vector_factory
    harness.worker_coordinator._vector_index_factory = vector_factory


class TestDocumentIsolationSQLite:
    """Test that queries are properly isolated to specific documents."""

    @pytest.fixture
    def mock_openai(
        self,
    ) -> Generator[tuple[AsyncOpenAI, AsyncOpenAI, AsyncOpenAI], None, None]:
        """Mock OpenAI API calls with specialized embedding rules."""
        embedding_rules = {
            "dragon": [0.9] * 1536,
            "wizard": [0.8] * 1536,
        }
        with mock_openai_context(embedding_rules) as mocks:
            yield mocks

    @pytest.mark.asyncio
    async def test_document_isolation(
        self,
        storage_backend: StorageBackend,
        mock_openai: tuple[AsyncOpenAI, AsyncOpenAI, AsyncOpenAI],
        indexer_runtime_harness: IndexerRuntimeHarness,
        base_config: BackwardCompatibilityConfig,
    ) -> None:
        """Test that queries only return results from the specified document."""
        index_config = IndexConfig.load(target_chunk_tokens=100)
        query_config = QueryConfig(budget_tokens=1000)
        vector_index = RecordingVectorIndex()

        _configure_runtime(indexer_runtime_harness, index_config, vector_index)
        indexer_runtime_harness.llm_service.client = mock_openai[0]

        storage_backend.clear_document("dragons.txt")
        storage_backend.clear_document("wizards.txt")
        dragons_store = storage_backend.for_document("dragons.txt")
        wizards_store = storage_backend.for_document("wizards.txt")

        dragons_store.set_metadata(
            file_path="dragons.txt",
            embedding_model=index_config.embedding_model,
            summary_model=index_config.summary_model,
        )
        wizards_store.set_metadata(
            file_path="wizards.txt",
            embedding_model=index_config.embedding_model,
            summary_model=index_config.summary_model,
        )

        doc1_text = "The mighty dragon breathed fire upon the castle. Dragons are powerful creatures."
        doc2_text = "The wise wizard cast a spell. Wizards study magic for many years."

        try:
            await indexer_runtime_harness.clear("dragons.txt")
            await indexer_runtime_harness.clear("wizards.txt")

            await indexer_runtime_harness.append(
                "dragons.txt",
                doc1_text,
                replace_existing=True,
                file_path="dragons.txt",
            )
            await indexer_runtime_harness.append(
                "wizards.txt",
                doc2_text,
                replace_existing=True,
                file_path="wizards.txt",
            )
            await indexer_runtime_harness.wait_for_idle()

            retriever1 = create_retriever(
                query_config,
                dragons_store,
                document_id="dragons.txt",
                api_key="test-key",
                client=mock_openai[1],
                vector_index=vector_index,
            )
            retriever2 = create_retriever(
                query_config,
                wizards_store,
                document_id="wizards.txt",
                api_key="test-key",
                client=mock_openai[1],
                vector_index=vector_index,
            )

            result1 = await retriever1.retrieve_async(
                "tell me about dragons", document_id="dragons.txt"
            )
            for node_id in result1.node_ids:
                retrieved_node = dragons_store.nodes.get_node(node_id)
                assert retrieved_node is not None
                assert retrieved_node.document_id == "dragons.txt"

            result2 = await retriever2.retrieve_async(
                "tell me about wizards", document_id="wizards.txt"
            )
            for node_id in result2.node_ids:
                retrieved_node = wizards_store.nodes.get_node(node_id)
                assert retrieved_node is not None
                assert retrieved_node.document_id == "wizards.txt"

            result3 = await retriever2.retrieve_async(
                "tell me about dragons", document_id="wizards.txt"
            )
            for node_id in result3.node_ids:
                retrieved_node = wizards_store.nodes.get_node(node_id)
                assert retrieved_node is not None
                assert retrieved_node.document_id == "wizards.txt"

            assembler = Assembler(wizards_store)
            summary = assembler.assemble(result3)
            assert "wizard" in summary.lower() or "magic" in summary.lower()
            assert "dragon" not in summary.lower()
        finally:
            await indexer_runtime_harness.clear("dragons.txt")
            await indexer_runtime_harness.clear("wizards.txt")

    @pytest.mark.asyncio
    async def test_filename_as_default_document_id(
        self,
        storage_backend: StorageBackend,
        mock_openai: tuple[AsyncOpenAI, AsyncOpenAI, AsyncOpenAI],
        indexer_runtime_harness: IndexerRuntimeHarness,
        base_config: BackwardCompatibilityConfig,
    ) -> None:
        """Test that filename is used as document_id when not specified."""
        index_config = IndexConfig.load(target_chunk_tokens=100)
        query_config = QueryConfig(budget_tokens=1000)
        vector_index = RecordingVectorIndex()

        _configure_runtime(indexer_runtime_harness, index_config, vector_index)
        indexer_runtime_harness.llm_service.client = mock_openai[0]

        document_id = "test_file.txt"
        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path=document_id,
            embedding_model=index_config.embedding_model,
            summary_model=index_config.summary_model,
        )

        text = "Test content for filename ID"

        try:
            await indexer_runtime_harness.clear(document_id)
            result = await indexer_runtime_harness.append(
                document_id,
                text,
                replace_existing=True,
                file_path=document_id,
            )
            await indexer_runtime_harness.wait_for_idle(document_id)

            assert result.document_id == document_id

            retriever = create_retriever(
                query_config,
                doc_store,
                document_id=document_id,
                api_key="test-key",
                client=mock_openai[1],
                vector_index=vector_index,
            )

            retrieval = await retriever.retrieve_async(
                "show me the content", document_id=document_id
            )
            assembler = Assembler(doc_store)
            summary = assembler.assemble(retrieval)
            assert "test content" in summary.lower()
        finally:
            await indexer_runtime_harness.clear(document_id)
