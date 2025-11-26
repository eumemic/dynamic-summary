"""Dynamic programming algorithm integration tests using the runtime harness."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

import pytest
from pytest import MonkeyPatch

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.splitter import TextSplitter
from tests.conftest import BackwardCompatibilityConfig, IndexerRuntimeHarness
from tests.utils import (
    create_hash_based_embedding_mock,
    create_predictable_summary_mock,
    create_retriever,
    mock_openai_context,
)
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
    harness.runtime._vector_index_factory = lambda _model: vector_index
    harness.worker_coordinator._vector_index_factory = lambda _doc_id: vector_index


class TestDPIntegration:
    """Exercise the DP tiling pipeline end-to-end against a runtime instance."""

    @pytest.fixture
    def config(
        self,
        config_factory: Callable[
            [int, int, int, str, str | None], BackwardCompatibilityConfig
        ],
    ) -> BackwardCompatibilityConfig:
        return config_factory(50, 0, 500, "test-key", None)

    @pytest.fixture
    def mock_openai(self, monkeypatch: MonkeyPatch) -> tuple[Mock, Mock]:
        """Provide deterministic embedding and summarisation clients."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with mock_openai_context() as (mock_index, mock_retrieve, mock_assemble):
            hash_sync, hash_async = create_hash_based_embedding_mock()
            mock_index.embeddings.create = hash_async
            mock_retrieve.embeddings.create = hash_sync

            chat_sync, chat_async = create_predictable_summary_mock()
            mock_index.chat.completions.create = chat_async

            return mock_retrieve, mock_index

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(2.0)
    async def test_no_duplicate_content(
        self,
        config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai: tuple[Mock, Mock],
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        mock_client, mock_index_client = mock_openai
        index_config = config.index_config
        query_config = config.query_config
        vector_index = RecordingVectorIndex()
        _configure_runtime(indexer_runtime_harness, index_config, vector_index)
        indexer_runtime_harness.llm_service.client = mock_index_client

        document_id = "dp-doc"
        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path="dp_integration_test.txt",
            embedding_model=index_config.embedding_model,
            summary_model=index_config.summary_model,
        )

        base_lines = [
            "First chunk of text that should appear once.",
            "Second chunk of text that should also appear once.",
            "Third chunk of text with unique content.",
            "Fourth chunk of text to complete the document.",
        ]
        document = "\n".join(base_lines * 8)

        try:
            await indexer_runtime_harness.clear(document_id)
            await indexer_runtime_harness.append(
                document_id,
                document,
                replace_existing=True,
                file_path="dp_integration_test.txt",
            )
            await indexer_runtime_harness.wait_for_idle(document_id)

            retriever = create_retriever(
                query_config,
                doc_store,
                document_id=document_id,
                api_key=config.openai_api_key,
                client=mock_client,
                vector_index=vector_index,
            )
            result = await retriever.retrieve_async(
                "First chunk Second chunk", document_id=document_id
            )

            assembler = Assembler(doc_store)
            assembled = assembler.assemble(result)
            lines = assembled.strip().split("\n")
            unique_lines = {line for line in lines if line}
            for line in unique_lines:
                assert (
                    lines.count(line) <= 8
                ), f"Line '{line}' appears more than source repetitions"
            assert "First chunk" in assembled
            assert "Second chunk" in assembled
        finally:
            await indexer_runtime_harness.clear(document_id)

    @pytest.mark.asyncio
    async def test_parent_child_deduplication(
        self,
        config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai: tuple[Mock, Mock],
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        mock_client, mock_index_client = mock_openai
        small_config = config.index_config.replace(target_chunk_tokens=10)
        vector_index = RecordingVectorIndex()
        _configure_runtime(indexer_runtime_harness, small_config, vector_index)
        indexer_runtime_harness.llm_service.client = mock_index_client

        document_id = "dp-doc"
        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path="dp_integration_test.txt",
            embedding_model=small_config.embedding_model,
            summary_model=small_config.summary_model,
        )

        document = "This is a test document with some content."

        try:
            await indexer_runtime_harness.clear(document_id)
            await indexer_runtime_harness.append(
                document_id,
                document,
                replace_existing=True,
                file_path="dp_integration_test.txt",
            )
            await indexer_runtime_harness.wait_for_idle(document_id)

            retriever = create_retriever(
                config.query_config,
                doc_store,
                document_id=document_id,
                api_key=config.openai_api_key,
                client=mock_client,
                vector_index=vector_index,
            )
            result = await retriever.retrieve_async(
                "test document", document_id=document_id
            )

            tiling_node_ids = list(set(result.tiling or []))
            tiling_nodes = [
                doc_store.nodes.get_node(node_id) for node_id in tiling_node_ids
            ]
            valid_nodes = [node for node in tiling_nodes if node is not None]
            for i, node in enumerate(valid_nodes):
                for j, other in enumerate(valid_nodes):
                    if i == j:
                        continue
                    if (
                        node.left_child_id == other.id
                        or node.right_child_id == other.id
                    ):
                        pytest.fail(
                            f"Tiling contains both parent {node.id} and child {other.id}"
                        )
        finally:
            await indexer_runtime_harness.clear(document_id)

    @pytest.mark.asyncio
    async def test_span_coverage(
        self,
        config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai: tuple[Mock, Mock],
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        mock_client, mock_index_client = mock_openai
        small_config = config.index_config.replace(target_chunk_tokens=5)
        vector_index = RecordingVectorIndex()
        _configure_runtime(indexer_runtime_harness, small_config, vector_index)
        indexer_runtime_harness.llm_service.client = mock_index_client

        document_id = "dp-doc"
        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path="dp_integration_test.txt",
            embedding_model=small_config.embedding_model,
            summary_model=small_config.summary_model,
        )

        document = "AAAA BBBB CCCC DDDD"

        try:
            await indexer_runtime_harness.clear(document_id)
            await indexer_runtime_harness.append(
                document_id,
                document,
                replace_existing=True,
                file_path="dp_integration_test.txt",
            )
            await indexer_runtime_harness.wait_for_idle(document_id)

            retriever = create_retriever(
                config.query_config,
                doc_store,
                document_id=document_id,
                api_key=config.openai_api_key,
                client=mock_client,
                vector_index=vector_index,
            )

            assembler = Assembler(doc_store)
            result1 = await retriever.retrieve_async(
                "AAAA BBBB", document_id=document_id
            )
            assembled1 = assembler.assemble(result1)
            assert assembled1

            result2 = await retriever.retrieve_async(
                "CCCC DDDD", document_id=document_id
            )
            assembled2 = assembler.assemble(result2)
            assert assembled2
        finally:
            await indexer_runtime_harness.clear(document_id)

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(2.0)
    async def test_budget_respected(
        self,
        config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai: tuple[Mock, Mock],
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        mock_client, mock_index_client = mock_openai
        vector_index = RecordingVectorIndex()
        _configure_runtime(indexer_runtime_harness, config.index_config, vector_index)
        indexer_runtime_harness.llm_service.client = mock_index_client

        document_id = "dp-doc"
        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path="dp_integration_test.txt",
            embedding_model=config.index_config.embedding_model,
            summary_model=config.index_config.summary_model,
        )

        document = " ".join([f"Sentence {i}." for i in range(100)])

        small_query_config = config.query_config.replace(
            budget_tokens=100, tiling_strategy="dp"
        )

        try:
            await indexer_runtime_harness.clear(document_id)
            await indexer_runtime_harness.append(
                document_id,
                document,
                replace_existing=True,
                file_path="dp_integration_test.txt",
            )
            await indexer_runtime_harness.wait_for_idle(document_id)

            retriever = create_retriever(
                small_query_config,
                doc_store,
                document_id=document_id,
                api_key=config.openai_api_key,
                client=mock_client,
                vector_index=vector_index,
            )
            result = await retriever.retrieve_async(
                "Sentence", document_id=document_id, budget_tokens=100
            )

            assembler = Assembler(doc_store)
            assembled = assembler.assemble(result)
            token_count = assembler.get_token_count(assembled)
            assert token_count <= small_query_config.budget_tokens * 1.1
        finally:
            await indexer_runtime_harness.clear(document_id)
