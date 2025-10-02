"""SQLite regression tests for chunk sizing using the runtime harness."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from openai import AsyncOpenAI

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.splitter import TextSplitter
from ragzoom.utils.tokenization import tokenizer
from tests.chunk_size_regression_harness import (
    SPLITTER_SAMPLE_PARAGRAPH,
    build_test_document,
    configure_runtime,
    seed_manual_chunk_tree,
)
from tests.conftest import IndexerRuntimeHarness


@pytest.mark.usefixtures("sqlite_backend")
class TestChunkSizeRegressionSQLite:
    """Validate chunk sizing end-to-end against the SQLite backend."""

    @pytest.fixture
    def config(self) -> IndexConfig:
        """Provide an IndexConfig tuned for chunk sizing assertions."""

        return IndexConfig.load(target_chunk_tokens=200)

    def test_splitter_creates_correct_chunk_size(self, config: IndexConfig) -> None:
        """TextSplitter should emit chunks near the configured token target."""

        splitter = TextSplitter(config)
        test_text = SPLITTER_SAMPLE_PARAGRAPH * 5

        chunks = splitter.split_text(test_text)
        assert len(chunks) > 1, "Text should split into multiple chunks"

        for index, chunk in enumerate(chunks):
            token_count = tokenizer.count_tokens(chunk)
            min_tokens = int(config.target_chunk_tokens * 0.3)
            max_tokens = int(config.target_chunk_tokens * 1.2)

            if index < len(chunks) - 1:
                assert (
                    min_tokens <= token_count <= max_tokens
                ), f"Chunk {index} has {token_count} tokens; expected near {config.target_chunk_tokens}"
            else:
                assert (
                    token_count > 0 and token_count <= max_tokens
                ), f"Last chunk {index} has {token_count} tokens; expected 1..{max_tokens}"

        avg_tokens = sum(tokenizer.count_tokens(chunk) for chunk in chunks) / len(
            chunks
        )
        assert 50 <= avg_tokens <= config.target_chunk_tokens * 1.2

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(3.0)
    async def test_indexed_chunks_have_correct_size(
        self,
        config: IndexConfig,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
        mock_openai_async_client: AsyncOpenAI,
    ) -> None:
        """End-to-end runtime append should respect chunk sizing constraints."""

        configure_runtime(indexer_runtime_harness, config)
        indexer_runtime_harness.llm_service.client = mock_openai_async_client

        document_id = "chunk-size-doc"
        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path=None,
            embedding_model=config.embedding_model,
            summary_model=config.summary_model,
        )

        test_doc = build_test_document(30)

        await indexer_runtime_harness.clear(document_id)
        await indexer_runtime_harness.append(
            document_id,
            test_doc,
            replace_existing=True,
            file_path=None,
        )

        leaf_nodes = doc_store.nodes.get_leaves()
        assert leaf_nodes, "Indexing should yield leaf nodes"

        for index, node in enumerate(leaf_nodes):
            token_count = tokenizer.count_tokens(node.text)
            if index == len(leaf_nodes) - 1:
                min_tokens = 20
                max_tokens = int(config.target_chunk_tokens * 1.2)
            else:
                min_tokens = int(config.target_chunk_tokens * 0.7)
                max_tokens = int(config.target_chunk_tokens * 1.3)

            assert (
                min_tokens <= token_count <= max_tokens
            ), f"Leaf {node.id} ({index + 1}/{len(leaf_nodes)}) has {token_count} tokens"

        parent_nodes = [
            node
            for node in doc_store.nodes.get_all()
            if hasattr(node, "left_child_id")
            and hasattr(node, "right_child_id")
            and (node.left_child_id is not None or node.right_child_id is not None)
        ]

        for node in parent_nodes:
            token_count = tokenizer.count_tokens(node.text)
            assert (
                token_count <= config.target_chunk_tokens * 2
            ), f"Parent node {node.id} has {token_count} tokens"

    @pytest.fixture
    def doc_store_with_chunks(
        self, storage_backend: StorageBackend
    ) -> Generator[DocumentStore, None, None]:
        """Populate a document with hand-crafted chunks for analysis tests."""

        document_id = "chunk-test-doc"
        storage_backend.clear_document(document_id)
        store = storage_backend.for_document(document_id)
        store.set_metadata(
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        seed_manual_chunk_tree(document_id, store)

        try:
            yield store
        finally:
            storage_backend.clear_document(document_id)

    def test_chunk_size_analysis(self, doc_store_with_chunks: DocumentStore) -> None:
        """Sanity check token counts on the seeded document store."""

        leaf_nodes = doc_store_with_chunks.nodes.get_leaves()
        assert len(leaf_nodes) == 4

        small_chunks = [node for node in leaf_nodes if node.token_count < 10]
        target_chunks = [node for node in leaf_nodes if node.token_count >= 190]
        assert len(small_chunks) == 2
        assert len(target_chunks) == 2

        parent_nodes = [
            node
            for node in doc_store_with_chunks.nodes.get_all()
            if hasattr(node, "left_child_id")
            and hasattr(node, "right_child_id")
            and (node.left_child_id is not None or node.right_child_id is not None)
        ]
        assert len(parent_nodes) == 3

        for node in parent_nodes:
            assert node.token_count <= 50

    def test_chunk_distribution_analysis(
        self, doc_store_with_chunks: DocumentStore
    ) -> None:
        """Compute distribution metrics over the seeded chunks."""

        leaf_nodes = doc_store_with_chunks.nodes.get_leaves()
        token_counts = [node.token_count for node in leaf_nodes]

        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / len(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0

        assert total_tokens == 400
        assert avg_tokens == 100.0
        assert min_tokens == 2
        assert max_tokens == 200

        small_chunks = [node for node in leaf_nodes if node.token_count < 10]
        large_chunks = [node for node in leaf_nodes if node.token_count >= 190]
        assert len(small_chunks) == 2
        assert len(large_chunks) == 2

        oversized_chunks = [node for node in leaf_nodes if node.token_count > 400]
        assert not oversized_chunks
