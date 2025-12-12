"""Backend-agnostic fast indexing tests using storage backend."""

from typing import cast
from unittest.mock import Mock, patch

import pytest

from ragzoom.contracts.storage_backend import StorageBackend
from tests.conftest import IndexerRuntimeHarness


class TestIndexingFast:
    """Fast indexing tests using storage backend instead of real database."""

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(2.5)
    async def test_full_document_gets_indexed(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """Test that the entire document is indexed, not just first 37%."""

        test_doc_parts = [
            f"Part {i}: This is test content that should be indexed. " * 5
            for i in range(12)
        ]
        test_document = "\n\n".join(test_doc_parts)
        doc_length = len(test_document)

        await indexer_runtime_harness.append(
            "test-doc",
            test_document,
            replace_existing=True,
            file_path="test-doc.txt",
            await_idle=False,
        )

        doc_store = storage_backend.for_document("test-doc")
        leaf_nodes = [node for node in doc_store.nodes.get_all() if node.height == 0]
        leaf_nodes.sort(key=lambda n: n.span_start)

        if leaf_nodes:
            last_span_end = leaf_nodes[-1].span_end
            coverage_ratio = last_span_end / doc_length
            assert (
                coverage_ratio > 0.95
            ), f"Only {coverage_ratio*100:.1f}% of document indexed!"

            for i in range(1, len(leaf_nodes)):
                prev_end = leaf_nodes[i - 1].span_end
                curr_start = leaf_nodes[i].span_start
                gap = curr_start - prev_end
                if gap > 0:
                    gap_text = test_document[prev_end:curr_start]
                    if gap_text.isspace():
                        continue

                assert (
                    gap < 150
                ), f"Large gap found: {gap} chars between positions {prev_end} and {curr_start}"

    @pytest.mark.asyncio
    async def test_small_document_indexing(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """Test indexing a very small document to isolate the issue."""

        test_document = (
            "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        )
        doc_length = len(test_document)

        await indexer_runtime_harness.append(
            "test-doc",
            test_document,
            replace_existing=True,
            file_path="test-doc.txt",
            await_idle=False,
        )

        doc_store = storage_backend.for_document("test-doc")
        leaf_nodes = [node for node in doc_store.nodes.get_all() if node.height == 0]
        leaf_nodes.sort(key=lambda n: n.span_start)

        if leaf_nodes:
            last_span_end = leaf_nodes[-1].span_end
            assert (
                last_span_end >= doc_length - 10
            ), f"Document not fully indexed: {last_span_end} < {doc_length}"

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(4.0)
    async def test_check_api_batch_limits(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
    ) -> None:
        """Test if there's a limit on API batching causing truncation."""

        chunks = [f"Chunk {i}: " + "word " * 20 for i in range(30)]
        test_document = " ".join(chunks)
        doc_length = len(test_document)

        api_call_count = 0
        texts_per_call: list[int] = []

        async def mock_embeddings_create(**kwargs: object) -> Mock:
            nonlocal api_call_count, texts_per_call
            api_call_count += 1
            input_texts = cast(list[str] | str, kwargs.get("input", []))
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            texts_per_call.append(len(input_texts))

            from types import SimpleNamespace

            return Mock(
                data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in input_texts]
            )

        llm_client = indexer_runtime_harness.llm_service.client

        with patch.object(llm_client.embeddings, "create", new=mock_embeddings_create):
            await indexer_runtime_harness.append(
                "test-doc",
                test_document,
                replace_existing=True,
                file_path="test-doc.txt",
                await_idle=True,  # Wait for async embedding to complete
            )

        doc_store = storage_backend.for_document("test-doc")
        leaf_nodes = [node for node in doc_store.nodes.get_all() if node.height == 0]
        leaf_nodes.sort(key=lambda n: n.span_start)

        if leaf_nodes:
            last_span_end = leaf_nodes[-1].span_end
            coverage_ratio = last_span_end / doc_length
            assert (
                coverage_ratio > 0.95
            ), f"Only {coverage_ratio*100:.1f}% indexed after {api_call_count} API calls"

        assert api_call_count > 0
        assert texts_per_call, "Expected to record embedding call batch sizes"
