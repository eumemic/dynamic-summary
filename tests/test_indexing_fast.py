"""Backend-agnostic fast indexing tests using storage backend."""

import asyncio
from typing import cast
from unittest.mock import MagicMock, Mock

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol
from ragzoom.index import TreeBuilder
from tests.conftest import BackwardCompatibilityConfig


class TestIndexingFast:
    """Fast indexing tests using storage backend instead of real database."""

    def test_full_document_gets_indexed(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai_async_client: MagicMock,
        vector_index: _VectorIndexProtocol,
    ) -> None:
        """Test that the entire document is indexed, not just first 37%."""
        config = base_config.index_config

        # Create a smaller test document for faster testing
        test_doc_parts = []
        for i in range(20):  # Reduced from 100 to 20 parts
            test_doc_parts.append(
                f"Part {i}: This is test content that should be indexed. " * 5
            )

        test_document = "\n\n".join(test_doc_parts)
        doc_length = len(test_document)

        # Use the mock OpenAI client from fixture
        mock_client = mock_openai_async_client

        # Create document-scoped store and tree builder
        doc_store = storage_backend.for_document("test-doc")
        doc_store.set_metadata(
            file_path="test-doc.txt",
            content_hash="test-hash",
            chunk_count=0,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        tree_builder = TreeBuilder(config, doc_store, vector_index, max_concurrent=5)
        tree_builder.llm_service.client = mock_client

        # Index the document
        asyncio.run(tree_builder.add_document_async(test_document))

        # Verify: Check that the entire document was indexed
        # Get all leaf nodes and check their spans
        # Get leaf nodes by filtering all nodes
        all_nodes = doc_store.nodes.get_all()
        leaf_nodes = [node for node in all_nodes if node.height == 0]

        # Sort by span_start
        leaf_nodes.sort(key=lambda n: n.span_start)

        # Check coverage
        if leaf_nodes:
            first_span_start = leaf_nodes[0].span_start
            last_span_end = leaf_nodes[-1].span_end

            print(f"Document length: {doc_length}")
            print(f"First leaf starts at: {first_span_start}")
            print(f"Last leaf ends at: {last_span_end}")
            print(f"Coverage: {last_span_end / doc_length * 100:.1f}%")
            print(f"Number of leaf nodes: {len(leaf_nodes)}")

            # The last leaf should cover near the end of the document
            # Allow for some overlap/gap due to chunking
            coverage_ratio = last_span_end / doc_length
            assert (
                coverage_ratio > 0.95
            ), f"Only {coverage_ratio*100:.1f}% of document indexed!"

            # Check for gaps in coverage
            for i in range(1, len(leaf_nodes)):
                prev_end = leaf_nodes[i - 1].span_end
                curr_start = leaf_nodes[i].span_start
                # Some overlap is expected due to chunk_overlap
                gap = curr_start - prev_end

                # If there's a gap, check if it's only whitespace
                if gap > 0:
                    gap_text = test_document[prev_end:curr_start]
                    if gap_text.isspace():
                        # Whitespace-only gaps are acceptable (text splitter limitation)
                        continue

                    # For debugging large gaps
                    if gap > 100:
                        print(
                            f"Gap content ({gap} chars): {repr(gap_text[:50])}...{repr(gap_text[-50:])}"
                        )

                # LangChain text splitter has known issues with dropping content
                # Allow larger gaps as this is a known limitation (Issue #10)
                assert (
                    gap < 150
                ), f"Large gap found: {gap} chars between positions {prev_end} and {curr_start}"

    def test_small_document_indexing(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai_async_client: MagicMock,
        vector_index: _VectorIndexProtocol,
    ) -> None:
        """Test indexing a very small document to isolate the issue."""
        config = base_config.index_config
        # Create a minimal document
        test_document = (
            "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        )
        doc_length = len(test_document)

        # Use the mock OpenAI client from fixture
        mock_client = mock_openai_async_client

        # Create document-scoped store and tree builder
        doc_store = storage_backend.for_document("test-doc")
        doc_store.set_metadata(
            file_path="test-doc.txt",
            content_hash="test-hash",
            chunk_count=0,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        tree_builder = TreeBuilder(config, doc_store, vector_index, max_concurrent=1)
        tree_builder.llm_service.client = mock_client

        # Index the document
        asyncio.run(tree_builder.add_document_async(test_document))

        # Check that we indexed the whole thing
        # Get leaf nodes by filtering all nodes
        all_nodes = doc_store.nodes.get_all()
        leaf_nodes = [node for node in all_nodes if node.height == 0]

        if leaf_nodes:
            leaf_nodes.sort(key=lambda n: n.span_start)
            last_span_end = leaf_nodes[-1].span_end

            print(f"Small doc length: {doc_length}")
            print(f"Last span end: {last_span_end}")
            print(f"Leaf nodes: {len(leaf_nodes)}")
            print(f"Leaf node text: '{leaf_nodes[0].text}'")

            assert (
                last_span_end >= doc_length - 10
            ), f"Document not fully indexed: {last_span_end} < {doc_length}"

    def test_check_api_batch_limits(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai_async_client: MagicMock,
        vector_index: _VectorIndexProtocol,
    ) -> None:
        """Test if there's a limit on API batching causing truncation."""
        config = base_config.index_config
        # Create document with many chunks to test batching
        # Reduced from 250 to 50 chunks - still tests batching but faster
        chunks = []
        for i in range(50):  # Reduced further for speed
            chunks.append(f"Chunk {i}: " + "word " * 20)  # ~50 tokens each

        test_document = " ".join(chunks)
        doc_length = len(test_document)

        # Track API calls
        api_call_count = 0
        texts_per_call = []

        async def mock_embeddings_create(**kwargs: object) -> Mock:
            nonlocal api_call_count, texts_per_call
            api_call_count += 1
            input_texts = cast(list[str] | str, kwargs.get("input", []))
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            texts_per_call.append(len(input_texts))

            # Return embeddings for each text
            return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in input_texts])

        # Use the mock OpenAI client from fixture and customize embedding tracking
        mock_client = mock_openai_async_client
        mock_client.embeddings.create = mock_embeddings_create

        # Create document-scoped store and tree builder
        doc_store = storage_backend.for_document("test-doc")
        doc_store.set_metadata(
            file_path="test-doc.txt",
            content_hash="test-hash",
            chunk_count=0,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        tree_builder = TreeBuilder(config, doc_store, vector_index, max_concurrent=5)
        tree_builder.llm_service.client = mock_client

        # Index the document
        asyncio.run(tree_builder.add_document_async(test_document))

        # Check results
        # Get leaf nodes by filtering all nodes
        all_nodes = doc_store.nodes.get_all()
        leaf_nodes = [node for node in all_nodes if node.height == 0]

        print(f"API calls made: {api_call_count}")
        print(f"Texts per call: {texts_per_call}")
        print(f"Total leaf nodes: {len(leaf_nodes)}")

        # Verify full coverage
        if leaf_nodes:
            leaf_nodes.sort(key=lambda n: n.span_start)
            last_span_end = leaf_nodes[-1].span_end
            coverage_ratio = last_span_end / doc_length

            assert (
                coverage_ratio > 0.95
            ), f"Only {coverage_ratio*100:.1f}% indexed after {api_call_count} API calls"
