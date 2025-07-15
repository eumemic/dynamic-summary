"""Test to reproduce incomplete document indexing issue."""

import asyncio
import tempfile
from unittest.mock import AsyncMock, Mock

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.store import Store


class TestIncompleteIndexing:
    """Test cases to reproduce incomplete document indexing."""

    @pytest.fixture
    def setup(self):
        """Create test configuration and store."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RagZoomConfig(
                openai_api_key="test-key",
                sqlite_database_url="sqlite:///:memory:",
                chroma_persist_directory=temp_dir,
                leaf_tokens=50,  # Small chunks for easier testing
                leaf_overlap_tokens=5,
                adjacent_context_tokens=25,  # Must be <= leaf_tokens
                budget_tokens=1000,
                embedding_dimensions=1536,
            )
            store = Store(config)
            yield config, store

            # Close store to prevent file handle leaks
            store.close()

    def test_full_document_gets_indexed(self, setup):
        """Test that the entire document is indexed, not just first 37%."""
        config, store = setup
        # Create a test document with known content
        # Make it long enough to create multiple chunks
        test_doc_parts = []
        for i in range(100):  # 100 parts
            test_doc_parts.append(f"Part {i}: This is test content that should be indexed. " * 5)

        test_document = "\n\n".join(test_doc_parts)
        doc_length = len(test_document)

        # Mock OpenAI client
        mock_client = AsyncMock()

        # Mock embeddings to return correct number of embeddings for batch requests
        async def mock_embeddings_create(**kwargs):
            input_texts = kwargs.get('input', [])
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            # Return one embedding for each input text
            return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in input_texts])

        mock_client.embeddings.create = mock_embeddings_create

        # Mock summarization
        mock_summary_response = Mock()
        mock_summary_response.choices = [Mock()]
        mock_summary_response.choices[0].message = Mock()
        mock_summary_response.choices[0].message.content = "Summary left <<<MID>>> Summary right"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_summary_response)

        # Create tree builder and index
        tree_builder = TreeBuilder(config, store, max_concurrent=5)
        tree_builder.client = mock_client

        # Index the document
        asyncio.run(tree_builder.add_document_async(test_document, document_id="test-doc"))

        # Verify: Check that the entire document was indexed
        # Get all leaf nodes and check their spans
        leaf_nodes = store.get_leaf_nodes()

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
            assert coverage_ratio > 0.95, f"Only {coverage_ratio*100:.1f}% of document indexed!"

            # Check for gaps in coverage
            for i in range(1, len(leaf_nodes)):
                prev_end = leaf_nodes[i-1].span_end
                curr_start = leaf_nodes[i].span_start
                # Some overlap is expected due to chunk_overlap
                gap = curr_start - prev_end
                assert gap < 100, f"Large gap found: {gap} chars between positions {prev_end} and {curr_start}"

    def test_small_document_indexing(self, setup):
        """Test indexing a very small document to isolate the issue."""
        config, store = setup
        # Create a minimal document
        test_document = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        doc_length = len(test_document)

        # Mock OpenAI client
        mock_client = AsyncMock()

        # Mock embeddings to return correct number of embeddings for batch requests
        async def mock_embeddings_create(**kwargs):
            input_texts = kwargs.get('input', [])
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            # Return one embedding for each input text
            return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in input_texts])

        mock_client.embeddings.create = mock_embeddings_create

        # Mock summarization
        mock_summary_response = Mock()
        mock_summary_response.choices = [Mock()]
        mock_summary_response.choices[0].message = Mock()
        mock_summary_response.choices[0].message.content = "Summary <<<MID>>> Summary"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_summary_response)

        # Create tree builder and index
        tree_builder = TreeBuilder(config, store, max_concurrent=1)
        tree_builder.client = mock_client

        # Index the document
        asyncio.run(tree_builder.add_document_async(test_document, document_id="test-doc"))

        # Check that we indexed the whole thing
        leaf_nodes = store.get_leaf_nodes()

        if leaf_nodes:
            leaf_nodes.sort(key=lambda n: n.span_start)
            last_span_end = leaf_nodes[-1].span_end

            print(f"Small doc length: {doc_length}")
            print(f"Last span end: {last_span_end}")
            print(f"Leaf nodes: {len(leaf_nodes)}")
            print(f"Leaf node text: '{leaf_nodes[0].text}'")

            assert last_span_end >= doc_length - 10, f"Document not fully indexed: {last_span_end} < {doc_length}"

    def test_check_api_batch_limits(self, setup):
        """Test if there's a limit on API batching causing truncation."""
        config, store = setup
        # Create document with many chunks (200+ to test batching)
        chunks = []
        for i in range(250):
            chunks.append(f"Chunk {i}: " + "word " * 20)  # ~50 tokens each

        test_document = " ".join(chunks)
        doc_length = len(test_document)

        # Track API calls
        api_call_count = 0
        texts_per_call = []

        async def mock_embeddings_create(**kwargs):
            nonlocal api_call_count, texts_per_call
            api_call_count += 1
            input_texts = kwargs.get('input', [])
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            texts_per_call.append(len(input_texts))

            # Return embeddings for each text
            return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in input_texts])

        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_client.embeddings.create = mock_embeddings_create

        # Mock summarization
        mock_summary_response = Mock()
        mock_summary_response.choices = [Mock()]
        mock_summary_response.choices[0].message = Mock()
        mock_summary_response.choices[0].message.content = "Summary <<<MID>>> Summary"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_summary_response)

        # Create tree builder and index
        tree_builder = TreeBuilder(config, store, max_concurrent=5)
        tree_builder.client = mock_client

        # Index the document
        asyncio.run(tree_builder.add_document_async(test_document, document_id="test-doc"))

        # Check results
        leaf_nodes = store.get_leaf_nodes()

        print(f"API calls made: {api_call_count}")
        print(f"Texts per call: {texts_per_call}")
        print(f"Total leaf nodes: {len(leaf_nodes)}")

        # Verify full coverage
        if leaf_nodes:
            leaf_nodes.sort(key=lambda n: n.span_start)
            last_span_end = leaf_nodes[-1].span_end
            coverage_ratio = last_span_end / doc_length

            assert coverage_ratio > 0.95, f"Only {coverage_ratio*100:.1f}% indexed after {api_call_count} API calls"
