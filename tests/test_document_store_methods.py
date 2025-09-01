"""Test new DocumentStore methods added in Phase 4."""

from tests.mock_store import SimpleMockStore


class TestDocumentStoreMethods:
    """Test the new methods added to DocumentStore for Phase 4."""

    def test_get_embedding_model(self) -> None:
        """Test that DocumentStore correctly retrieves embedding model."""
        store = SimpleMockStore()

        # Add document with metadata
        from types import SimpleNamespace

        store.documents["doc1"] = SimpleNamespace(
            id="doc1",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4",
        )

        # Create document store
        doc_store = store.for_document("doc1")

        # Test getting embedding model
        model = doc_store.get_embedding_model()
        assert model == "text-embedding-3-small"

    def test_get_embedding_model_missing(self) -> None:
        """Test that DocumentStore returns None when embedding model is missing."""
        store = SimpleMockStore()

        # Add document without embedding_model
        from types import SimpleNamespace

        store.documents["doc1"] = SimpleNamespace(
            id="doc1",
            summary_model="gpt-4",
        )

        # Create document store
        doc_store = store.for_document("doc1")

        # Test getting embedding model
        model = doc_store.get_embedding_model()
        assert model is None

    def test_get_avg_leaf_tokens(self) -> None:
        """Test that DocumentStore correctly calculates average leaf tokens."""
        store = SimpleMockStore()

        # Add leaf nodes with different token counts
        for i in range(3):
            store.add_node(
                node_id=f"leaf_{i}",
                text=f"Leaf text {i}",
                span_start=i * 100,
                span_end=(i + 1) * 100,
                document_id="doc1",
                embedding=[0.5] * 1536,
                token_count=100 + i * 50,  # 100, 150, 200
            )

        # Add a parent node (not a leaf)
        store.add_node(
            node_id="parent",
            text="Parent text",
            span_start=0,
            span_end=300,
            document_id="doc1",
            embedding=[0.5] * 1536,
            token_count=300,
            left_child_id="leaf_0",
            right_child_id="leaf_1",
        )

        # Create document store
        doc_store = store.for_document("doc1")

        # Test getting average leaf tokens
        avg_tokens = doc_store.get_avg_leaf_tokens()
        # Average of 100, 150, 200 = 150
        assert avg_tokens == 150

    def test_get_avg_leaf_tokens_no_leaves(self) -> None:
        """Test that DocumentStore returns None when no leaf nodes exist."""
        store = SimpleMockStore()

        # Create document store for empty document
        doc_store = store.for_document("doc1")

        # Test getting average leaf tokens
        avg_tokens = doc_store.get_avg_leaf_tokens()
        assert avg_tokens is None

    def test_document_id_mismatch_safety(self) -> None:
        """Test that DocumentStore validates document ID matches."""
        store = SimpleMockStore()

        # Add nodes to different documents
        store.add_node(
            node_id="doc1_node",
            text="Doc 1 content",
            span_start=0,
            span_end=100,
            document_id="doc1",
            embedding=[0.5] * 1536,
        )
        store.add_node(
            node_id="doc2_node",
            text="Doc 2 content",
            span_start=0,
            span_end=100,
            document_id="doc2",
            embedding=[0.5] * 1536,
        )

        # Create document store for doc1
        doc1_store = store.for_document("doc1")

        # Verify can get doc1 node
        node1 = doc1_store.nodes.get("doc1_node")
        assert node1 is not None
        assert node1.id == "doc1_node"

        # Verify cannot get doc2 node through doc1 store
        node2 = doc1_store.nodes.get("doc2_node")
        assert node2 is None  # Should be filtered out

    def test_cross_document_store(self) -> None:
        """Test that DocumentStore with None document_id allows cross-document access."""
        store = SimpleMockStore()

        # Add nodes to different documents
        store.add_node(
            node_id="doc1_node",
            text="Doc 1 content",
            span_start=0,
            span_end=100,
            document_id="doc1",
            embedding=[0.5] * 1536,
        )
        store.add_node(
            node_id="doc2_node",
            text="Doc 2 content",
            span_start=0,
            span_end=100,
            document_id="doc2",
            embedding=[0.5] * 1536,
        )

        # Create cross-document store
        cross_store = store.for_document(None)

        # Should be able to access both documents
        node1 = cross_store.nodes.get("doc1_node")
        assert node1 is not None

        node2 = cross_store.nodes.get("doc2_node")
        assert node2 is not None
