"""Tests for BM25 hybrid search integration with Retriever.

Spec: specs/bm25-hybrid-search.md § Integration with Retriever
"""

import asyncio
from collections.abc import Callable, Generator
from collections.abc import Sequence as Seq
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ragzoom.retrieve import Retriever

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.vector_filter import VectorFilter
from ragzoom.document_store import DocumentStore
from ragzoom.vector_api import Vector


@pytest.mark.usefixtures("sqlite_backend")
class TestRetrieverBM25Parameter:
    """Tests that Retriever.retrieve_async() accepts use_bm25 parameter."""

    @pytest.fixture
    def setup_simple_tree(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> Generator[tuple[IndexConfig, DocumentStore, "Retriever"], None, None]:
        """Set up a simple tree structure for testing."""
        index_config = IndexConfig.load(target_chunk_tokens=100)
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Create document store
        doc_store = sqlite_store_factory("test-doc")

        # Create simple tree structure:
        #         root
        #        /    \
        #       L1     L2

        nodes: list[NodeDataDict] = [
            {
                "node_id": "L1",
                "text": "First leaf content about error code E1234",
                "span_start": 0,
                "span_end": 40,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "L2",
                "text": "Second leaf content about other topics",
                "span_start": 40,
                "span_end": 80,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 0,
                "level_index": 1,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "root",
                "text": "Full document summary about error code E1234 and other topics",
                "span_start": 0,
                "span_end": 80,
                "document_id": "test-doc",
                "token_count": 100,
                "height": 1,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": "L1",
                "right_child_id": "L2",
            },
        ]

        doc_store.nodes.add_batch(nodes)

        # Set parent references
        doc_store.nodes.update_parent_references_batch(
            [
                ("L1", "root"),
                ("L2", "root"),
            ]
        )

        from ragzoom.vector_factory import create_vector_index
        from tests.utils import create_retriever

        vi = create_vector_index(
            "python", "sqlite:///:memory:", query_config.embedding_model
        )
        retriever = create_retriever(
            query_config=query_config,
            store=doc_store,
            document_id="test-doc",
            api_key=operational_config.openai_api_key.get_secret_value(),
            vector_index=vi,
        )
        yield index_config, doc_store, retriever

    def _mock_retriever(self, retriever: "Retriever") -> None:
        """Apply standard mocks to retriever for testing."""

        def mock_search_similar(
            query_embedding: list[float] | NDArray[np.float64],
            k: int,
            filters: Seq[VectorFilter] | None = None,
        ) -> list[Vector]:
            import numpy as _np

            return [
                Vector(
                    id="L1",
                    vec=_np.ones(1536, dtype=_np.float32),
                    meta={
                        "document_id": "test-doc",
                        "span_start": 0,
                        "span_end": 40,
                        "parent_id": "root",
                        "is_leaf": 1,
                    },
                    model_id="text-embedding-3-small",
                    dim=1536,
                )
            ]

        retriever.vector_index.search_similar = mock_search_similar  # type: ignore[method-assign]

        retriever.embedding_service.get_query_embedding = (  # type: ignore[method-assign]
            lambda query, document_id=None: [0.3] * 1536
        )

    def test_retrieve_async_accepts_use_bm25(
        self,
        setup_simple_tree: tuple[IndexConfig, DocumentStore, "Retriever"],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that retrieve_async() accepts use_bm25 parameter.

        Spec: specs/bm25-hybrid-search.md § Integration with Retriever
        Success: retrieve_async(..., use_bm25=True) enables hybrid search
        """
        _, _, retriever = setup_simple_tree
        self._mock_retriever(retriever)

        # Should not raise TypeError for unexpected keyword argument
        result = asyncio.run(
            retriever.retrieve_async(
                query="error code E1234",
                num_seeds=1,
                budget_tokens=1000,
                document_id="test-doc",
                use_bm25=True,  # NEW parameter
            )
        )

        assert result.tiling is not None
        assert len(result.tiling) > 0

    def test_retrieve_async_accepts_use_bm25_false(
        self,
        setup_simple_tree: tuple[IndexConfig, DocumentStore, "Retriever"],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that retrieve_async() accepts use_bm25=False to disable BM25."""
        _, _, retriever = setup_simple_tree
        self._mock_retriever(retriever)

        # Should work with use_bm25=False (pure vector search)
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id="test-doc",
                use_bm25=False,
            )
        )

        assert result.tiling is not None
        assert len(result.tiling) > 0

    def test_retrieve_async_defaults_to_query_config_use_bm25(
        self,
        setup_simple_tree: tuple[IndexConfig, DocumentStore, "Retriever"],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that retrieve_async() defaults to QueryConfig.use_bm25."""
        _, _, retriever = setup_simple_tree
        self._mock_retriever(retriever)

        # When not specified, should use QueryConfig.use_bm25 (default True)
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id="test-doc",
                # use_bm25 not specified - should default to config value
            )
        )

        assert result.tiling is not None
        assert len(result.tiling) > 0

    def test_retrieve_async_use_bm25_overrides_query_config(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
    ) -> None:
        """Test that use_bm25 parameter overrides QueryConfig.use_bm25."""
        # Create retriever with use_bm25=False in QueryConfig
        query_config = QueryConfig(budget_tokens=1000, use_bm25=False)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))
        doc_store = sqlite_store_factory("test-doc")

        # Create minimal tree
        nodes: list[NodeDataDict] = [
            {
                "node_id": "L1",
                "text": "Leaf content",
                "span_start": 0,
                "span_end": 40,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
        ]
        doc_store.nodes.add_batch(nodes)

        from ragzoom.vector_factory import create_vector_index
        from tests.utils import create_retriever

        vi = create_vector_index(
            "python", "sqlite:///:memory:", query_config.embedding_model
        )
        retriever = create_retriever(
            query_config=query_config,
            store=doc_store,
            document_id="test-doc",
            api_key=operational_config.openai_api_key.get_secret_value(),
            vector_index=vi,
        )

        self._mock_retriever(retriever)

        # use_bm25=True should override QueryConfig's False
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id="test-doc",
                use_bm25=True,  # Override config
            )
        )

        assert result.tiling is not None
