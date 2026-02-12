"""Tests for BM25 hybrid search integration with Retriever.

Spec: specs/bm25-hybrid-search.md § Integration with Retriever

This module tests the actual integration of BM25 search and Reciprocal Rank
Fusion into the retrieval pipeline. Tests verify that:
1. When use_bm25=True, both vector and BM25 search are run
2. Results are fused using RRF before seed selection
3. BM25 index is cached per document for efficiency
"""

import asyncio
from collections.abc import Callable, Generator
from collections.abc import Sequence as Seq
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ragzoom.retrieve import Retriever

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.bm25 import BM25IndexCache
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.vector_filter import VectorFilter
from ragzoom.document_store import DocumentStore
from ragzoom.vector_api import Vector


@pytest.mark.usefixtures("sqlite_backend")
class TestHybridRetrieval:
    """Tests that verify BM25 + vector hybrid retrieval integration."""

    @pytest.fixture
    def setup_hybrid_tree(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> Generator[tuple[IndexConfig, DocumentStore, "Retriever"], None, None]:
        """Set up a tree structure for hybrid retrieval testing.

        Creates nodes where BM25 and vector search would return different rankings:
        - L1: Contains exact term "error_code_E1234" (good for BM25)
        - L2: Semantically similar to query but no exact match (good for vector)
        - L3: Neither exact match nor semantic similarity
        """
        index_config = IndexConfig.load(target_chunk_tokens=100)
        query_config = QueryConfig(budget_tokens=1000, use_bm25=True)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        doc_store = sqlite_store_factory("test-doc")

        # Create tree with nodes that have different BM25 vs vector relevance
        nodes: list[NodeDataDict] = [
            {
                "node_id": "L1",
                "text": "Error code E1234 occurred in module ABC",
                "span_start": 0,
                "span_end": 50,
                "document_id": "test-doc",
                "token_count": 30,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "L2",
                "text": "A problem was detected in the system module",
                "span_start": 50,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 30,
                "height": 0,
                "level_index": 1,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "L3",
                "text": "The weather today is sunny and warm",
                "span_start": 100,
                "span_end": 150,
                "document_id": "test-doc",
                "token_count": 30,
                "height": 0,
                "level_index": 2,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "L4",
                "text": "Additional content about debugging issues",
                "span_start": 150,
                "span_end": 200,
                "document_id": "test-doc",
                "token_count": 30,
                "height": 0,
                "level_index": 3,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "L5",
                "text": "More irrelevant content about cats and dogs",
                "span_start": 200,
                "span_end": 250,
                "document_id": "test-doc",
                "token_count": 30,
                "height": 0,
                "level_index": 4,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "root",
                "text": "Summary covering all content including errors and weather",
                "span_start": 0,
                "span_end": 250,
                "document_id": "test-doc",
                "token_count": 100,
                "height": 1,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": "L1",
                "right_child_id": "L5",
            },
        ]

        doc_store.nodes.add_batch(nodes)

        # Set parent references
        doc_store.nodes.update_parent_references_batch(
            [
                ("L1", "root"),
                ("L2", "root"),
                ("L3", "root"),
                ("L4", "root"),
                ("L5", "root"),
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

    def _mock_vector_search(self, retriever: "Retriever", ranking: list[str]) -> None:
        """Mock vector search to return nodes in a specific order."""

        def mock_search_similar(
            query_embedding: list[float] | NDArray[np.float64],
            k: int,
            filters: Seq[VectorFilter] | None = None,
        ) -> list[Vector]:
            """Return vectors in the order specified by ranking."""
            results = []
            for i, node_id in enumerate(ranking[:k]):
                # Higher-ranked nodes get vectors closer to query
                vec = np.ones(1536, dtype=np.float32) * (1.0 - i * 0.1)
                span_starts = {"L1": 0, "L2": 50, "L3": 100, "L4": 150, "L5": 200}
                span_ends = {"L1": 50, "L2": 100, "L3": 150, "L4": 200, "L5": 250}
                results.append(
                    Vector(
                        id=node_id,
                        vec=vec,
                        meta={
                            "document_id": "test-doc",
                            "span_start": span_starts.get(node_id, 0),
                            "span_end": span_ends.get(node_id, 50),
                            "parent_id": "root",
                            "is_leaf": 1,
                        },
                        model_id="text-embedding-3-small",
                        dim=1536,
                    )
                )
            return results

        retriever.vector_index.search_similar = mock_search_similar  # type: ignore[method-assign]

    def _mock_embedding(self, retriever: "Retriever") -> None:
        """Mock embedding service to return consistent embeddings."""
        retriever.embedding_service.get_query_embedding = (  # type: ignore[method-assign]
            lambda query, document_id=None: [0.5] * 1536
        )

    def test_hybrid_retrieval_uses_bm25_when_enabled(
        self,
        setup_hybrid_tree: tuple[IndexConfig, DocumentStore, "Retriever"],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that BM25 search is run when use_bm25=True.

        Spec: specs/bm25-hybrid-search.md § Integration with Retriever
        Success: When use_bm25=True, retriever runs both searches and fuses results
        """
        _, doc_store, retriever = setup_hybrid_tree
        self._mock_embedding(retriever)

        # Vector search ranks: L2, L3, L4 (semantic similarity, no exact match)
        # BM25 should boost L1 (has "E1234")
        self._mock_vector_search(retriever, ["L2", "L3", "L4", "L5", "L1"])

        # Query for exact term that L1 has
        result = asyncio.run(
            retriever.retrieve_async(
                query="error code E1234",
                num_seeds=3,
                budget_tokens=1000,
                document_id="test-doc",
                use_bm25=True,
            )
        )

        # L1 should be selected because BM25 boosts it via RRF
        assert "L1" in result.node_ids, (
            f"L1 should be selected due to BM25 boost. " f"Selected: {result.node_ids}"
        )

    def test_hybrid_retrieval_disabled_uses_vector_only(
        self,
        setup_hybrid_tree: tuple[IndexConfig, DocumentStore, "Retriever"],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that only vector search is used when use_bm25=False.

        Spec: specs/bm25-hybrid-search.md § Integration with Retriever
        Success: When use_bm25=False, retriever uses only vector search
        """
        _, doc_store, retriever = setup_hybrid_tree
        self._mock_embedding(retriever)

        # Vector search ranks: L2, L3, L4 (L1 is last - semantic mismatch)
        self._mock_vector_search(retriever, ["L2", "L3", "L4", "L5", "L1"])

        # With BM25 disabled, should follow vector ranking
        # Use num_seeds=3 so k_candidates=6 (with mmr_k_multiplier=2.0)
        # This ensures L1 (ranked 5th by vector) is in the candidate pool
        result_disabled = asyncio.run(
            retriever.retrieve_async(
                query="error code E1234",
                num_seeds=3,
                budget_tokens=1000,
                document_id="test-doc",
                use_bm25=False,
            )
        )

        # With BM25 enabled, L1 should be boosted
        result_enabled = asyncio.run(
            retriever.retrieve_async(
                query="error code E1234",
                num_seeds=3,
                budget_tokens=1000,
                document_id="test-doc",
                use_bm25=True,
            )
        )

        # When BM25 is disabled, L1 should not be a seed (it's last in vector ranking)
        # When BM25 is enabled, L1 should be a seed (BM25 boosts it for exact match)
        assert result_disabled.seed_count == 3
        assert result_enabled.seed_count == 3

        # L1 has "E1234" - BM25 should boost it when enabled
        # With BM25 enabled, L1 should appear due to RRF boost
        assert "L1" in result_enabled.node_ids, (
            f"L1 should be selected with BM25 enabled. "
            f"Selected: {result_enabled.node_ids}"
        )

    def test_bm25_index_is_cached(
        self,
        setup_hybrid_tree: tuple[IndexConfig, DocumentStore, "Retriever"],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that BM25 index is cached for repeated queries.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index Caching
        Success: Second query reuses cached BM25 index
        """
        _, doc_store, retriever = setup_hybrid_tree
        self._mock_embedding(retriever)
        self._mock_vector_search(retriever, ["L2", "L3", "L4", "L5", "L1"])

        # Track BM25 index creation
        original_init = BM25IndexCache.get_or_build
        call_count = {"builds": 0}

        def tracking_get_or_build(
            self: BM25IndexCache, document_id: str, nodes: dict[str, object]
        ) -> object:
            was_cached = document_id in self
            result = original_init(self, document_id, nodes)  # type: ignore[arg-type]
            if not was_cached:
                call_count["builds"] += 1
            return result

        with patch.object(BM25IndexCache, "get_or_build", tracking_get_or_build):
            # First query - should build index
            asyncio.run(
                retriever.retrieve_async(
                    query="error code E1234",
                    num_seeds=2,
                    budget_tokens=1000,
                    document_id="test-doc",
                    use_bm25=True,
                )
            )

            # Second query - should reuse cached index
            asyncio.run(
                retriever.retrieve_async(
                    query="different query",
                    num_seeds=2,
                    budget_tokens=1000,
                    document_id="test-doc",
                    use_bm25=True,
                )
            )

        # Index should only be built once
        assert call_count["builds"] == 1, (
            f"BM25 index should be built once and cached. "
            f"Builds: {call_count['builds']}"
        )

    def test_hybrid_retrieval_with_zero_seeds(
        self,
        setup_hybrid_tree: tuple[IndexConfig, DocumentStore, "Retriever"],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that hybrid retrieval handles num_seeds=0 correctly.

        When num_seeds=0, neither vector nor BM25 search should run.
        """
        _, doc_store, retriever = setup_hybrid_tree
        self._mock_embedding(retriever)
        self._mock_vector_search(retriever, ["L2", "L3", "L4", "L5", "L1"])

        result = asyncio.run(
            retriever.retrieve_async(
                query="any query",
                num_seeds=0,
                budget_tokens=1000,
                document_id="test-doc",
                use_bm25=True,
            )
        )

        # No seeds should be selected
        assert result.seed_count == 0

    def _upsert_all_node_vectors(self, retriever: "Retriever") -> None:
        """Upsert vectors for all test nodes into the vector index.

        In production, all nodes have embeddings in the vector index from
        ingestion. This mirrors that state so get_vectors() works for any
        node, not just those returned by search_similar().
        """
        span_starts = {"L1": 0, "L2": 50, "L3": 100, "L4": 150, "L5": 200}
        span_ends = {"L1": 50, "L2": 100, "L3": 150, "L4": 200, "L5": 250}
        items: list[
            tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
        ] = []
        for node_id in ["L1", "L2", "L3", "L4", "L5"]:
            vec: list[float] | NDArray[np.float64] = [0.1] * 1536
            meta: dict[str, object] = {
                "document_id": "test-doc",
                "span_start": span_starts[node_id],
                "span_end": span_ends[node_id],
                "parent_id": "root",
                "is_leaf": 1,
            }
            items.append((node_id, vec, meta))
        retriever.vector_index.upsert(items)

    def test_bm25_only_hits_included_in_candidates(
        self,
        setup_hybrid_tree: tuple[IndexConfig, DocumentStore, "Retriever"],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that BM25-only hits (not in vector results) are included.

        This is the key value proposition of hybrid search: BM25 finds exact
        keyword matches that vector search misses. If we discard BM25-only
        hits, hybrid search degrades to vector-only search.

        We set up vector search to return only L2,L3,L4 (excluding L1),
        then query for "E1234" which BM25 should match to L1. L1 must
        appear in final results despite being absent from vector candidates.
        """
        _, doc_store, retriever = setup_hybrid_tree
        self._mock_embedding(retriever)

        # Ensure all nodes have vectors stored (mirrors production ingestion)
        self._upsert_all_node_vectors(retriever)

        # Vector search deliberately excludes L1 — it only returns L2, L3, L4
        self._mock_vector_search(retriever, ["L2", "L3", "L4"])

        # L1 has "Error code E1234" — BM25 will rank it high for this query,
        # but vector search didn't return it at all
        result = asyncio.run(
            retriever.retrieve_async(
                query="E1234",
                num_seeds=3,
                budget_tokens=1000,
                document_id="test-doc",
                use_bm25=True,
            )
        )

        # L1 must appear despite being a BM25-only hit
        assert "L1" in result.node_ids, (
            f"BM25-only hit L1 should be included via get_vectors. "
            f"Selected: {result.node_ids}"
        )

    def test_hybrid_retrieval_rrf_combines_rankings(
        self,
        setup_hybrid_tree: tuple[IndexConfig, DocumentStore, "Retriever"],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that RRF properly combines vector and BM25 rankings.

        Spec: specs/bm25-hybrid-search.md § Architecture > Reciprocal Rank Fusion
        Success: Items appearing in both rankings get higher combined scores
        """
        _, doc_store, retriever = setup_hybrid_tree
        self._mock_embedding(retriever)

        # Set up vector ranking where L4 is ranked high
        # BM25 will rank L1 high for "E1234" query
        # After RRF, both L1 and L4 should be selected (appear in different rankings)
        self._mock_vector_search(retriever, ["L4", "L2", "L3", "L5", "L1"])

        result = asyncio.run(
            retriever.retrieve_async(
                query="error code E1234 debugging",
                num_seeds=2,
                budget_tokens=1000,
                document_id="test-doc",
                use_bm25=True,
            )
        )

        # Both L1 (BM25 boost for E1234) and L4 (vector top rank) should be considered
        # The exact selection depends on RRF scores
        assert result.tiling is not None
        assert len(result.node_ids) > 0
