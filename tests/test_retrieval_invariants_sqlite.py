"""Ensure retrieval invariants hold without extra fetching.

This test simulates a vector index returning a stale node ID that doesn't exist
in storage. Retrieval must:
  - Filter out stale candidates from selection/coverage/tiling
  - Return a RetrievalResult whose `nodes` contains every ID in `tiling`
  - Keep `node_ids` free of stale IDs
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.config import OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.vector_filter import VectorFilter
from ragzoom.document_store import DocumentStore


@pytest.mark.usefixtures("sqlite_backend")
class TestRetrievalInvariantsSQLite:
    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("docX")

    def test_retrieval_filters_stale_candidates(self, doc_store: DocumentStore) -> None:
        # Build a tiny tree (root with two leaves)
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "leafL",
                "text": "left leaf",
                "embedding": np.array([0.9] * 1536),
                "span_start": 0,
                "span_end": 10,
                "document_id": "docX",
                "token_count": 5,
                "height": 0,
                "level_index": 0,
                "parent_id": "root",
            },
            {
                "node_id": "leafR",
                "text": "right leaf",
                "embedding": np.array([0.9] * 1536),
                "span_start": 10,
                "span_end": 20,
                "document_id": "docX",
                "token_count": 5,
                "height": 0,
                "level_index": 1,
                "parent_id": "root",
            },
            {
                "node_id": "root",
                "text": "root",
                "embedding": np.array([0.1] * 1536),
                "span_start": 0,
                "span_end": 20,
                "document_id": "docX",
                "token_count": 9,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leafL",
                "right_child_id": "leafR",
                "parent_id": None,
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("leafL", "root"), ("leafR", "root")]
        )

        # Prepare retriever with a python vector index and a stale candidate
        qcfg = QueryConfig(budget_tokens=100)
        _ocfg = OperationalConfig(openai_api_key=SecretStr("test-key"))

        with patch("openai.OpenAI") as mock_client:
            mock_embeddings = Mock()
            mock_embeddings.create = Mock(
                return_value=Mock(data=[Mock(embedding=[0.9] * 1536)])
            )
            mock_instance = Mock()
            mock_instance.embeddings = mock_embeddings
            mock_client.return_value = mock_instance

            from ragzoom.config import IndexConfig
            from ragzoom.retrieval.budget_planner import BudgetPlanner
            from ragzoom.retrieval.embedding_service import EmbeddingService
            from ragzoom.retrieve import Retriever
            from ragzoom.vector_api import Vector
            from ragzoom.vector_factory import create_vector_index

            vi = create_vector_index(
                "python", "sqlite:///:memory:", qcfg.embedding_model
            )
            emb = EmbeddingService(mock_instance, doc_store, qcfg.embedding_model)
            planner = BudgetPlanner(doc_store, IndexConfig.load().target_chunk_tokens)
            retriever = Retriever(qcfg, doc_store, emb, planner, vi)

            # Return one real and one stale vector (staleX does not exist in store)
            import numpy as _np

            def _mock_search_similar(
                query_embedding: list[float] | NDArray[np.float64],
                k: int,
                filters: Sequence[VectorFilter] | None = None,
            ) -> list[Vector]:
                return [
                    Vector(
                        "staleX",
                        _np.ones(1536, dtype=_np.float32),
                        {
                            "document_id": "docX",
                            "span_start": 0,
                            "span_end": 0,
                            "parent_id": "root",
                            "is_leaf": 1,
                        },
                        "m",
                        3,
                    ),
                    Vector(
                        "leafL",
                        _np.ones(1536, dtype=_np.float32),
                        {
                            "document_id": "docX",
                            "span_start": 0,
                            "span_end": 0,
                            "parent_id": "root",
                            "is_leaf": 1,
                        },
                        "m",
                        3,
                    ),
                ]

            retriever.vector_index.search_similar = _mock_search_similar  # type: ignore[method-assign]

            result = retriever.retrieve("dragon", num_seeds=1, document_id="docX")

        # Invariants:
        # 1) No stale ids in node_ids (selection)
        assert "staleX" not in result.node_ids
        # 2) coverage_map contains only existing nodes
        for nid in result.coverage_map.keys():
            assert doc_store.nodes.get_node(nid) is not None
        # 3) tiling nodes all present in result.nodes and exist in store
        assert result.tiling is not None
        # Make a local copy to satisfy mypy about potential None types
        tiling_ids = list(result.tiling)
        assert result.nodes is not None
        for nid in tiling_ids:
            assert nid in result.nodes
            assert doc_store.nodes.get_node(nid) is not None
