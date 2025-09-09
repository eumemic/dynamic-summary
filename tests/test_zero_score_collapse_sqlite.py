"""SQLite-based tests for zero score collapse scenarios.

SQLite-based tests ensuring the DP tiling algorithm returns a valid tiling
when ancestors have lower scores and budget cannot fit deeper nodes,
with the real in-memory SQLite backend.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.config import QueryConfig
from ragzoom.document_store import DocumentStore
from ragzoom.dynamic_tiling import DynamicTilingGenerator


@pytest.mark.usefixtures("sqlite_backend")
class TestZeroScoreCollapseSQLite:
    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("test-doc")

    def test_zero_score_collapse_empty_result(self, doc_store: DocumentStore) -> None:
        """When budget can't fit the leaf, algorithm should select root.

        Tree:
            root (token_count=10)
              └─ parent (token_count=20)
                   └─ leaf (token_count=50)
        """
        # Seed nodes; set token_count explicitly to control DP costs
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "leaf",
                "text": "x",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 0,
                "parent_id": "parent",
                "path": "00",
            },
            {
                "node_id": "parent",
                "text": "y",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 20,
                "height": 1,
                "left_child_id": "leaf",
                "right_child_id": None,
                "parent_id": "root",
                "path": "0",
            },
            {
                "node_id": "root",
                "text": "z",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 2,
                "left_child_id": "parent",
                "right_child_id": None,
                "path": "",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("leaf", "parent"), ("parent", "root")]
        )

        # Scores chosen so root quality > parent quality for this scenario
        scores = {"leaf": 1.0, "parent": 0.2, "root": 0.6}
        coverage_map = {"root": True, "parent": True, "leaf": True}

        generator = DynamicTilingGenerator(QueryConfig())

        # Load nodes and find root
        nodes_map: dict[str, object] = {}
        for nid in coverage_map:
            node = doc_store.nodes.get_node(nid)
            if node:
                nodes_map[nid] = node
        root_id = next(
            (
                nid
                for nid, n in nodes_map.items()
                if getattr(n, "parent_id", None) not in nodes_map
            ),
            None,
        )
        assert root_id is not None

        # Costs
        leaf_cost = generator._get_node_cost(nodes_map["leaf"])  # type: ignore[arg-type]
        # parent_cost intentionally unused in assertion; kept cost computation concise
        root_cost = generator._get_node_cost(nodes_map["root"])  # type: ignore[arg-type]

        # Budget just under the leaf cost -> should select root
        budget = leaf_cost - 1
        result = generator.find_optimal_tiling(budget, scores, nodes_map, root_id)  # type: ignore[arg-type]

        assert len(result.tiling.node_ids) == 1
        assert result.tiling.node_ids[0] == "root"
        total_tokens = sum(ni.token_cost for ni in result.node_infos)
        assert total_tokens == root_cost

    def test_zero_score_collapse_to_root(self, doc_store: DocumentStore) -> None:
        """Deeper tree: with constrained budget, algorithm collapses to root."""
        # Seed deeper left-chain tree with explicit token counts (single child each level)
        seeds: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "leaf",
                "text": "leaf",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 30,
                "height": 0,
                "parent_id": "level3",
                "path": "0000",
            },
            {
                "node_id": "level3",
                "text": "l3",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 25,
                "height": 1,
                "left_child_id": "leaf",
                "right_child_id": None,
                "parent_id": "level2",
                "path": "000",
            },
            {
                "node_id": "level2",
                "text": "l2",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 20,
                "height": 2,
                "left_child_id": "level3",
                "right_child_id": None,
                "parent_id": "level1",
                "path": "00",
            },
            {
                "node_id": "level1",
                "text": "l1",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 15,
                "height": 3,
                "left_child_id": "level2",
                "right_child_id": None,
                "parent_id": "root",
                "path": "0",
            },
            {
                "node_id": "root",
                "text": "root",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 4,
                "left_child_id": "level1",
                "right_child_id": None,
                "path": "",
            },
        ]
        doc_store.nodes.add_batch(seeds)
        doc_store.nodes.update_parent_references_batch(
            [
                ("leaf", "level3"),
                ("level3", "level2"),
                ("level2", "level1"),
                ("level1", "root"),
            ]
        )

        # Scores chosen so root is best under constrained budget
        scores = {
            "leaf": 1.0,
            "level3": 0.01,
            "level2": 0.01,
            "level1": 0.01,
            "root": 1.0,
        }
        coverage_map = {k: True for k in scores}

        generator = DynamicTilingGenerator(QueryConfig())

        nodes_map: dict[str, object] = {}
        for nid in coverage_map:
            node = doc_store.nodes.get_node(nid)
            if node:
                nodes_map[nid] = node
        root_id = next(
            (
                nid
                for nid, n in nodes_map.items()
                if getattr(n, "parent_id", None) not in nodes_map
            ),
            None,
        )
        assert root_id is not None

        # Costs
        level3_cost = generator._get_node_cost(nodes_map["level3"])  # type: ignore[arg-type]
        root_cost = generator._get_node_cost(nodes_map["root"])  # type: ignore[arg-type]

        # Budget just under level3 cost should push selection to root
        budget = level3_cost - 1
        result = generator.find_optimal_tiling(budget, scores, nodes_map, root_id)  # type: ignore[arg-type]

        assert len(result.tiling.node_ids) == 1
        assert result.tiling.node_ids[0] == "root"
        total_tokens = sum(ni.token_cost for ni in result.node_infos)
        assert total_tokens == root_cost
