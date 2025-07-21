"""Tests for the Retriever class and its methods."""

from unittest.mock import MagicMock

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.retrieve import Retriever
from ragzoom.store import Store


class TestRetriever:
    """Test suite for the Retriever class."""

    @pytest.fixture
    def setup_retriever(self, tmp_path, monkeypatch):
        """Set up a Retriever instance with a mock store."""
        monkeypatch.setenv(
            "RAGZOOM_SQLITE_DATABASE_URL", f"sqlite:///{tmp_path}/test.db"
        )
        monkeypatch.setenv("RAGZOOM_CHROMA_DB_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("RAGZOOM_OPENAI_API_KEY", "test-key")

        config = RagZoomConfig()
        store = MagicMock(spec=Store)
        retriever = Retriever(config, store)
        return retriever, store

    @pytest.mark.skip(
        reason="Testing legacy _extract_frontier method - needs analysis: does DP algorithm maintain these invariants?"
    )
    def test_extract_tiling_logic_legacy(self, setup_retriever):
        """Verify the logic of the _extract_frontier method."""
        retriever, store = setup_retriever

        # Mock the node structure
        #      P1
        #     /  \
        #    C1   C2
        #        /  \
        #       GC1  GC2
        nodes = {
            "P1": MagicMock(id="P1", span_start=0),
            "C1": MagicMock(id="C1", span_start=0),
            "C2": MagicMock(id="C2", span_start=50),
            "GC1": MagicMock(id="GC1", span_start=50),
            "GC2": MagicMock(id="GC2", span_start=75),
        }
        store.get_node.side_effect = lambda node_id: nodes.get(node_id)

        children_map = {
            "P1": (nodes["C1"], nodes["C2"]),
            "C1": (None, None),
            "C2": (nodes["GC1"], nodes["GC2"]),
            "GC1": (None, None),
            "GC2": (None, None),
        }
        store.get_children.side_effect = lambda node_id: children_map.get(
            node_id, (None, None)
        )

        # Case 1: P1 is covered, but its children are not. P1 is the tiling.
        coverage_map = {"P1": True}
        frontier = retriever._extract_frontier(coverage_map)
        assert frontier == ["P1"]

        # Case 2: P1 and C1 are covered. C2 is not. Tiling is C1 and P1.
        # P1 is still on the tiling because it's not fully represented by its covered children.
        coverage_map = {"P1": True, "C1": True}
        frontier = retriever._extract_frontier(coverage_map)
        assert sorted(frontier) == sorted(["C1", "P1"])

        # Case 3: P1, C1, C2 are covered. GC1, GC2 are not.
        # P1 is no longer on the tiling. C1 and C2 are.
        coverage_map = {"P1": True, "C1": True, "C2": True}
        frontier = retriever._extract_frontier(coverage_map)
        assert sorted(frontier) == sorted(["C1", "C2"])

        # Case 4: Full branch is covered.
        # Tiling is the deepest covered nodes: C1, GC1, GC2
        coverage_map = {"P1": True, "C1": True, "C2": True, "GC1": True, "GC2": True}
        frontier = retriever._extract_frontier(coverage_map)
        assert sorted(frontier) == sorted(["C1", "GC1", "GC2"])
