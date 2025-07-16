"""Test that demonstrates the exact assembly ordering bug from Issue #7.

This test fails with the OLD code but passes with the NEW code.
"""


import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.retrieve import RetrievalResult
from ragzoom.store import Store


class TestAssemblyOrderingBug:
    """Test the specific bug where sorting by node.span_start gives wrong order."""

    @pytest.fixture
    def setup_components(self, tmp_path, monkeypatch):
        """Set up test components."""
        monkeypatch.setenv("RAGZOOM_SQLITE_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
        monkeypatch.setenv("RAGZOOM_CHROMA_DB_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("RAGZOOM_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("RAGZOOM_SLOPE_CAP", "false")

        config = RagZoomConfig()
        store = Store(config)

        yield config, store
        store.close()

    def test_assembly_ordering_bug_exact_scenario(self, setup_components):
        """Test the exact scenario where old code fails but new code succeeds."""
        config, store = setup_components

        # Create scenario where node.span_start order != actual coverage order
        #
        # Node A (parent): span (0-100), when left child in frontier, outputs RIGHT half at (50-100)
        # Node B (leaf): span (40-80), outputs full content at (40-80)
        #
        # _sort_nodes_chronologically() order: A (span_start=0), B (span_start=40)
        # Correct chronological order: B (actual 40-80), A (actual 50-100)
        #
        # OLD BUG: Outputs A's content first, then B's content = WRONG
        # NEW FIX: Outputs B's content first, then A's content = CORRECT

        # Left child of parent (will be in frontier)
        store.add_node(
            node_id="left_child",
            text="FIRST: The story begins here.",
            embedding=[0.1] * 1536,
            depth=0,
            span_start=0,
            span_end=50,
            document_id="doc1",
        )

        # Right child of parent
        store.add_node(
            node_id="right_child",
            text="THIRD: The story continues.",
            embedding=[0.2] * 1536,
            depth=0,
            span_start=50,
            span_end=100,
            document_id="doc1",
        )

        # Parent node with <<<MID>>> delimiter
        parent_text = "Summary of beginning. <<<MID>>> THIRD: Summary of continuation."
        parent_mid_offset = parent_text.find("<<<MID>>>")

        store.add_node(
            node_id="parent_node",
            text=parent_text,
            embedding=[0.15] * 1536,
            depth=1,
            span_start=0,  # This will be sorted first by _sort_nodes_chronologically
            span_end=100,
            left_child_id="left_child",
            right_child_id="right_child",
            summary=parent_text,
            mid_offset=parent_mid_offset,
            document_id="doc1",
        )

        # Middle content node that should come BEFORE parent's right half
        store.add_node(
            node_id="middle_node",
            text="SECOND: Important middle content.",
            embedding=[0.3] * 1536,
            depth=0,
            span_start=40,  # This will be sorted second by _sort_nodes_chronologically
            span_end=80,    # But should appear BEFORE parent's right half (50-100)
            document_id="doc1",
        )

        # Frontier: left_child + parent_node + middle_node
        # Since left_child is in frontier, parent_node outputs only RIGHT half (50-100)
        # middle_node outputs full content (40-80)
        #
        # OLD: parent_node (span_start=0) then middle_node (span_start=40)
        #      Result: "THIRD: Summary..." then "SECOND: Important..."  = WRONG ORDER!
        #
        # NEW: middle_node (actual 40-80) then parent_node (actual 50-100)
        #      Result: "SECOND: Important..." then "THIRD: Summary..." = CORRECT ORDER!

        retrieval_result = RetrievalResult(
            node_ids=["left_child", "parent_node", "middle_node"],
            scores={"left_child": 0.9, "parent_node": 0.8, "middle_node": 0.7},
            coverage_map={"parent_node": True, "left_child": True, "middle_node": True},
            frontier_nodes=["left_child", "parent_node", "middle_node"],
        )

        assembler = Assembler(config, store)
        result = assembler.assemble(retrieval_result)

        lines = result.strip().split('\n\n')

        # With the NEW code (fix), should be in correct chronological order:
        assert len(lines) == 3
        assert "FIRST:" in lines[0]   # left_child (0-50)
        assert "SECOND:" in lines[1]  # middle_node (40-80)
        assert "THIRD:" in lines[2]   # parent_node right half (50-100)

        # With OLD code (bug), this assertion would FAIL because:
        # - parent_node would be sorted before middle_node due to span_start=0 < 40
        # - Result would be: "FIRST:", "THIRD:", "SECOND:" = wrong order
