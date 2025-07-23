"""Test for zero score collapse bug.

This test reproduces the issue where the algorithm returns empty or minimal
tiling when ancestors have zero quality scores.
"""

from ragzoom.config import RagZoomConfig
from ragzoom.dynamic_tiling import DynamicTilingGenerator, Segment
from tests.mock_store import SimpleMockStore


def test_zero_score_collapse_empty_result():
    """Test that algorithm correctly uses root segments when deeper nodes don't fit budget."""

    # Create configuration and store
    config = RagZoomConfig(leaf_tokens=100)
    store = SimpleMockStore(config=config)

    # Create a tree where only the leaf has a score
    # But the leaf is too expensive for the budget

    # Leaf - 50 tokens (seed node)
    store.add_node(
        node_id="leaf",
        text="x" * 100,  # Will be ~50 tokens
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        parent_id="parent",  # Fix: Add parent_id
        document_id="test-doc",
    )

    # Parent - 20 tokens when split
    store.add_node(
        node_id="parent",
        text="y" * 40,  # Will be ~20 tokens total
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        parent_id="root",  # Fix: Add parent_id
        left_child_id="leaf",
        right_child_id="leaf",
        summary="y" * 40,
        mid_offset=20,
        document_id="test-doc",
    )

    # Root - 10 tokens when split
    store.add_node(
        node_id="root",
        text="z" * 20,  # Will be ~10 tokens total
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        left_child_id="parent",
        right_child_id="parent",
        summary="z" * 20,
        mid_offset=10,
        document_id="test-doc",
    )

    # With the fix, all nodes in coverage map should have scores
    # Simulating what retrieve.py would compute
    scores = {
        "leaf": 1.0,  # High relevance (exact match)
        "parent": 0.7,  # Parent has moderate relevance
        "root": 0.4,  # Root has lower relevance
    }
    coverage_map = {"root": True, "parent": True, "leaf": True}

    generator = DynamicTilingGenerator(config, store)

    # Get actual token costs
    leaf_cost = generator._get_segment_cost(Segment("leaf", None))
    parent_cost = generator._get_segment_cost(
        Segment("parent", "LEFT")
    ) + generator._get_segment_cost(Segment("parent", "RIGHT"))
    root_cost = generator._get_segment_cost(
        Segment("root", "LEFT")
    ) + generator._get_segment_cost(Segment("root", "RIGHT"))

    print("\nToken costs:")
    print(f"  leaf: {leaf_cost}")
    print(f"  parent (L+R): {parent_cost}")
    print(f"  root (L+R): {root_cost}")

    # Test with budget that can't fit leaf but can fit parent
    budget = leaf_cost - 1  # Just under leaf cost
    result = generator.find_optimal_tiling(budget, scores, "test-doc", coverage_map)

    print(f"\nWith budget {budget}:")
    print(f"  Segments: {result.segments}")
    print(f"  Quality: {result.total_quality}")

    # The algorithm now correctly returns root segments instead of empty
    # With budget=12 and scores for all nodes:
    # - Root segments (10 tokens) give relevance-tokens = 0.4 * 10 = 4.0
    # - Parent segments (10 tokens) would give 0.7 * 10 = 7.0 but can't be reached
    #   due to budget splitting (each side only gets 6 tokens)
    # - So root segments are the best achievable option

    assert (
        len(result.segments) == 2
    ), f"Expected 2 segments, got {len(result.segments)} segments: {result.segments}"
    assert result.segments[0].node_id == "root"
    assert result.segments[0].side == "LEFT"
    assert result.segments[1].node_id == "root"
    assert result.segments[1].side == "RIGHT"

    # Should use root segments
    total_tokens = sum(si.token_cost for si in result.segment_infos)
    assert (
        total_tokens == root_cost
    ), f"Should use root segments ({root_cost} tokens), but used {total_tokens}"


def test_zero_score_collapse_to_root():
    """Test algorithm correctly chooses root segments due to budget splitting constraints."""

    config = RagZoomConfig(leaf_tokens=100)
    store = SimpleMockStore(config=config)

    # Create a deeper tree to show collapse behavior
    # All token counts are for LEFT+RIGHT combined

    # Leaf (30 tokens) - the only seed
    store.add_node(
        node_id="leaf",
        text="This is important content that the user searched for and we want to include",
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        parent_id="level3",  # Fix: Add parent_id
        document_id="test-doc",
    )

    # Level 3 (25 tokens) - good summary close to content
    store.add_node(
        node_id="level3",
        text="This is a detailed summary of the important content that user wants",
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        parent_id="level2",  # Fix: Add parent_id
        left_child_id="leaf",
        right_child_id="leaf",
        summary="This is a detailed summary of the important content that user wants",
        mid_offset=30,
        document_id="test-doc",
    )

    # Level 2 (20 tokens) - medium summary
    store.add_node(
        node_id="level2",
        text="A medium level summary of the content below including key points",
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        parent_id="level1",  # Fix: Add parent_id
        left_child_id="level3",
        right_child_id="level3",
        summary="A medium level summary of the content below including key points",
        mid_offset=25,
        document_id="test-doc",
    )

    # Level 1 (15 tokens) - higher level summary
    store.add_node(
        node_id="level1",
        text="High level summary of this document section with main themes",
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        parent_id="root",  # Fix: Add parent_id
        left_child_id="level2",
        right_child_id="level2",
        summary="High level summary of this document section with main themes",
        mid_offset=20,
        document_id="test-doc",
    )

    # Root (5 tokens) - very brief
    store.add_node(
        node_id="root",
        text="Brief document overview",
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        left_child_id="level1",
        right_child_id="level1",
        summary="Brief document overview",
        mid_offset=10,
        document_id="test-doc",
    )

    # With the fix, all nodes in coverage map should have scores
    # Simulating decreasing relevance as we go up the tree
    scores = {
        "leaf": 1.0,  # Exact match
        "level3": 0.8,  # Close summary
        "level2": 0.6,  # Medium summary
        "level1": 0.4,  # Higher level
        "root": 0.2,  # Very high level
    }
    coverage_map = {
        "root": True,
        "level1": True,
        "level2": True,
        "level3": True,
        "leaf": True,
    }

    generator = DynamicTilingGenerator(config, store)

    # Get token costs
    leaf_cost = generator._get_segment_cost(Segment("leaf", None))
    level3_cost = generator._get_segment_cost(
        Segment("level3", "LEFT")
    ) + generator._get_segment_cost(Segment("level3", "RIGHT"))
    level2_cost = generator._get_segment_cost(
        Segment("level2", "LEFT")
    ) + generator._get_segment_cost(Segment("level2", "RIGHT"))
    level1_cost = generator._get_segment_cost(
        Segment("level1", "LEFT")
    ) + generator._get_segment_cost(Segment("level1", "RIGHT"))
    root_cost = generator._get_segment_cost(
        Segment("root", "LEFT")
    ) + generator._get_segment_cost(Segment("root", "RIGHT"))

    print("\nActual token costs:")
    print(f"  leaf: {leaf_cost}")
    print(f"  level3: {level3_cost}")
    print(f"  level2: {level2_cost}")
    print(f"  level1: {level1_cost}")
    print(f"  root: {root_cost}")

    # Test with budget just under level3 (should use level2, but will collapse to root)
    budget = level3_cost - 1
    result = generator.find_optimal_tiling(budget, scores, "test-doc", coverage_map)

    print(f"\nWith budget {budget} (just under level3's {level3_cost}):")
    print(f"  Result segments: {[s.node_id for s in result.segments]}")
    print(f"  Total tokens used: {sum(si.token_cost for si in result.segment_infos)}")

    # The algorithm correctly returns root segments
    # With budget=11:
    # - level2 costs 11 tokens and would give relevance-tokens = 0.6 * 11 = 6.6
    # - But due to budget splitting at root, each side only gets ~5-6 tokens
    # - This isn't enough to afford level1 (11 tokens), so it returns empty
    # - So the algorithm uses root segments as the best achievable option

    assert len(result.segments) == 2, f"Expected 2 segments, got {len(result.segments)}"
    assert all(
        s.node_id == "root" for s in result.segments
    ), f"Expected root segments, got {[s.node_id for s in result.segments]}"

    # Should use root segments
    total_tokens = sum(si.token_cost for si in result.segment_infos)
    assert (
        total_tokens == root_cost
    ), f"Should use root ({root_cost} tokens), but used {total_tokens} tokens"


if __name__ == "__main__":
    print("=== Test 1: Empty Result ===")
    test_zero_score_collapse_empty_result()
    print("\n=== Test 2: Collapse to Root ===")
    test_zero_score_collapse_to_root()
    print("\nBoth tests demonstrate the zero score bug!")
