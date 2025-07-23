"""Test for zero score collapse bug.

This test reproduces the issue where the algorithm returns empty or minimal
tiling when ancestors have zero quality scores.
"""

from ragzoom.config import RagZoomConfig
from ragzoom.dynamic_tiling import DynamicTilingGenerator
from tests.mock_store import SimpleMockStore


def test_zero_score_collapse_empty_result():
    """Test that algorithm returns empty tiling when it can't reach seed nodes."""

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
        parent_id="parent",  # Set parent
        document_id="test-doc",
    )

    # Parent - 20 tokens when split
    store.add_node(
        node_id="parent",
        text="y" * 40,  # Will be ~20 tokens total
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        left_child_id="leaf",
        right_child_id="leaf",
        summary="y" * 40,
        parent_id="root",  # Set parent
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
        parent_id=None,  # This is the root
        document_id="test-doc",
    )

    # Only leaf has score
    scores = {"leaf": 1.0}
    coverage_map = {"root": True, "parent": True, "leaf": True}

    generator = DynamicTilingGenerator(store, config)

    # Get actual token costs
    leaf = store.get_node("leaf")
    parent = store.get_node("parent")
    root = store.get_node("root")

    leaf_cost = generator._get_node_cost(leaf)
    parent_cost = generator._get_node_cost(parent)
    root_cost = generator._get_node_cost(root)

    print("\nToken costs:")
    print(f"  leaf: {leaf_cost}")
    print(f"  parent: {parent_cost}")
    print(f"  root: {root_cost}")

    # Test with budget that can't fit leaf but can fit parent
    budget = leaf_cost - 1  # Just under leaf cost
    result = generator.find_optimal_tiling(budget, scores, "test-doc", coverage_map)

    print(f"\nWith budget {budget}:")
    print(f"  Tiling: {result.tiling}")
    print(f"  Quality: {result.total_quality}")

    # Previously returned empty tiling, now returns root for coverage
    # The algorithm correctly chooses root because:
    # - Root costs 10 tokens (fits in budget of 12)
    # - Using both children (parent nodes) would cost 20 tokens (exceeds budget)
    # - So root is the optimal choice for coverage

    assert (
        len(result.tiling) == 1
    ), f"Expected 1 node, got {len(result.tiling)} nodes: {result.tiling}"
    assert result.tiling[0] == "root", f"Expected root, got {result.tiling[0]}"

    # Should use the root node's tokens
    # Calculate tokens from the tiling
    total_tokens = sum(
        generator._get_node_cost(store.get_node(node_id)) for node_id in result.tiling
    )
    assert (
        total_tokens == root_cost
    ), f"Should use root node ({root_cost} tokens), but used {total_tokens}"


def test_zero_score_collapse_to_root():
    """Test algorithm collapses to root level when intermediate nodes have zero scores."""

    config = RagZoomConfig(leaf_tokens=100)
    store = SimpleMockStore(config=config)

    # Create a deeper tree to show collapse behavior

    # Leaf (30 tokens) - the only seed
    store.add_node(
        node_id="leaf",
        text="This is important content that the user searched for and we want to include",
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        parent_id="level3",  # Set parent
        document_id="test-doc",
    )

    # Level 3 (25 tokens) - good summary close to content
    store.add_node(
        node_id="level3",
        text="This is a detailed summary of the important content that user wants",
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        left_child_id="leaf",
        right_child_id="leaf",
        summary="This is a detailed summary of the important content that user wants",
        parent_id="level2",  # Set parent
        document_id="test-doc",
    )

    # Level 2 (20 tokens) - medium summary
    store.add_node(
        node_id="level2",
        text="A medium level summary of the content below including key points",
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        left_child_id="level3",
        right_child_id="level3",
        summary="A medium level summary of the content below including key points",
        parent_id="level1",  # Set parent
        document_id="test-doc",
    )

    # Level 1 (15 tokens) - higher level summary
    store.add_node(
        node_id="level1",
        text="High level summary of this document section with main themes",
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=100,
        left_child_id="level2",
        right_child_id="level2",
        summary="High level summary of this document section with main themes",
        parent_id="root",  # Set parent
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
        parent_id=None,  # This is the root
        document_id="test-doc",
    )

    scores = {"leaf": 1.0}
    coverage_map = {
        "root": True,
        "level1": True,
        "level2": True,
        "level3": True,
        "leaf": True,
    }

    generator = DynamicTilingGenerator(store, config)

    # Get token costs
    leaf = store.get_node("leaf")
    level3 = store.get_node("level3")
    level2 = store.get_node("level2")
    level1 = store.get_node("level1")
    root = store.get_node("root")

    leaf_cost = generator._get_node_cost(leaf)
    level3_cost = generator._get_node_cost(level3)
    level2_cost = generator._get_node_cost(level2)
    level1_cost = generator._get_node_cost(level1)
    root_cost = generator._get_node_cost(root)

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
    print(f"  Result tiling: {result.tiling}")
    # Calculate tokens from the tiling
    total_tokens = sum(
        generator._get_node_cost(store.get_node(node_id)) for node_id in result.tiling
    )
    print(f"  Total tokens used: {total_tokens}")

    # BUG: Should use level2 summary but returns empty or collapses to root
    # Since level2 costs 11 tokens and budget is 11, it should use level2

    # With the new algorithm, it should choose root since:
    # - Only leaf has score (1.0)
    # - Budget is 11, can't afford leaf (14) or level3 (12)
    # - Between level2 (11 tokens, 0 score) and root (3 tokens, 0 score)
    # - Both have 0 score, so algorithm picks cheaper option (root)
    assert len(result.tiling) == 1, f"Expected 1 node, got {len(result.tiling)}"
    assert result.tiling[0] == "root", f"Expected root, got {result.tiling}"

    # Verify token usage
    assert (
        total_tokens == root_cost
    ), f"Should use root ({root_cost} tokens), but used {total_tokens} tokens"


if __name__ == "__main__":
    print("=== Test 1: Empty Result ===")
    test_zero_score_collapse_empty_result()
    print("\n=== Test 2: Collapse to Root ===")
    test_zero_score_collapse_to_root()
    print("\nBoth tests demonstrate the zero score bug!")
