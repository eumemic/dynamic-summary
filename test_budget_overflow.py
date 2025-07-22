#!/usr/bin/env python3
"""Test case that forces the budget overflow scenario.

Key insight: We need to make the parent segment unattractive 
but the combined child frontier very attractive.
"""

import sys
sys.path.insert(0, '.')

from ragzoom.config import RagZoomConfig
from ragzoom.dynamic_tiling import DynamicTilingGenerator
from tests.mock_store import SimpleMockStore
import tiktoken


def create_budget_overflow_test():
    """Create a test that demonstrates budget overflow."""
    config = RagZoomConfig(leaf_tokens=100)
    store = SimpleMockStore(config=config)
    dp_generator = DynamicTilingGenerator(config, store)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Key to triggering the bug:
    # 1. Parent P costs 80 tokens (40 left + 40 right)
    # 2. All relevance is on right side, so left gets budget 0
    # 3. Right side has a very attractive child frontier that costs 90
    # 4. Left side is forced to use parent segment (40 tokens)
    # 5. Combined: 40 + 90 = 130 > 100
    
    # Create parent with specific costs
    left_40 = "a " * 40  # Will tokenize to ~40 tokens
    right_40 = "b " * 40  # Will tokenize to ~40 tokens
    p_text = left_40 + right_40
    
    print(f"Creating parent P with costs:")
    print(f"  Left half: {len(tokenizer.encode(left_40))} tokens")
    print(f"  Right half: {len(tokenizer.encode(right_40))} tokens") 
    print(f"  Total: {len(tokenizer.encode(p_text))} tokens")
    
    store.add_node(
        node_id="P",
        text=p_text,
        embedding=[0.1] * 1536,
        span_start=0,
        span_end=200,
        document_id="test-doc",
        left_child_id="L",
        right_child_id="R",
        summary=p_text,
        mid_offset=len(left_40),
    )
    
    # Left child - make it worse than parent
    store.add_node(
        node_id="L",
        text="x",  # Minimal
        embedding=[0.2] * 1536,
        span_start=0,
        span_end=100,
        parent_id="P",
        document_id="test-doc",
    )
    
    # Right child - gateway to good content
    store.add_node(
        node_id="R",
        text="y",
        embedding=[0.3] * 1536,
        span_start=100,
        span_end=200,
        parent_id="P",
        document_id="test-doc",
        left_child_id="R1",
        right_child_id="R2",
        summary="y",
        mid_offset=0,
    )
    
    # Create a very attractive right frontier that costs exactly 90 tokens
    # This is the key - we need R1 and R2 combined to be very high quality
    r1_text = "c " * 45  # ~45 tokens
    r2_text = "d " * 45  # ~45 tokens
    
    store.add_node(
        node_id="R1",
        text=r1_text,
        embedding=[0.9] * 1536,
        span_start=100,
        span_end=150,
        parent_id="R",
        document_id="test-doc",
    )
    
    store.add_node(
        node_id="R2",
        text=r2_text,
        embedding=[0.95] * 1536,
        span_start=150,
        span_end=200,
        parent_id="R",
        document_id="test-doc",
    )
    
    print(f"\nRight subtree costs:")
    print(f"  R1: {len(tokenizer.encode(r1_text))} tokens")
    print(f"  R2: {len(tokenizer.encode(r2_text))} tokens")
    print(f"  R1+R2: {len(tokenizer.encode(r1_text)) + len(tokenizer.encode(r2_text))} tokens")
    
    # Critical: Give R1 and R2 very high scores so algorithm prefers them
    scores = {
        "R1": 100.0,  # Extremely high
        "R2": 100.0,  # Extremely high
        # Parent nodes get 0 by default
    }
    
    # Coverage map
    coverage_map = {
        "P": True, "L": True, "R": True, "R1": True, "R2": True
    }
    
    # Debug: manually trace what should happen
    print("\n--- Manual trace of what should happen ---")
    print("1. Start at P with budget=100")
    print("2. P costs 80, so we continue")
    print("3. Budget split: left=0 (no relevance), right=100 (all relevance)")
    print("4. Left side with budget=0:")
    print("   - Can't afford child L")
    print("   - Returns P.LEFT segment (40 tokens)")
    print("5. Right side with budget=100:")
    print("   - Can afford R1+R2 (90 tokens) with quality=200")
    print("   - Much better than P.RIGHT with quality=0")
    print("   - Returns [R1, R2]")
    print("6. Combined: P.LEFT + [R1, R2] = 40 + 90 = 130 tokens")
    print("7. This exceeds budget of 100!")
    
    # Run the algorithm
    print("\n--- Running DP algorithm ---")
    dp_result = dp_generator.find_optimal_tiling(
        budget_tokens=100,
        scores=scores,
        document_id="test-doc", 
        coverage_map=coverage_map,
    )
    
    # Check results
    total_cost = sum(info.token_cost for info in dp_result.segment_infos)
    
    print(f"\nResults:")
    print(f"Total cost: {total_cost} tokens (budget: 100)")
    print(f"Total quality: {dp_result.total_quality}")
    
    print(f"\nSegments chosen:")
    for info in dp_result.segment_infos:
        seg = info.segment
        print(f"  {seg.node_id} (side={seg.side}): {info.token_cost} tokens")
    
    if total_cost > 100:
        print(f"\n❌ BUDGET OVERFLOW! {total_cost} > 100")
        print("This proves the final budget check is necessary!")
        return False
    else:
        print(f"\n✅ Budget satisfied: {total_cost} <= 100")
        print("The algorithm found a safe tiling.")
        # Let's check if it chose the parent fallback
        if any(seg.node_id == "P" for seg in dp_result.segments):
            print("It fell back to using parent segments (the safe choice).")
        return True


if __name__ == "__main__":
    success = create_budget_overflow_test()
    if not success:
        sys.exit(1)