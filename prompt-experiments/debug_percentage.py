#!/usr/bin/env python3
"""Debug the percentage strategy to see what prompts it's generating."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.strategies import PercentageStrategy

# Load corpus to get some test chunks
with open("experiments/results/corpus.json", "r") as f:
    corpus = json.load(f)

# Test the percentage strategy with different inputs
strategy = PercentageStrategy()

# Test cases: different input sizes and target compressions
test_cases = [
    (200, 100),   # 200 input, target 100 (50%)
    (500, 250),   # 500 input, target 250 (50%)
    (1000, 300),  # 1000 input, target 300 (30%)
    (800, 160),   # 800 input, target 160 (20%)
    (400, 360),   # 400 input, target 360 (90%)
]

print("Testing Percentage Strategy Prompts:\n")
print("-" * 80)

for input_tokens, target_tokens in test_cases:
    # Create fake metrics
    metrics = {
        "tokens": input_tokens,
        "characters": input_tokens * 5,
        "words": int(input_tokens * 0.75)
    }
    
    instruction = strategy.get_length_instruction(metrics, target_tokens)
    percentage = (target_tokens / input_tokens) * 100
    
    print(f"Input: {input_tokens} tokens, Target: {target_tokens} tokens ({percentage:.1f}%)")
    print(f"Instruction: {instruction}")
    print("-" * 80)

# Now test with actual corpus chunks
print("\nTesting with real corpus chunks:\n")
print("-" * 80)

for i in range(3):
    chunk = corpus["chunks"][i]
    input_tokens = chunk["metrics"]["tokens"]
    
    # Test at 50% compression
    target_tokens = input_tokens // 2
    
    instruction = strategy.get_length_instruction(chunk["metrics"], target_tokens)
    
    print(f"Chunk {chunk['id']}: {input_tokens} tokens -> {target_tokens} tokens")
    print(f"Instruction: {instruction}")
    print(f"First 100 chars of text: {chunk['text'][:100]}...")
    print("-" * 80)