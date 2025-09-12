#!/usr/bin/env python3
# ruff: noqa: E402
"""Quick test of a single strategy to verify the setup works."""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tiktoken
from experiments.strategies import AbsoluteTokenStrategy
from openai import AsyncOpenAI


async def test_single():
    """Test a single summarization with absolute token strategy."""

    # Load one chunk from corpus
    with open("experiments/results/corpus.json") as f:
        corpus = json.load(f)

    chunk = corpus["chunks"][0]
    print(f"Testing with chunk: {chunk['id']}")
    print(f"Input tokens: {chunk['metrics']['tokens']}")

    # Create strategy
    strategy = AbsoluteTokenStrategy()

    # Set target to 50% of input
    target_tokens = chunk['metrics']['tokens'] // 2
    print(f"Target tokens: {target_tokens}")

    # Get prompt
    prompt = strategy.get_prompt(chunk["text"], chunk["metrics"], target_tokens)
    print(f"\nPrompt instruction: {strategy.get_length_instruction(chunk['metrics'], target_tokens)}")

    # Initialize client
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    # Make API call
    print("\nCalling GPT-5-nano...")
    response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="minimal",
    )

    summary = response.choices[0].message.content

    # Measure results
    tokenizer = tiktoken.get_encoding("cl100k_base")
    actual_tokens = len(tokenizer.encode(summary))

    print("\nResults:")
    print(f"  Target: {target_tokens} tokens")
    print(f"  Actual: {actual_tokens} tokens")
    print(f"  Error: {actual_tokens - target_tokens} ({(actual_tokens - target_tokens) / target_tokens * 100:.1f}%)")
    print("\nSummary preview (first 200 chars):")
    print(summary[:200] + "...")


if __name__ == "__main__":
    asyncio.run(test_single())
