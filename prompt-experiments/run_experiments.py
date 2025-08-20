#!/usr/bin/env python3
"""
Run summarization length targeting experiments.
Tests different strategies for hitting target summary lengths.
"""

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tiktoken
from active_strategies import ACTIVE_STRATEGIES
from openai import AsyncOpenAI


class ExperimentRunner:
    """Run summarization experiments with different targeting strategies."""

    def __init__(self, corpus_path: str = "results/corpus.json",
                 max_concurrent: int = 30):
        """Initialize the experiment runner.
        
        Args:
            corpus_path: Path to the test corpus JSON file
            max_concurrent: Maximum concurrent API requests
        """
        # Handle both relative and absolute paths
        if Path(corpus_path).is_absolute():
            self.corpus_path = Path(corpus_path)
        else:
            # Assume relative to prompt-experiments directory
            self.corpus_path = Path(__file__).parent / corpus_path
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = AsyncOpenAI(api_key=api_key)

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Load corpus
        self.load_corpus()

    def load_corpus(self):
        """Load the test corpus from JSON."""
        with open(self.corpus_path, encoding="utf-8") as f:
            corpus_data = json.load(f)
        self.chunks = corpus_data["chunks"]
        print(f"Loaded {len(self.chunks)} chunks from corpus")

    async def run_single_experiment(self, chunk: dict, strategy: Any,
                                   target_tokens: int, pbar: tqdm = None) -> dict[str, Any]:
        """Run a single summarization experiment.
        
        Args:
            chunk: Chunk data with text and metrics
            strategy: The targeting strategy to use
            target_tokens: Target token count for the summary
            pbar: Optional progress bar to update
            
        Returns:
            Dictionary with experiment results
        """
        async with self.semaphore:
            # Get the prompt for this strategy
            prompt = strategy.get_prompt(
                chunk["text"],
                chunk["metrics"],
                target_tokens
            )

            # Record start time
            start_time = time.time()

            try:
                # Make API call to GPT-5-nano
                response = await self.client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    reasoning_effort="minimal",  # GPT-5 parameter
                )

                # Get the summary
                summary = response.choices[0].message.content
                if not summary:
                    raise ValueError("Empty response from model")

                # Measure actual tokens
                actual_tokens = len(self.tokenizer.encode(summary))
                actual_chars = len(summary)
                actual_words = len(summary.split())

                # Calculate metrics
                token_error = actual_tokens - target_tokens
                token_error_pct = (token_error / target_tokens) * 100

                result = {
                    "chunk_id": chunk["id"],
                    "strategy": strategy.name,
                    "target_tokens": target_tokens,
                    "input_tokens": chunk["metrics"]["tokens"],
                    "actual_tokens": actual_tokens,
                    "actual_chars": actual_chars,
                    "actual_words": actual_words,
                    "token_error": token_error,
                    "token_error_pct": token_error_pct,
                    "summary": summary,
                    "success": True,
                    "duration": time.time() - start_time,
                }

            except Exception as e:
                result = {
                    "chunk_id": chunk["id"],
                    "strategy": strategy.name,
                    "target_tokens": target_tokens,
                    "input_tokens": chunk["metrics"]["tokens"],
                    "error": str(e),
                    "success": False,
                    "duration": time.time() - start_time,
                }

            # Update progress bar if provided
            if pbar:
                pbar.update(1)

            return result

    def get_compression_ratios(self):
        """Get list of compression ratios to test."""
        # Focused range: 30% to 80% in 10% increments
        # This covers practical summarization use cases
        return [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    async def run_all_experiments(self,
                                  sample_size: int = None,
                                  strategies: list = None,
                                  compression_ratios: list[float] = None):
        """Run all experiments across strategies and compression ratios.
        
        Args:
            sample_size: If set, sample this many chunks with balanced representation
            strategies: List of strategies to test (default: ALL_STRATEGIES)
            compression_ratios: List of compression ratios (default: standard set)
        """
        # Optionally sample chunks
        if sample_size and sample_size < len(self.chunks):
            # Always use balanced sampling
            # Create balanced sample across chunk sizes
            by_size = {}
            for chunk in self.chunks:
                size = chunk.get("target_chunk_size", 200)
                if size not in by_size:
                    by_size[size] = []
                by_size[size].append(chunk)

            # Sample equally from each size category
            per_size = sample_size // len(by_size)
            remainder = sample_size % len(by_size)

            test_chunks = []
            for i, (size, chunks) in enumerate(sorted(by_size.items())):
                n = per_size + (1 if i < remainder else 0)
                n = min(n, len(chunks))  # Don't oversample
                test_chunks.extend(random.sample(chunks, n))

            print(f"Using balanced sample of {len(test_chunks)} chunks")
            # Show distribution
            sizes = [c.get("target_chunk_size", 200) for c in test_chunks]
            for size in sorted(set(sizes)):
                count = sizes.count(size)
                print(f"  {size}-token chunks: {count}")
        else:
            test_chunks = self.chunks
            print(f"Using all {len(test_chunks)} chunks")

        # Use provided strategies or default to active strategies
        if strategies is None:
            strategies = ACTIVE_STRATEGIES

        # Get compression ratios
        if compression_ratios is None:
            compression_ratios = self.get_compression_ratios()

        # Calculate total experiments
        total_experiments = len(test_chunks) * len(strategies) * len(compression_ratios)
        print(f"\nTotal experiments to run: {total_experiments}")
        print(f"  Chunks: {len(test_chunks)}")
        print(f"  Compression ratios: {len(compression_ratios)}")
        print(f"  Strategies ({len(strategies)}):")
        for s in strategies:
            print(f"    - {s.name}")

        # Collect all experiment configurations first
        experiment_configs = []

        for chunk in test_chunks:
            input_tokens = chunk["metrics"]["tokens"]

            for ratio in compression_ratios:
                # Calculate target tokens for this ratio
                target_tokens = int(input_tokens * ratio)

                # Skip if target is too small or same as input
                if target_tokens < 10 or target_tokens >= input_tokens:
                    continue

                for strategy in strategies:
                    experiment_configs.append({
                        "chunk": chunk,
                        "strategy": strategy,
                        "compression_ratio": ratio,
                        "target_tokens": target_tokens,
                    })

        print(f"\nActual experiments after filtering: {len(experiment_configs)}")
        print("Starting experiments...\n")

        # Create progress bar
        pbar = tqdm(total=len(experiment_configs), desc="Running experiments", unit="exp")

        # Create tasks with progress bar
        tasks = []
        for config in experiment_configs:
            task = self.run_single_experiment(
                config["chunk"],
                config["strategy"],
                config["target_tokens"],
                pbar
            )
            tasks.append(task)

        # Run all experiments concurrently
        results = await asyncio.gather(*tasks)

        # Close progress bar
        pbar.close()

        # Add compression ratio to results
        for result, config in zip(results, experiment_configs):
            result["compression_ratio"] = config["compression_ratio"]

        return results

    def save_results(self, results: list[dict], output_path: str = None):
        """Save experiment results to JSON.
        
        Args:
            results: List of experiment results
            output_path: Path to save results (default: experiments/results/raw_results.json)
        """
        if output_path is None:
            output_path = "experiments/results/raw_results.json"

        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)

        # Calculate summary statistics
        successful = [r for r in results if r.get("success", False)]
        failed = len(results) - len(successful)

        output_data = {
            "timestamp": time.time(),
            "total_experiments": len(results),
            "successful": len(successful),
            "failed": failed,
            "strategies": [s.name for s in ACTIVE_STRATEGIES],
            "compression_ratios": self.get_compression_ratios(),
            "results": results
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Results saved to {output_file}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {failed}")


async def main():
    """Main entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run summarization length targeting experiments")
    parser.add_argument("--sample", type=int, help="Sample size with balanced chunk representation (default: use all chunks)")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--max-concurrent", type=int, default=30,
                       help="Maximum concurrent API requests (default: 30)")

    args = parser.parse_args()

    # Create runner
    runner = ExperimentRunner(max_concurrent=args.max_concurrent)

    # Run experiments
    print("🚀 Starting summarization experiments...")
    start_time = time.time()

    results = await runner.run_all_experiments(sample_size=args.sample)

    elapsed = time.time() - start_time
    print(f"\n⏱️  Experiments completed in {elapsed:.1f} seconds")

    # Save results with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    output_file = output_dir / "raw_results.json"
    runner.save_results(results, str(output_file))

    # Auto-run analysis and generate visualizations
    print("\n📊 Generating analysis and visualizations...")
    # Run analysis on the results we just saved
    import sys

    from analyze_results import main as analyze_main
    old_argv = sys.argv
    sys.argv = ["analyze_results.py", str(output_file)]

    try:
        analyze_main()
    except Exception as e:
        print(f"Warning: Analysis failed with error: {e}")
    finally:
        sys.argv = old_argv

    # Create symlink to latest results
    latest_link = Path("results/latest")
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.name)

    print(f"\n✅ Results saved to {output_dir}/")

    # List all generated charts with absolute paths for easy clicking
    print("\n📈 Generated visualizations (command-click to open):")
    for png_file in sorted(output_dir.glob("*.png")):
        print(f"  - {png_file.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
