#!/usr/bin/env python3
"""Analyze experiment results by input chunk size."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Load results
with open("experiments/results/raw_results.json") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame([r for r in data["results"] if r.get("success", False)])

# Load corpus to get chunk size info
with open("experiments/results/corpus.json") as f:
    corpus = json.load(f)

# Create mapping of chunk_id to target_chunk_size
chunk_size_map = {c["id"]: c["target_chunk_size"] for c in corpus["chunks"]}

# Add chunk size to dataframe
df["chunk_size"] = df["chunk_id"].map(chunk_size_map)

print("Performance Analysis by Chunk Size")
print("=" * 80)

# Analyze each strategy by chunk size
strategies = df["strategy"].unique()

for strategy in sorted(strategies):
    print(f"\n{strategy.upper()}")
    print("-" * 80)

    strategy_df = df[df["strategy"] == strategy]

    for chunk_size in sorted(strategy_df["chunk_size"].unique()):
        size_df = strategy_df[strategy_df["chunk_size"] == chunk_size]

        abs_errors = size_df["token_error_pct"].abs()

        print(f"\nChunk size: {chunk_size} tokens")
        print(f"  Samples: {len(size_df)}")
        print(f"  Mean error: {size_df['token_error_pct'].mean():.1f}%")
        print(f"  Mean abs error: {abs_errors.mean():.1f}%")
        print(f"  Median abs error: {abs_errors.median():.1f}%")
        print(f"  Within ±10%: {(abs_errors <= 10).mean() * 100:.1f}%")
        print(f"  Within ±20%: {(abs_errors <= 20).mean() * 100:.1f}%")

# Create summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE: Mean Absolute Error (%) by Strategy and Chunk Size")
print("-" * 80)

# Create pivot table
pivot = df.pivot_table(
    values="token_error_pct",
    index="strategy",
    columns="chunk_size",
    aggfunc=lambda x: np.abs(x).mean(),
)

# Format as string with 1 decimal place
pivot_str = pivot.round(1).to_string()
print(pivot_str)

# Find best strategy for each chunk size
print("\n" + "=" * 80)
print("BEST STRATEGY FOR EACH CHUNK SIZE")
print("-" * 80)

for chunk_size in sorted(df["chunk_size"].unique()):
    size_df = df[df["chunk_size"] == chunk_size]

    # Calculate mean absolute error for each strategy
    strategy_performance = {}
    for strategy in strategies:
        strat_df = size_df[size_df["strategy"] == strategy]
        if len(strat_df) > 0:
            strategy_performance[strategy] = strat_df["token_error_pct"].abs().mean()

    # Find best
    best_strategy = min(strategy_performance, key=strategy_performance.get)
    best_error = strategy_performance[best_strategy]

    print(f"\n{chunk_size} tokens: {best_strategy} (error: {best_error:.1f}%)")

    # Show top 3
    sorted_strategies = sorted(strategy_performance.items(), key=lambda x: x[1])
    print("  Top 3:")
    for i, (strat, error) in enumerate(sorted_strategies[:3], 1):
        print(f"    {i}. {strat}: {error:.1f}%")

# Check if performance varies by compression ratio within chunk sizes
print("\n" + "=" * 80)
print("INTERACTION EFFECTS: Does optimal strategy depend on compression ratio?")
print("-" * 80)

# Focus on word_count vs absolute_token for different scenarios
for chunk_size in [200, 500, 1000]:
    print(f"\nChunk size: {chunk_size} tokens")

    size_df = df[df["chunk_size"] == chunk_size]
    if len(size_df) == 0:
        continue

    # Group by compression ratio bins
    size_df["compression_bin"] = pd.cut(
        size_df["compression_ratio"],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=[
            "Heavy (10-30%)",
            "Medium (30-50%)",
            "Light (50-70%)",
            "Minimal (70-90%)",
        ],
    )

    for strategy in ["word_count", "absolute_token", "absolute_char"]:
        strat_df = size_df[size_df["strategy"] == strategy]
        if len(strat_df) > 0:
            print(f"\n  {strategy}:")
            for comp_bin in strat_df["compression_bin"].dropna().unique():
                bin_df = strat_df[strat_df["compression_bin"] == comp_bin]
                if len(bin_df) > 0:
                    abs_error = bin_df["token_error_pct"].abs().mean()
                    within_20 = (bin_df["token_error_pct"].abs() <= 20).mean() * 100
                    print(
                        f"    {comp_bin}: {abs_error:.1f}% error, {within_20:.0f}% within ±20%"
                    )
