#!/usr/bin/env python3
"""Analyze why percentage strategy is performing so poorly."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Load results
with open("experiments/results/raw_results.json", "r") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame([r for r in data["results"] if r.get("success", False)])

# Filter for percentage strategy
pct_df = df[df["strategy"] == "percentage"]

print("Percentage Strategy Analysis")
print("=" * 80)
print(f"Total percentage experiments: {len(pct_df)}")
print()

# Group by compression ratio
print("Results by Compression Ratio:")
print("-" * 80)
for ratio in sorted(pct_df["compression_ratio"].unique()):
    ratio_df = pct_df[pct_df["compression_ratio"] == ratio]
    print(f"\nCompression Ratio: {ratio:.1%}")
    print(f"  Count: {len(ratio_df)}")
    print(f"  Mean target: {ratio_df['target_tokens'].mean():.1f} tokens")
    print(f"  Mean actual: {ratio_df['actual_tokens'].mean():.1f} tokens")
    print(f"  Mean error: {ratio_df['token_error'].mean():.1f} tokens ({ratio_df['token_error_pct'].mean():.1f}%)")
    
    # Show a few examples
    if len(ratio_df) >= 2:
        print("\n  Examples:")
        for idx, row in ratio_df.head(2).iterrows():
            print(f"    Input: {row['input_tokens']} -> Target: {row['target_tokens']} -> Actual: {row['actual_tokens']} (error: {row['token_error']:+d})")

# Look for patterns in the actual output sizes
print("\n" + "=" * 80)
print("Actual Token Distribution:")
print("-" * 80)
print(f"Min actual tokens: {pct_df['actual_tokens'].min()}")
print(f"Max actual tokens: {pct_df['actual_tokens'].max()}")
print(f"Mean actual tokens: {pct_df['actual_tokens'].mean():.1f}")
print(f"Median actual tokens: {pct_df['actual_tokens'].median():.1f}")
print(f"Std actual tokens: {pct_df['actual_tokens'].std():.1f}")

# Check if summaries are suspiciously similar in length
print("\nActual token counts histogram:")
hist, bins = np.histogram(pct_df['actual_tokens'], bins=20)
for i in range(len(hist)):
    if hist[i] > 0:
        print(f"  {bins[i]:.0f}-{bins[i+1]:.0f} tokens: {'█' * int(hist[i]/2)} ({hist[i]} summaries)")

# Check a few actual summaries
print("\n" + "=" * 80)
print("Sample Summaries (first 200 chars):")
print("-" * 80)
for i, row in pct_df.head(3).iterrows():
    print(f"\nInput: {row['input_tokens']} tokens, Target: {row['target_tokens']} tokens ({row['compression_ratio']:.1%})")
    print(f"Actual: {row['actual_tokens']} tokens (error: {row['token_error']:+d})")
    if 'summary' in row and row['summary']:
        print(f"Summary: {row['summary'][:200]}...")
    else:
        print("Summary: [Not available]")