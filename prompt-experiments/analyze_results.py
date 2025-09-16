#!/usr/bin/env python3
"""
Analyze results from summarization length targeting experiments.
Generates statistics and visualizations to identify optimal strategies.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(results_path: str = "experiments/results/raw_results.json") -> dict:
    """Load experiment results from JSON."""
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


def calculate_strategy_statistics(results: list[dict]) -> pd.DataFrame:
    """Calculate statistics for each strategy.

    Returns:
        DataFrame with statistics per strategy
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([r for r in results if r.get("success", False)])

    if df.empty:
        print("No successful results to analyze")
        return pd.DataFrame()

    # Group by strategy
    stats = []
    for strategy in df["strategy"].unique():
        strategy_df = df[df["strategy"] == strategy]

        # Calculate metrics
        errors = strategy_df["token_error"].values
        errors_pct = strategy_df["token_error_pct"].values
        abs_errors = np.abs(errors)
        abs_errors_pct = np.abs(errors_pct)

        stats.append(
            {
                "strategy": strategy,
                "count": len(strategy_df),
                "mean_error": np.mean(errors),
                "mean_error_pct": np.mean(errors_pct),
                "median_error": np.median(errors),
                "median_error_pct": np.median(errors_pct),
                "mean_abs_error": np.mean(abs_errors),
                "mean_abs_error_pct": np.mean(abs_errors_pct),
                "median_abs_error": np.median(abs_errors),
                "median_abs_error_pct": np.median(abs_errors_pct),
                "std_error": np.std(errors),
                "std_error_pct": np.std(errors_pct),
                "within_10_pct": np.sum(abs_errors_pct <= 10)
                / len(abs_errors_pct)
                * 100,
                "within_20_pct": np.sum(abs_errors_pct <= 20)
                / len(abs_errors_pct)
                * 100,
                "max_overshoot": np.max(errors),
                "max_undershoot": np.min(errors),
            }
        )

    return pd.DataFrame(stats).sort_values("mean_abs_error_pct")


def analyze_by_compression_ratio(results: list[dict]) -> pd.DataFrame:
    """Analyze performance by compression ratio.

    Returns:
        DataFrame with statistics per strategy per compression ratio
    """
    df = pd.DataFrame([r for r in results if r.get("success", False)])

    if df.empty:
        return pd.DataFrame()

    # Group by strategy and compression ratio
    grouped = df.groupby(["strategy", "compression_ratio"])

    stats = []
    for (strategy, ratio), group in grouped:
        errors_pct = group["token_error_pct"].values
        abs_errors_pct = np.abs(errors_pct)

        stats.append(
            {
                "strategy": strategy,
                "compression_ratio": ratio,
                "count": len(group),
                "mean_error_pct": np.mean(errors_pct),
                "mean_abs_error_pct": np.mean(abs_errors_pct),
                "median_abs_error_pct": np.median(abs_errors_pct),
                "std_error_pct": np.std(errors_pct),
                "within_10_pct": np.sum(abs_errors_pct <= 10)
                / len(abs_errors_pct)
                * 100,
                "within_20_pct": np.sum(abs_errors_pct <= 20)
                / len(abs_errors_pct)
                * 100,
            }
        )

    return pd.DataFrame(stats)


def analyze_by_input_size(results: list[dict]) -> pd.DataFrame:
    """Analyze performance by input size category.

    Returns:
        DataFrame with statistics per strategy per input size category
    """
    df = pd.DataFrame([r for r in results if r.get("success", False)])

    if df.empty:
        return pd.DataFrame()

    # Categorize input sizes
    def categorize_size(tokens):
        if tokens < 250:
            return "tiny (< 250)"
        elif tokens < 450:
            return "small (250-450)"
        elif tokens < 750:
            return "medium (450-750)"
        elif tokens < 1200:
            return "large (750-1200)"
        else:
            return "huge (1200+)"

    df["size_category"] = df["input_tokens"].apply(categorize_size)

    # Group by strategy and size
    grouped = df.groupby(["strategy", "size_category"])

    stats = []
    for (strategy, size), group in grouped:
        errors_pct = group["token_error_pct"].values
        abs_errors_pct = np.abs(errors_pct)

        stats.append(
            {
                "strategy": strategy,
                "size_category": size,
                "count": len(group),
                "mean_error_pct": np.mean(errors_pct),
                "mean_abs_error_pct": np.mean(abs_errors_pct),
                "median_abs_error_pct": np.median(abs_errors_pct),
                "within_10_pct": np.sum(abs_errors_pct <= 10)
                / len(abs_errors_pct)
                * 100,
                "within_20_pct": np.sum(abs_errors_pct <= 20)
                / len(abs_errors_pct)
                * 100,
            }
        )

    return pd.DataFrame(stats)


def test_nice_fractions_hypothesis(results: list[dict]) -> dict[str, Any]:
    """Test if 'nice' fractions perform better than 'messy' ones.

    Returns:
        Dictionary with analysis results
    """
    df = pd.DataFrame([r for r in results if r.get("success", False)])

    if df.empty:
        return {}

    # Define nice and messy ratios
    nice_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Categorize
    df["is_nice"] = df["compression_ratio"].apply(lambda x: x in nice_ratios)

    # Compare performance
    nice_df = df[df["is_nice"]]
    messy_df = df[~df["is_nice"]]

    return {
        "nice_ratios": {
            "count": len(nice_df),
            "mean_abs_error_pct": nice_df["token_error_pct"].abs().mean(),
            "median_abs_error_pct": nice_df["token_error_pct"].abs().median(),
            "within_10_pct": (nice_df["token_error_pct"].abs() <= 10).mean() * 100,
            "within_20_pct": (nice_df["token_error_pct"].abs() <= 20).mean() * 100,
        },
        "messy_ratios": {
            "count": len(messy_df),
            "mean_abs_error_pct": messy_df["token_error_pct"].abs().mean(),
            "median_abs_error_pct": messy_df["token_error_pct"].abs().median(),
            "within_10_pct": (messy_df["token_error_pct"].abs() <= 10).mean() * 100,
            "within_20_pct": (messy_df["token_error_pct"].abs() <= 20).mean() * 100,
        },
    }


def identify_compensation_factors(results: list[dict]) -> pd.DataFrame:
    """Identify systematic biases that can be compensated for.

    Returns:
        DataFrame with compensation factors per strategy
    """
    df = pd.DataFrame([r for r in results if r.get("success", False)])

    if df.empty:
        return pd.DataFrame()

    compensations = []
    for strategy in df["strategy"].unique():
        strategy_df = df[df["strategy"] == strategy]

        # Calculate mean bias (systematic over/under-shooting)
        mean_error_pct = strategy_df["token_error_pct"].mean()

        # Calculate compensation factor
        # If mean error is +10%, we need to ask for 90.9% of what we want
        compensation_factor = 100 / (100 + mean_error_pct)

        compensations.append(
            {
                "strategy": strategy,
                "mean_bias_pct": mean_error_pct,
                "compensation_factor": compensation_factor,
                "recommendation": (
                    f"Ask for {compensation_factor:.2f}x the target"
                    if abs(mean_error_pct) > 5
                    else "No compensation needed"
                ),
            }
        )

    return pd.DataFrame(compensations)


def create_visualizations(results: list[dict], output_dir: str = "experiments/results"):
    """Create visualization plots for the analysis.

    Args:
        results: List of experiment results
        output_dir: Directory to save visualizations
    """
    df = pd.DataFrame([r for r in results if r.get("success", False)])

    if df.empty:
        print("No data for visualizations")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. Error distribution by strategy (violin plot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Violin plot of errors
    sns.violinplot(data=df, x="strategy", y="token_error_pct", ax=ax1)
    ax1.axhline(0, color="green", linestyle="--", alpha=0.5)
    ax1.axhline(20, color="red", linestyle="--", alpha=0.3)
    ax1.axhline(-20, color="red", linestyle="--", alpha=0.3)
    ax1.set_title("Error Distribution by Strategy")
    ax1.set_xlabel("Strategy")
    ax1.set_ylabel("Token Error (%)")
    ax1.tick_params(axis="x", rotation=45)

    # Box plot of absolute errors
    df["abs_error_pct"] = df["token_error_pct"].abs()
    sns.boxplot(data=df, x="strategy", y="abs_error_pct", ax=ax2)
    ax2.axhline(10, color="green", linestyle="--", alpha=0.5, label="±10% threshold")
    ax2.axhline(20, color="orange", linestyle="--", alpha=0.5, label="±20% threshold")
    ax2.set_title("Absolute Error Distribution by Strategy")
    ax2.set_xlabel("Strategy")
    ax2.set_ylabel("Absolute Token Error (%)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path / "error_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Performance heatmap by compression ratio
    pivot_data = df.pivot_table(
        values="abs_error_pct",
        index="compression_ratio",
        columns="strategy",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=50,
        ax=ax,
        cbar_kws={"label": "Mean Absolute Error (%)"},
    )
    ax.set_title(
        "Performance Heatmap: Mean Absolute Error by Strategy and Compression Ratio"
    )
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Compression Ratio")
    plt.tight_layout()
    plt.savefig(output_path / "performance_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Scatter plot: Target vs Actual
    strategies = df["strategy"].unique()
    n_strategies = len(strategies)
    n_cols = 3
    n_rows = (n_strategies + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_strategies > 1 else [axes]

    for i, strategy in enumerate(strategies):
        strategy_df = df[df["strategy"] == strategy]
        ax = axes[i]

        ax.scatter(
            strategy_df["target_tokens"], strategy_df["actual_tokens"], alpha=0.5, s=20
        )

        # Add diagonal line
        max_val = max(
            strategy_df["target_tokens"].max(), strategy_df["actual_tokens"].max()
        )
        ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5)

        # Add ±20% bands
        ax.fill_between(
            [0, max_val],
            [0, max_val * 0.8],
            [0, max_val * 1.2],
            alpha=0.1,
            color="green",
        )

        ax.set_title(f"{strategy}")
        ax.set_xlabel("Target Tokens")
        ax.set_ylabel("Actual Tokens")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for i in range(n_strategies, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Target vs Actual Tokens by Strategy")
    plt.tight_layout()
    plt.savefig(output_path / "target_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Visualizations saved to {output_path}")


def generate_report(results_path: str = "experiments/results/raw_results.json"):
    """Generate a comprehensive analysis report.

    Args:
        results_path: Path to the raw results JSON file
    """
    # Load results
    data = load_results(results_path)
    results = data["results"]

    # Calculate statistics
    strategy_stats = calculate_strategy_statistics(results)
    # ratio_stats currently not used; can be added to report later
    size_stats = analyze_by_input_size(results)
    nice_fractions = test_nice_fractions_hypothesis(results)
    compensation = identify_compensation_factors(results)

    # Generate markdown report
    report = []
    report.append("# Summarization Length Targeting Experiment Results\n")
    report.append(f"Total experiments: {data['total_experiments']}")
    report.append(f"Successful: {data['successful']}")
    report.append(f"Failed: {data['failed']}\n")

    # Overall strategy performance
    report.append("## Overall Strategy Performance\n")
    report.append("Sorted by mean absolute error (best first):\n")
    report.append("```")
    report.append(strategy_stats.to_string())
    report.append("```\n")

    # Best strategy identification
    if not strategy_stats.empty:
        best_strategy = strategy_stats.iloc[0]
        report.append(f"### 🏆 Best Strategy: {best_strategy['strategy']}")
        report.append(
            f"- Mean absolute error: {best_strategy['mean_abs_error_pct']:.2f}%"
        )
        report.append(
            f"- Within ±10%: {best_strategy['within_10_pct']:.1f}% of attempts"
        )
        report.append(
            f"- Within ±20%: {best_strategy['within_20_pct']:.1f}% of attempts\n"
        )

    # Nice fractions hypothesis
    report.append("## Nice Fractions Hypothesis\n")
    if nice_fractions:
        report.append("### Nice Ratios (10%, 20%, ..., 90%)")
        for key, value in nice_fractions["nice_ratios"].items():
            report.append(f"- {key}: {value:.2f}")
        report.append("\n### Messy Ratios (37%, 42%, 58%, 63%, 73%)")
        for key, value in nice_fractions["messy_ratios"].items():
            report.append(f"- {key}: {value:.2f}")

        # Compare
        nice_perf = nice_fractions["nice_ratios"]["mean_abs_error_pct"]
        messy_perf = nice_fractions["messy_ratios"]["mean_abs_error_pct"]
        if nice_perf < messy_perf:
            report.append(
                f"\n✅ **Nice fractions perform {messy_perf - nice_perf:.1f}% better**"
            )
        else:
            report.append(
                f"\n❌ **Messy fractions perform {nice_perf - messy_perf:.1f}% better**"
            )

    # Compensation factors
    report.append("\n## Systematic Bias & Compensation\n")
    report.append("```")
    report.append(compensation.to_string())
    report.append("```\n")

    # Performance by input size
    report.append("## Performance by Input Size\n")
    if not size_stats.empty:
        for size in size_stats["size_category"].unique():
            size_df = size_stats[size_stats["size_category"] == size]
            best = size_df.nsmallest(1, "mean_abs_error_pct").iloc[0]
            report.append(f"### {size}")
            report.append(
                f"Best strategy: {best['strategy']} (mean error: {best['mean_abs_error_pct']:.2f}%)\n"
            )

    # Save report to same directory as input file
    input_dir = Path(results_path).parent
    output_path = input_dir / "analysis_report.md"
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    print(f"✅ Report saved to {output_path}")

    # Create visualizations in same directory
    create_visualizations(results, output_dir=input_dir)


def main():
    """Main entry point for analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze summarization experiment results"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="experiments/results/raw_results.json",
        help="Path to raw results JSON",
    )

    args = parser.parse_args()

    print("📊 Analyzing experiment results...")
    generate_report(args.input)
    print("✨ Analysis complete!")


if __name__ == "__main__":
    main()
