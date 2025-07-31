"""Telemetry visualization classes and functions."""

import json
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry import (
    analyze_retry_patterns,
    compute_amplification_metrics,
    compute_batch_efficiency,
    compute_metrics_from_telemetry,
    get_telemetry_thresholds,
)
from ragzoom.telemetry_config import (
    DEFAULT_FONT_SIZE,
    DISPLAY_DPI,
    EMBEDDING_COST_PER_1K,
    FIGURE_HEIGHT,
    FIGURE_WIDTH,
    SAVE_DPI,
    SUMMARY_INPUT_COST_PER_1K,
    SUMMARY_OUTPUT_COST_PER_1K,
)

# Set style for professional-looking plots
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    # Fallback to a default style if seaborn-darkgrid style is deprecated/unavailable
    plt.style.use("ggplot")
sns.set_palette("husl")
matplotlib.rcParams["figure.dpi"] = DISPLAY_DPI
matplotlib.rcParams["savefig.dpi"] = SAVE_DPI
matplotlib.rcParams["font.size"] = DEFAULT_FONT_SIZE


class TelemetryVisualizer:
    """Generate visualizations from telemetry data."""

    # Histogram binning constants
    SMALL_BIN_THRESHOLD = 20
    MEDIUM_BIN_THRESHOLD = 100
    SMALL_BIN_WIDTH = 5
    MEDIUM_BIN_WIDTH = 10
    LARGE_BIN_COUNT = 20

    def __init__(self, output_dir: Path) -> None:
        """Initialize visualizer with output directory."""
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.thresholds = get_telemetry_thresholds()

    def load_benchmark_data(self, file_path: Path) -> dict[str, Any]:
        """Load benchmark data from JSON file."""
        with open(file_path) as f:
            data: dict[str, Any] = json.load(f)
            return data

    def visualize_single_benchmark(
        self, benchmark_path: Path, output_format: str = "png"
    ) -> None:
        """Create visualizations for a single benchmark file."""
        print(f"Analyzing {benchmark_path.name}...")

        # Load data
        data = self.load_benchmark_data(benchmark_path)

        if "telemetry" not in data:
            print(f"Warning: No telemetry data found in {benchmark_path}")
            return

        telemetry = data["telemetry"]
        config = self._create_config_from_metrics(data.get("metrics", {}))

        # Create figure with subplots
        fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        gs = GridSpec(6, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Amplification by Level
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_amplification_by_level(telemetry, config, ax1)

        # 2. Cost Breakdown
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_cost_breakdown(telemetry, config, ax2)

        # 3. Batch Efficiency
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_batch_efficiency(telemetry, ax3)

        # 4. Retry Patterns
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_retry_patterns(telemetry, ax4)

        # 5. Summary Accuracy Distribution
        ax5 = fig.add_subplot(gs[3, :])
        self._plot_summary_accuracy(telemetry, ax5)

        # 6. Node Creation Timeline
        ax6 = fig.add_subplot(gs[4, :])
        self._plot_node_timeline(telemetry, ax6)

        # 7. Token Count Distributions
        ax7 = fig.add_subplot(gs[5, :])
        self._plot_token_distributions(telemetry, ax7)

        # Add title and metadata
        chunk_size = data["config"]["leaf_tokens"]
        fig.suptitle(
            f"Telemetry Analysis - {chunk_size} Token Chunks", fontsize=16, y=0.995
        )

        # Save figure
        output_path = self.output_dir / f"telemetry_{chunk_size}_tokens.{output_format}"
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Saved visualization to {output_path}")

        # Also generate markdown report
        self._generate_markdown_report(data, telemetry, config, chunk_size)

    def _create_config_from_metrics(self, metrics: dict[str, Any]) -> RagZoomConfig:
        """Create a config object from metrics data for cost calculations."""
        return RagZoomConfig(
            openai_api_key="dummy",  # Not needed for analysis
            embedding_cost_per_1k=EMBEDDING_COST_PER_1K,
            summary_input_cost_per_1k=SUMMARY_INPUT_COST_PER_1K,
            summary_output_cost_per_1k=SUMMARY_OUTPUT_COST_PER_1K,
        )

    def _calculate_histogram_bins(
        self, batch_sizes: list[float]
    ) -> tuple[list[int] | int, Literal["left", "mid", "right"]]:
        """Calculate appropriate histogram bins based on data distribution.

        Args:
            batch_sizes: List of batch sizes to analyze

        Returns:
            Tuple of (bins, align) where:
            - bins: Either a list of bin edges or an integer number of bins
            - align: 'left' for discrete bins, 'mid' for continuous bins
        """
        unique_sizes = sorted(set(batch_sizes))
        max_size = int(max(batch_sizes))

        if len(unique_sizes) <= 10 and max_size <= self.SMALL_BIN_THRESHOLD:
            # For small discrete values, use exact bins for each value
            bins: list[int] | int = list(range(0, max_size + 2))  # 0, 1, 2, ..., max+1
            align: Literal["left", "mid", "right"] = "left"
        elif max_size <= self.MEDIUM_BIN_THRESHOLD:
            # For medium ranges, use fixed-width bins
            bin_width = (
                self.SMALL_BIN_WIDTH if max_size <= 50 else self.MEDIUM_BIN_WIDTH
            )
            bins = list(range(0, max_size + bin_width, bin_width))
            align = "left"
        else:
            # For large ranges, use automatic binning
            bins = self.LARGE_BIN_COUNT
            align = "mid"

        return bins, align

    def _plot_amplification_by_level(
        self, telemetry: dict, config: RagZoomConfig, ax: plt.Axes
    ) -> None:
        """Plot amplification metrics by tree level."""
        amplification = compute_amplification_metrics(telemetry, config)

        if not amplification["by_height"]:
            ax.text(
                0.5,
                0.5,
                "No amplification data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Amplification by Tree Level")
            return

        levels = sorted(amplification["by_height"].keys())
        cost_medians = []
        input_medians = []
        output_medians = []

        for level in levels:
            level_data = amplification["by_height"][level]
            cost_medians.append(
                np.median(level_data["cost"]) if level_data["cost"] else 0
            )
            input_medians.append(
                np.median(level_data["input"]) if level_data["input"] else 0
            )
            output_medians.append(
                np.median(level_data["output"]) if level_data["output"] else 0
            )

        x = np.arange(len(levels))
        width = 0.25

        ax.bar(x - width, cost_medians, width, label="Cost", alpha=0.8)
        ax.bar(x, input_medians, width, label="Input", alpha=0.8)
        ax.bar(x + width, output_medians, width, label="Output", alpha=0.8)

        ax.set_xlabel("Tree Level")
        ax.set_ylabel("Amplification Factor")
        ax.set_title(
            "Token & Cost Amplification by Tree Level\n(Lower is better - shows summarization efficiency)"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add threshold line with better labeling
        ax.axhline(
            y=self.thresholds.high_cost_amplification,
            color="r",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
        )

        # Add threshold annotation
        ax.text(
            0.98,
            self.thresholds.high_cost_amplification + 0.1,
            f"Alert threshold: {self.thresholds.high_cost_amplification:.1f}x",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.2),
            fontsize=8,
        )

    def _plot_cost_breakdown(
        self, telemetry: dict, config: RagZoomConfig, ax: plt.Axes
    ) -> None:
        """Plot cost breakdown pie chart."""
        metrics = compute_metrics_from_telemetry(telemetry, config)

        # Calculate costs from metrics
        embedding_cost = (
            metrics.total_embedding_tokens / 1000
        ) * metrics.embedding_cost_per_1k
        summary_input_cost = (
            metrics.total_summary_prompt_tokens / 1000
        ) * metrics.summary_input_cost_per_1k
        summary_output_cost = (
            metrics.total_summary_completion_tokens / 1000
        ) * metrics.summary_output_cost_per_1k

        total_cost = embedding_cost + summary_input_cost + summary_output_cost

        if total_cost == 0:
            ax.text(
                0.5,
                0.5,
                "No cost data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Cost Breakdown")
            return

        costs = [embedding_cost, summary_input_cost, summary_output_cost]
        labels = ["Embeddings", "Summary Input", "Summary Output"]
        colors = ["#ff9999", "#66b3ff", "#99ff99"]

        ax.pie(costs, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Cost Breakdown (Total: ${total_cost:.4f})")

    def _plot_batch_efficiency(self, telemetry: dict, ax: plt.Axes) -> None:
        """Plot embedding batch efficiency with clear explanations."""
        batch_eff = compute_batch_efficiency(telemetry)

        if not batch_eff["batch_sizes"]:
            ax.text(
                0.5,
                0.5,
                "No embedding batch data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Embedding Batch Efficiency")
            return

        batch_sizes = batch_eff["batch_sizes"]
        avg_batch_size = batch_eff["avg_batch_size"]

        # Calculate appropriate histogram bins
        hist_bins, align = self._calculate_histogram_bins(batch_sizes)

        # Create histogram with intelligent binning
        _, _, patches = ax.hist(
            batch_sizes,
            bins=hist_bins,
            alpha=0.7,
            edgecolor="black",
            color="skyblue",
            align=align,
        )

        # Add average line
        ax.axvline(
            avg_batch_size,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Average: {avg_batch_size:.1f}",
        )

        # Add optimal batch size reference (theoretical maximum from the data)
        max_batch_size = max(batch_sizes) if batch_sizes else 1
        ax.axvline(
            max_batch_size,
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"Peak: {max_batch_size}",
        )

        # Calculate and display efficiency metrics
        utilization_pct = batch_eff["batch_utilization"]
        total_batches = batch_eff["total_batches"]
        total_embeddings = batch_eff["total_embeddings"]

        ax.set_xlabel("Embedding Batch Size")
        ax.set_ylabel("Number of Batches")
        ax.set_title(
            f"Embedding Batch Efficiency\n"
            f"Utilization: {utilization_pct:.1f}% "
            f"({total_embeddings} embeddings in {total_batches} batches)"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add text explanation of utilization metric
        ax.text(
            0.02,
            0.98,
            "Utilization: Average batch size vs 95th percentile\n"
            "Higher values = better API efficiency",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            fontsize=8,
        )

    def _plot_retry_patterns(self, telemetry: dict, ax: plt.Axes) -> None:
        """Plot retry pattern analysis, or show success message if no retries."""
        retry_data = analyze_retry_patterns(telemetry)

        retry_rate = retry_data["retry_rate"]
        total_attempts = retry_data["total_attempts"]

        # If no retries occurred, show a success message instead of empty chart
        if retry_rate == 0.0 or retry_data["retry_attempts"] == 0:
            ax.text(
                0.5,
                0.5,
                "✅ No Retries Needed\n\nAll summary attempts succeeded on first try.\n"
                f"Total successful attempts: {retry_data['successful_attempts']}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
            )
            ax.set_title("Summary Retry Analysis - Excellent Performance!")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return

        # Show detailed retry analysis when retries occurred
        categories = ["Total Attempts", "Successful", "Retries"]
        values = [
            total_attempts,
            retry_data["successful_attempts"],
            retry_data["retry_attempts"],
        ]

        bars = ax.bar(categories, values, alpha=0.8)

        # Color code bars
        bars[0].set_color("#90cdf4")  # Total - blue
        bars[1].set_color("#86efac")  # Successful - green
        bars[2].set_color("#fca5a5")  # Retries - red

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show labels for non-zero values
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                )

        ax.set_ylabel("Count")
        ax.set_title(
            f"Summary Retry Analysis\n"
            f"Retry Rate: {retry_rate:.1f}% "
            f'(Success Rate: {retry_data["retry_success_rate"]:.1f}%)'
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Show rejection reasons if available
        if retry_data["rejection_reasons"]:
            reasons_text = "Rejection reasons:\n" + "\n".join(
                [
                    f"• {reason}: {count}"
                    for reason, count in sorted(
                        retry_data["rejection_reasons"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[
                        :3
                    ]  # Top 3
                ]
            )
            ax.text(
                0.98,
                0.02,
                reasons_text,
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
                fontsize=8,
            )

    def _extract_summary_deviations_from_telemetry(
        self, telemetry: dict
    ) -> list[float]:
        """Extract summary accuracy deviations from telemetry data.

        Returns:
            List of deviation percentages from chunk_size target
        """
        deviations = []
        parsed_data = telemetry if isinstance(telemetry, dict) else {}

        # Process all documents
        for doc_name, doc_data in parsed_data.get("documents", {}).items():
            # Get the chunk size for this document
            chunk_size = doc_data.get("metadata", {}).get("chunk_size", 0)
            if chunk_size <= 0:
                continue

            # Process all nodes
            nodes = doc_data.get("nodes", [])
            for node in nodes:
                # Only process summary nodes (height > 0)
                height = node.get("height", node.get("level", 0))
                if height > 0:
                    # Look for accepted summary attempts
                    summary_attempts = node.get("summary_attempts", [])
                    for attempt in summary_attempts:
                        if attempt.get("status") == "accepted":
                            actual_tokens = attempt.get("actual_tokens", 0)
                            if actual_tokens > 0:
                                # Calculate deviation percentage
                                deviation = (
                                    (actual_tokens - chunk_size) / chunk_size * 100
                                )
                                deviations.append(deviation)
                                break  # Only use the accepted attempt

        return deviations

    def _plot_summary_accuracy(self, telemetry: dict, ax: plt.Axes) -> None:
        """Plot summary accuracy distribution."""
        # Extract deviations from telemetry
        deviations = self._extract_summary_deviations_from_telemetry(telemetry)

        if not deviations:
            ax.text(
                0.5,
                0.5,
                "No deviation data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Summary Accuracy Distribution")
            return

        # Create histogram
        ax.hist(deviations, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="green", linestyle="--", label="Target", linewidth=2)

        # Add median line
        median_dev = float(np.median(deviations))
        ax.axvline(
            median_dev,
            color="red",
            linestyle="--",
            label=f"Median: {median_dev:.1f}%",
            linewidth=2,
        )

        ax.set_xlabel("Deviation from Target Token Count (%)")
        ax.set_ylabel("Frequency")
        ax.set_title(
            "Summary Length Accuracy\n(How well summaries hit target token counts)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_node_timeline(self, telemetry: dict, ax: plt.Axes) -> None:
        """Plot node creation timeline."""
        # Extract node creation times
        creation_times = []
        for doc_data in telemetry.get("documents", {}).values():
            for node in doc_data.get("nodes", []):
                if "created_at" in node:
                    creation_times.append(node["created_at"])

        if not creation_times:
            ax.text(
                0.5,
                0.5,
                "No timeline data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Node Creation Timeline")
            return

        creation_times.sort()
        min_time = creation_times[0]
        relative_times = [(t - min_time) for t in creation_times]

        ax.plot(
            relative_times,
            range(len(relative_times)),
            alpha=0.8,
            linewidth=2,
            color="purple",
        )
        ax.set_xlabel("Time Since Start (seconds)")
        ax.set_ylabel("Cumulative Nodes Created")
        ax.set_title(
            "Document Processing Timeline\n(Shows indexing progress over time)"
        )
        ax.grid(True, alpha=0.3)

        # Add total processing time annotation
        total_time = max(relative_times) if relative_times else 0
        total_nodes = len(relative_times)
        ax.text(
            0.02,
            0.98,
            (
                f"Total: {total_nodes} nodes in {total_time:.1f}s\n"
                f"Rate: {total_nodes/total_time:.1f} nodes/sec"
                if total_time > 0
                else f"Total: {total_nodes} nodes"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            fontsize=8,
        )

    def _plot_token_distributions(self, telemetry: dict, ax: plt.Axes) -> None:
        """Plot token count distributions by tree level using violin plots."""
        # TODO: Consider refactoring this method to extract data preparation logic
        # into a separate method for better readability and testability
        import pandas as pd

        # Extract token data by level from telemetry
        token_data = []
        parsed_data = telemetry if isinstance(telemetry, dict) else {}

        for doc_type, doc_data in parsed_data.get("documents", {}).items():
            nodes = doc_data.get("nodes", [])

            for node in nodes:
                # Get height (compatible with both v1.0 and v2.0)
                height = node.get("height", node.get("level", 0))

                # Extract token counts from summary attempts for summary nodes
                if height > 0:  # Summary nodes
                    summary_attempts = node.get("summary_attempts", [])
                    for attempt in summary_attempts:
                        if attempt.get("status") == "accepted":
                            actual_tokens = attempt.get("actual_tokens", 0)
                            target_tokens = attempt.get("target_tokens", 0)
                            if actual_tokens > 0:
                                token_data.append(
                                    {
                                        "level": f"Level {height}",
                                        "actual_tokens": actual_tokens,
                                        "target_tokens": target_tokens,
                                        "node_type": "Summary",
                                    }
                                )
                else:  # Leaf nodes
                    # For leaf nodes, we can estimate from embedding data or use default
                    embedding = node.get("embedding", {})
                    text_tokens = embedding.get("text_tokens", 0)
                    if text_tokens > 0:
                        token_data.append(
                            {
                                "level": "Level 0 (Leaves)",
                                "actual_tokens": text_tokens,
                                "target_tokens": text_tokens,
                                "node_type": "Leaf",
                            }
                        )

        if not token_data:
            ax.text(
                0.5,
                0.5,
                "No token distribution data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Token Count Distributions by Tree Level")
            return

        # Create DataFrame for plotting
        df = pd.DataFrame(token_data)

        # Create violin plot
        sns.violinplot(
            data=df,
            x="level",
            y="actual_tokens",
            hue="level",  # Assign x to hue to fix deprecation warning
            ax=ax,
            inner="quartile",  # Show quartiles
            palette="Set2",
            legend=False,  # Don't show legend since it's redundant with x-axis
        )

        # Add target token reference lines if available
        if "target_tokens" in df.columns:
            level_names = df["level"].unique()
            for i, level_name in enumerate(level_names):
                level_data = df[df["level"] == level_name]
                if len(level_data) > 0:
                    target = level_data["target_tokens"].iloc[0]
                    if target > 0:
                        ax.axhline(
                            y=target,
                            color="red",
                            linestyle="--",
                            alpha=0.5,
                            label="Target" if i == 0 else "",
                        )

        ax.set_xlabel("Tree Level")
        ax.set_ylabel("Token Count")
        ax.set_title("Token Count Distributions by Tree Level")
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Add legend if target lines were added
        if ax.lines:
            ax.legend(loc="upper right")

    def _generate_markdown_report(
        self, data: dict, telemetry: dict, config: RagZoomConfig, chunk_size: int
    ) -> None:
        """Generate a markdown report alongside visualizations."""
        report_path = self.output_dir / f"telemetry_report_{chunk_size}_tokens.md"

        with open(report_path, "w") as f:
            f.write(f"# Telemetry Report - {chunk_size} Token Chunks\n\n")

            # Add metrics summary
            amplification = compute_amplification_metrics(telemetry, config)
            batch_eff = compute_batch_efficiency(telemetry)
            retry_patterns = analyze_retry_patterns(telemetry)

            f.write("## Summary Metrics\n\n")
            f.write(
                f"- **Median Cost Amplification**: {amplification['median_cost']:.2f}x\n"
            )
            f.write(f"- **Batch Utilization**: {batch_eff['batch_utilization']:.1f}%\n")
            f.write(f"- **Retry Rate**: {retry_patterns['retry_rate']:.1f}%\n")
            f.write("\n")

            f.write("## Visualizations\n\n")
            f.write(f"![Telemetry Analysis](telemetry_{chunk_size}_tokens.png)\n")

        print(f"Saved markdown report to {report_path}")

    def visualize_comparison(
        self, results_dir: Path, output_format: str = "png"
    ) -> None:
        """Create comparison visualizations between multiple benchmarks."""
        json_files = list(results_dir.glob("telemetry_*_tokens.json"))

        if len(json_files) < 2:
            print("Need at least 2 benchmark files for comparison")
            return

        # Load all benchmarks
        benchmarks = {}
        for file in json_files:
            data = self.load_benchmark_data(file)
            if "telemetry" in data and "config" in data:
                chunk_size = data["config"]["leaf_tokens"]
                benchmarks[chunk_size] = data

        if len(benchmarks) < 2:
            print("Need at least 2 benchmarks with telemetry data for comparison")
            return

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Benchmark Comparison", fontsize=16)

        # Plot comparisons
        self._plot_comparison_metrics(benchmarks, axes)

        plt.tight_layout()
        output_path = self.output_dir / f"comparison.{output_format}"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Saved comparison visualization to {output_path}")

    def _plot_comparison_metrics(
        self, benchmarks: dict[int, dict], axes: np.ndarray
    ) -> None:
        """Plot comparison metrics across benchmarks."""
        chunk_sizes = sorted(benchmarks.keys())

        # Collect metrics for each benchmark
        cost_amps = []
        batch_utils = []
        retry_rates = []
        total_costs = []

        for chunk_size in chunk_sizes:
            data = benchmarks[chunk_size]
            telemetry = data["telemetry"]
            config = self._create_config_from_metrics(data.get("metrics", {}))

            amp = compute_amplification_metrics(telemetry, config)
            batch = compute_batch_efficiency(telemetry)
            retry = analyze_retry_patterns(telemetry)
            metrics = compute_metrics_from_telemetry(telemetry, config)

            cost_amps.append(amp["median_cost"])
            batch_utils.append(batch["batch_utilization"])
            retry_rates.append(retry["retry_rate"])
            # Calculate total cost from metrics
            embedding_cost = (
                metrics.total_embedding_tokens / 1000
            ) * metrics.embedding_cost_per_1k
            summary_cost = (
                metrics.total_summary_prompt_tokens / 1000
            ) * metrics.summary_input_cost_per_1k + (
                metrics.total_summary_completion_tokens / 1000
            ) * metrics.summary_output_cost_per_1k
            total_costs.append(embedding_cost + summary_cost)

        # Plot 1: Cost Amplification
        ax = axes[0, 0]
        ax.plot(chunk_sizes, cost_amps, "o-", markersize=8)
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Median Cost Amplification")
        ax.set_title("Cost Amplification vs Chunk Size")
        ax.grid(True, alpha=0.3)

        # Plot 2: Batch Utilization
        ax = axes[0, 1]
        ax.plot(chunk_sizes, batch_utils, "o-", markersize=8, color="green")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Batch Utilization (%)")
        ax.set_title("Batch Utilization vs Chunk Size")
        ax.grid(True, alpha=0.3)

        # Plot 3: Retry Rate
        ax = axes[1, 0]
        ax.plot(chunk_sizes, retry_rates, "o-", markersize=8, color="red")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Retry Rate (%)")
        ax.set_title("Retry Rate vs Chunk Size")
        ax.grid(True, alpha=0.3)

        # Plot 4: Total Cost
        ax = axes[1, 1]
        ax.plot(chunk_sizes, total_costs, "o-", markersize=8, color="purple")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Total Cost ($)")
        ax.set_title("Total Cost vs Chunk Size")
        ax.grid(True, alpha=0.3)
