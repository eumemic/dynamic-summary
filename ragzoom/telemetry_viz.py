"""Telemetry visualization classes and functions.

NOTE: This module provides visualization for telemetry data, focusing on token usage,
costs, batch efficiency, and retry patterns. For programmatic analysis, use the
simplified metrics in telemetry_cli.py.
"""

import json
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry_analysis import (
    analyze_retry_patterns,
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
        gs = GridSpec(6, 2, figure=fig, hspace=0.3, wspace=0.3, top=0.96)

        # 1. Token usage by Tree Level
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_token_usage_by_tree_level(telemetry, config, ax1)

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
            f"Telemetry Analysis - {chunk_size} Token Chunks", fontsize=16, y=0.98
        )

        # Save figure
        output_path = self.output_dir / f"telemetry_{chunk_size}_tokens.{output_format}"
        # Suppress layout and font warnings for cleaner output
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout",
            )
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Saved visualization to {output_path}")

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

    def visualize_side_by_side(
        self, file1: Path, file2: Path, output_format: str = "png"
    ) -> None:
        """Create side-by-side visualizations of two telemetry files."""
        print(f"Creating side-by-side comparison: {file1.name} vs {file2.name}")

        # Load both datasets
        data1 = self.load_benchmark_data(file1)
        data2 = self.load_benchmark_data(file2)

        if "telemetry" not in data1:
            print(f"Warning: No telemetry data found in {file1}")
            return
        if "telemetry" not in data2:
            print(f"Warning: No telemetry data found in {file2}")
            return

        telemetry1 = data1["telemetry"]
        telemetry2 = data2["telemetry"]
        config1 = self._create_config_from_metrics(data1.get("metrics", {}))
        config2 = self._create_config_from_metrics(data2.get("metrics", {}))

        # Create figure with side-by-side subplots (7 rows × 2 columns)
        fig = plt.figure(figsize=(20, 28))  # Wider and taller for side-by-side
        gs = GridSpec(7, 2, figure=fig, hspace=0.3, wspace=0.3, top=0.96)

        # Add super title
        fig.suptitle(
            "Side-by-Side Comparison: Baseline vs Current",
            fontsize=16,
            y=0.98,
        )

        # 1. Token usage by Tree Level
        ax1_left = fig.add_subplot(gs[0, 0])
        self._plot_token_usage_by_tree_level(telemetry1, config1, ax1_left)
        ax1_left.set_title("Token Usage by Tree Level", fontsize=12)

        ax1_right = fig.add_subplot(gs[0, 1])
        self._plot_token_usage_by_tree_level(telemetry2, config2, ax1_right)
        ax1_right.set_title("Token Usage by Tree Level", fontsize=12)

        # Share y-axis scale for better comparison
        max_y = max(ax1_left.get_ylim()[1], ax1_right.get_ylim()[1])
        ax1_left.set_ylim(0, max_y)
        ax1_right.set_ylim(0, max_y)

        # 2. Cost Breakdown
        ax2_left = fig.add_subplot(gs[1, 0])
        self._plot_cost_breakdown(telemetry1, config1, ax2_left)
        ax2_left.set_title("Cost Breakdown", fontsize=12)

        ax2_right = fig.add_subplot(gs[1, 1])
        self._plot_cost_breakdown(telemetry2, config2, ax2_right)
        ax2_right.set_title("Cost Breakdown", fontsize=12)

        # 3. Batch Efficiency
        ax3_left = fig.add_subplot(gs[2, 0])
        self._plot_batch_efficiency(telemetry1, ax3_left)
        ax3_left.set_title("Batch Efficiency", fontsize=12)

        ax3_right = fig.add_subplot(gs[2, 1])
        self._plot_batch_efficiency(telemetry2, ax3_right)
        ax3_right.set_title("Batch Efficiency", fontsize=12)

        # 4. Retry Patterns
        ax4_left = fig.add_subplot(gs[3, 0])
        self._plot_retry_patterns(telemetry1, ax4_left)
        ax4_left.set_title("Retry Patterns", fontsize=12)

        ax4_right = fig.add_subplot(gs[3, 1])
        self._plot_retry_patterns(telemetry2, ax4_right)
        ax4_right.set_title("Retry Patterns", fontsize=12)

        # 5. Summary Accuracy
        ax5_left = fig.add_subplot(gs[4, 0])
        self._plot_summary_accuracy(telemetry1, ax5_left)
        ax5_left.set_title("Summary Accuracy", fontsize=12)

        ax5_right = fig.add_subplot(gs[4, 1])
        self._plot_summary_accuracy(telemetry2, ax5_right)
        ax5_right.set_title("Summary Accuracy", fontsize=12)

        # Share x-axis scale for accuracy plots
        min_x = min(ax5_left.get_xlim()[0], ax5_right.get_xlim()[0])
        max_x = max(ax5_left.get_xlim()[1], ax5_right.get_xlim()[1])
        ax5_left.set_xlim(min_x, max_x)
        ax5_right.set_xlim(min_x, max_x)

        # 6. Node Timeline
        ax6_left = fig.add_subplot(gs[5, 0])
        self._plot_node_timeline(telemetry1, ax6_left)
        ax6_left.set_title("Node Creation Timeline", fontsize=12)

        ax6_right = fig.add_subplot(gs[5, 1])
        self._plot_node_timeline(telemetry2, ax6_right)
        ax6_right.set_title("Node Creation Timeline", fontsize=12)

        # 7. Token Distributions
        ax7_left = fig.add_subplot(gs[6, 0])
        self._plot_token_distributions(telemetry1, ax7_left)
        ax7_left.set_title("Token Distributions", fontsize=12)

        ax7_right = fig.add_subplot(gs[6, 1])
        self._plot_token_distributions(telemetry2, ax7_right)
        ax7_right.set_title("Token Distributions", fontsize=12)

        # Save figure
        output_path = (
            self.output_dir / f"comparison_{file1.stem}_vs_{file2.stem}.{output_format}"
        )

        # Suppress warnings about tight_layout
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout",
            )
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight", dpi=SAVE_DPI)
        plt.close()

        print(f"Saved side-by-side comparison to {output_path}")

    def _plot_token_usage_by_tree_level(
        self, telemetry: dict, config: RagZoomConfig, ax: plt.Axes
    ) -> None:
        """Plot token usage by tree level with stacked bars."""
        # Group tokens by height
        tokens_by_height: dict[int, dict[str, list[float]]] = {}

        # Parse telemetry to extract tokens per level
        parsed_data = (
            telemetry
            if "format_version" in telemetry
            else telemetry.get("telemetry", {})
        )

        for doc_type, doc_data in parsed_data.get("documents", {}).items():
            for node in doc_data.get("nodes", []):
                height = node.get("height", node.get("level", 0))
                if height == 0:
                    continue  # Skip leaf nodes

                # Get token counts for this node
                for attempt in node.get("summary_attempts", []):
                    if attempt.get("status") == "accepted":
                        prompt_tokens = attempt.get("prompt_tokens", 0)
                        completion_tokens = attempt.get("completion_tokens", 0)

                        if height not in tokens_by_height:
                            tokens_by_height[height] = {
                                "prompt_tokens": [],
                                "completion_tokens": [],
                            }

                        tokens_by_height[height]["prompt_tokens"].append(prompt_tokens)
                        tokens_by_height[height]["completion_tokens"].append(
                            completion_tokens
                        )
                        break

        if not tokens_by_height:
            ax.text(
                0.5,
                0.5,
                "No token data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Token Usage by Tree Level")
            return

        levels = sorted(tokens_by_height.keys())
        avg_prompt_tokens = []
        avg_completion_tokens = []

        for level in levels:
            level_data = tokens_by_height[level]
            avg_prompt_tokens.append(np.mean(level_data["prompt_tokens"]))
            avg_completion_tokens.append(np.mean(level_data["completion_tokens"]))

        x = np.arange(len(levels))
        width = 0.6

        # Create stacked bars
        ax.bar(
            x,
            avg_prompt_tokens,
            width,
            label="Input Tokens",
            alpha=0.8,
            color="#66b3ff",
        )
        ax.bar(
            x,
            avg_completion_tokens,
            width,
            bottom=avg_prompt_tokens,
            label="Output Tokens",
            alpha=0.8,
            color="#99ff99",
        )

        ax.set_xlabel("Tree Level")
        ax.set_ylabel("Average Tokens per Node")
        ax.set_title("Token Usage by Tree Level")
        ax.set_xticks(x)
        ax.set_xticklabels([str(level) for level in levels])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

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
        hist_bins, align = self._calculate_histogram_bins(
            [float(size) for size in batch_sizes]
        )

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
        efficiency_pct = batch_eff["batch_utilization"]
        total_batches = batch_eff["total_batches"]
        total_embeddings = batch_eff["total_embeddings"]

        ax.set_xlabel("Embedding Batch Size")
        ax.set_ylabel("Number of Batches")
        ax.set_title(
            f"Embedding Batch Efficiency\n"
            f"Efficiency: {efficiency_pct:.1f}% "
            f"({total_embeddings} embeddings in {total_batches} batches)"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add text explanation of efficiency metric
        ax.text(
            0.02,
            0.98,
            "Efficiency: % of embeddings that were batched\n"
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
                "No Retries Needed\n\nAll summary attempts succeeded on first try.\n"
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
            "Summary Length Accuracy\n(Distribution of deviations from target chunk size)"
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
            inner=None,  # Remove noisy quartile lines
            palette="Set2",
            legend=False,  # Don't show legend since it's redundant with x-axis
        )

        # Add single target line at chunk_size from telemetry
        # Get chunk_size from telemetry metadata
        chunk_size = None
        parsed_data = telemetry if isinstance(telemetry, dict) else {}
        for doc_name, doc_data in parsed_data.get("documents", {}).items():
            chunk_size = doc_data.get("metadata", {}).get("chunk_size", 0)
            if chunk_size > 0:
                break  # Use the first valid chunk_size found

        if chunk_size and chunk_size > 0:
            ax.axhline(
                y=chunk_size,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label="Target",
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
