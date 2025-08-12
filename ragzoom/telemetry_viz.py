"""Telemetry visualization classes and functions.

NOTE: This module provides visualization for telemetry data, focusing on token usage,
costs, batch efficiency, and retry patterns. For programmatic analysis, use the
simplified metrics in telemetry_cli.py.
"""

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from ragzoom.config import get_embedding_cost, get_llm_costs
from ragzoom.telemetry_analysis import (
    compute_batch_efficiency,
    get_accepted_attempt,
    get_telemetry_thresholds,
)
from ragzoom.telemetry_config import (
    DEFAULT_FONT_SIZE,
    DISPLAY_DPI,
    FIGURE_HEIGHT,
    FIGURE_WIDTH,
    SAVE_DPI,
)
from ragzoom.telemetry_types import NodeTelemetryDict

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

    def __init__(self, output_path: Path) -> None:
        """Initialize visualizer with output file path."""
        self.output_path = output_path
        self.thresholds = get_telemetry_thresholds()

    def _extract_nodes_from_telemetry(self, telemetry: dict) -> list[dict[str, Any]]:
        """Extract nodes from telemetry data, handling both v1.0/v2.0 and v3.0 formats.

        Args:
            telemetry: Telemetry data dictionary

        Returns:
            List of node dictionaries
        """
        if "documents" in telemetry:
            # v1.0/v2.0 format with documents dictionary
            nodes: list[dict[str, Any]] = []
            for doc_data in telemetry.get("documents", {}).values():
                nodes.extend(doc_data.get("nodes", []))
            return nodes
        else:
            # v3.0 flat format
            nodes_data = telemetry.get("nodes", [])
            return nodes_data if isinstance(nodes_data, list) else []

    def _extract_chunk_size_from_telemetry(self, telemetry: dict) -> int:
        """Extract chunk size from telemetry data, handling both formats.

        Args:
            telemetry: Telemetry data dictionary

        Returns:
            Chunk size in tokens, or 0 if not found
        """
        if "documents" in telemetry:
            # v1.0/v2.0 format - use first document with valid chunk_size
            for doc_data in telemetry.get("documents", {}).values():
                chunk_size = doc_data.get("metadata", {}).get("chunk_size", 0)
                if chunk_size > 0:
                    return int(chunk_size)
            return 0
        else:
            # v3.0+ format - read from config
            config = telemetry.get("config", {})
            chunk_size = config.get("target_chunk_tokens", 0)
            return int(chunk_size) if chunk_size else 0

    def _ensure_output_dir(self) -> None:
        """Ensure the output directory exists, creating it if necessary."""
        self.output_path.parent.mkdir(exist_ok=True, parents=True)

    def _suppress_matplotlib_warnings(self) -> AbstractContextManager[None]:
        """Context manager to suppress common matplotlib warnings."""
        import warnings
        from collections.abc import Iterator
        from contextlib import contextmanager

        @contextmanager
        def suppress() -> Iterator[None]:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This figure includes Axes that are not compatible with tight_layout",
                )
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="matplotlib"
                )
                yield

        return suppress()

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

        # Handle both wrapped and direct v3.0 formats
        if "telemetry" in data:
            # Legacy wrapped format: {"telemetry": {...}, "config": {...}}
            telemetry = data["telemetry"]
        elif "format_version" in data:
            # Direct v3.0 format: {"format_version": "3.0", ...}
            telemetry = data
        else:
            print(f"Warning: No telemetry data found in {benchmark_path}")
            return

        # Create figure with subplots (3 rows only)
        fig = plt.figure(
            figsize=(FIGURE_WIDTH * 0.33, FIGURE_HEIGHT * 0.6)
        )  # Reduce width by 2/3 and height
        gs = GridSpec(3, 1, figure=fig, hspace=0.3, top=0.94)

        # 1. Cost Breakdown
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_cost_breakdown(telemetry, ax1)

        # 2. Summary Compression Patterns
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_summary_scatter(telemetry, ax2)

        # 3. Node Creation Timeline
        ax3 = fig.add_subplot(gs[2, :])
        self._plot_node_timeline(telemetry, ax3)

        # Add title and metadata
        if "config" in data:
            # Get chunk size from config
            if "leaf_tokens" in data["config"]:
                # Legacy format
                chunk_size = data["config"]["leaf_tokens"]
            elif "target_chunk_tokens" in data["config"]:
                # Current format
                chunk_size = data["config"]["target_chunk_tokens"]
            else:
                chunk_size = "Unknown"
        elif "chunk_size" in telemetry:
            # v3.0 format has chunk_size directly
            chunk_size = telemetry["chunk_size"]
        else:
            chunk_size = "Unknown"
        fig.suptitle(
            f"Telemetry Analysis - {chunk_size} Token Chunks", fontsize=16, y=0.98
        )

        # Save figure
        self._ensure_output_dir()
        with self._suppress_matplotlib_warnings():
            plt.tight_layout()
            plt.savefig(self.output_path, bbox_inches="tight")
        plt.close()

        print(f"Saved visualization to {self.output_path}")

    def _get_cost_functions(self, telemetry: dict) -> tuple:
        """Get cost calculation functions for models in telemetry."""
        # Get models from config
        config = telemetry.get("config", {})
        embedding_model = config.get("embedding_model")
        summary_model = config.get("summary_model")

        if not embedding_model or not summary_model:
            raise ValueError(
                "Cannot determine models from telemetry. "
                "Expected config.embedding_model and config.summary_model."
            )

        # Get costs
        embedding_cost_per_1k = get_embedding_cost(embedding_model)
        summary_input_cost_per_1k, summary_output_cost_per_1k = get_llm_costs(
            summary_model
        )

        return (
            embedding_cost_per_1k,
            summary_input_cost_per_1k,
            summary_output_cost_per_1k,
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
        self,
        file1: Path,
        file2: Path,
        output_format: str = "png",
        figsize: tuple[int, int] | None = None,
    ) -> None:
        """Create side-by-side visualizations of two telemetry files.

        Args:
            file1: Path to first telemetry file
            file2: Path to second telemetry file
            output_format: Output format (png, pdf, svg)
            figsize: Optional figure size (width, height) in inches. Defaults to (20, 28).
        """
        print(f"Creating side-by-side comparison: {file1.name} vs {file2.name}")

        # Load both datasets
        data1 = self.load_benchmark_data(file1)
        data2 = self.load_benchmark_data(file2)

        # Handle both wrapped and direct v3.0 formats for file1
        if "telemetry" in data1:
            telemetry1 = data1["telemetry"]
        elif "format_version" in data1:
            telemetry1 = data1
        else:
            print(f"Warning: No telemetry data found in {file1}")
            return

        # Handle both wrapped and direct v3.0 formats for file2
        if "telemetry" in data2:
            telemetry2 = data2["telemetry"]
        elif "format_version" in data2:
            telemetry2 = data2
        else:
            print(f"Warning: No telemetry data found in {file2}")
            return

        # Telemetry data already contains model information for cost calculations

        # Create figure with side-by-side subplots (only 3 rows)
        if figsize is None:
            figsize = (
                10,
                14,
            )  # Half the width, slightly taller for double Summary Accuracy
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            3, 2, figure=fig, hspace=0.2, wspace=0.15, top=0.92, height_ratios=[1, 2, 1]
        )

        # Add super title
        fig.suptitle(
            "Side-by-Side Comparison: Baseline vs Current",
            fontsize=16,
            y=0.97,
        )

        # 1. Cost Breakdown
        ax1_left = fig.add_subplot(gs[0, 0])
        self._plot_cost_breakdown(telemetry1, ax1_left)
        ax1_left.set_title("Cost Breakdown", fontsize=12)

        ax1_right = fig.add_subplot(gs[0, 1])
        self._plot_cost_breakdown(telemetry2, ax1_right)
        ax1_right.set_title("Cost Breakdown", fontsize=12)
        ax1_right.set_ylabel("")  # Remove y-axis label

        # Share y-axis scale for cost comparison
        max_y = max(ax1_left.get_ylim()[1], ax1_right.get_ylim()[1])
        ax1_left.set_ylim(0, max_y)
        ax1_right.set_ylim(0, max_y)

        # 2. Summary Compression Patterns
        ax2_left = fig.add_subplot(gs[1, 0])
        self._plot_summary_scatter(telemetry1, ax2_left)
        ax2_left.set_title("Summary Compression Patterns", fontsize=12)

        ax2_right = fig.add_subplot(gs[1, 1])
        self._plot_summary_scatter(telemetry2, ax2_right)
        ax2_right.set_title("Summary Compression Patterns", fontsize=12)
        ax2_right.set_ylabel("")  # Remove y-axis label

        # Share both axes for scatter plots comparison
        max_x = max(ax2_left.get_xlim()[1], ax2_right.get_xlim()[1])
        min_x = min(ax2_left.get_xlim()[0], ax2_right.get_xlim()[0])
        ax2_left.set_xlim(min_x, max_x)
        ax2_right.set_xlim(min_x, max_x)

        max_y = max(ax2_left.get_ylim()[1], ax2_right.get_ylim()[1])
        min_y = min(ax2_left.get_ylim()[0], ax2_right.get_ylim()[0])
        ax2_left.set_ylim(min_y, max_y)
        ax2_right.set_ylim(min_y, max_y)

        # 3. Summary Creation Timeline
        ax3_left = fig.add_subplot(gs[2, 0])
        self._plot_node_timeline(telemetry1, ax3_left)
        ax3_left.set_title("Summary Creation Timeline", fontsize=12)

        ax3_right = fig.add_subplot(gs[2, 1])
        self._plot_node_timeline(telemetry2, ax3_right)
        ax3_right.set_title("Summary Creation Timeline", fontsize=12)
        ax3_right.set_ylabel("")  # Remove y-axis label

        # Share both axes for timeline comparison
        max_x = max(ax3_left.get_xlim()[1], ax3_right.get_xlim()[1])
        ax3_left.set_xlim(0, max_x)
        ax3_right.set_xlim(0, max_x)

        max_y = max(ax3_left.get_ylim()[1], ax3_right.get_ylim()[1])
        ax3_left.set_ylim(0, max_y)
        ax3_right.set_ylim(0, max_y)

        # Save figure
        self._ensure_output_dir()
        with self._suppress_matplotlib_warnings():
            plt.tight_layout()
            plt.savefig(self.output_path, bbox_inches="tight", dpi=SAVE_DPI)
        plt.close()

        print(f"Saved side-by-side comparison to {self.output_path}")

    def _plot_token_usage_by_tree_level(self, telemetry: dict, ax: plt.Axes) -> None:
        """Plot token usage by tree level with stacked bars."""
        # Group tokens by height
        tokens_by_height: dict[int, dict[str, list[float]]] = {}

        # Extract nodes from telemetry data
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # Process nodes
        for node in nodes:
            height = node.get("height", node.get("level", 0))
            if height == 0:
                continue  # Skip leaf nodes

            # Get token counts for this node
            # Cast to NodeTelemetryDict for type safety
            node_typed = cast(NodeTelemetryDict, node)
            accepted_attempt, _ = get_accepted_attempt(node_typed)
            if accepted_attempt:
                prompt_tokens = accepted_attempt.get("prompt_tokens", 0)
                completion_tokens = accepted_attempt.get("completion_tokens", 0)

                if height not in tokens_by_height:
                    tokens_by_height[height] = {
                        "prompt_tokens": [],
                        "completion_tokens": [],
                    }

                tokens_by_height[height]["prompt_tokens"].append(prompt_tokens)
                tokens_by_height[height]["completion_tokens"].append(completion_tokens)

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

    def _plot_cost_breakdown(self, telemetry: dict, ax: plt.Axes) -> None:
        """Plot cost breakdown by attempt number as vertical stacked bar."""
        # Get cost functions for models in telemetry
        embedding_cost_per_1k, summary_input_cost_per_1k, summary_output_cost_per_1k = (
            self._get_cost_functions(telemetry)
        )

        # Calculate costs by attempt number
        costs_by_attempt = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # 5 = 5+

        # Extract nodes from telemetry data
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # Process all attempts from summary nodes
        for node in nodes:
            height = node.get("height", node.get("level", 0))
            if height > 0:  # Summary nodes only
                attempts = node.get("summary_attempts", [])
                for attempt_num, attempt in enumerate(attempts, 1):
                    prompt_tokens = attempt.get("prompt_tokens", 0)
                    completion_tokens = attempt.get("completion_tokens", 0)

                    # Calculate cost for this attempt
                    input_cost = (prompt_tokens / 1000) * summary_input_cost_per_1k
                    output_cost = (
                        completion_tokens / 1000
                    ) * summary_output_cost_per_1k
                    attempt_cost = input_cost + output_cost

                    # Group attempts 5+ together
                    display_num = min(attempt_num, 5)
                    costs_by_attempt[display_num] += attempt_cost

        # Calculate total cost (excluding embeddings)
        total_cost = sum(costs_by_attempt.values())

        if total_cost == 0:
            ax.text(
                0.5,
                0.5,
                "No cost data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Cost Breakdown by Attempt")
            return

        # Create vertical stacked bar
        colors = [
            "#2563eb",
            "#10b981",
            "#f59e0b",
            "#ef4444",
            "#991b1b",
        ]  # Same as retry patterns
        labels = ["Attempt 1", "Attempt 2", "Attempt 3", "Attempt 4", "Attempt 5+"]

        # Single vertical bar centered with black border
        bottom = 0
        for attempt_num in range(1, 6):
            cost = costs_by_attempt[attempt_num]
            if cost > 0:  # Only plot if there's cost
                label = labels[attempt_num - 1]
                color = colors[attempt_num - 1]
                ax.bar(
                    0.5,
                    cost,
                    bottom=bottom,
                    color=color,
                    label=label,  # Simplified label without cost
                    width=0.3,
                )
                bottom += cost

        ax.set_ylim(0, total_cost * 1.2)  # Add 20% padding for legend
        ax.set_xlim(0, 1)
        ax.set_ylabel("Cost ($)")
        ax.set_title(f"Cost Breakdown by Attempt\nTotal: ${total_cost:.4f}")
        ax.set_xticks([])  # Hide x-axis ticks
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

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
        """Plot retry attempt distribution as stacked bar chart (cumulative)."""
        # Count nodes by number of attempts
        attempt_counts: dict[int, int] = {}

        # Extract nodes from telemetry data
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # Count attempts for each summary node
        total_summary_nodes = 0
        for node in nodes:
            height = node.get("height", node.get("level", 0))
            if height > 0:  # Summary nodes only
                total_summary_nodes += 1
                attempts = node.get("summary_attempts", [])
                num_attempts = len(attempts)
                if num_attempts > 0:
                    attempt_counts[num_attempts] = (
                        attempt_counts.get(num_attempts, 0) + 1
                    )

        if total_summary_nodes == 0:
            ax.text(
                0.5,
                0.5,
                "No retry data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Retry Patterns")
            return

        # Calculate cumulative counts (nodes with at least N attempts)
        max_attempts = max(attempt_counts.keys()) if attempt_counts else 1
        cumulative_counts = []
        labels = []

        # Build cumulative data (e.g., [31, 24, 13, 9])
        for threshold in range(1, min(max_attempts + 1, 6)):  # Show up to 5 categories
            # Count nodes with at least 'threshold' attempts
            cumulative_count = sum(
                v for k, v in attempt_counts.items() if k >= threshold
            )

            if threshold <= 4:
                labels.append(f"≥{threshold}")
                cumulative_counts.append(cumulative_count)
            elif threshold == 5:
                labels.append("≥5")
                cumulative_counts.append(cumulative_count)
                break

        # Don't reverse - keep ≥1 at bottom
        # Use same colors as Summary Accuracy (blue to red gradient)
        colors = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#991b1b"][: len(labels)]

        # Create stacked bar - each full cumulative count stacked on top
        bar_width = 0.4
        bottom = 0

        for i, (label, count, color) in enumerate(
            zip(labels, cumulative_counts, colors)
        ):
            ax.bar(
                0.5,
                count,
                bottom=bottom,
                width=bar_width,
                color=color,
                label=f"{label} attempts: {count} ({count/total_summary_nodes*100:.0f}%)",
                edgecolor="black",
                linewidth=1,
            )

            bottom += count

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel("Number of Nodes")
        ax.set_title(
            f"Retry Pattern Distribution\n{total_summary_nodes} summary nodes total"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    def _extract_summary_deviations_from_telemetry(
        self, telemetry: dict
    ) -> list[float]:
        """Extract summary accuracy deviations from telemetry data.

        Returns:
            List of deviation percentages from chunk_size target
        """
        deviations: list[float] = []

        # Extract chunk size and nodes from telemetry data
        chunk_size = self._extract_chunk_size_from_telemetry(telemetry)
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # If no valid chunk_size found, return empty deviations
        if chunk_size <= 0:
            return deviations

        # Process nodes
        for node in nodes:
            # Only process summary nodes (height > 0)
            height = node.get("height", node.get("level", 0))
            if height > 0:
                # Look for accepted summary attempts
                # Cast to NodeTelemetryDict for type safety
                node_typed = cast(NodeTelemetryDict, node)
                accepted_attempt, _ = get_accepted_attempt(node_typed)
                if accepted_attempt:
                    actual_tokens = accepted_attempt.get("actual_tokens", 0)
                    if actual_tokens > 0:
                        # Calculate deviation percentage
                        deviation = (actual_tokens - chunk_size) / chunk_size * 100
                        deviations.append(deviation)

        return deviations

    def _plot_summary_scatter(self, telemetry: dict, ax: plt.Axes) -> None:
        """Plot input vs output token scatter plot, color-coded by retry count."""
        # Extract chunk size (target) and nodes from telemetry data
        chunk_size = self._extract_chunk_size_from_telemetry(telemetry)
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # Prepare data for scatter plot - one dot per attempt
        input_tokens = []
        output_tokens = []
        attempt_numbers = []

        # Process nodes to extract ALL attempts (not just accepted ones)
        for node in nodes:
            height = node.get("height", node.get("level", 0))
            if height > 0:  # Summary nodes only
                # Get input tokens (tokens being summarized)
                input_text_tokens = node.get("input_text_tokens")
                if input_text_tokens is None or input_text_tokens <= 0:
                    # Skip nodes without input token data
                    continue

                # Process ALL summary attempts for this node
                attempts = node.get("summary_attempts", [])
                for attempt_num, attempt in enumerate(attempts, 1):
                    actual_tokens = attempt.get("actual_tokens", 0)
                    if actual_tokens > 0:
                        input_tokens.append(input_text_tokens)
                        output_tokens.append(actual_tokens)
                        attempt_numbers.append(attempt_num)

        if not input_tokens:
            ax.text(
                0.5,
                0.5,
                "No input/output token data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Summary Compression Patterns")
            return

        # Create color map for attempt numbers (same colors as cost breakdown)
        colors = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#991b1b"]  # Blue to red
        attempt_colors = []
        for attempt_num in attempt_numbers:
            if attempt_num >= len(colors):
                attempt_colors.append(colors[-1])  # 5+ attempts = darkest red
            else:
                attempt_colors.append(colors[attempt_num - 1])  # Convert to 0-indexed

        # Create scatter plot
        ax.scatter(
            input_tokens,
            output_tokens,
            c=attempt_colors,
            alpha=0.6,
            s=50,
            edgecolors="none",
        )

        # Set axis limits with margin around data (calculate early for use in other elements)
        x_margin = (max(input_tokens) - min(input_tokens)) * 0.05
        y_margin = (max(output_tokens) - min(output_tokens)) * 0.05

        x_min, x_max = min(input_tokens) - x_margin, max(input_tokens) + x_margin
        y_min, y_max = min(output_tokens) - y_margin, max(output_tokens) + y_margin

        # Extract retry threshold from telemetry for dynamic acceptable range
        retry_threshold = None
        if "config" in telemetry:
            retry_threshold = telemetry["config"].get("retry_threshold")

        # Add target line (horizontal at chunk_size)
        if chunk_size > 0:
            ax.axhline(
                chunk_size,
                color="green",
                linestyle="--",
                label=f"Target ({chunk_size} tokens)",
                linewidth=2,
            )

        # Apply axis limits first
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Add acceptable range band based on retry_threshold (full width of plot)
        # Use very large x-coordinates to ensure it extends beyond any possible axis limits
        if chunk_size > 0:
            if retry_threshold is not None:
                threshold_tokens = chunk_size * retry_threshold
                ax.fill_between(
                    [-10000, 10000],  # Extend far beyond any possible data range
                    chunk_size - threshold_tokens,
                    chunk_size + threshold_tokens,
                    alpha=0.1,
                    color="green",
                    label=f"±{retry_threshold*100:.0f}% retry threshold",
                )
            else:
                # Fallback to ±10 tokens if no retry_threshold found
                ax.fill_between(
                    [-10000, 10000],  # Extend far beyond any possible data range
                    chunk_size - 10,
                    chunk_size + 10,
                    alpha=0.1,
                    color="green",
                    label="±10 token range",
                )

        # Add diagonal reference line showing 1:1 ratio (full plot extent)
        ax.plot(
            [x_min, x_max],
            [x_min, x_max],
            "k:",
            alpha=0.3,
            linewidth=1,
            label="1:1 ratio",
        )

        # Create custom legend for attempt numbers (matching cost breakdown)
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=colors[0], label="Attempt 1", alpha=0.6),
            Patch(facecolor=colors[1], label="Attempt 2", alpha=0.6),
            Patch(facecolor=colors[2], label="Attempt 3", alpha=0.6),
            Patch(facecolor=colors[3], label="Attempt 4", alpha=0.6),
            Patch(facecolor=colors[4], label="Attempt 5+", alpha=0.6),
        ]

        # Only include legend items for attempt numbers that exist in the data
        max_attempts = max(attempt_numbers) if attempt_numbers else 0
        legend_elements = legend_elements[: min(max_attempts, 5)]

        # Add the legend elements to the main legend
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

        # Labels and title
        ax.set_xlabel("Input Tokens (text to summarize)")
        ax.set_ylabel("Output Tokens (summary)")
        ax.set_title("Summary Compression Patterns")
        ax.grid(True, alpha=0.3)

        # Add statistics annotation
        # Calculate deviations from target (chunk_size) as percentages
        if chunk_size > 0:
            deviations_pct = [
                (output - chunk_size) / chunk_size * 100 for output in output_tokens
            ]
            avg_deviation_pct = np.mean(deviations_pct)
            median_deviation_pct = np.median(deviations_pct)

            # Calculate actual token positions for the lines
            avg_position = chunk_size * (1 + avg_deviation_pct / 100)
            median_position = chunk_size * (1 + median_deviation_pct / 100)

            # Draw horizontal lines for average and median deviations
            ax.axhline(
                avg_position,
                color="blue",
                linestyle=":",
                alpha=0.5,
                linewidth=1.5,
                label=f"Avg: {avg_deviation_pct:+.1f}%",
            )
            ax.axhline(
                median_position,
                color="red",
                linestyle="-.",
                alpha=0.5,
                linewidth=1.5,
                label=f"Median: {median_deviation_pct:+.1f}%",
            )
        else:
            avg_deviation_pct = 0.0
            median_deviation_pct = 0.0

        avg_attempts = np.mean(attempt_numbers)

        # Count unique nodes for summary count
        unique_inputs = len(set(input_tokens))

        stats_text = (
            f"Avg deviation: {avg_deviation_pct:+.1f}%\n"
            f"Median deviation: {median_deviation_pct:+.1f}%\n"
            f"Avg attempts: {avg_attempts:.2f}\n"
            f"Total attempts: {len(input_tokens)} ({unique_inputs} nodes)"
        )
        ax.text(
            0.98,
            0.02,
            stats_text,
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            fontsize=8,
        )

    def _plot_summary_accuracy(self, telemetry: dict, ax: plt.Axes) -> None:
        """Plot summary accuracy distribution for all attempts, color-coded by attempt number."""
        # Constants for attempt grouping
        max_attempt_groups = 5  # Maximum number of attempt groups to track

        # Extract deviations from all attempts
        deviations_by_attempt: dict[int, list[float]] = {
            i: [] for i in range(1, max_attempt_groups + 1)
        }

        # Extract chunk size and nodes from telemetry data
        chunk_size = self._extract_chunk_size_from_telemetry(telemetry)
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # If no valid chunk_size found, return empty deviations
        if chunk_size <= 0:
            ax.text(
                0.5,
                0.5,
                "No deviation data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Summary Accuracy")
            return

        # Process all attempts from all summary nodes
        for node in nodes:
            height = node.get("height", node.get("level", 0))
            if height > 0:  # Summary nodes only
                attempts = node.get("summary_attempts", [])
                for attempt_num, attempt in enumerate(attempts, 1):
                    actual_tokens = attempt.get("actual_tokens", 0)
                    if actual_tokens > 0:
                        deviation = (actual_tokens - chunk_size) / chunk_size * 100
                        if attempt_num <= max_attempt_groups:
                            deviations_by_attempt[attempt_num].append(deviation)
                        else:
                            # Group attempts beyond max_attempt_groups together
                            deviations_by_attempt[max_attempt_groups].append(deviation)

        # Combine all deviations to determine bin edges
        all_deviations = []
        for devs in deviations_by_attempt.values():
            all_deviations.extend(devs)

        if not all_deviations:
            ax.text(
                0.5,
                0.5,
                "No deviation data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Summary Accuracy")
            return

        # Create consistent bins for all attempts
        bins = np.linspace(min(all_deviations), max(all_deviations), 31)

        # Create stacked histogram data with horizontal bars
        colors = [
            "#2563eb",
            "#10b981",
            "#f59e0b",
            "#ef4444",
            "#991b1b",
        ]  # Blue to red gradient
        labels = ["Attempt 1", "Attempt 2", "Attempt 3", "Attempt 4", "Attempt 5+"]

        # Calculate histogram data for each attempt
        left = np.zeros(len(bins) - 1)

        for attempt_num in range(1, 6):
            if deviations_by_attempt[attempt_num]:
                counts, _ = np.histogram(deviations_by_attempt[attempt_num], bins=bins)
                ax.barh(
                    bins[:-1],
                    counts,
                    height=np.diff(bins),
                    left=left,
                    color=colors[attempt_num - 1],
                    label=labels[attempt_num - 1],
                    alpha=0.8,
                    edgecolor="none",
                )
                left += counts

        # Add target line (now horizontal)
        ax.axhline(0, color="green", linestyle="--", label="Target", linewidth=2)

        # Add statistics for all deviations (not just final)
        if all_deviations:
            median_dev = float(np.median(all_deviations))
            avg_dev = float(np.mean(all_deviations))
            ax.axhline(
                median_dev,
                color="red",
                linestyle="-.",
                label=f"Median: {median_dev:.1f}%",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.axhline(
                avg_dev,
                color="blue",
                linestyle=":",
                label=f"Average: {avg_dev:.1f}%",
                linewidth=1.5,
                alpha=0.7,
            )

        ax.set_ylabel("Deviation from Target Token Count (%)")
        ax.set_xlabel("Frequency")
        ax.set_title("Summary Accuracy\n(All attempts, stacked by attempt number)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_node_timeline(self, telemetry: dict, ax: plt.Axes) -> None:
        """Plot summary node creation timeline."""
        # Extract summary completion times (when summaries actually finished)
        creation_times = []

        # Extract nodes and get summary completion times
        nodes = self._extract_nodes_from_telemetry(telemetry)
        for node in nodes:
            # Only include nodes that have summary attempts (i.e., actually performed summaries)
            if node.get("summary_attempts"):
                # Get the accepted attempt (usually the last one)
                accepted_idx = node.get(
                    "accepted_attempt", len(node["summary_attempts"]) - 1
                )
                if 0 <= accepted_idx < len(node["summary_attempts"]):
                    attempt = node["summary_attempts"][accepted_idx]
                    # Use the end_time of the summary attempt
                    if "end_time" in attempt:
                        creation_times.append(attempt["end_time"])

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
        ax.set_ylabel("Cumulative Summaries Created")
        ax.set_title(
            "Summary Creation Timeline\n(Shows summarization progress over time)"
        )
        ax.grid(True, alpha=0.3)

        # Add total processing time annotation (bottom right)
        total_time = max(relative_times) if relative_times else 0
        total_nodes = len(relative_times)
        ax.text(
            0.98,
            0.02,
            (
                f"Total: {total_nodes} summaries in {total_time:.1f}s\n"
                f"Rate: {total_nodes/total_time:.1f} summaries/sec"
                if total_time > 0
                else f"Total: {total_nodes} nodes"
            ),
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            fontsize=8,
        )

    def _plot_token_distributions(self, telemetry: dict, ax: plt.Axes) -> None:
        """Plot token count distributions by attempt number using violin plots."""
        import pandas as pd

        # Extract token data by attempt number from telemetry
        token_data = []

        # Extract nodes from telemetry data
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # Process all attempts from summary nodes
        for node in nodes:
            height = node.get("height", node.get("level", 0))
            if height > 0:  # Summary nodes only
                attempts = node.get("summary_attempts", [])
                for attempt_num, attempt in enumerate(attempts, 1):
                    actual_tokens = attempt.get("actual_tokens", 0)
                    if actual_tokens > 0:
                        # Group attempts 5+ together
                        display_num = min(attempt_num, 5)
                        token_data.append(
                            {
                                "attempt": (
                                    f"Attempt {display_num}"
                                    if display_num < 5
                                    else "Attempt 5+"
                                ),
                                "actual_tokens": actual_tokens,
                                "attempt_order": display_num,  # For sorting
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
            ax.set_title("Token Distributions by Attempt")
            return

        # Create DataFrame for plotting
        df = pd.DataFrame(token_data)

        # Sort by attempt order
        df = df.sort_values("attempt_order")
        attempt_order = df["attempt"].unique()

        # Create violin plot with same colors as Summary Accuracy
        colors = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#991b1b"]

        # Map colors to attempts
        palette = {attempt: colors[i] for i, attempt in enumerate(attempt_order[:5])}

        sns.violinplot(
            data=df,
            x="attempt",
            y="actual_tokens",
            hue="attempt",
            order=attempt_order,
            ax=ax,
            inner=None,  # Remove noisy quartile lines
            palette=palette,
            legend=False,  # Don't show legend since it's redundant with x-axis
            density_norm="count",  # Scale violin width by number of observations
            common_norm=True,  # Use same scaling across all violins
        )

        # Get chunk_size for target line
        chunk_size = self._extract_chunk_size_from_telemetry(telemetry)

        if chunk_size and chunk_size > 0:
            ax.axhline(
                y=chunk_size,
                color="green",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label="Target",
            )

        ax.set_xlabel("Attempt Number")
        ax.set_ylabel("Token Count")
        ax.set_title("Token Distributions by Attempt")
        ax.grid(True, alpha=0.3, axis="y")

        # Add legend if target line was added
        if ax.lines:
            ax.legend(loc="upper right")
