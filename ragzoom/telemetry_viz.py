"""Telemetry visualization classes and functions.

NOTE: This module provides visualization for telemetry data, focusing on token usage,
costs, batch efficiency, and retry patterns. For programmatic analysis, use the
simplified metrics in telemetry_cli.py.
"""

import json
import logging
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ragzoom.config import get_embedding_cost, get_llm_costs
from ragzoom.telemetry_analysis import (
    compute_batch_efficiency,
    get_accepted_attempt,
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

# Color constants for visualization consistency
EMBEDDINGS_COLOR = "#9333ea"  # Purple for embeddings
ATTEMPT_COLORS = [
    "#2563eb",  # Blue for initial attempt
    "#10b981",  # Green for retry 1
    "#f59e0b",  # Yellow for retry 2
    "#ef4444",  # Orange for retry 3
    "#991b1b",  # Red for retry 4+
]


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

    def _extract_nodes_from_telemetry(
        self, telemetry: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract nodes from telemetry data.

        Args:
            telemetry: Telemetry data dictionary

        Returns:
            List of node dictionaries
        """
        # Format 4.2: nodes are at the top level
        nodes_data = telemetry.get("nodes", [])
        return nodes_data if isinstance(nodes_data, list) else []

    def _extract_chunk_size_from_telemetry(self, telemetry: dict[str, Any]) -> int:
        """Extract chunk size from telemetry data.

        Args:
            telemetry: Telemetry data dictionary

        Returns:
            Chunk size in tokens, or 0 if not found
        """
        # Format 4.2: read from config
        config = telemetry.get("config", {})
        chunk_size = config.get("target_chunk_tokens", 0)
        return int(chunk_size) if chunk_size else 0

    def _calculate_max_time_from_telemetry(self, telemetry: dict[str, Any]) -> float:
        """Calculate the maximum end time from telemetry data.

        Args:
            telemetry: Telemetry data dictionary

        Returns:
            Maximum end time found in the telemetry, relative to baseline
        """
        nodes = self._extract_nodes_from_telemetry(telemetry)
        indexing_start_time = telemetry.get("indexed_at", None)

        max_time = 0
        min_time = None

        # Process each node to find the maximum end time
        for node in nodes:
            # Skip leaf nodes (height 0) - they're raw text chunks
            if node["height"] == 0:
                continue

            # Handle passthrough nodes (no summary attempts)
            if not node.get("summary_attempts"):
                created_at = node.get("created_at", 0)
                max_time = max(max_time, created_at)
                if min_time is None:
                    min_time = created_at
                continue

            # Process summary nodes with attempts
            attempts = node["summary_attempts"]
            for attempt in attempts:
                start_time = attempt.get("start_time")
                end_time = attempt.get("end_time")

                if start_time is None or end_time is None:
                    continue

                if min_time is None:
                    min_time = start_time

                max_time = max(max_time, end_time)

        # Calculate baseline and return relative max time
        baseline = indexing_start_time if indexing_start_time is not None else min_time
        if baseline is not None and max_time > baseline:
            return float(max_time - baseline)
        else:
            return 1.0  # Fallback value

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

        # Handle format 4.2
        if "format_version" in data:
            # Standard format 4.2: {"format_version": "4.2", ...}
            telemetry = data
        else:
            print(f"Warning: No telemetry data found in {benchmark_path}")
            return

        # Create figure with subplots (3 rows only)
        fig = plt.figure(
            figsize=(FIGURE_WIDTH * 0.33, FIGURE_HEIGHT * 0.6)
        )  # Reduce width by 2/3 and height
        # Use GridSpecFromSubplotSpec for different gaps between rows

        # Create main grid with 2 sections for different spacing
        main_gs = GridSpec(
            2, 1, figure=fig, hspace=0.25, top=0.92, height_ratios=[2, 2]
        )

        # Top section: Cost Breakdown and Summary Compression (closer together)
        top_gs = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=main_gs[0], hspace=0.3, height_ratios=[0.6, 1.4]
        )

        # Bottom section: Tree Construction Timeline
        bottom_gs = GridSpecFromSubplotSpec(1, 1, subplot_spec=main_gs[1])

        # 1. Cost Breakdown
        ax1 = fig.add_subplot(top_gs[0])
        self._plot_cost_breakdown(telemetry, ax1)

        # 2. Summary Compression Patterns
        ax2 = fig.add_subplot(top_gs[1])
        self._plot_summary_scatter(telemetry, ax2)

        # 3. Tree Construction Timeline
        ax3 = fig.add_subplot(bottom_gs[0])
        self._plot_tree_construction_timeline(telemetry, ax3)

        # Add title and metadata
        if "config" in data:
            # Get chunk size from config
            chunk_size = data["config"].get("target_chunk_tokens", "Unknown")
        elif "chunk_size" in telemetry:
            # Get chunk size from metadata
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

    def _get_cost_functions(
        self, telemetry: dict[str, Any]
    ) -> tuple[float, float, float]:
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

        # Handle format 4.2 for file1
        if "format_version" in data1:
            telemetry1 = data1
        else:
            print(f"Warning: No telemetry data found in {file1}")
            return

        # Handle format 4.2 for file2
        if "format_version" in data2:
            telemetry2 = data2
        else:
            print(f"Warning: No telemetry data found in {file2}")
            return

        # Telemetry data already contains model information for cost calculations

        # Pre-calculate the maximum time across both datasets for consistent y-axis scaling
        max_time_1 = self._calculate_max_time_from_telemetry(telemetry1)
        max_time_2 = self._calculate_max_time_from_telemetry(telemetry2)
        global_max_time = max(max_time_1, max_time_2)

        # Create figure with side-by-side subplots using built-in axis sharing
        if figsize is None:
            figsize = (
                10,
                14,
            )  # Half the width, slightly taller for double Summary Accuracy

        # Create figure with GridSpec for flexible subplot arrangement
        # Note: We can't use simple sharex='row' because each row has different x-axis semantics
        # Row 1: Cost breakdown (categorical x-axis) - no x-sharing
        # Row 2: Summary scatter (numeric x-axis) - needs x-sharing
        # Row 3: Timeline (numeric x-axis) - needs x-sharing
        fig = plt.figure(figsize=figsize)

        # Create main grid with 2 sections for different spacing
        main_gs = GridSpec(
            2, 1, figure=fig, hspace=0.25, top=0.92, height_ratios=[2, 2]
        )

        # Top section: Cost Breakdown and Summary Compression (closer together)
        top_gs = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=main_gs[0],
            hspace=0.3,
            wspace=0.15,
            height_ratios=[0.6, 1.4],
        )

        # Bottom section: Tree Construction Timeline
        bottom_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[1], wspace=0.15)

        # Add super title
        fig.suptitle(
            "Side-by-Side Comparison: Baseline vs Current",
            fontsize=16,
            y=0.97,
        )

        # 1. Cost Breakdown (no axis sharing needed)
        ax1_left = fig.add_subplot(top_gs[0, 0])
        ax1_right = fig.add_subplot(top_gs[0, 1], sharey=ax1_left)

        self._plot_cost_breakdown(telemetry1, ax1_left)
        ax1_left.set_title("Cost Breakdown", fontsize=12)

        self._plot_cost_breakdown(telemetry2, ax1_right)
        ax1_right.set_title("Cost Breakdown", fontsize=12)
        ax1_right.set_ylabel("")  # Remove y-axis label

        # 2. Summary Compression Patterns (share both axes)
        ax2_left = fig.add_subplot(top_gs[1, 0])
        ax2_right = fig.add_subplot(top_gs[1, 1], sharex=ax2_left, sharey=ax2_left)

        self._plot_summary_scatter(telemetry1, ax2_left)
        ax2_left.set_title("Summary Compression Patterns", fontsize=12)

        self._plot_summary_scatter(telemetry2, ax2_right)
        ax2_right.set_title("Summary Compression Patterns", fontsize=12)
        ax2_right.set_ylabel("")  # Remove y-axis label

        # 3. Tree Construction Timeline (share both axes)
        ax3_left = fig.add_subplot(bottom_gs[0, 0])
        ax3_right = fig.add_subplot(bottom_gs[0, 1], sharex=ax3_left, sharey=ax3_left)

        self._plot_tree_construction_timeline(
            telemetry1, ax3_left, max_y_limit=global_max_time
        )
        ax3_left.set_title("Tree Construction Timeline", fontsize=12, pad=25)

        self._plot_tree_construction_timeline(
            telemetry2, ax3_right, max_y_limit=global_max_time
        )
        ax3_right.set_title("Tree Construction Timeline", fontsize=12, pad=25)
        ax3_right.set_ylabel("")  # Remove y-axis label

        # Save figure
        self._ensure_output_dir()
        with self._suppress_matplotlib_warnings():
            plt.tight_layout()
            plt.savefig(self.output_path, bbox_inches="tight", dpi=SAVE_DPI)
        plt.close()

        print(f"Saved side-by-side comparison to {self.output_path}")

    def _plot_token_usage_by_tree_level(
        self, telemetry: dict[str, Any], ax: Axes
    ) -> None:
        """Plot token usage by tree level with stacked bars."""
        # Group tokens by height
        tokens_by_height: dict[int, dict[str, list[float]]] = {}

        # Extract nodes from telemetry data
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # Process nodes
        for node in nodes:
            height = node["height"]
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

    def _plot_cost_breakdown(self, telemetry: dict[str, Any], ax: Axes) -> None:
        """Plot cost breakdown by attempt number as vertical stacked bar."""
        # Get cost functions for models in telemetry
        embedding_cost_per_1k, summary_input_cost_per_1k, summary_output_cost_per_1k = (
            self._get_cost_functions(telemetry)
        )

        # Calculate costs by attempt number
        costs_by_attempt = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # 5 = 5+

        # Extract nodes from telemetry data
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # Calculate embedding cost
        total_embedding_tokens = 0
        for node in nodes:
            embedding = node.get("embedding")
            if embedding:
                total_embedding_tokens += embedding.get("text_tokens", 0)

        embedding_cost = (total_embedding_tokens / 1000) * embedding_cost_per_1k

        # Process all attempts from summary nodes
        for node in nodes:
            height = node["height"]
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

        # Calculate total cost (including embeddings now)
        summary_cost = sum(costs_by_attempt.values())
        total_cost = embedding_cost + summary_cost

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
        # Use purple for embeddings, then the standard retry colors
        colors = [EMBEDDINGS_COLOR] + ATTEMPT_COLORS
        labels = [
            "Embeddings",
            "Initial attempt",
            "Retry 1",
            "Retry 2",
            "Retry 3",
            "Retry 4+",
        ]

        # Single vertical bar centered - start with embeddings at bottom
        bottom = 0.0

        # Add embeddings bar first (at the bottom)
        if embedding_cost > 0:
            ax.bar(
                0.5,
                embedding_cost,
                bottom=bottom,
                color=colors[0],
                label=labels[0],
                width=0.3,
            )
            bottom += embedding_cost

        # Then add summary attempt costs on top
        for attempt_num in range(1, 6):
            cost = costs_by_attempt[attempt_num]
            if cost > 0:  # Only plot if there's cost
                label = labels[attempt_num]  # Adjusted index for new labels list
                color = colors[attempt_num]  # Adjusted index for new colors list
                ax.bar(
                    0.5,
                    cost,
                    bottom=bottom,
                    color=color,
                    label=label,
                    width=0.3,
                )
                bottom += cost

        ax.set_ylim(0, total_cost * 1.2)  # Add 20% padding for legend
        ax.set_xlim(0, 1)
        ax.set_ylabel("Cost ($)")
        ax.set_title(f"Cost Breakdown\nTotal: ${total_cost:.4f}")
        ax.set_xticks([])  # Hide x-axis ticks
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_batch_efficiency(self, telemetry: dict[str, Any], ax: Axes) -> None:
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

    def _plot_retry_patterns(self, telemetry: dict[str, Any], ax: Axes) -> None:
        """Plot retry attempt distribution as stacked bar chart (cumulative)."""
        # Count nodes by number of attempts
        attempt_counts: dict[int, int] = {}

        # Extract nodes from telemetry data
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # Count attempts for each summary node
        total_summary_nodes = 0
        for node in nodes:
            height = node["height"]
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
        colors = ATTEMPT_COLORS[: len(labels)]

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
        self, telemetry: dict[str, Any]
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
            height = node["height"]
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

    def _plot_summary_scatter(self, telemetry: dict[str, Any], ax: Axes) -> None:
        """Plot input vs output token scatter plot, color-coded by attempt number."""
        # Extract chunk size (target) and nodes from telemetry data
        chunk_size = self._extract_chunk_size_from_telemetry(telemetry)
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # Prepare data for scatter plot - one dot per attempt
        input_tokens = []
        output_tokens = []
        attempt_numbers = []
        is_accepted = []  # Track which attempts are accepted
        node_count = 0  # Track actual number of nodes processed

        # Process nodes to extract ALL attempts (not just accepted ones)
        for node in nodes:
            height = node["height"]
            if height > 0:  # Summary nodes only
                # Get input tokens (tokens being summarized)
                input_text_tokens = node.get("input_text_tokens")
                if input_text_tokens is None or input_text_tokens <= 0:
                    # Skip nodes without input token data
                    continue

                # Process ALL summary attempts for this node
                attempts = node.get("summary_attempts", [])
                if attempts:  # Only count nodes that have attempts
                    node_count += 1
                    # Get the accepted attempt index (defaults to last attempt)
                    accepted_idx = node.get("accepted_attempt", len(attempts) - 1)
                    for attempt_num, attempt in enumerate(attempts, 1):
                        actual_tokens = attempt.get("actual_tokens", 0)
                        if actual_tokens > 0:
                            input_tokens.append(input_text_tokens)
                            output_tokens.append(actual_tokens)
                            attempt_numbers.append(attempt_num)
                            # Check if this is the accepted attempt (0-based index)
                            is_accepted.append(attempt_num - 1 == accepted_idx)

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
        colors = ATTEMPT_COLORS  # Blue to red gradient for attempts

        # Map each attempt to a color
        attempt_colors = []
        for attempt_num in attempt_numbers:
            if attempt_num >= len(colors):
                attempt_colors.append(colors[-1])  # 5+ attempts = darkest red
            else:
                attempt_colors.append(colors[attempt_num - 1])  # Convert to 0-indexed

        # Create scatter plot - first plot all attempts
        ax.scatter(
            input_tokens,
            output_tokens,
            c=attempt_colors,
            alpha=0.6,
            s=50,
            edgecolors="none",
        )

        # Then plot accepted attempts with black borders on top
        accepted_inputs = [inp for inp, acc in zip(input_tokens, is_accepted) if acc]
        accepted_outputs = [out for out, acc in zip(output_tokens, is_accepted) if acc]
        accepted_colors = [col for col, acc in zip(attempt_colors, is_accepted) if acc]
        if accepted_inputs:
            ax.scatter(
                accepted_inputs,
                accepted_outputs,
                c=accepted_colors,
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidths=1,
                zorder=10,  # Draw on top
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
        # Get current x-axis limits and extend well beyond them to ensure full coverage
        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        # Extend by 10x the range on each side to ensure the green zone covers the entire visible area
        x_extend = [xlim[0] - x_range * 10, xlim[1] + x_range * 10]

        if chunk_size > 0:
            if retry_threshold is not None:
                threshold_tokens = chunk_size * retry_threshold
                # With undershoot elimination, we accept all undershoots (0 to target)
                # and only retry overshoots beyond target + threshold
                ax.fill_between(
                    x_extend,
                    0,  # Accept all undershoots
                    chunk_size + threshold_tokens,  # Retry beyond this
                    alpha=0.1,
                    color="green",
                    label=f"Acceptance range (0 to +{retry_threshold*100:.0f}%)",
                )
            else:
                # Warn when retry_threshold is missing and use fallback
                logging.warning(
                    "retry_threshold not found in telemetry config. "
                    "Using fallback acceptance range 0 to target+10 tokens."
                )
                ax.fill_between(
                    x_extend,
                    0,  # Accept all undershoots
                    chunk_size + 10,  # Retry beyond target + 10
                    alpha=0.1,
                    color="green",
                    label="Acceptance range (0 to +10 tokens)",
                )

        # Add diagonal reference line showing 1:1 ratio (extend well beyond visible area)
        # Use a very large range to ensure the diagonal covers any zoom level
        diagonal_extent = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)) * 100
        ax.plot(
            [-diagonal_extent, diagonal_extent],
            [-diagonal_extent, diagonal_extent],
            "k:",
            alpha=0.3,
            linewidth=1,
            label="1:1 ratio",
        )

        # Create custom legend for attempt numbers (matching cost breakdown)
        # Use circles instead of rectangles
        legend_elements: list[Line2D] = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[0],
                markersize=8,
                label="Initial attempt",
                linestyle="None",
                alpha=0.6,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[1],
                markersize=8,
                label="Retry 1",
                linestyle="None",
                alpha=0.6,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[2],
                markersize=8,
                label="Retry 2",
                linestyle="None",
                alpha=0.6,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[3],
                markersize=8,
                label="Retry 3",
                linestyle="None",
                alpha=0.6,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[4],
                markersize=8,
                label="Retry 4+",
                linestyle="None",
                alpha=0.6,
            ),
        ]

        # Only include legend items for attempt numbers that exist in the data
        max_attempts = max(attempt_numbers) if attempt_numbers else 0
        legend_elements = legend_elements[: min(max_attempts, 5)]

        # Add accepted attempt indicator to legend with transparent fill
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",  # Transparent fill
                markeredgecolor="black",
                markersize=8,
                markeredgewidth=1,
                label="Accepted attempt",
                linestyle="None",
            )
        )

        # Add statistics annotation
        # Calculate deviations from target (chunk_size) as percentages for ACCEPTED attempts only
        if chunk_size > 0 and accepted_outputs:
            accepted_deviations_pct = [
                (output - chunk_size) / chunk_size * 100 for output in accepted_outputs
            ]
            avg_deviation_pct = np.mean(accepted_deviations_pct)
            median_deviation_pct = np.median(accepted_deviations_pct)

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

            # Add the line references to legend elements

            legend_elements.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        color="blue",
                        linestyle=":",
                        alpha=0.5,
                        linewidth=1.5,
                        label=f"Avg: {avg_deviation_pct:+.1f}%",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="red",
                        linestyle="-.",
                        alpha=0.5,
                        linewidth=1.5,
                        label=f"Median: {median_deviation_pct:+.1f}%",
                    ),
                ]
            )

        elif chunk_size > 0:
            # No accepted outputs, but chunk_size is valid
            avg_deviation_pct = 0.0
            median_deviation_pct = 0.0
        else:
            avg_deviation_pct = 0.0
            median_deviation_pct = 0.0

        # Add the legend with all elements
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

        # Labels and title
        ax.set_xlabel("Input Tokens (text to summarize)")
        ax.set_ylabel("Output Tokens (summary)")
        ax.set_title("Summary Compression Patterns")
        ax.grid(True, alpha=0.3)

        avg_attempts = np.mean(attempt_numbers)

        stats_text = (
            f"Avg attempts per node: {avg_attempts:.2f}\n"
            f"Total attempts: {len(input_tokens)} ({node_count} nodes)"
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

    def _plot_node_timeline(self, telemetry: dict[str, Any], ax: Axes) -> None:
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

    def _plot_tree_construction_timeline(
        self, telemetry: dict[str, Any], ax: Axes, max_y_limit: float | None = None
    ) -> None:
        """Plot tree construction as rectangles showing span coverage over time.

        X-axis: Document span position
        Y-axis: Time (seconds) from actual indexing start
        Rectangles: Each node with width=span coverage, height=processing duration
        Colors: Retry attempts (blue→green→yellow→orange→red)
        """
        from matplotlib.patches import Rectangle

        # Extract nodes from telemetry
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # Define retry attempt colors (1=blue, 2=green, 3=yellow, 4=orange, 5+=red)
        attempt_colors = ATTEMPT_COLORS

        # Only show visualization if we have real spans from telemetry
        node_spans = {}  # node_id -> (start, end)

        # Check if we have real spans from telemetry
        has_real_spans = all(node.get("span") is not None for node in nodes)

        if not has_real_spans:
            # Don't show visualization without real spans
            ax.text(
                0.5,
                0.5,
                "Tree Construction Timeline not available\n(telemetry format too old - missing span data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            return

        # Use the real spans from TreeNode data
        for node in nodes:
            span = node.get("span")
            if span:
                node_spans[node["node_id"]] = tuple(span)

        # Get the actual indexing start time from telemetry
        # This is when the TelemetryCollector was initialized
        indexing_start_time = telemetry.get("indexed_at", None)

        # Track min/max for axis limits
        min_time = None  # Will be set to indexed_at or first node time as baseline
        max_time = 0
        min_span = float("inf")
        max_span = 0

        # First pass: calculate span range for gap sizing
        for node in nodes:
            node_id = node.get("node_id")
            if node_id in node_spans:
                span_start, span_end = node_spans[node_id]
                min_span = min(min_span, span_start)
                max_span = max(max_span, span_end)

        # Calculate gap size: use fixed 20 chars for visual separation
        # This provides consistent visual spacing regardless of document length
        gap = 20

        # Process each node
        for node in nodes:
            # Get span from our calculated dictionary
            node_id = node.get("node_id")
            if node_id not in node_spans:
                raise ValueError(
                    f"Node {node_id} has no calculated span - this should not happen"
                )

            span_start, span_end = node_spans[node_id]

            # Skip only leaf nodes (height 0) - they're raw text chunks
            if node["height"] == 0:
                continue

            # Handle passthrough nodes (no summary attempts) - draw as single pixel line
            if not node.get("summary_attempts"):
                # Draw a single pixel horizontal line for passthrough nodes
                created_at = node.get("created_at", 0)

                # Update max_time
                max_time = max(max_time, created_at)

                # Calculate relative time from indexing start
                # Three-level fallback: indexed_at -> min_time -> current node time
                # This handles telemetry without indexed_at (older versions)
                if indexing_start_time is not None:
                    baseline = indexing_start_time
                elif min_time is not None:
                    baseline = min_time
                else:
                    baseline = created_at
                    min_time = created_at
                relative_time = created_at - baseline

                # For passthrough nodes, use first attempt color (blue)
                color = attempt_colors[0]

                # Add gap between adjacent nodes for visual clarity
                rect = Rectangle(
                    (span_start, relative_time),  # Position at relative time from start
                    max(
                        1, span_end - span_start - gap
                    ),  # Width = span coverage minus gap
                    0.5,  # Minimal height (0.5 seconds for visibility)
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.9,
                )
                ax.add_patch(rect)
                continue

            # Process summary nodes with attempts
            attempts = node["summary_attempts"]
            accepted_idx = node.get("accepted_attempt", len(attempts) - 1)

            # Color will be determined per attempt

            cumulative_start = None
            for attempt_idx, attempt in enumerate(attempts):
                start_time = attempt.get("start_time")
                end_time = attempt.get("end_time")

                if start_time is None or end_time is None:
                    continue

                # Update min_time tracking for fallback
                if indexing_start_time is None and min_time is None:
                    min_time = start_time

                if cumulative_start is None:
                    cumulative_start = start_time

                max_time = max(max_time, end_time)

                # Determine color based on attempt number
                attempt_num = attempt_idx + 1  # Convert to 1-based
                if attempt_num >= len(attempt_colors):
                    color = attempt_colors[-1]  # 5+ attempts = darkest red
                else:
                    color = attempt_colors[attempt_num - 1]  # Convert to 0-indexed
                is_accepted = attempt_idx == accepted_idx

                # Calculate baseline for relative time
                # Three-level fallback: indexed_at -> min_time -> current attempt time
                if indexing_start_time is not None:
                    baseline = indexing_start_time
                elif min_time is not None:
                    baseline = min_time
                else:
                    baseline = cumulative_start
                    min_time = cumulative_start

                # Draw rectangle for this attempt with gap
                rect = Rectangle(
                    (
                        span_start,
                        cumulative_start - baseline,
                    ),  # (x, y) = (document position, relative time)
                    max(
                        1, span_end - span_start - gap
                    ),  # width = span coverage minus gap
                    end_time - cumulative_start,  # height = duration
                    facecolor=color,
                    edgecolor="black" if is_accepted else "none",
                    linewidth=0.5 if is_accepted else 0,
                    alpha=0.9,
                )
                ax.add_patch(rect)
                cumulative_start = end_time

        # Set axis limits and labels
        if max_span > min_span:
            # Determine final baseline for Y-axis: prefer indexed_at, fallback to min_time
            baseline = (
                indexing_start_time if indexing_start_time is not None else min_time
            )
            if baseline is not None:
                ax.set_xlim(min_span, max_span)
                # Use provided max_y_limit if available, otherwise calculate from current data
                if max_y_limit is not None:
                    ax.set_ylim(0, max_y_limit)
                else:
                    ax.set_ylim(0, max_time - baseline if max_time > baseline else 1)
            else:
                ax.set_xlim(min_span, max_span)
                ax.set_ylim(0, max_y_limit if max_y_limit is not None else 1)
            ax.set_xlabel("Document Position (characters)")
            ax.set_ylabel("Time Since Start (seconds)")
            # Add extra padding at the top for the legend
            ax.set_title(
                "Tree Construction Timeline",
                pad=25,  # Add padding to make room for legend
            )
            ax.grid(True, alpha=0.3)

            # Add legend for attempt colors
            legend_elements = [
                Patch(facecolor=attempt_colors[0], label="Initial attempt", alpha=0.9),
                Patch(facecolor=attempt_colors[1], label="Retry 1", alpha=0.9),
                Patch(facecolor=attempt_colors[2], label="Retry 2", alpha=0.9),
                Patch(facecolor=attempt_colors[3], label="Retry 3", alpha=0.9),
                Patch(facecolor=attempt_colors[4], label="Retry 4+", alpha=0.9),
            ]

            # Only include legend items for attempts that exist
            max_attempts = 0
            for node in nodes:
                if node.get("summary_attempts"):
                    max_attempts = max(max_attempts, len(node["summary_attempts"]))

            if max_attempts > 0:
                legend_elements = legend_elements[:max_attempts]
                # Place legend horizontally between title and chart
                ax.legend(
                    handles=legend_elements,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.05),  # Position above chart, below title
                    ncol=min(len(legend_elements), 6),  # Horizontal layout
                    fontsize=8,
                    frameon=False,  # Remove frame for cleaner look
                )
        else:
            # No valid data to plot
            ax.text(
                0.5,
                0.5,
                "No timeline data available\n(no nodes found in telemetry)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Tree Construction Timeline")

    def _plot_token_distributions(self, telemetry: dict[str, Any], ax: Axes) -> None:
        """Plot token count distributions by attempt number using violin plots."""
        import pandas as pd

        # Extract token data by attempt number from telemetry
        token_data = []

        # Extract nodes from telemetry data
        nodes = self._extract_nodes_from_telemetry(telemetry)

        # Process all attempts from summary nodes
        for node in nodes:
            height = node["height"]
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
                                    "Initial attempt"
                                    if display_num == 1
                                    else (
                                        f"Retry {display_num - 1}"
                                        if display_num < 5
                                        else "Retry 4+"
                                    )
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
        colors = ATTEMPT_COLORS

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
