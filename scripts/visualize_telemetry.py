#!/usr/bin/env python3
"""Visualize telemetry data from RagZoom benchmarks.

This script generates comprehensive visualizations from telemetry data collected
during indexing operations. It supports both single benchmark analysis and
comparison between multiple benchmarks.

Usage:
    python scripts/visualize_telemetry.py benchmark_results/metrics_200_tokens.json
    python scripts/visualize_telemetry.py benchmark_results/ --output-dir reports/
    python scripts/visualize_telemetry.py benchmark_results/ --format html --compare
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for importing ragzoom
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
except ImportError as e:
    print(f"Error: Missing required visualization dependencies: {e}")
    print("Please install: pip install matplotlib seaborn pandas numpy")
    sys.exit(1)

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry import (
    analyze_retry_patterns,
    compute_amplification_metrics,
    compute_batch_efficiency,
    compute_metrics_from_telemetry,
    get_telemetry_thresholds,
)

# Set style for professional-looking plots
try:
    plt.style.use('seaborn-darkgrid')
except OSError:
    # Fallback to a default style if seaborn style is not available
    plt.style.use('ggplot')
sns.set_palette("husl")
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 10


class TelemetryVisualizer:
    """Generate visualizations from telemetry data."""

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

    def visualize_single_benchmark(self, benchmark_path: Path, output_format: str = "png") -> None:
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
        fig = plt.figure(figsize=(20, 24))
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
        self._plot_summary_accuracy(data.get("metrics", {}), ax5)

        # 6. Node Creation Timeline
        ax6 = fig.add_subplot(gs[4, :])
        self._plot_node_timeline(telemetry, ax6)

        # 7. Token Usage Heatmap
        ax7 = fig.add_subplot(gs[5, :])
        self._plot_token_heatmap(telemetry, ax7)

        # Add title and metadata
        chunk_size = data["config"]["leaf_tokens"]
        fig.suptitle(f'Telemetry Analysis - {chunk_size} Token Chunks', fontsize=16, y=0.995)

        # Save figure
        output_path = self.output_dir / f"telemetry_{chunk_size}_tokens.{output_format}"
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization to {output_path}")

        # Also generate markdown report
        self._generate_markdown_report(data, telemetry, config, chunk_size)

    def _create_config_from_metrics(self, metrics: dict[str, Any]) -> RagZoomConfig:
        """Create a config object from metrics data for cost calculations.

        Uses default values based on typical API costs as of January 2025:
        - text-embedding-3-small: $0.02 per 1M tokens ($0.00002 per 1K)
        - gpt-4o-mini input: $0.15 per 1M tokens ($0.00015 per 1K)
        - gpt-4o-mini output: $0.60 per 1M tokens ($0.0006 per 1K)

        Note: These are default values for visualization purposes. Actual costs
        in the benchmark data were calculated using the exact costs at runtime.
        """
        # Use actual values from metrics if available, otherwise use defaults
        # Note: metrics use older costs, so we use those for consistency
        return RagZoomConfig(
            openai_api_key="dummy",  # Not needed for analysis
            embedding_cost_per_1k=0.0001,  # text-embedding-3-small (older pricing)
            summary_input_cost_per_1k=0.0025,  # gpt-4o-mini input (older pricing)
            summary_output_cost_per_1k=0.01,   # gpt-4o-mini output (older pricing)
        )

    def _plot_amplification_by_level(self, telemetry: dict[str, Any], config: RagZoomConfig, ax: plt.Axes) -> None:
        """Plot amplification metrics by tree level."""
        try:
            amplification = compute_amplification_metrics(telemetry, config)
            by_level = amplification.get("by_level", {})

            if not by_level:
                ax.text(0.5, 0.5, "No amplification data available",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Amplification by Tree Level")
                return

            # Prepare data for plotting
            levels = sorted(by_level.keys())
            input_medians = []
            output_medians = []
            cost_medians = []

            for level in levels:
                level_data = by_level[level]
                input_medians.append(np.median(level_data.get("input", [1.0])))
                output_medians.append(np.median(level_data.get("output", [1.0])))
                cost_medians.append(np.median(level_data.get("cost", [1.0])))

            # Plot lines
            x = np.arange(len(levels))
            width = 0.25

            ax.bar(x - width, input_medians, width, label='Input Amplification', alpha=0.8)
            ax.bar(x, output_medians, width, label='Output Amplification', alpha=0.8)
            ax.bar(x + width, cost_medians, width, label='Cost Amplification', alpha=0.8)

            ax.set_xlabel('Tree Level')
            ax.set_ylabel('Amplification Factor')
            ax.set_title('Median Amplification Factors by Tree Level')
            ax.set_xticks(x)
            ax.set_xticklabels([f"Level {level}" for level in levels])
            ax.legend()
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1.0)')
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Amplification by Tree Level (Error)")

    def _plot_cost_breakdown(self, telemetry: dict[str, Any], config: RagZoomConfig, ax: plt.Axes) -> None:
        """Plot cost breakdown pie chart."""
        try:
            metrics = compute_metrics_from_telemetry(telemetry, config)

            # Calculate costs
            embedding_cost = (metrics.total_embedding_tokens / 1000) * config.embedding_cost_per_1k
            summary_input_cost = (metrics.total_summary_prompt_tokens / 1000) * config.summary_input_cost_per_1k
            summary_output_cost = (metrics.total_summary_completion_tokens / 1000) * config.summary_output_cost_per_1k

            # Create pie chart
            costs = [embedding_cost, summary_input_cost, summary_output_cost]
            labels = ['Embeddings', 'Summary Input', 'Summary Output']
            colors = ['#ff9999', '#66b3ff', '#99ff99']

            # Filter out zero costs
            non_zero = [(c, lbl, col) for c, lbl, col in zip(costs, labels, colors) if c > 0]
            if non_zero:
                costs_tuple, labels_tuple, colors_tuple = zip(*non_zero)
                costs = list(costs_tuple)
                labels = list(labels_tuple)
                colors = list(colors_tuple)

                ax.pie(costs, labels=labels, colors=colors,
                       autopct='%1.1f%%', startangle=90)
                ax.set_title('Cost Breakdown by API Type')

                # Add total cost annotation
                total_cost = sum(costs)
                ax.text(0, -1.3, f'Total Cost: ${total_cost:.4f}',
                       ha='center', transform=ax.transAxes, fontsize=10)
            else:
                ax.text(0.5, 0.5, "No cost data available",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Cost Breakdown")

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Cost Breakdown (Error)")

    def _plot_batch_efficiency(self, telemetry: dict[str, Any], ax: plt.Axes) -> None:
        """Plot embedding batch efficiency."""
        try:
            batch_data = compute_batch_efficiency(telemetry)

            if not batch_data["batch_sizes"]:
                ax.text(0.5, 0.5, "No batch data available",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Batch Efficiency")
                return

            # Create histogram of batch sizes
            batch_sizes = batch_data["batch_sizes"]

            ax.hist(batch_sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(batch_data["avg_batch_size"], color='red', linestyle='--',
                      label=f'Average: {batch_data["avg_batch_size"]:.1f}')

            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Embedding Batch Size Distribution\n'
                        f'Utilization: {batch_data["batch_utilization"]:.1f}%')
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Batch Efficiency (Error)")

    def _plot_retry_patterns(self, telemetry: dict[str, Any], ax: plt.Axes) -> None:
        """Plot retry patterns analysis."""
        try:
            retry_data = analyze_retry_patterns(telemetry)

            if retry_data["total_attempts"] == 0:
                ax.text(0.5, 0.5, "No summary attempts found",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Retry Patterns")
                return

            # Create subplot layout
            gs_inner = GridSpec(1, 2, width_ratios=[1, 2], wspace=0.3)

            # Left: Retry rate metrics
            ax_left = plt.subplot(gs_inner[0])
            metrics_text = (
                f"Total Attempts: {retry_data['total_attempts']}\n"
                f"Successful: {retry_data['successful_attempts']}\n"
                f"Retry Attempts: {retry_data['retry_attempts']}\n"
                f"Retry Success Rate: {retry_data['retry_success_rate']:.1f}%\n\n"
                f"Nodes with Retries: {retry_data['nodes_with_retries']}\n"
                f"Total Summary Nodes: {retry_data['total_nodes_with_summaries']}\n"
                f"Retry Rate: {retry_data['retry_rate']:.1f}%"
            )
            ax_left.text(0.1, 0.5, metrics_text, transform=ax_left.transAxes,
                        fontsize=10, verticalalignment='center')
            ax_left.axis('off')
            ax_left.set_title('Retry Metrics', fontsize=12)

            # Right: Rejection reasons
            ax_right = plt.subplot(gs_inner[1])
            reasons = retry_data.get("rejection_reasons", {})
            if reasons:
                reasons_sorted = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
                reasons_list, counts = zip(*reasons_sorted)

                y_pos = np.arange(len(reasons_list))
                ax_right.barh(y_pos, counts, alpha=0.7, color='coral')
                ax_right.set_yticks(y_pos)
                ax_right.set_yticklabels(reasons_list)
                ax_right.set_xlabel('Count')
                ax_right.set_title('Rejection Reasons', fontsize=12)
                ax_right.grid(True, alpha=0.3, axis='x')
            else:
                ax_right.text(0.5, 0.5, "No rejections",
                            ha='center', va='center', transform=ax_right.transAxes)
                ax_right.set_title('Rejection Reasons', fontsize=12)
                ax_right.axis('off')

            # Remove the main axis
            ax.axis('off')
            ax.set_title('Summary Retry Analysis', fontsize=14, pad=20)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Retry Patterns (Error)")

    def _plot_summary_accuracy(self, metrics: dict[str, Any], ax: plt.Axes) -> None:
        """Plot summary accuracy distribution."""
        accuracy_data = metrics.get("summary_accuracy", {})

        if not accuracy_data:
            ax.text(0.5, 0.5, "No summary accuracy data available",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Summary Accuracy Distribution")
            return

        # Prepare data for visualization
        all_deviations = []
        labels = []

        for target_size, stats in sorted(accuracy_data.items()):
            if stats["count"] > 0:
                # Create data points for each bucket
                for bucket, data in stats["histogram"].items():
                    count = data["count"]
                    if count > 0:
                        # Map bucket to representative deviation values
                        if bucket == "0-10%":
                            deviations = [5.0] * count  # Use midpoint
                        elif bucket == "10-25%":
                            deviations = [17.5] * count
                        elif bucket == "25-50%":
                            deviations = [37.5] * count
                        elif bucket == "50-100%":
                            deviations = [75.0] * count
                        else:  # 100%+
                            deviations = [125.0] * count

                        all_deviations.extend(deviations)
                        labels.extend([f"{target_size} tokens"] * count)

        if all_deviations:
            # Create violin plot
            df = pd.DataFrame({'Deviation (%)': all_deviations, 'Target Size': labels})
            sns.violinplot(data=df, x='Target Size', y='Deviation (%)', ax=ax, inner='box')

            ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10% threshold')
            ax.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='25% threshold')
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')

            ax.set_title('Summary Size Accuracy Distribution by Target Size')
            ax.set_ylabel('Absolute Deviation from Target (%)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No deviation data available",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Summary Accuracy Distribution")

    def _plot_node_timeline(self, telemetry: dict[str, Any], ax: plt.Axes) -> None:
        """Plot node creation timeline."""
        try:
            # Extract node creation times
            nodes_data = []
            for doc_data in telemetry.get("documents", {}).values():
                for node in doc_data.get("nodes", []):
                    nodes_data.append({
                        'created_at': node.get('created_at', 0),
                        'level': node.get('level', 0),
                        'type': node.get('node_type', 'unknown')
                    })

            if not nodes_data:
                ax.text(0.5, 0.5, "No node timeline data available",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Node Creation Timeline")
                return

            # Sort by creation time
            nodes_data.sort(key=lambda x: x['created_at'])

            # Normalize timestamps
            start_time = nodes_data[0]['created_at']
            times = [(n['created_at'] - start_time) for n in nodes_data]
            levels = [n['level'] for n in nodes_data]
            types = [n['type'] for n in nodes_data]

            # Create scatter plot
            colors = ['blue' if t == 'leaf' else 'red' for t in types]
            ax.scatter(times, levels, c=colors, alpha=0.6, s=50)

            # Add legend
            blue_patch = plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='blue', markersize=8, label='Leaf nodes')
            red_patch = plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='red', markersize=8, label='Summary nodes')
            ax.legend(handles=[blue_patch, red_patch])

            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Tree Level')
            ax.set_title('Node Creation Timeline')
            ax.grid(True, alpha=0.3)

            # Set integer y-ticks
            if levels:
                ax.set_yticks(range(max(levels) + 1))

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Node Creation Timeline (Error)")

    def _plot_token_heatmap(self, telemetry: dict[str, Any], ax: plt.Axes) -> None:
        """Plot token usage heatmap by node level and type."""
        try:
            # Collect token data by level and type
            token_data: dict[int, dict[str, list[int]]] = {}

            for doc_data in telemetry.get("documents", {}).values():
                for node in doc_data.get("nodes", []):
                    level = node.get('level', 0)
                    node_type = node.get('node_type', 'unknown')

                    if level not in token_data:
                        token_data[level] = {'leaf': [], 'summary': []}

                    # Get token count from embedding or summary data
                    tokens = 0
                    if 'embedding' in node and node['embedding']:
                        tokens = node['embedding'].get('text_tokens', 0)
                    elif 'summary_attempts' in node and node['summary_attempts']:
                        # Use the accepted summary's actual tokens
                        for attempt in node['summary_attempts']:
                            if attempt.get('status') == 'accepted':
                                tokens = attempt.get('actual_tokens', 0)
                                break

                    if tokens > 0 and node_type in token_data[level]:
                        token_data[level][node_type].append(tokens)

            if not token_data:
                ax.text(0.5, 0.5, "No token data available",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Token Usage Heatmap")
                return

            # Prepare data for heatmap
            levels = sorted(token_data.keys())
            types = ['leaf', 'summary']

            # Calculate average tokens for each cell
            heatmap_data: list[list[float]] = []
            for node_type in types:
                row: list[float] = []
                for level in levels:
                    tokens_list: list[int] = token_data[level].get(node_type, [])
                    avg_tokens = float(np.mean(tokens_list)) if tokens_list else 0.0
                    row.append(avg_tokens)
                heatmap_data.append(row)

            # Create heatmap
            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

            # Set ticks and labels
            ax.set_xticks(np.arange(len(levels)))
            ax.set_yticks(np.arange(len(types)))
            ax.set_xticklabels([f"Level {level}" for level in levels])
            ax.set_yticklabels(['Leaf Nodes', 'Summary Nodes'])

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Average Tokens', rotation=270, labelpad=15)

            # Add text annotations
            for i in range(len(types)):
                for j in range(len(levels)):
                    ax.text(j, i, f'{heatmap_data[i][j]:.0f}',
                           ha="center", va="center", color="black" if heatmap_data[i][j] < 50 else "white")

            ax.set_title('Average Token Count by Node Type and Level')
            ax.set_xlabel('Tree Level')

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Token Usage Heatmap (Error)")

    def _generate_markdown_report(self, data: dict[str, Any], telemetry: dict[str, Any], config: RagZoomConfig, chunk_size: int) -> None:
        """Generate a markdown report with analysis summary."""
        try:
            # Compute all metrics
            amplification = compute_amplification_metrics(telemetry, config)
            batch_efficiency = compute_batch_efficiency(telemetry)
            retry_patterns = analyze_retry_patterns(telemetry)
            metrics = compute_metrics_from_telemetry(telemetry, config)

            # Create markdown content
            report = f"""# Telemetry Analysis Report - {chunk_size} Token Chunks

Generated from: {data.get('timestamp', 'Unknown')}

## Executive Summary

- **Total Source Tokens**: {metrics.source_document_tokens:,}
- **Total API Calls**: {metrics.total_api_calls}
- **Total Cost**: ${metrics.cost_per_1k_tokens * (metrics.source_document_tokens / 1000):.4f}
- **Processing Time**: {metrics.total_duration_seconds:.2f} seconds
- **Throughput**: {metrics.tokens_per_second:.1f} tokens/second

## Amplification Analysis

### Overall Metrics
- **Median Cost Amplification**: {amplification['median_cost']:.2f}x
- **90th Percentile Cost**: {amplification['cost_p90']:.2f}x
- **95th Percentile Cost**: {amplification['cost_p95']:.2f}x
- **Median Input Amplification**: {amplification['median_input']:.2f}x
- **Median Output Amplification**: {amplification['median_output']:.2f}x

### By Level
"""
            for level in sorted(amplification.get('by_level', {}).keys()):
                level_data = amplification['by_level'][level]
                report += f"\n#### Level {level}\n"
                report += f"- Input: {np.median(level_data.get('input', [0])):.2f}x\n"
                report += f"- Output: {np.median(level_data.get('output', [0])):.2f}x\n"
                report += f"- Cost: {np.median(level_data.get('cost', [0])):.2f}x\n"

            report += f"""
## Batch Efficiency

- **Total Batches**: {batch_efficiency['total_batches']}
- **Total Embeddings**: {batch_efficiency['total_embeddings']}
- **Average Batch Size**: {batch_efficiency['avg_batch_size']:.1f}
- **Batch Utilization**: {batch_efficiency['batch_utilization']:.1f}%

## Retry Analysis

- **Total Summary Attempts**: {retry_patterns['total_attempts']}
- **Successful Attempts**: {retry_patterns['successful_attempts']}
- **Retry Rate**: {retry_patterns['retry_rate']:.1f}%
- **Retry Success Rate**: {retry_patterns['retry_success_rate']:.1f}%

### Rejection Reasons
"""
            for reason, count in sorted(retry_patterns.get('rejection_reasons', {}).items(),
                                       key=lambda x: x[1], reverse=True):
                report += f"- {reason}: {count} occurrences\n"

            report += """
## Recommendations

"""
            # Add recommendations based on metrics
            if amplification['median_cost'] > self.thresholds.high_cost_amplification:
                report += "- ⚠️ High cost amplification detected. Consider optimizing prompt templates.\n"

            if batch_efficiency['batch_utilization'] < self.thresholds.low_batch_utilization:
                report += "- ⚠️ Low batch utilization. Consider increasing batch sizes for better efficiency.\n"

            if retry_patterns['retry_rate'] > self.thresholds.high_retry_rate:
                report += "- ⚠️ High retry rate. Review summary generation parameters and constraints.\n"

            if amplification['median_cost'] <= self.thresholds.good_cost_amplification and batch_efficiency['batch_utilization'] >= self.thresholds.good_batch_utilization:
                report += "- ✅ System is operating efficiently with good cost control.\n"

            # Save report
            report_path = self.output_dir / f"telemetry_report_{chunk_size}_tokens.md"
            with open(report_path, 'w') as f:
                f.write(report)

            print(f"Saved markdown report to {report_path}")

        except Exception as e:
            print(f"Error generating markdown report: {e}")

    def visualize_comparison(self, benchmark_dir: Path, output_format: str = "png") -> None:
        """Create comparison visualizations across multiple benchmarks."""
        print(f"Loading benchmarks from {benchmark_dir}...")

        # Load all benchmark files
        benchmarks = {}
        for file in benchmark_dir.glob("metrics_*_tokens.json"):
            try:
                data = self.load_benchmark_data(file)
                chunk_size = data["config"]["leaf_tokens"]
                benchmarks[chunk_size] = data
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if len(benchmarks) < 2:
            print("Need at least 2 benchmarks for comparison")
            return

        # Create comparison figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Cost per 1K tokens comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_cost_comparison(benchmarks, ax1)

        # 2. Throughput comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_throughput_comparison(benchmarks, ax2)

        # 3. Amplification comparison
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_amplification_comparison(benchmarks, ax3)

        # 4. Batch efficiency comparison
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_batch_efficiency_comparison(benchmarks, ax4)

        # 5. Retry rate comparison
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_retry_rate_comparison(benchmarks, ax5)

        # 6. Summary accuracy comparison
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_accuracy_comparison(benchmarks, ax6)

        fig.suptitle('Telemetry Comparison Across Chunk Sizes', fontsize=16, y=0.995)

        # Save figure
        output_path = self.output_dir / f"telemetry_comparison.{output_format}"
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved comparison visualization to {output_path}")

    def _plot_cost_comparison(self, benchmarks: dict[int, dict[str, Any]], ax: plt.Axes) -> None:
        """Plot cost comparison across chunk sizes."""
        chunk_sizes = sorted(benchmarks.keys())
        costs = []

        for size in chunk_sizes:
            metrics = benchmarks[size].get("metrics", {})
            efficiency = metrics.get("efficiency", {})
            costs.append(efficiency.get("cost_per_1k_tokens", 0))

        ax.bar(range(len(chunk_sizes)), costs, color='lightcoral', alpha=0.7)
        ax.set_xticks(range(len(chunk_sizes)))
        ax.set_xticklabels([f"{size}" for size in chunk_sizes])
        ax.set_xlabel('Chunk Size (tokens)')
        ax.set_ylabel('Cost per 1K tokens ($)')
        ax.set_title('Cost Efficiency by Chunk Size')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, cost in enumerate(costs):
            ax.text(i, cost + 0.0001, f'${cost:.4f}', ha='center', va='bottom')

    def _plot_throughput_comparison(self, benchmarks: dict[int, dict[str, Any]], ax: plt.Axes) -> None:
        """Plot throughput comparison across chunk sizes."""
        chunk_sizes = sorted(benchmarks.keys())
        throughputs = []

        for size in chunk_sizes:
            metrics = benchmarks[size].get("metrics", {})
            timing = metrics.get("timing", {})
            throughputs.append(timing.get("tokens_per_second", 0))

        ax.plot(chunk_sizes, throughputs, 'o-', color='green', markersize=8, linewidth=2)
        ax.set_xlabel('Chunk Size (tokens)')
        ax.set_ylabel('Tokens per Second')
        ax.set_title('Processing Throughput by Chunk Size')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for size, throughput in zip(chunk_sizes, throughputs):
            ax.annotate(f'{throughput:.1f}', (size, throughput),
                       textcoords="offset points", xytext=(0,10), ha='center')

    def _plot_amplification_comparison(self, benchmarks: dict[int, dict[str, Any]], ax: plt.Axes) -> None:
        """Plot amplification comparison across chunk sizes."""
        chunk_sizes = sorted(benchmarks.keys())

        # Initialize data collectors
        cost_amps = []
        input_amps = []
        output_amps = []

        for size in chunk_sizes:
            if "telemetry" in benchmarks[size]:
                try:
                    config = self._create_config_from_metrics(benchmarks[size].get("metrics", {}))
                    amp_data = compute_amplification_metrics(benchmarks[size]["telemetry"], config)
                    cost_amps.append(amp_data.get("median_cost", 0))
                    input_amps.append(amp_data.get("median_input", 0))
                    output_amps.append(amp_data.get("median_output", 0))
                except Exception:
                    cost_amps.append(0)
                    input_amps.append(0)
                    output_amps.append(0)
            else:
                # Fallback to metrics if telemetry not available
                metrics = benchmarks[size].get("metrics", {})
                amp = metrics.get("amplification", {})
                cost_amps.append(amp.get("median_cost", 0))
                input_amps.append(amp.get("median_input", 0))
                output_amps.append(amp.get("median_output", 0))

        # Plot lines
        x = np.arange(len(chunk_sizes))
        width = 0.25

        ax.bar(x - width, input_amps, width, label='Input', alpha=0.8, color='skyblue')
        ax.bar(x, output_amps, width, label='Output', alpha=0.8, color='lightgreen')
        ax.bar(x + width, cost_amps, width, label='Cost', alpha=0.8, color='salmon')

        ax.set_xlabel('Chunk Size (tokens)')
        ax.set_ylabel('Amplification Factor')
        ax.set_title('Median Amplification Factors by Chunk Size')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in chunk_sizes])
        ax.legend()
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_batch_efficiency_comparison(self, benchmarks: dict[int, dict[str, Any]], ax: plt.Axes) -> None:
        """Plot batch efficiency comparison."""
        chunk_sizes = sorted(benchmarks.keys())
        utilizations = []
        avg_batch_sizes = []

        for size in chunk_sizes:
            if "telemetry" in benchmarks[size]:
                try:
                    batch_data = compute_batch_efficiency(benchmarks[size]["telemetry"])
                    utilizations.append(batch_data.get("batch_utilization", 0))
                    avg_batch_sizes.append(batch_data.get("avg_batch_size", 0))
                except Exception:
                    utilizations.append(0)
                    avg_batch_sizes.append(0)
            else:
                # Fallback to metrics
                metrics = benchmarks[size].get("metrics", {})
                efficiency = metrics.get("efficiency", {})
                avg_batch_sizes.append(efficiency.get("avg_embedding_batch_size", 0))
                utilizations.append(0)  # Not available in old format

        # Create dual y-axis plot
        ax2 = ax.twinx()

        # Plot average batch size as bars
        bars = ax.bar(range(len(chunk_sizes)), avg_batch_sizes, alpha=0.6, color='blue', label='Avg Batch Size')

        # Plot utilization as line
        line = ax2.plot(range(len(chunk_sizes)), utilizations, 'ro-', markersize=8, linewidth=2, label='Utilization %')

        ax.set_xticks(range(len(chunk_sizes)))
        ax.set_xticklabels([str(s) for s in chunk_sizes])
        ax.set_xlabel('Chunk Size (tokens)')
        ax.set_ylabel('Average Batch Size', color='blue')
        ax2.set_ylabel('Batch Utilization (%)', color='red')
        ax.set_title('Embedding Batch Efficiency by Chunk Size')

        # Combine legends
        # bars is a BarContainer, we need to convert it to a list for legend
        bar_patch = bars[0]  # Get the first bar patch for legend
        line_obj = line[0]   # line is a list with one Line2D object
        ax.legend([bar_patch, line_obj], ['Avg Batch Size', 'Utilization %'], loc='upper left')

        ax.grid(True, alpha=0.3, axis='y')

    def _plot_retry_rate_comparison(self, benchmarks: dict[int, dict[str, Any]], ax: plt.Axes) -> None:
        """Plot retry rate comparison."""
        chunk_sizes = sorted(benchmarks.keys())
        retry_rates = []

        for size in chunk_sizes:
            if "telemetry" in benchmarks[size]:
                try:
                    retry_data = analyze_retry_patterns(benchmarks[size]["telemetry"])
                    retry_rates.append(retry_data.get("retry_rate", 0))
                except Exception:
                    retry_rates.append(0)
            else:
                retry_rates.append(0)  # Not available in old format

        ax.bar(range(len(chunk_sizes)), retry_rates, color='orange', alpha=0.7)
        ax.set_xticks(range(len(chunk_sizes)))
        ax.set_xticklabels([str(s) for s in chunk_sizes])
        ax.set_xlabel('Chunk Size (tokens)')
        ax.set_ylabel('Retry Rate (%)')
        ax.set_title('Summary Retry Rate by Chunk Size')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, rate in enumerate(retry_rates):
            ax.text(i, rate + 0.5, f'{rate:.1f}%', ha='center', va='bottom')

        # Add warning line at threshold
        ax.axhline(y=self.thresholds.high_retry_rate, color='red', linestyle='--', alpha=0.5, label=f'{self.thresholds.high_retry_rate}% threshold')
        ax.legend()

    def _plot_accuracy_comparison(self, benchmarks: dict[int, dict[str, Any]], ax: plt.Axes) -> None:
        """Plot summary accuracy comparison."""
        chunk_sizes = sorted(benchmarks.keys())

        # Collect median deviations for each target size
        target_sizes: set[str] = set()
        for size in chunk_sizes:
            metrics = benchmarks[size].get("metrics", {})
            accuracy = metrics.get("summary_accuracy", {})
            target_sizes.update(accuracy.keys())

        if not target_sizes:
            ax.text(0.5, 0.5, "No accuracy data available",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Summary Accuracy Comparison")
            return

        # Plot grouped bars for each target size
        target_sizes_int: list[int] = sorted([int(t) for t in target_sizes if t.isdigit()])
        x = np.arange(len(chunk_sizes))
        width = 0.8 / len(target_sizes_int)

        for i, target in enumerate(target_sizes_int):
            deviations = []
            for chunk_size in chunk_sizes:
                metrics = benchmarks[chunk_size].get("metrics", {})
                accuracy = metrics.get("summary_accuracy", {})
                stats = accuracy.get(str(target), {})
                deviations.append(stats.get("median_deviation_percent", 0))

            offset = (i - len(target_sizes)/2 + 0.5) * width
            ax.bar(x + offset, deviations, width, label=f'{target} tokens', alpha=0.8)

        ax.set_xlabel('Chunk Size (tokens)')
        ax.set_ylabel('Median Deviation (%)')
        ax.set_title('Summary Accuracy by Chunk Size and Target Size')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in chunk_sizes])
        ax.legend(title='Target Size')
        ax.grid(True, alpha=0.3, axis='y')

        # Add reference lines
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=25, color='orange', linestyle='--', alpha=0.5)


def main() -> int:
    """Main entry point for the visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize telemetry data from RagZoom benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a single benchmark
  python scripts/visualize_telemetry.py benchmark_results/metrics_200_tokens.json

  # Visualize all benchmarks in a directory
  python scripts/visualize_telemetry.py benchmark_results/

  # Generate HTML output with comparison
  python scripts/visualize_telemetry.py benchmark_results/ --format html --compare

  # Specify output directory
  python scripts/visualize_telemetry.py benchmark_results/ --output-dir reports/
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input file or directory containing benchmark JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("telemetry_reports"),
        help="Output directory for visualizations (default: telemetry_reports)"
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for visualizations (default: png)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison visualizations when input is a directory"
    )

    args = parser.parse_args()

    # Create visualizer
    visualizer = TelemetryVisualizer(args.output_dir)

    # Process input
    if args.input.is_file():
        # Single file visualization
        visualizer.visualize_single_benchmark(args.input, args.format)
    elif args.input.is_dir():
        # Directory of benchmarks
        json_files = list(args.input.glob("metrics_*_tokens.json"))

        if not json_files:
            print(f"No benchmark files found in {args.input}")
            return 1

        # Visualize each file
        for file in json_files:
            visualizer.visualize_single_benchmark(file, args.format)

        # Generate comparison if requested
        if args.compare and len(json_files) >= 2:
            visualizer.visualize_comparison(args.input, args.format)
    else:
        print(f"Error: {args.input} not found")
        return 1

    print("\nVisualization complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

