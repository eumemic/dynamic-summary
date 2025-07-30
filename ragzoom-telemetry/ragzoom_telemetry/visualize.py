"""Telemetry visualization command."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click

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

    def load_benchmark_data(self, file_path: Path) -> Dict[str, Any]:
        """Load benchmark data from JSON file."""
        with open(file_path) as f:
            data: Dict[str, Any] = json.load(f)
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

    def _create_config_from_metrics(self, metrics: Dict[str, Any]) -> RagZoomConfig:
        """Create a config object from metrics data for cost calculations."""
        return RagZoomConfig(
            openai_api_key="dummy",  # Not needed for analysis
            embedding_cost_per_1k=0.0001,  # text-embedding-3-small (older pricing)
            summary_input_cost_per_1k=0.0025,  # gpt-4o-mini input (older pricing)
            summary_output_cost_per_1k=0.01,   # gpt-4o-mini output (older pricing)
        )

    def _plot_amplification_by_level(self, telemetry: Dict, config: RagZoomConfig, ax: plt.Axes) -> None:
        """Plot amplification metrics by tree level."""
        amplification = compute_amplification_metrics(telemetry, config)
        
        if not amplification["by_level"]:
            ax.text(0.5, 0.5, 'No amplification data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Amplification by Tree Level")
            return

        levels = sorted(amplification["by_level"].keys())
        cost_medians = []
        input_medians = []
        output_medians = []
        
        for level in levels:
            level_data = amplification["by_level"][level]
            cost_medians.append(np.median(level_data["cost"]) if level_data["cost"] else 0)
            input_medians.append(np.median(level_data["input"]) if level_data["input"] else 0)
            output_medians.append(np.median(level_data["output"]) if level_data["output"] else 0)

        x = np.arange(len(levels))
        width = 0.25

        bars1 = ax.bar(x - width, cost_medians, width, label='Cost', alpha=0.8)
        bars2 = ax.bar(x, input_medians, width, label='Input', alpha=0.8)
        bars3 = ax.bar(x + width, output_medians, width, label='Output', alpha=0.8)

        ax.set_xlabel('Tree Level')
        ax.set_ylabel('Amplification Factor')
        ax.set_title('Amplification Metrics by Tree Level')
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add threshold line
        ax.axhline(y=self.thresholds.high_cost_amplification, color='r', 
                  linestyle='--', alpha=0.5, label='High threshold')

    def _plot_cost_breakdown(self, telemetry: Dict, config: RagZoomConfig, ax: plt.Axes) -> None:
        """Plot cost breakdown pie chart."""
        metrics = compute_metrics_from_telemetry(telemetry, config)
        
        embedding_cost = metrics.total_embedding_cost
        summary_cost = metrics.total_summary_cost
        
        if embedding_cost == 0 and summary_cost == 0:
            ax.text(0.5, 0.5, 'No cost data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Cost Breakdown")
            return

        costs = [embedding_cost, summary_cost]
        labels = ['Embeddings', 'Summaries']
        colors = ['#ff9999', '#66b3ff']
        
        wedges, texts, autotexts = ax.pie(costs, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Cost Breakdown (Total: ${sum(costs):.4f})')

    def _plot_batch_efficiency(self, telemetry: Dict, ax: plt.Axes) -> None:
        """Plot batch efficiency metrics."""
        batch_eff = compute_batch_efficiency(telemetry)
        
        if not batch_eff["batch_sizes"]:
            ax.text(0.5, 0.5, 'No batch data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Batch Efficiency")
            return

        batch_sizes = batch_eff["batch_sizes"]
        
        ax.hist(batch_sizes, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(batch_eff["avg_batch_size"], color='red', linestyle='--', 
                  label=f'Avg: {batch_eff["avg_batch_size"]:.1f}')
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Embedding Batch Distribution (Utilization: {batch_eff["batch_utilization"]:.1f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_retry_patterns(self, telemetry: Dict, ax: plt.Axes) -> None:
        """Plot retry pattern analysis."""
        retry_data = analyze_retry_patterns(telemetry)
        
        categories = ['Total Attempts', 'Successful', 'Retries']
        values = [
            retry_data["total_attempts"],
            retry_data["successful_attempts"],
            retry_data["retry_attempts"]
        ]
        
        bars = ax.bar(categories, values, alpha=0.8)
        
        # Color code bars
        bars[0].set_color('#90cdf4')  # Total - blue
        bars[1].set_color('#86efac')  # Successful - green
        bars[2].set_color('#fca5a5')  # Retries - red
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        ax.set_ylabel('Count')
        ax.set_title(f'Summary Retry Patterns (Retry Rate: {retry_data["retry_rate"]:.1f}%)')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_summary_accuracy(self, metrics: Dict, ax: plt.Axes) -> None:
        """Plot summary accuracy distribution."""
        if "summary_accuracy" not in metrics or not metrics["summary_accuracy"]:
            ax.text(0.5, 0.5, 'No summary accuracy data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Summary Accuracy Distribution")
            return

        # Extract deviation data
        deviations = []
        for target_size, stats in metrics["summary_accuracy"].items():
            if "deviations" in stats:
                deviations.extend(stats["deviations"])

        if not deviations:
            ax.text(0.5, 0.5, 'No deviation data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Summary Accuracy Distribution")
            return

        # Create histogram
        ax.hist(deviations, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='green', linestyle='--', label='Target', linewidth=2)
        
        # Add median line
        median_dev = np.median(deviations)
        ax.axvline(median_dev, color='red', linestyle='--', 
                  label=f'Median: {median_dev:.1f}%', linewidth=2)
        
        ax.set_xlabel('Deviation from Target (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Summary Token Count Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_node_timeline(self, telemetry: Dict, ax: plt.Axes) -> None:
        """Plot node creation timeline."""
        # Extract node creation times
        creation_times = []
        for doc_data in telemetry.get("documents", {}).values():
            for node in doc_data.get("nodes", []):
                if "created_at" in node:
                    creation_times.append(node["created_at"])

        if not creation_times:
            ax.text(0.5, 0.5, 'No timeline data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Node Creation Timeline")
            return

        creation_times.sort()
        min_time = creation_times[0]
        relative_times = [(t - min_time) for t in creation_times]
        
        ax.plot(relative_times, range(len(relative_times)), alpha=0.8)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Cumulative Nodes Created')
        ax.set_title('Node Creation Timeline')
        ax.grid(True, alpha=0.3)

    def _plot_token_heatmap(self, telemetry: Dict, ax: plt.Axes) -> None:
        """Plot token usage heatmap by level and node."""
        # This is a simplified version - you could make it more sophisticated
        amplification = compute_amplification_metrics(telemetry, RagZoomConfig())
        
        if not amplification["by_level"]:
            ax.text(0.5, 0.5, 'No token usage data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Token Usage Heatmap")
            return

        # Create a simple heatmap of cost amplification by level
        levels = sorted(amplification["by_level"].keys())
        data_matrix = []
        
        for level in levels:
            level_data = amplification["by_level"][level]
            # Take first 10 nodes or pad with zeros
            costs = level_data["cost"][:10] if "cost" in level_data else []
            while len(costs) < 10:
                costs.append(0)
            data_matrix.append(costs)

        if data_matrix:
            im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd')
            ax.set_xlabel('Node Index')
            ax.set_ylabel('Tree Level')
            ax.set_title('Cost Amplification Heatmap')
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels(levels)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Cost Amplification Factor')

    def _generate_markdown_report(self, data: Dict, telemetry: Dict, 
                                 config: RagZoomConfig, chunk_size: int) -> None:
        """Generate a markdown report alongside visualizations."""
        report_path = self.output_dir / f"telemetry_report_{chunk_size}_tokens.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Telemetry Report - {chunk_size} Token Chunks\n\n")
            
            # Add metrics summary
            amplification = compute_amplification_metrics(telemetry, config)
            batch_eff = compute_batch_efficiency(telemetry)
            retry_patterns = analyze_retry_patterns(telemetry)
            
            f.write("## Summary Metrics\n\n")
            f.write(f"- **Median Cost Amplification**: {amplification['median_cost']:.2f}x\n")
            f.write(f"- **Batch Utilization**: {batch_eff['batch_utilization']:.1f}%\n")
            f.write(f"- **Retry Rate**: {retry_patterns['retry_rate']:.1f}%\n")
            f.write("\n")
            
            f.write("## Visualizations\n\n")
            f.write(f"![Telemetry Analysis](telemetry_{chunk_size}_tokens.png)\n")

        print(f"Saved markdown report to {report_path}")

    def visualize_comparison(self, results_dir: Path, output_format: str = "png") -> None:
        """Create comparison visualizations between multiple benchmarks."""
        json_files = list(results_dir.glob("metrics_*_tokens.json"))
        
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
        fig.suptitle('Benchmark Comparison', fontsize=16)

        # Plot comparisons
        self._plot_comparison_metrics(benchmarks, axes)

        plt.tight_layout()
        output_path = self.output_dir / f"comparison.{output_format}"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved comparison visualization to {output_path}")

    def _plot_comparison_metrics(self, benchmarks: Dict[int, Dict], axes: np.ndarray) -> None:
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
            total_costs.append(metrics.total_cost)

        # Plot 1: Cost Amplification
        ax = axes[0, 0]
        ax.plot(chunk_sizes, cost_amps, 'o-', markersize=8)
        ax.set_xlabel('Chunk Size (tokens)')
        ax.set_ylabel('Median Cost Amplification')
        ax.set_title('Cost Amplification vs Chunk Size')
        ax.grid(True, alpha=0.3)

        # Plot 2: Batch Utilization
        ax = axes[0, 1]
        ax.plot(chunk_sizes, batch_utils, 'o-', markersize=8, color='green')
        ax.set_xlabel('Chunk Size (tokens)')
        ax.set_ylabel('Batch Utilization (%)')
        ax.set_title('Batch Utilization vs Chunk Size')
        ax.grid(True, alpha=0.3)

        # Plot 3: Retry Rate
        ax = axes[1, 0]
        ax.plot(chunk_sizes, retry_rates, 'o-', markersize=8, color='red')
        ax.set_xlabel('Chunk Size (tokens)')
        ax.set_ylabel('Retry Rate (%)')
        ax.set_title('Retry Rate vs Chunk Size')
        ax.grid(True, alpha=0.3)

        # Plot 4: Total Cost
        ax = axes[1, 1]
        ax.plot(chunk_sizes, total_costs, 'o-', markersize=8, color='purple')
        ax.set_xlabel('Chunk Size (tokens)')
        ax.set_ylabel('Total Cost ($)')
        ax.set_title('Total Cost vs Chunk Size')
        ax.grid(True, alpha=0.3)


@click.command("visualize")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    type=click.Path(),
    default="telemetry_reports",
    help="Output directory for visualizations",
)
@click.option(
    "--format",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    help="Output format (default: png)",
)
@click.option(
    "--compare",
    is_flag=True,
    help="Generate comparison visualizations when input is a directory",
)
def visualize(
    input_path: str, output_dir: str, format: str, compare: bool
) -> None:
    """Generate visualizations from telemetry data."""
    try:
        visualizer = TelemetryVisualizer(Path(output_dir))
        input_path_obj = Path(input_path)

        if input_path_obj.is_file():
            # Single file visualization
            visualizer.visualize_single_benchmark(input_path_obj, format)
        elif input_path_obj.is_dir():
            # Directory of benchmarks
            json_files = list(input_path_obj.glob("metrics_*_tokens.json"))
            # Also support new telemetry.json files
            json_files.extend(input_path_obj.glob("telemetry*.json"))

            if not json_files:
                click.echo(f"❌ No benchmark files found in {input_path}")
                sys.exit(1)

            # Visualize each file
            for file in json_files:
                visualizer.visualize_single_benchmark(file, format)

            # Generate comparison if requested
            if compare and len(json_files) >= 2:
                visualizer.visualize_comparison(input_path_obj, format)
        else:
            click.echo(f"❌ Error: {input_path} not found")
            sys.exit(1)

        click.echo("\n✅ Visualization complete!")

    except Exception as e:
        click.echo(f"❌ Error generating visualizations: {e}", err=True)
        sys.exit(1)