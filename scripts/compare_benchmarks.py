#!/usr/bin/env python3
"""Compare benchmark results between branches and generate markdown report."""

import json
import sys
from pathlib import Path
from typing import Dict, Optional


def load_benchmark_results(results_dir: Path) -> Dict[int, dict]:
    """Load benchmark results from JSON files."""
    results = {}
    
    for file in results_dir.glob("metrics_*_tokens.json"):
        try:
            with open(file) as f:
                data = json.load(f)
                chunk_size = data["config"]["leaf_tokens"]
                results[chunk_size] = data
        except Exception as e:
            print(f"Error loading {file}: {e}", file=sys.stderr)
    
    return results


def calculate_change(old_value: float, new_value: float) -> tuple[float, str]:
    """Calculate percentage change and return (percentage, emoji)."""
    if old_value == 0:
        return 0, ""
    
    change = ((new_value - old_value) / old_value) * 100
    
    # Determine emoji based on metric type and direction
    if abs(change) < 1:
        emoji = ""
    elif change > 0:
        # For cost and time metrics, increase is bad
        emoji = "⚠️"
    else:
        # For cost and time metrics, decrease is good
        emoji = "✅"
    
    return change, emoji


def format_value(value: float, metric_type: str) -> str:
    """Format value based on metric type."""
    if metric_type == "cost":
        return f"${value:.4f}"
    elif metric_type == "percent":
        return f"{value:.1f}%"
    elif metric_type == "time":
        return f"{value:.2f}s"
    else:
        return f"{value:.1f}"


def generate_comparison_table(
    baseline: Dict[int, dict],
    current: Dict[int, dict],
    output_format: str = "markdown"
) -> str:
    """Generate comparison table between baseline and current results."""
    
    # Get all chunk sizes present in both sets
    chunk_sizes = sorted(set(baseline.keys()) & set(current.keys()))
    
    if not chunk_sizes:
        return "❌ No matching chunk sizes found between baseline and current results"
    
    lines = []
    
    # Header
    lines.append("## 📊 Performance Report\n")
    
    # Throughput comparison
    lines.append("### Throughput Comparison")
    lines.append("| Chunk Size | Baseline | Current | Change |")
    lines.append("|------------|----------|---------|--------|")
    
    throughput_regression = False
    
    for size in chunk_sizes:
        base_tps = baseline[size]["metrics"]["timing"]["tokens_per_second"]
        curr_tps = current[size]["metrics"]["timing"]["tokens_per_second"]
        change, emoji = calculate_change(base_tps, curr_tps)
        
        # For throughput, higher is better, so flip the emoji logic
        if change > 0:
            emoji = "✅"
        elif change < -10:  # More than 10% regression
            emoji = "⚠️"
            throughput_regression = True
        elif change < 0:
            emoji = ""
        
        lines.append(
            f"| {size} tokens | {base_tps:.1f} tok/s | {curr_tps:.1f} tok/s | "
            f"{emoji} {change:+.1f}% |"
        )
    
    # Token usage comparison
    lines.append("\n### Token Usage (per 1K source tokens)")
    lines.append("| Chunk Size | Metric | Baseline | Current | Change |")
    lines.append("|------------|--------|----------|---------|--------|")
    
    cost_regression = False
    
    for size in chunk_sizes:
        base_m = baseline[size]["metrics"]["efficiency"]
        curr_m = current[size]["metrics"]["efficiency"]
        
        # Embedding tokens
        base_embed = base_m["embedding_tokens_per_1k"]
        curr_embed = curr_m["embedding_tokens_per_1k"]
        change, emoji = calculate_change(base_embed, curr_embed)
        
        lines.append(
            f"| {size} tokens | Embedding | {base_embed:.1f} | {curr_embed:.1f} | "
            f"{emoji if abs(change) > 1 else ''} {change:+.1f}% |"
        )
        
        # Summary tokens
        base_summary = base_m["summary_tokens_per_1k"]
        curr_summary = curr_m["summary_tokens_per_1k"]
        change, emoji = calculate_change(base_summary, curr_summary)
        
        lines.append(
            f"| | Summary | {base_summary:.1f} | {curr_summary:.1f} | "
            f"{emoji if abs(change) > 1 else ''} {change:+.1f}% |"
        )
        
        # Total cost
        base_cost = base_m["cost_per_1k_tokens"]
        curr_cost = curr_m["cost_per_1k_tokens"]
        change, emoji = calculate_change(base_cost, curr_cost)
        
        if change > 10:  # More than 10% cost increase
            cost_regression = True
            emoji = "❌"
        
        lines.append(
            f"| | **Total Cost** | ${base_cost:.4f} | ${curr_cost:.4f} | "
            f"{emoji if abs(change) > 1 else ''} {change:+.1f}% |"
        )
    
    # Summary accuracy if available
    if any("summary_accuracy" in current[size]["metrics"] for size in chunk_sizes):
        lines.append("\n### Summary Size Accuracy")
        lines.append("| Chunk Size | Avg Deviation | Over Target | Under Target |")
        lines.append("|------------|---------------|-------------|--------------|")
        
        for size in chunk_sizes:
            if "summary_accuracy" not in current[size]["metrics"]:
                continue
                
            # Get summary stats for the chunk size (same as target)
            stats_dict = current[size]["metrics"]["summary_accuracy"]
            if str(size) in stats_dict:
                stats = stats_dict[str(size)]
                lines.append(
                    f"| {size} tokens | {stats['avg_deviation_percent']:.1f}% | "
                    f"{stats['percent_over_target']:.1f}% | "
                    f"{stats['percent_under_target']:.1f}% |"
                )
    
    # Summary
    lines.append("\n### Summary")
    
    issues = []
    if throughput_regression:
        issues.append("⚠️ Throughput regression detected (>10% decrease)")
    if cost_regression:
        issues.append("❌ Cost regression detected (>10% increase)")
    
    if issues:
        lines.extend(issues)
    else:
        lines.append("✅ No significant performance regressions detected")
    
    return "\n".join(lines)


def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 3:
        print("Usage: python compare_benchmarks.py <baseline_dir> <current_dir> [output_file]")
        print("Example: python compare_benchmarks.py baseline_results/ current_results/ report.md")
        sys.exit(1)
    
    baseline_dir = Path(sys.argv[1])
    current_dir = Path(sys.argv[2])
    output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # Load results
    baseline_results = load_benchmark_results(baseline_dir)
    current_results = load_benchmark_results(current_dir)
    
    if not baseline_results:
        print(f"Error: No benchmark results found in {baseline_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not current_results:
        print(f"Error: No benchmark results found in {current_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Generate comparison
    report = generate_comparison_table(baseline_results, current_results)
    
    # Output
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report written to {output_file}")
    else:
        print(report)
    
    # Exit with error code if regressions detected
    if "❌" in report or "⚠️" in report:
        sys.exit(1)


if __name__ == "__main__":
    main()