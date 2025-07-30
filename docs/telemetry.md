# RagZoom Telemetry Guide

## Overview

RagZoom's telemetry system provides detailed insights into the indexing process by capturing node-level metrics during document processing. This data enables performance analysis, cost optimization, and debugging without requiring expensive re-indexing operations.

## Table of Contents

1. [Telemetry Data Format](#telemetry-data-format)
2. [Using Telemetry for Analysis](#using-telemetry-for-analysis)
3. [Visualization Tools](#visualization-tools)
4. [Performance Optimization](#performance-optimization)
5. [Debugging with Telemetry](#debugging-with-telemetry)
6. [Format Migration](#format-migration)

## Telemetry Data Format

### Format Version

The current telemetry format version is **1.0**. The version is stored in all telemetry data to ensure backward compatibility:

```json
{
  "format_version": "1.0",
  "documents": { ... }
}
```

### Structure

Telemetry data follows this hierarchical structure:

```json
{
  "format_version": "1.0",
  "documents": {
    "<document_type>": {
      "nodes": [
        {
          "node_id": "node-123",
          "node_type": "leaf|summary",
          "level": 0,
          "span": [start, end],
          "created_at": 1234567890.0,
          "embedding": { ... },
          "summary_attempts": [ ... ]
        }
      ],
      "metadata": {
        "total_nodes": 100,
        "leaf_nodes": 64,
        "summary_nodes": 36,
        "source_document_tokens": 7500
      }
    }
  }
}
```

### Node Telemetry

Each node contains:

- **node_id**: Unique identifier
- **node_type**: Either "leaf" or "summary"
- **level**: Tree level (0 for leaves, increases up the tree)
- **span**: Character positions [start, end] in source document
- **created_at**: Timestamp when node was created
- **embedding**: (Optional) Embedding generation details
- **summary_attempts**: (Optional) List of summary generation attempts

### Embedding Telemetry

```json
{
  "embedding": {
    "text_tokens": 150,
    "batch_size": 10,
    "batch_position": 3,
    "model": "text-embedding-3-small",
    "timestamp": 1234567891.0
  }
}
```

### Summary Attempt Telemetry

```json
{
  "summary_attempts": [
    {
      "is_retry": false,
      "target_tokens": 100,
      "input_text_tokens": 200,
      "prompt_tokens": 250,
      "completion_tokens": 95,
      "actual_tokens": 90,
      "status": "accepted",
      "model": "gpt-4o-mini",
      "timestamp": 1234567892.0,
      "rejection_reason": null
    }
  ]
}
```

Status values:
- `"accepted"`: Summary met constraints
- `"rejected_over"`: Summary exceeded target size
- `"rejected_under"`: Summary fell short of target
- `"error"`: API or processing error

## Using Telemetry for Analysis

### Loading Telemetry Data

```python
import json
from ragzoom.telemetry import (
    compute_amplification_metrics,
    analyze_retry_patterns,
    compute_batch_efficiency,
    compute_metrics_from_telemetry
)
from ragzoom.config import RagZoomConfig

# Load benchmark data
with open("benchmark_results/metrics_200_tokens.json") as f:
    data = json.load(f)
    
telemetry = data["telemetry"]
config = RagZoomConfig(
    openai_api_key="your-key",
    embedding_cost_per_1k=0.0001,
    summary_input_cost_per_1k=0.0025,
    summary_output_cost_per_1k=0.01
)
```

### Computing Amplification Metrics

Amplification metrics show how much overhead is introduced by the API calls:

```python
amplification = compute_amplification_metrics(telemetry, config)

print(f"Median cost amplification: {amplification['median_cost']:.2f}x")
print(f"90th percentile cost: {amplification['cost_p90']:.2f}x")
print(f"Median input amplification: {amplification['median_input']:.2f}x")
print(f"Median output amplification: {amplification['median_output']:.2f}x")

# Analyze by level
for level, data in amplification['by_level'].items():
    print(f"\nLevel {level}:")
    print(f"  Input: {np.median(data['input']):.2f}x")
    print(f"  Output: {np.median(data['output']):.2f}x")
    print(f"  Cost: {np.median(data['cost']):.2f}x")
```

### Analyzing Batch Efficiency

```python
batch_efficiency = compute_batch_efficiency(telemetry)

print(f"Total batches: {batch_efficiency['total_batches']}")
print(f"Average batch size: {batch_efficiency['avg_batch_size']:.1f}")
print(f"Batch utilization: {batch_efficiency['batch_utilization']:.1f}%")
```

### Retry Pattern Analysis

```python
retry_patterns = analyze_retry_patterns(telemetry)

print(f"Retry rate: {retry_patterns['retry_rate']:.1f}%")
print(f"Retry success rate: {retry_patterns['retry_success_rate']:.1f}%")
print("\nRejection reasons:")
for reason, count in retry_patterns['rejection_reasons'].items():
    print(f"  {reason}: {count}")
```

### Full Metrics Reconstruction

```python
# Reconstruct complete IndexingMetrics from telemetry
metrics = compute_metrics_from_telemetry(telemetry, config)

print(f"Total duration: {metrics.total_duration_seconds:.2f}s")
print(f"Tokens per second: {metrics.tokens_per_second:.1f}")
print(f"Cost per 1K tokens: ${metrics.cost_per_1k_tokens:.4f}")
print(f"Peak memory: {metrics.peak_memory_mb:.1f} MB")
```

## Visualization Tools

### Basic Visualization

Visualize a single benchmark:

```bash
python scripts/visualize_telemetry.py benchmark_results/metrics_200_tokens.json
```

This generates:
- Amplification patterns by tree level
- Cost breakdown pie chart
- Batch efficiency histogram
- Retry pattern analysis
- Summary accuracy distribution
- Node creation timeline
- Token usage heatmap

### Comparison Visualization

Compare multiple benchmarks:

```bash
python scripts/visualize_telemetry.py benchmark_results/ --compare
```

### Output Formats

```bash
# Generate PDF reports
python scripts/visualize_telemetry.py benchmark_results/ --format pdf

# Specify output directory
python scripts/visualize_telemetry.py benchmark_results/ --output-dir reports/
```

### Generated Reports

The visualization script also generates markdown reports with:
- Executive summary of key metrics
- Detailed amplification analysis
- Batch efficiency statistics
- Retry pattern breakdown
- Actionable recommendations

## Performance Optimization

### Identifying Bottlenecks

1. **High Cost Amplification** (>2.0x)
   - Indicates inefficient prompt templates
   - Check for unnecessary context in prompts
   - Consider prompt optimization

2. **Low Batch Utilization** (<50%)
   - Increase `embedding_batch_size` in config
   - Ensure documents are large enough to fill batches

3. **High Retry Rate** (>20%)
   - Review summary size constraints
   - Adjust prompt instructions
   - Consider more flexible target ranges

### Optimization Strategies

```python
# Example: Find nodes with highest amplification
from ragzoom.telemetry import parse_telemetry_format

parsed = parse_telemetry_format(telemetry)
high_amp_nodes = []

for doc_data in parsed["documents"].values():
    for node in doc_data["nodes"]:
        if node["node_type"] == "summary":
            for attempt in node.get("summary_attempts", []):
                if attempt["status"] == "accepted":
                    amp = attempt["prompt_tokens"] / attempt["input_text_tokens"]
                    if amp > 3.0:
                        high_amp_nodes.append({
                            "node_id": node["node_id"],
                            "level": node["level"],
                            "amplification": amp
                        })
```

## Debugging with Telemetry

### Common Issues

1. **Unexpected Token Counts**
   ```python
   # Check for outliers in token usage
   for doc_data in parsed["documents"].values():
       for node in doc_data["nodes"]:
           if "embedding" in node:
               tokens = node["embedding"]["text_tokens"]
               if tokens > expected_max or tokens < expected_min:
                   print(f"Outlier: {node['node_id']} has {tokens} tokens")
   ```

2. **Retry Failures**
   ```python
   # Find nodes with multiple failed retries
   problem_nodes = []
   for doc_data in parsed["documents"].values():
       for node in doc_data["nodes"]:
           attempts = node.get("summary_attempts", [])
           if len(attempts) > 2:  # Multiple retries
               problem_nodes.append(node)
   ```

3. **Batch Inefficiency**
   ```python
   # Analyze batch size distribution
   batch_sizes = []
   for doc_data in parsed["documents"].values():
       for node in doc_data["nodes"]:
           if "embedding" in node:
               batch_sizes.append(node["embedding"]["batch_size"])
   
   print(f"Batch size distribution: {np.histogram(batch_sizes, bins=10)}")
   ```

### Adding Telemetry to Query Operations

For debugging query issues, you can enable telemetry collection:

```python
# Future feature - not yet implemented
ragzoom query "search text" --document-id doc1 --telemetry

# This would output telemetry showing:
# - Which nodes were retrieved
# - Relevance scores
# - Token budget allocation
# - Assembly decisions
```

## Format Migration

### Handling Format Changes

When telemetry format changes occur:

1. **Minor Version Changes** (e.g., 1.0 → 1.1)
   - Backward compatible additions
   - Old analysis tools continue to work
   - New fields are optional

2. **Major Version Changes** (e.g., 1.0 → 2.0)
   - Breaking changes requiring migration
   - Use migration tools to update old data

### Migration Example

```python
# Future migration tool usage
python scripts/migrate_telemetry.py old_benchmark.json --to-version 2.0

# Or migrate entire directory
python scripts/migrate_telemetry.py benchmark_results/ --to-version 2.0
```

### Writing Compatible Analysis Code

```python
def analyze_telemetry_safely(telemetry_data):
    """Example of version-aware analysis."""
    version = telemetry_data.get("format_version", "1.0")
    
    if version == "1.0":
        return analyze_v1_telemetry(telemetry_data)
    elif version.startswith("2."):
        return analyze_v2_telemetry(telemetry_data)
    else:
        raise ValueError(f"Unsupported telemetry version: {version}")
```

## Best Practices

1. **Regular Benchmarking**
   - Run benchmarks after significant changes
   - Compare telemetry to detect regressions
   - Use CI integration for automated checks

2. **Cost Monitoring**
   - Set up alerts for cost amplification > 3.0x
   - Track cost trends over time
   - Optimize high-cost operations first

3. **Performance Tracking**
   - Monitor throughput (tokens/second)
   - Track memory usage patterns
   - Identify scaling bottlenecks

4. **Debugging Workflow**
   - Start with visualization overview
   - Drill down to specific metrics
   - Use telemetry to reproduce issues
   - Validate fixes with new benchmarks

## Future Enhancements

Planned telemetry features:
- Real-time telemetry streaming
- Query operation telemetry
- Advanced anomaly detection
- Telemetry-based optimization suggestions
- Integration with monitoring systems

For the latest updates, see the [GitHub repository](https://github.com/tom-p-reichel/ragzoom).