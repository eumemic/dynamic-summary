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

The current telemetry format version is **2.0**. The version is stored in all telemetry data to ensure backward compatibility:

```json
{
  "format_version": "2.0",
  "documents": { ... }
}
```

### Version History

- **v1.0**: Initial telemetry format with node-level tracking
- **v2.0**: Improved telemetry format:
  - Removed redundant fields: `is_retry`, `node_type`, `span` fields
  - Renamed `level` to `height` throughout for clarity
  - Added `start_time`/`end_time` to EmbeddingTelemetry and SummaryAttempt
  - Replaced single `timestamp` field with start/end times for precise timing

### Structure

Telemetry data follows this hierarchical structure:

```json
{
  "format_version": "2.0",
  "documents": {
    "<document_type>": {
      "nodes": [
        {
          "node_id": "node-123",
          "height": 0,
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
- **height**: Tree height (0 for leaves, increases up the tree)
- **created_at**: Timestamp when node was created
- **embedding**: (Optional) Embedding generation details
- **summary_attempts**: (Optional) List of summary generation attempts

Note: Node type is derived from height (height 0 = leaf, height > 0 = summary)

### Embedding Telemetry

```json
{
  "embedding": {
    "text_tokens": 150,
    "batch_size": 10,
    "batch_position": 3,
    "model": "text-embedding-3-small",
    "start_time": 1234567891.0,
    "end_time": 1234567891.5
  }
}
```

### Summary Attempt Telemetry

```json
{
  "summary_attempts": [
    {
      "target_tokens": 100,
      "input_text_tokens": 200,
      "prompt_tokens": 250,
      "completion_tokens": 95,
      "actual_tokens": 90,
      "status": "accepted",
      "model": "gpt-4o-mini",
      "start_time": 1234567892.0,
      "end_time": 1234567893.5,
      "rejection_reason": null
    }
  ]
}
```

Note: Whether an attempt is a retry is determined by its position in the array (index 0 = initial attempt, index > 0 = retry)

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
with open("benchmark_results/telemetry_200_tokens.json") as f:
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

# Analyze by height
for height, data in amplification['by_height'].items():
    print(f"\nHeight {height}:")
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
ragzoom-telemetry visualize benchmark_results/telemetry_200_tokens.json
```

This generates:
- Amplification patterns by tree height
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
        if node["height"] > 0:  # Summary nodes have height > 0
            for attempt in node.get("summary_attempts", []):
                if attempt["status"] == "accepted":
                    amp = attempt["prompt_tokens"] / attempt["input_text_tokens"]
                    if amp > 3.0:
                        high_amp_nodes.append({
                            "node_id": node["node_id"],
                            "height": node["height"],
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

### CLI Integration

RagZoom provides a comprehensive set of telemetry commands through the CLI:

```bash
# Analyze telemetry data
ragzoom-telemetry analyze benchmark_results/telemetry_200_tokens.json

# Compare two benchmarks
ragzoom-telemetry compare baseline.json current.json

# Generate visualizations
ragzoom-telemetry visualize benchmark_results/telemetry_200_tokens.json
ragzoom-telemetry visualize benchmark_results/ --compare --format pdf
```

### Configuration Options

#### Threshold Configuration

The telemetry analysis tools use configurable thresholds for identifying performance issues and generating recommendations. These can be customized via environment variables:

```bash
# Analysis thresholds (defaults shown)
export RAGZOOM_HIGH_INPUT_AMPLIFICATION_THRESHOLD=3.0       # High input amplification warning
export RAGZOOM_HIGH_COST_AMPLIFICATION_THRESHOLD=2.0        # High cost amplification warning  
export RAGZOOM_GOOD_COST_AMPLIFICATION_THRESHOLD=1.5        # Good cost amplification target
export RAGZOOM_HIGH_RETRY_RATE_THRESHOLD=20                 # High retry rate warning (%)
export RAGZOOM_GOOD_BATCH_UTILIZATION_THRESHOLD=70          # Good batch utilization target (%)
export RAGZOOM_LOW_BATCH_UTILIZATION_THRESHOLD=50           # Low batch utilization warning (%)
export RAGZOOM_MULTIPLE_RETRY_THRESHOLD=1                   # Multiple retry detection
```

**Examples:**

```bash
# Use stricter thresholds for production monitoring
export RAGZOOM_HIGH_COST_AMPLIFICATION_THRESHOLD=1.8
export RAGZOOM_HIGH_RETRY_RATE_THRESHOLD=15
ragzoom telemetry analyze production_metrics.json

# Relaxed thresholds for development
export RAGZOOM_HIGH_COST_AMPLIFICATION_THRESHOLD=3.0
export RAGZOOM_HIGH_RETRY_RATE_THRESHOLD=30
ragzoom telemetry visualize dev_metrics.json
```

These thresholds affect:
- Warning messages in analysis reports
- Colored indicators in visualizations  
- Threshold lines on charts
- Recommendation generation

### Future: Query Telemetry

For debugging query issues, telemetry collection during queries is planned:

```bash
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
   - Use migration tools to update old data (if available)
   - v2.0 changes: removed redundant fields, added timing, renamed level→height

### Migration from v1.0 to v2.0

The analysis tools support both v1.0 and v2.0 formats automatically. When processing v1.0 data:
- `node_type` field is used if present
- `level` is treated as `height`
- Single `timestamp` fields are used for timing analysis

No manual migration is required - the tools handle version differences transparently.

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