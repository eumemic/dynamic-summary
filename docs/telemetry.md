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
from ragzoom.telemetry_analysis import (
    compute_simplified_metrics,
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

### Computing Simplified Metrics

The simplified metrics system provides actionable insights at the chunk-size level:

```python
result = compute_simplified_metrics(telemetry, config)

for chunk_size, metrics in result.metrics_by_chunk_size.items():
    print(f"\nChunk size: {chunk_size} tokens")
    
    # Target-fit metrics (how close summaries are to target)
    print(f"  Target-fit median error: {metrics['target_fit']['median_error']:.1f}%")
    print(f"  Target-fit p95 error: {metrics['target_fit']['p95_error']:.1f}%")
    print(f"  Within 10% of target: {metrics['target_fit']['percent_within_10']:.1f}%")
    
    # Retry metrics
    print(f"  Retry rate: {metrics['retries']['retry_rate']:.1f}%")
    print(f"  Max retries: {metrics['retries']['max_retries']}")
    
    # Cost metrics (actual USD costs)
    print(f"  Avg cost per node: ${metrics['cost']['avg_cost_per_node']:.6f}")
    print(f"  Total cost: ${metrics['cost']['total_cost']:.4f}")
    
    # Latency metrics
    print(f"  Avg latency: {metrics['latency']['avg_ms']:.0f}ms")
    print(f"  P95 latency: {metrics['latency']['p95_ms']:.0f}ms")
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
# Compute metrics from telemetry data
metrics = compute_metrics_from_telemetry(telemetry, config)

print(f"Total duration: {metrics.total_duration_seconds:.2f}s")
print(f"Tokens per second: {metrics.tokens_per_second:.1f}")
print(f"Cost per 1K tokens: ${metrics.cost_per_1k_tokens:.4f}")
print(f"Peak memory: {metrics.peak_memory_mb:.1f} MB")
```

## Visualization Tools

### Basic Visualization

The `visualize` command supports two modes:

#### Single File Visualization

```bash
ragzoom-telemetry visualize telemetry.json
```

This generates a comprehensive report with:
- Token usage and cost by tree level
- Cost breakdown pie chart
- Batch efficiency histogram
- Retry pattern analysis
- Summary accuracy distribution
- Node creation timeline
- Token count distributions by level

#### Side-by-Side Comparison

Compare two telemetry files directly:

```bash
ragzoom-telemetry visualize baseline.json current.json
```

This creates a side-by-side visualization showing both telemetry results in parallel, making it easy to spot differences in:
- Performance characteristics
- Cost efficiency
- Retry patterns
- Token distributions

The plots share scales where appropriate for direct visual comparison.

### Output Options

```bash
# Default output: visualization.png in current directory
ragzoom-telemetry visualize telemetry.json

# Specify custom output path
ragzoom-telemetry visualize baseline.json -o analysis.png
ragzoom-telemetry visualize baseline.json current.json -o comparison.pdf

# Format is inferred from extension, or use --format
ragzoom-telemetry visualize telemetry.json --format pdf
ragzoom-telemetry visualize telemetry.json -o report --format svg
```

### Generated Reports

The visualization script also generates markdown reports with:
- Executive summary of key metrics
- Target-fit accuracy analysis
- Batch efficiency statistics
- Retry pattern breakdown
- Cost analysis per chunk size
- Actionable recommendations

## Performance Optimization

### Identifying Bottlenecks

1. **Poor Target-Fit** (>20% median error)
   - Summaries consistently missing target size
   - Review prompt instructions for clarity
   - Consider adjusting chunk size targets

2. **Low Batch Utilization** (<50%)
   - Increase `embedding_batch_size` in config
   - Ensure documents are large enough to fill batches

3. **High Retry Rate** (>20%)
   - Review summary size constraints
   - Adjust prompt instructions
   - Consider more flexible target ranges

4. **High Cost per Node** (track over time)
   - Compare costs across chunk sizes
   - Identify optimal chunk size for your use case
   - Monitor for cost regressions

### Optimization Strategies

```python
# Example: Find nodes with poor target-fit
from ragzoom.telemetry_analysis import parse_telemetry_format

parsed = parse_telemetry_format(telemetry)
poor_fit_nodes = []

for doc_data in parsed["documents"].values():
    chunk_size = doc_data.get("metadata", {}).get("chunk_size", 100)
    for node in doc_data["nodes"]:
        if node["height"] > 0:  # Summary nodes have height > 0
            for attempt in node.get("summary_attempts", []):
                if attempt["status"] == "accepted":
                    actual = attempt["actual_tokens"]
                    error_pct = abs((actual - chunk_size) / chunk_size * 100)
                    if error_pct > 20:
                        poor_fit_nodes.append({
                            "node_id": node["node_id"],
                            "height": node["height"],
                            "error_pct": error_pct,
                            "actual": actual,
                            "target": chunk_size
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

# Generate visualizations for a single file
ragzoom-telemetry visualize baseline.json

# Generate side-by-side comparison of two files
ragzoom-telemetry visualize baseline.json current.json

# Specify output format and path
ragzoom-telemetry visualize before.json after.json -o reports/comparison.pdf
```

### Comparison Output Format

The `compare` command provides a detailed performance comparison between baseline and current telemetry with:

#### Inline Variance Display
Metrics now show variance inline using the ±format:
- `50.0 ±2.0 tok` - Median error of 50 tokens with MAD (Median Absolute Deviation) of 2
- `$0.0010 ±0.0001` - Cost with variance

#### Two-Line Change Format
Changes are displayed on two lines for clarity:
```
Line 1: [emoji] absolute_change (percentage%)
Line 2: [emoji] σ±variance_change (percentage%)
```

Example:
```
🟢 -247.0 tok (-58.7%)
🟡 σ+18 (+30%)
```

#### Color-Coded Significance Indicators

**Metric Changes:**
- 🔴 = **Regression detected** - Exceeds dynamic threshold (>5σ baseline variance)
- 🟡 = **Significant undesirable change** - Notable but not a regression (>1σ)
- 🟢 = **Significant improvement** - Desirable change (>1σ)
- ⚪ = **Insignificant change** - Within normal variance (<1σ)

**Variance Changes:**
- 🟡 = **Variance increase** - Notable but doesn't trigger regression
- 🟢 = **Variance decrease** - Improved stability
- ⚪ = **Insignificant variance change** - Within normal fluctuation (<50% of baseline)

#### Dynamic Thresholds

RagZoom uses **variance-based dynamic thresholds** instead of fixed percentages:

```
threshold = (k1 + k2) × baseline_variance

Where:
- k1 = 3.0 (covers 99.7% of normal distribution for between-run variance)
- k2 = 2.0 (additional margin for baseline measurement uncertainty)
- Total = 5σ threshold (>99.99% confidence for regression detection)
```

This approach:
- **Eliminates false positives** from natural LLM non-determinism
- **Adapts to each metric's inherent variability**
- **CI environments** get 1.5x multiplier for higher variance

#### Example Output

```
Performance Comparison Report
================================================================================

Chunk Size | Metric              |         Baseline |          Current | Change                                     | Threshold
-----------|---------------------|------------------|------------------|--------------------------------------------|-----------
100 tok    | Median Error        |   -23.0 ±2.0 tok |   -24.2 ±2.4 tok | ⚪ +0.0 tok (+0.0%)                        | 10.0 tok
           |                     |                  |                  | 🟡 σ+0.4 (+20%)                            |
100 tok    | p95 Error           |   +36.0 ±5.0 tok |   +59.0 ±7.0 tok | 🟡 +23.0 tok (+63.9%)                      | 25.0 tok
           |                     |                  |                  | 🟡 σ+2.0 (+40%)                            |
100 tok    | Cost                | $0.0002 ±0.0000  | $0.0002 ±0.0000  | ⚪ +$0.0000 (+0.5%)                        | $0.0001
           |                     |                  |                  | ⚪ σ-0.0000 (-2%)                           |
```

### Configuration Options

#### Threshold Configuration

The telemetry analysis tools use configurable thresholds for identifying performance issues and generating recommendations. These can be customized via environment variables:

```bash
# Analysis thresholds (defaults shown)
export RAGZOOM_HIGH_TARGET_FIT_ERROR_THRESHOLD=20           # High target-fit error warning (%)
export RAGZOOM_GOOD_TARGET_FIT_THRESHOLD=10                 # Good target-fit threshold (%)
export RAGZOOM_HIGH_RETRY_RATE_THRESHOLD=20                 # High retry rate warning (%)
export RAGZOOM_GOOD_BATCH_UTILIZATION_THRESHOLD=70          # Good batch utilization target (%)
export RAGZOOM_LOW_BATCH_UTILIZATION_THRESHOLD=50           # Low batch utilization warning (%)
export RAGZOOM_MULTIPLE_RETRY_THRESHOLD=1                   # Multiple retry detection
export RAGZOOM_HIGH_COST_PER_NODE_THRESHOLD=0.001           # High cost per node warning ($)
```

**Examples:**

```bash
# Use stricter thresholds for production monitoring
export RAGZOOM_HIGH_TARGET_FIT_ERROR_THRESHOLD=15
export RAGZOOM_HIGH_RETRY_RATE_THRESHOLD=15
ragzoom-telemetry analyze production_metrics.json

# Relaxed thresholds for development
export RAGZOOM_HIGH_TARGET_FIT_ERROR_THRESHOLD=30
export RAGZOOM_HIGH_RETRY_RATE_THRESHOLD=30
ragzoom-telemetry visualize dev_metrics.json
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
   - Track cost per node across chunk sizes
   - Monitor total costs for budget management
   - Compare costs between different models

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