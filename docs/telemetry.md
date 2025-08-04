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

The current telemetry format version is **3.0**. The version is stored in all telemetry data to ensure backward compatibility:

```json
{
  "format_version": "3.0",
  "document_id": "example.pdf",
  ...
}
```

### Version History

- **v1.0**: Initial telemetry format with node-level tracking
- **v2.0**: Improved telemetry format:
  - Removed redundant fields: `is_retry`, `node_type`, `span` fields
  - Renamed `level` to `height` throughout for clarity
  - Added `start_time`/`end_time` to EmbeddingTelemetry and SummaryAttempt
  - Replaced single `timestamp` field with start/end times for precise timing
- **v3.0**: Flattened structure to eliminate redundancy:
  - Removed nested "documents" dict (always single document)
  - Moved metadata fields to top level
  - Added models field at top level
  - Eliminated duplicate document_id and chunk_size fields

### Structure

Telemetry data follows this flat structure in v3.0:

```json
{
  "format_version": "3.0",
  "document_id": "example.pdf",
  "source_document_tokens": 7500,
  "chunk_size": 200,
  "indexed_at": 1234567890.0,
  "models": {
    "summary": "gpt-4o-mini",
    "embedding": "text-embedding-3-small"
  },
  "nodes": [
    {
      "node_id": "node-123",
      "height": 0,
      "created_at": 1234567890.0,
      "embedding": { ... },
      "summary_attempts": [ ... ]
    }
  ]
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

Visualize a single benchmark:

```bash
ragzoom-telemetry visualize benchmark_results/telemetry_200_tokens.json
```

This generates:
- Token usage and cost by tree level
- Cost breakdown pie chart
- Batch efficiency histogram
- Retry pattern analysis
- Summary accuracy distribution
- Node creation timeline
- Token count distributions by level

### Comparison Visualization

Compare multiple benchmarks:

```bash
ragzoom-telemetry visualize benchmark_results/ --compare
```

### Output Formats

```bash
# Generate PDF reports
ragzoom-telemetry visualize benchmark_results/ --format pdf

# Specify output directory
ragzoom-telemetry visualize benchmark_results/ --output-dir reports/
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

# Generate visualizations
ragzoom-telemetry visualize benchmark_results/telemetry_200_tokens.json
ragzoom-telemetry visualize benchmark_results/ --compare --format pdf
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

### Automatic Migration to v3.0

All RagZoom telemetry analysis tools automatically migrate v1.0 and v2.0 formats to v3.0 during parsing:

- **v1.0/v2.0 → v3.0**: Automatic migration extracts metadata to top level, flattens structure
- **CLI wrapper format**: Handles old CLI output that wrapped telemetry in config/document/telemetry structure
- **Models inference**: Attempts to extract model names from nodes if not provided
- **Backward compatibility**: All v1.0 and v2.0 fields are preserved

No manual migration is required - the tools handle version differences transparently.

### Writing Compatible Analysis Code

```python
from ragzoom.telemetry_analysis import parse_telemetry_format

def analyze_telemetry(telemetry_data):
    """Example of version-agnostic analysis."""
    # Always returns v3.0 format
    parsed = parse_telemetry_format(telemetry_data)
    
    # Access data consistently regardless of input version
    doc_id = parsed["document_id"]
    nodes = parsed["nodes"]
    models = parsed["models"]
    source_tokens = parsed["source_document_tokens"]
    
    # Process nodes uniformly
    for node in nodes:
        height = node["height"]  # Works for all versions
        if node.get("embedding"):
            # Process embedding data
            pass
        if node.get("summary_attempts"):
            # Process summary data
            pass
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