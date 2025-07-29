# RagZoom Benchmarks

This directory contains performance benchmarking tools for RagZoom.

## Overview

The benchmarking infrastructure has been improved to provide:
- More realistic testing with actual documents
- Faster CI through saved baseline artifacts
- Memory usage tracking
- Standalone scripts for local benchmarking

## Running Benchmarks

### Standalone Script

```bash
# Run with default settings (100, 200, 400 token chunks)
python benchmarks/run_indexing_benchmark.py

# Custom chunk sizes
python benchmarks/run_indexing_benchmark.py --chunk-sizes 50,100,200,400,800

# Custom output directory
python benchmarks/run_indexing_benchmark.py --output-dir my_results/
```

### Pytest Benchmarks

```bash
# Run all benchmarks
pytest tests/benchmarks/ -v

# Run specific chunk size
pytest tests/benchmarks/test_indexing_performance.py -k "leaf_tokens==200"
```

## CI Integration

### Baseline Generation

The `baseline-benchmarks.yml` workflow runs on every push to master and:
1. Runs benchmarks with standard settings
2. Saves results as GitHub Actions artifacts
3. Artifacts are retained for 90 days

### PR Performance Testing

The `performance.yml` workflow runs on every PR and:
1. Downloads the latest baseline artifacts (if available)
2. Falls back to generating baselines if artifacts are missing
3. Runs benchmarks on the PR branch
4. Compares results and comments on the PR
5. Fails if regression thresholds are exceeded

### Regression Thresholds

Default thresholds (configurable via environment variables):
- Throughput: 10% decrease triggers warning
- Cost: 10% increase triggers warning

Set custom thresholds:
```bash
export PERF_THROUGHPUT_REGRESSION_THRESHOLD=15.0
export PERF_COST_REGRESSION_THRESHOLD=5.0
```

## Metrics Collected

### Performance Metrics
- **Throughput**: Tokens processed per second
- **Time per 1K tokens**: Processing time normalized by document size
- **API calls**: Total calls and breakdown by type

### Cost Analysis
- **Embedding costs**: Based on model pricing
- **Summary costs**: Input and output token costs
- **Total cost per 1K tokens**: Normalized cost metric

### Memory Usage
- **Peak memory**: Maximum RSS during indexing
- **Memory growth**: Difference from start to peak
- **Start/end memory**: For leak detection

### Accuracy Metrics
- **Summary size accuracy**: How well summaries match target sizes
- **Deviation percentages**: Over/under target statistics

## Test Documents

Benchmarks use real documents from `test_data/`:
- **The Hobbit Chapter 1** (~7K tokens): Narrative prose
- **Technical Documentation**: Code and technical content
- **Moby Dick Sample**: Classic literature

## Output Format

Results are saved as JSON files compatible with the comparison script:
```json
{
  "config": {
    "leaf_tokens": 200,
    "embedding_model": "text-embedding-3-small",
    "summary_model": "gpt-4o-mini",
    "document": "The Hobbit Ch1"
  },
  "metrics": {
    "timing": { ... },
    "document": { ... },
    "api_usage": { ... },
    "efficiency": { ... },
    "memory": { ... }
  },
  "timestamp": 1234567890.123
}
```

## Future Improvements

- Query performance benchmarks
- Visualization tools for results
- Historical trend tracking
- Multiple document size testing (1K, 10K, 100K tokens)
- Concurrent operation benchmarks