# ragzoom-telemetry

**⚠️ Developer Tools Only**: This package contains telemetry analysis tools for RagZoom developers and CI/CD pipelines. End users should use the main `ragzoom` package.

## Overview

`ragzoom-telemetry` provides specialized tools for analyzing performance telemetry data collected during RagZoom indexing operations. These tools are primarily used for:

- Performance regression detection in CI/CD
- Benchmarking different configurations
- Visualizing system behavior and costs
- Debugging performance issues

## Installation

```bash
# From the ragzoom repository
pip install ./ragzoom-telemetry

# Or for development
cd ragzoom-telemetry
pip install -e .
```

## Dependencies

This package has heavy dependencies not needed by end users:
- `matplotlib` - For visualization generation
- `seaborn` - For statistical plots
- `pandas` - For data analysis
- `numpy` - For numerical operations
- `ragzoom` - For accessing telemetry analysis functions

## Usage

### Analyze Telemetry Data

Generate a text report from telemetry data:

```bash
ragzoom-telemetry analyze telemetry.json

# Save report to file
ragzoom-telemetry analyze telemetry.json --output report.txt
```

The analyze command provides:
- Amplification metrics (cost, input, output)
- Batch efficiency statistics
- Retry patterns and success rates

### Compare Benchmarks

Compare two telemetry files for performance regression detection:

```bash
ragzoom-telemetry compare baseline.json current.json

# Save comparison report
ragzoom-telemetry compare baseline.json current.json --output comparison.md
```

Exit codes:
- 0: No regression detected
- 1: Performance regression detected

Regression thresholds can be configured via environment variables:
- `PERF_SUMMARY_TOKEN_REGRESSION_THRESHOLD` (default: 10.0%)
- `PERF_AVG_DEVIATION_REGRESSION_THRESHOLD` (default: 20.0%)
- `PERF_MEDIAN_DEVIATION_REGRESSION_THRESHOLD` (default: 20.0%)
- `PERF_STD_DEVIATION_REGRESSION_THRESHOLD` (default: 30.0%)
- `PERF_P95_REGRESSION_THRESHOLD` (default: 25.0%)

### Visualize Performance

Generate comprehensive visualizations:

```bash
# Single file visualization
ragzoom-telemetry visualize telemetry.json

# Directory of benchmarks
ragzoom-telemetry visualize benchmark_results/

# Generate comparison charts
ragzoom-telemetry visualize benchmark_results/ --compare

# Different output formats
ragzoom-telemetry visualize telemetry.json --format pdf
ragzoom-telemetry visualize telemetry.json --format svg

# Custom output directory
ragzoom-telemetry visualize telemetry.json --output-dir my_reports/
```

Visualizations include:
1. Amplification metrics by tree level
2. Cost breakdown (embeddings vs summaries)
3. Batch efficiency distribution
4. Retry patterns
5. Summary accuracy distribution
6. Node creation timeline
7. Token usage heatmap

## Telemetry Format Support

This package supports both telemetry format versions:
- **v1.0**: Initial format with comprehensive node tracking
- **v2.0**: Improved format with better timing precision (PR #50)

Key differences handled automatically:
- `level` → `height` (v2.0)
- Removed fields in v2.0: `is_retry`, `node_type`, `span_start`, `span_end`
- `timestamp` → `start_time` + `end_time` (v2.0)

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
- name: Install telemetry tools
  run: pip install ./ragzoom-telemetry

- name: Run benchmarks
  run: |
    ragzoom index document.txt --telemetry telemetry.json

- name: Compare with baseline
  run: |
    ragzoom-telemetry compare baseline.json telemetry.json --output report.md || echo "REGRESSION=true" >> $GITHUB_OUTPUT

- name: Generate visualizations
  run: |
    ragzoom-telemetry visualize telemetry.json --output-dir reports/
```

## Development

To work on ragzoom-telemetry:

```bash
cd ragzoom-telemetry
pip install -e .
```

Run tests (when available):
```bash
pytest tests/
```

## Telemetry Thresholds

Analysis thresholds can be configured via environment variables:
- `RAGZOOM_HIGH_INPUT_AMPLIFICATION_THRESHOLD` (default: 3.0)
- `RAGZOOM_HIGH_COST_AMPLIFICATION_THRESHOLD` (default: 2.0)
- `RAGZOOM_GOOD_COST_AMPLIFICATION_THRESHOLD` (default: 1.5)
- `RAGZOOM_HIGH_RETRY_RATE_THRESHOLD` (default: 20%)
- `RAGZOOM_GOOD_BATCH_UTILIZATION_THRESHOLD` (default: 70%)
- `RAGZOOM_LOW_BATCH_UTILIZATION_THRESHOLD` (default: 50%)

## Related Tools

The main `ragzoom` package provides:
- `ragzoom index --telemetry` - Collect telemetry during indexing
- `ragzoom.telemetry` module - Core analysis functions used by this package

## License

Same as the main RagZoom project.