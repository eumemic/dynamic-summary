# Performance Baseline Storage

This directory contains performance baseline telemetry files used for regression detection and historical analysis.

## Directory Structure

```
benchmarks/
├── baselines/
│   ├── telemetry-2025-01-22-143052-abc123f.json  # Historical baselines
│   ├── telemetry-2025-01-22-144030-def456a.json
│   ├── ...
│   └── latest.json -> telemetry-2025-01-22-144030-def456a.json  # Symlink to latest
└── README.md
```

## File Naming Convention

Baseline files follow the pattern:
```
telemetry-YYYY-MM-DD-HHMMSS-<short-commit>.json
```

- **YYYY-MM-DD-HHMMSS**: UTC timestamp when the baseline was generated
- **short-commit**: First 7 characters of the Git commit hash
- **latest.json**: Symlink pointing to the most recent baseline

## How Baselines Are Generated

1. **On every merge to master**: The `baseline-benchmarks.yml` workflow runs
2. **Indexing benchmark**: Runs `run-indexing-benchmarks` with the full Moby Dick text
3. **Storage**: Saves the telemetry as a new baseline file
4. **Cleanup**: Automatically keeps only the 100 most recent baselines
5. **Commit**: Pushes the new baseline directly to master

## Using Baselines Locally

### Automatic Baseline Detection

The `run-indexing-benchmarks` script automatically finds baselines in this order:

1. `benchmarks/baselines/latest.json` (preferred)
2. `benchmarks/latest/telemetry.json` (previous run)
3. `./telemetry.json` (current directory)
4. Repository root variations (for worktrees)

### Run a Comparison

```bash
# Uses latest repository baseline automatically
./scripts/run-indexing-benchmarks test_data/moby_dick_ci.txt

# Use a specific baseline
./scripts/run-indexing-benchmarks \
  --baseline benchmarks/baselines/telemetry-2025-01-22-143052-abc123f.json \
  test_data/moby_dick_ci.txt

# Compare against different document or settings
./scripts/run-indexing-benchmarks \
  --baseline benchmarks/baselines/latest.json \
  --target-chunk-tokens 400 \
  test_data/custom_document.txt
```

## Git LFS

Baseline files are stored using Git LFS to prevent repository bloat:

- **Storage**: ~880KB per baseline file
- **Retention**: 100 most recent baselines (~88MB total)
- **Free tier**: GitHub provides 1GB LFS storage, sufficient for >1000 baselines

### Setting Up Git LFS (for new contributors)

```bash
# Install Git LFS (if not already installed)
git lfs install

# Verify LFS files are tracked
git lfs ls-files

# Clone with LFS files
git lfs pull
```

## Historical Analysis

With 100 baselines providing ~3 months of history, you can:

### Analyze Performance Trends

```bash
# Extract metrics from all baselines
for file in benchmarks/baselines/telemetry-*.json; do
  echo -n "$(basename $file .json | cut -d- -f2-4): "
  jq -r '.telemetry | map(select(.operation_type == "index_document")) | .[0].duration_seconds' "$file"
done

# Compare specific time periods
ragzoom-telemetry compare \
  benchmarks/baselines/telemetry-2025-01-15-*.json \
  benchmarks/baselines/latest.json
```

### Investigate Regressions

```bash
# Find when a regression was introduced
git log --oneline benchmarks/baselines/ | head -10
```

## Maintenance

The system is designed to be maintenance-free:

- **Automatic cleanup**: Keeps exactly 100 baselines
- **Self-managing**: No manual intervention required
- **Git LFS pruning**: Use `git lfs prune` to clean up orphaned objects

## Performance Targets

Current performance benchmarks use:
- **Document**: Moby Dick CI slice (~514KB, ~75K tokens — size-matched to the former corpus so baseline scales stay comparable)
- **Chunk size**: 200 tokens (from default_config.json)
- **Comparison**: Uses dynamic thresholds based on baseline variance
- **CI frequency**: Every merge to master