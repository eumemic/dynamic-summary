# Testing Performance Optimization

## Overview

RagZoom's test suite has been optimized for fast local development while maintaining test fidelity. We use a mock storage layer for unit tests and reserve real I/O operations for integration tests.

## Performance Improvements

- **Unit tests**: 4.5x faster using SimpleMockStore (5.38s → 1.20s per test)
- **Full test suite**: 2.1x faster (23.15s → 10.88s)
- **Pre-commit hooks**: Run only fast tests (~7.5s)

## Test Categories

### Fast Tests (Default)
- Use SimpleMockStore for in-memory operations
- Run on every commit via pre-commit hook
- Complete in < 8 seconds

### Integration Tests (@pytest.mark.integration)
- Use real SQLite + ChromaDB
- Test actual I/O operations
- Run in CI/CD pipeline

### Benchmark Tests (@pytest.mark.benchmark)
- Tests that take > 5 seconds with real store
- Currently: test_incomplete_indexing.py
- Excluded from pre-commit, run in CI

## Running Tests

```bash
# Run fast tests only (pre-commit default)
pytest tests/ -m "not benchmark and not integration" -n 8

# Run all tests including integration
pytest tests/ -n 8

# Run with real store for all tests
pytest tests/ --use-real-store -n 8

# Run only integration tests
pytest tests/ -m integration --use-real-store
```

## Git Hooks

### Pre-commit Hook
1. Runs fast tests (excludes integration and benchmarks)
2. Runs linting (ruff + black) 
3. Runs type checking (mypy - warning only)
4. Takes ~8-10 seconds total

### Pre-push Hook
- Removed - all checks now in pre-commit

## Mock Store

The SimpleMockStore (`tests/mock_store.py`) provides:
- Full tree structure operations
- Basic vector search simulation
- State tracking (dirty nodes, pinning)
- Embedding dimension validation
- Compatible with all Store methods used in tests

## Best Practices

1. **New tests should use mock store by default** - just use the `store` fixture
2. **Mark I/O-heavy tests as @pytest.mark.integration** - they'll use real store automatically
3. **Tag performance tests as @pytest.mark.benchmark** - excluded from pre-commit
4. **Use --use-real-store flag** to verify tests work with real storage
