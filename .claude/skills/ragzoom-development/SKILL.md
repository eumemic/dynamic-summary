---
name: ragzoom-development
description: This skill should be used when the user asks to "run tests", "run checks", "debug a test", "benchmark performance", "set up development environment", "understand the architecture", "how does indexing work", "how does the tiling algorithm work", or mentions testing, linting, type checking, or development workflows.
---

# RagZoom Development

Guidance for developing, testing, and understanding the RagZoom codebase.

## Quality Checks

**Most checks run automatically - manual runs are rarely needed.**

### Automatic Checks
- **On every Python edit**: `dmypy`, `ruff`, and `black` run automatically (~750ms)
- **On every commit**: Pre-commit hook runs all checks (tests, linting, formatting, security, duplication)

### When to Run Manually

1. **Don't run checks before committing** - the pre-commit hook handles this. Just commit and let it fail if there are issues.
2. **To test specific functionality during development**: Use `./scripts/run-checks.sh`
3. **Never use `pytest` directly** - use `run-checks.sh` which ensures proper environment setup

### Common Commands

```bash
# Run quality checks (excludes integration/benchmarks by default)
./scripts/run-checks.sh

# Include integration tests
./scripts/run-checks.sh --include-integration-tests

# Run only tests impacted by specific files
./scripts/run-checks.sh --impacted-only path/to/changed1.py path/to/changed2.py

# Stop at first error (useful for debugging)
./scripts/run-checks.sh --fail-fast
```

## Type Safety

The codebase enforces strict type checking:

- `strict = true` in mypy configuration
- `disallow_any_explicit = true` - no explicit `Any` types
- All functions, methods, and class attributes must have type hints
- **Never add `# type: ignore`** without explicit user permission
- Tests are type-checked as strictly as production code

To debug type errors:
```bash
dmypy run -- ragzoom/
# Or for a clean check:
dmypy stop && mypy ragzoom --ignore-missing-imports --no-error-summary
```

## Code Duplication

Zero-duplication policy enforced by jscpd. Run manually:
```bash
npx jscpd@latest ragzoom/
```

Mark legitimate false positives (like async/sync wrappers) with:
```python
# jscpd:ignore-start
def retrieve(self, ...):  # Intentional: async/sync wrapper pattern
# jscpd:ignore-end
```

## Error Handling Standards

Follow the "No Fallback Code" principle:

**Do:**
```python
if not validate_email(email):
    raise ValidationError(field="email", value=email, reason="invalid format")
```

**Don't:**
```python
try:
    validate_data()
except Exception:
    pass  # Silent failure!
```

Use specific exceptions from `ragzoom.exceptions`:
- `ValidationError` - Data validation failures
- `DatabaseError` - Database operation failures
- `LLMError` - AI service failures
- `ConfigurationError` - Configuration issues
- `NodeNotFoundError`, `DocumentNotFoundError` - Domain errors

## Benchmarking

Run performance benchmarks:
```bash
./scripts/run-indexing-benchmarks --baseline baseline.json document.txt
```

Outputs in `benchmarks/`:
- `telemetry.json` - Raw telemetry data
- `log.txt` - Complete indexing logs
- `comparison.md` - Performance comparison report

## Protobuf Code Generation

After modifying `proto/dynamic_summary.proto`:
```bash
./scripts/compile-proto.sh
```

Generated stubs in `ragzoom/rpc/` are committed but never edited manually.

## Docker Dev Stack

Run the full system (gRPC, REST API, inspector UI):
```bash
export OPENAI_API_KEY="sk-..."
./scripts/devstack start      # Start stack
./scripts/devstack logs       # Tail logs
./scripts/devstack exec-cli index README.md --document-id readme
./scripts/devstack stop       # Tear down
```

For hot reload during development:
```bash
./scripts/devstack watch
```

## Additional Resources

### Reference Files

For detailed architecture and algorithm documentation:
- **`references/architecture.md`** - System components, data flow, design principles
- **`references/tiling-algorithm.md`** - Deep dive into the DP/greedy tiling algorithm

### Test Organization

| Source File | Test File | Coverage Focus |
|-------------|-----------|----------------|
| `store.py` | `test_store.py` | Database operations, caching |
| `index.py` | `test_indexing_fast.py` | Tree building logic |
| `retrieve.py` | `test_retrieve.py` | Query processing |
| `greedy_tiling.py` | `test_greedy_tiling.py` | Tiling algorithm |
| `assemble.py` | `test_assemble.py` | Summary assembly |
| `api.py` | `test_api_*.py` | REST endpoints |
