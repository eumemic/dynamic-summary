---
name: ragzoom-development
description: This skill should be used when the user asks to "run tests", "run checks", "debug a test", "benchmark performance", "set up development environment", "understand the architecture", "how does indexing work", "how does the tiling algorithm work", "study the codebase", "learn the codebase", or mentions testing, linting, type checking, or development workflows.
---

# RagZoom Development

Guidance for developing, testing, and understanding the RagZoom codebase.

## Quick Start: Understanding the Codebase

The core algorithm lives in `ragzoom/greedy_tiling.py`. Start there to understand how RagZoom works:

1. `GreedyTilingGenerator.find_optimal_tiling_over_roots()` - main entry point
2. Starts with all leaves, rolls up least-valuable sibling pairs until within budget
3. Priority: `quality_lost / tokens_saved` (lower = better to roll up)

For a complete learning path, see **`references/codebase-guide.md`**.

## Quality Checks

**Most checks run automatically - manual runs are rarely needed.**

### Automatic Checks
- **On every Python edit**: `dmypy`, `ruff`, and `black` (~750ms)
- **On every commit**: Pre-commit hook runs all checks

### Manual Commands

```bash
# Run quality checks (excludes integration/benchmarks)
./scripts/run-checks.sh

# Include integration tests
./scripts/run-checks.sh --include-integration-tests

# Test files impacted by specific changes
./scripts/run-checks.sh --impacted-only path/to/changed.py

# Stop at first error
./scripts/run-checks.sh --fail-fast
```

**Never use `pytest` directly** - use `run-checks.sh` for proper environment setup.

## Type Safety

Strict type checking is enforced:

- `strict = true` and `disallow_any_explicit = true` in mypy
- All functions, methods, and class attributes require type hints
- **Never add `# type: ignore`** without explicit user permission
- Tests are type-checked as strictly as production code

Debug type errors:
```bash
dmypy run -- ragzoom/
# Or for a clean check:
dmypy stop && mypy ragzoom --ignore-missing-imports --no-error-summary
```

## Code Duplication

Zero-duplication policy enforced by jscpd:
```bash
npx jscpd@latest ragzoom/
```

Mark legitimate false positives:
```python
# jscpd:ignore-start
def retrieve(self, ...):  # Intentional: async/sync wrapper pattern
# jscpd:ignore-end
```

## Error Handling

Follow the "No Fallback Code" principle - fail hard with clear messages:

```python
# Do:
if not validate_email(email):
    raise ValidationError(field="email", value=email, reason="invalid format")

# Don't:
try:
    validate_data()
except Exception:
    pass  # Silent failure!
```

Use exceptions from `ragzoom.exceptions`:
- `ValidationError`, `DatabaseError`, `LLMError`, `ConfigurationError`
- `NodeNotFoundError`, `DocumentNotFoundError`

## Benchmarking

```bash
./scripts/run-indexing-benchmarks --baseline baseline.json document.txt
```

Outputs in `benchmarks/`: `telemetry.json`, `log.txt`, `comparison.md`

## Protobuf Generation

After modifying `proto/dynamic_summary.proto`:
```bash
./scripts/compile-proto.sh
```

Generated stubs in `ragzoom/rpc/` are committed but never edited manually.

## Docker Dev Stack

```bash
export OPENAI_API_KEY="sk-..."
./scripts/devstack start      # Start stack
./scripts/devstack logs       # Tail logs
./scripts/devstack exec-cli index README.md --document-id readme
./scripts/devstack stop       # Tear down
./scripts/devstack watch      # Hot reload mode
```

## Reference Files

For detailed documentation:
- **`references/codebase-guide.md`** - Systematic learning path, module structure, test organization
- **`references/architecture.md`** - System components, data flow, design principles
- **`references/tiling-algorithm.md`** - Deep dive into the greedy tiling algorithm
