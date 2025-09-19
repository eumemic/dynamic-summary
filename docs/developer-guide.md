# Developer Guide

**Last Updated**: January 2025

Welcome to the RagZoom project! This comprehensive guide covers the technology stack, development environment setup, testing strategies, and best practices for contributing to the project.

## 1. Technology Stack

-   **Core Language:** Python 3.11+
-   **API Framework:** FastAPI
-   **CLI Framework:** `click`
-   **Database:** PostgreSQL with the `pgvector` extension storing both metadata and embeddings.
-   **LLM Interaction:** `openai` for generating summaries and embeddings.
-   **Tokenization:** `tiktoken` for accurately counting tokens.

## 2. Development Environment Setup

Getting your environment set up correctly is the most important first step.

### 2.1 Backend & Data Layout (Update)

- Default backend: SQLite (no Docker). Files are created under:
  - Database: `data/sqlite.db`
  - Vector index (Chroma): `data/chroma/` (requires `chromadb`)
- The `data/` directory is ignored by Git.
- Switch to PostgreSQL by setting `RAGZOOM_BACKEND=postgres` and `RAGZOOM_DATABASE_URL`.
- Document‑level lock prevents concurrent indexing of the same document (lock files under `data/.ragzoom/locks/`).

1.  **Create a Virtual Environment:** It is strongly recommended to use a Python virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
2.  **Run the Setup Script:** The `scripts/setup-dev.sh` script is the one-stop shop for getting everything installed and configured.
    ```bash
    ./scripts/setup-dev.sh
    ```
    This script will:
    - Install all base and development requirements from `requirements.txt` and `requirements-dev.txt`.
    - Install the `ragzoom` package in editable mode (`pip install -e .`).
    - Create a `.env` file from the example for you to add your API keys.
    - **Crucially, it sets up the Git pre-commit hooks.**

## 3. Testing Strategy

### 3.1. Test Organization

Our test suite is organized to balance thoroughness with development speed:

- **Unit Tests**: Fast tests focused on functional correctness
- **Integration Tests**: End-to-end tests that exercise full app wiring (marked with `@pytest.mark.integration`)
- **Concurrency Tests**: Thread safety and concurrent behavior

For the complete list of test markers and pytest configuration options, see [API Reference - Testing](api-reference.md#testing).

### 3.2. Running Tests

```bash
# Preferred: use the unified checks script
./scripts/run-checks.sh                       # Default: excludes integration and benchmarks
./scripts/run-checks.sh --include-integration-tests  # Include integration tests (benchmarks still excluded)

# Run pytest directly (advanced)
pytest tests/ -m "not benchmark and not integration" -n 8
pytest tests/ -m "not benchmark" -n 8  # include integration

# Run specific test file or test
pytest tests/test_integration.py -n 8
pytest tests/test_integration.py::TestIntegration::test_token_budget_enforcement -vv -n 8
```

### 3.3. Test Performance

- **Mock Store Performance**: 4.5x faster than real store (1.20s vs 5.38s per test)
- **Full Suite**: ~137 tests complete in ~8.5 seconds with 8 workers
- **Pre-commit**: Runs only fast tests to keep commits quick

### 3.4. Coverage Analysis

```bash
# Generate coverage report
pytest tests/ --cov=ragzoom --cov-report=html

# View in browser
open htmlcov/index.html

# Quick coverage summary
pytest tests/ --cov=ragzoom --cov-report=term
```

### 3.5. Debugging Failed Tests

```bash
# Drop into debugger on failure
pytest tests/ --pdb

# Show local variables on failure
pytest tests/ -l

# Run with logging enabled
pytest tests/ -s --log-cli-level=DEBUG

# Profile specific operations
python -m cProfile -o profile.stats ragzoom/index.py
```

## 4. Development Process & Tooling

We use a suite of tools to ensure code quality and consistency. These are run automatically by the pre-commit hook, so it's important to understand what they do.

### 4.1. Pre-Commit Hook

The pre-commit hook is defined in `scripts/git-hooks/pre-commit` and is the guardian of our codebase quality. Before any commit is finalized, it runs the following checks in parallel:

-   **Tests (`pytest`):** Runs only tests downstream of staged files (`--impacted-only <staged files>`). Defaults exclude integration and benchmarks.
-   **Formatting (`black`):** Automatically reformats your code to be consistent.
-   **Linting (`ruff`):** Checks for common errors, style issues, and automatically fixes what it can.
-   **Type Checking (`mypy`):** Statically analyzes type hints to catch potential bugs.

**Important:** The hook is configured to **auto-fix** formatting and simple linting errors. After it runs, it will re-stage any files it modified. If there are still errors (e.g., a failing test or a complex `mypy` error), the commit will be aborted, and you will need to fix the issues manually.

### 4.2. Running Checks Manually

You can and should run these checks yourself as you code:
-   **Run all quality checks:** `./scripts/run-checks.sh` (excludes integration/benchmarks by default)
-   **Include integration tests:** `./scripts/run-checks.sh --include-integration-tests`
-   **Run only downstream tests for files:** `./scripts/run-checks.sh --impacted-only path/to/file1.py path/to/file2.py`
-   **Skip specific checks:** `./scripts/run-checks.sh --skip tests,jscpd`
-   **Stop at first error:** `./scripts/run-checks.sh --fail-fast`

Dev tools (optional, speeds up duplicate detection):
-   Install jscpd globally: `npm install -g jscpd` (optional; avoids npx startup)
-   **Run specific test patterns:** `pytest tests/ -k "pattern"`
-   **Auto-format your code:** `black ragzoom/ tests/`
-   **Auto-fix linting issues:** `ruff check ragzoom/ tests/ --fix`
-   **Run the type checker:** `dmypy run -- ragzoom/` (11x faster after first run)

### 4.3. Claude Code Hooks

If you're using Claude Code (claude.ai/code), the project includes intelligent hooks that run tests automatically when you edit files:

- **Configuration**: `.claude/project_settings.json`
- **Behavior**: When you edit a Python file, it automatically:
  - Runs type checking with dmypy
  - Runs linting with ruff
  - Executes relevant tests based on the file you changed

**Example Output**:
```
🔍 Checking Python file: ragzoom/store.py
  📝 Type checking...
  ✅ Success: no issues found in 1 source file
  🧹 Linting...
  ✅ All checks passed!
```

For comprehensive Claude Code agent instructions and collaboration guidelines, see [CLAUDE.md](../CLAUDE.md) (symlinked to AGENT_INSTRUCTIONS.md).

### 4.4. Debugging Type Errors

The `mypy` check can sometimes be noisy, flagging pre-existing issues in files you haven't touched. If you're struggling with a persistent type error, the following command can be very helpful, as it provides a clean, stateless check on just the `ragzoom` directory:
```bash
mypy ragzoom --ignore-missing-imports --no-error-summary --check-untyped-defs
```

### 4.5. Code Duplication Detection

The project maintains a **zero-duplication policy** using [jscpd](https://github.com/kucherenko/jscpd).

**Configuration**: `.jscpd.json` (threshold: 0%, min 11 lines/15 tokens)

**Running**:
```bash
npx jscpd@latest ragzoom/
```

**Fixing Duplications**:
- Extract common code into utilities
- Use inheritance/composition
- Create shared constants
- Apply DRY principle

**Legitimate False Positives**:

Some patterns are intentionally similar. Use your judgment - common examples include:
- Async/sync wrapper methods
- Protocol/interface implementations
- Generated code that can't be refactored

Mark with `jscpd:ignore` comments and explain why:
```python
# jscpd:ignore-start
def retrieve(self, ...):  # Intentional: async/sync wrapper pattern
# jscpd:ignore-end
```

**Avoid marking as false positive**:
- Deprecated code (delete it instead)
- "Almost identical" logic (parameterize the differences)
- Copy-pasted implementations (extract shared code)

### 4.6. Error Handling Standards

The project follows strict error handling principles aligned with our "No Fallback Code" craftsmanship rule.

#### Exception Hierarchy

Use specific exception types from `ragzoom.exceptions`:

```python
from ragzoom.exceptions import (
    ValidationError,      # Data validation failures
    DatabaseError,        # Database operation failures  
    LLMError,            # AI service failures
    ConfigurationError,   # Configuration issues
    ResourceError,       # Resource allocation failures
    NodeNotFoundError,    # Specific domain errors
    DocumentNotFoundError
)
```

#### Error Handling Patterns

**✅ DO: Use specific exceptions**
```python
if not validate_email(email):
    raise ValidationError(
        field="email",
        value=email,
        reason="invalid format"
    )
```

**✅ DO: Preserve exception chains**
```python
try:
    result = external_service.call()
except Exception as e:
    service_error = LLMError(
        operation="summarize",
        model="gpt-4o", 
        message=f"Service failed: {e}"
    )
    raise preserve_exception_chain(service_error, e)
```

**❌ DON'T: Silent failures**
```python
# BAD - swallows errors
try:
    validate_data()
except Exception:
    pass  # Silent failure!
```

**❌ DON'T: String-based error categorization**
```python
# BAD - brittle string parsing
if "connection" in str(e).lower():
    handle_database_error()
```

**❌ DON'T: Generic Exception catches without re-raising**
```python
# BAD - loses error context
try:
    operation()
except Exception as e:
    logger.error(f"Error: {e}")
    return None  # Should raise instead!
```

#### API Error Handling

The API uses centralized error handling middleware (`ErrorHandlingMiddleware`). Endpoints should:

1. **Let exceptions propagate** - don't catch them
2. **Use domain-specific exceptions** - middleware converts to HTTP codes
3. **Include rich context** - middleware creates structured responses

```python
@app.post("/endpoint")
async def my_endpoint(request: Request):
    # DON'T do this:
    # try:
    #     result = process(request)
    #     return result
    # except Exception as e:
    #     raise HTTPException(500, str(e))
    
    # DO this instead - let middleware handle errors:
    result = process(request)  # May raise domain exceptions
    return result
```

#### CLI Error Handling

Use the `handle_cli_error()` helper for consistent user-friendly error messages:

```python
try:
    operation()
except Exception as e:
    handle_cli_error(e, "performing operation")
```

#### Testing Error Paths

Always test error conditions:

```python
def test_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        validate_data("invalid")
    
    assert exc_info.value.field == "data"
    assert exc_info.value.reason == "invalid format"
```

#### Logging Errors

Use structured logging with context:

```python
from ragzoom.error_utils import log_error_with_context

try:
    operation()
except Exception as e:
    log_error_with_context(
        logger, e, "user_operation",
        user_id=user_id,
        request_id=request_id
    )
    raise  # Always re-raise
```

#### Success Criteria

Valid error handling ensures:
- ✅ Zero silent failures
- ✅ Zero string-based error categorization  
- ✅ No generic Exception catches without justification
- ✅ Rich error context for debugging
- ✅ Consistent user-facing error messages
- ✅ Complete test coverage for error paths

## 5. Test Markers and Categories

### Test Markers

- `@pytest.mark.integration` - End-to-end tests exercising full app wiring
- `@pytest.mark.asyncio` - Async test functions

### Test File Mapping

| Source File | Test File | Coverage Focus |
|-------------|-----------|----------------|
| `store.py` | `test_store.py` | Database operations, caching |
| `index.py` | `test_indexing_fast.py` | Tree building logic |
| `retrieve.py` | `test_retrieve.py` | Query processing |
| `dynamic_tiling.py` | `test_dp_*.py` | DP algorithm variants |
| `assemble.py` | `test_dp_assembly.py` | Summary assembly |
| `api.py` | `test_api_*.py`, `test_concurrency.py` | REST endpoints |

## 6. Performance Optimization

### 6.1. Use Mock Store for Unit Tests

The `SimpleMockStore` provides:
- In-memory tree operations (4.5x faster)
- Basic vector search simulation
- State tracking (dirty nodes, pinning)
- No external dependencies

### 6.2. Parallel Test Execution

```bash
# Use pytest-xdist for parallel execution
pytest tests/ -n auto  # Auto-detect CPU cores
pytest tests/ -n 8     # Use 8 workers
```

### 6.3. Test Timing Analysis

```bash
# Show slowest tests
pytest tests/ --durations=10

# Time entire test run
time pytest tests/ -n 8
```

## 7. Benchmarking and Performance Analysis

### 7.1. Running Benchmarks

RagZoom includes a comprehensive benchmarking script for performance regression testing:

```bash
# Basic benchmark with baseline comparison
./scripts/run-indexing-benchmarks --baseline baseline.json document.txt

# Specify custom output directory
./scripts/run-indexing-benchmarks --baseline old.json --output-dir benchmarks/ document.txt

# Include document ID and other indexing options
./scripts/run-indexing-benchmarks --baseline baseline.json --document-id test-doc file.txt

# View available options
./scripts/run-indexing-benchmarks --help
```

### 7.2. Benchmark Outputs

The script generates comprehensive analysis in the specified output directory (default: `benchmarks/`):

- **telemetry.json**: Raw telemetry data from current run
- **log.txt**: Complete indexing logs with debug information
- **comparison.md**: Markdown report comparing performance metrics

## Incremental Append (Beta)

Incremental append lets the engine mutate only the rightmost frontier of an existing
document tree instead of rebuilding from scratch. The feature ships enabled by default
while we continue rollout hardening.

- Use the CLI with `ragzoom index --append --document-id <id>` to stream new files into
  an existing document without rebuilding it. The non-append mode continues to clear the
  document before indexing.
- Ensure the schema migrations for `documents.version` and `node_vectors.doc_version`
  have been applied (automatic for new environments).
- Retrieval calls now supply `(document_id, doc_version)` so queries see atomic snapshots.
- Telemetry emitted during append runs includes an `append_metadata` block describing the
  patch span, version, and node counts. Use this for diff-based validation during testing.
- The validation framework (`ragzoom.validate`) can be toggled on to byte-compare the tail
  and assert span/height invariants after each append.

If the schema prerequisites are missing, the service raises a descriptive error prompting
you to run migrations before trying the incremental path again.
- **visualization.png**: Visual charts showing performance differences

File paths are displayed as clickable links in supported terminals.

### 7.3. CI Integration

The benchmarking script is designed for CI environments:
- Uses `--telemetry --validate --debug --no-progress` flags automatically
- Generates machine-readable outputs for automated analysis
- Creates visual reports for human review in pull requests

### 7.4. Performance Regression Detection

The comparison system uses dynamic thresholds based on baseline variance:
- 🔴 **Regression**: >5σ performance degradation from baseline
- 🟡 **Significant change**: >1σ notable difference
- 🟢 **Improvement**: Positive performance changes
- ⚪ **Normal variance**: Within expected fluctuation

For detailed telemetry analysis, see [Telemetry Guide](telemetry.md).

## 8. Git Workflow

### 8.1. Pre-commit Hook Details

The hook runs in this order:
1. Impacted tests only (downstream of staged files), excluding integration and benchmarks
2. Black formatting (auto-fixes)
3. Ruff linting (auto-fixes simple issues)
4. MyPy type checking (blocking on errors)

**Timing**: ~8 seconds total


## 8. Common Issues and Solutions

### 8.1. Segmentation Faults

If `pytest` crashes with a segmentation fault:
```bash
pytest tests/      # Retry
```

### 8.2. Persistent MyPy Errors

The `dmypy` daemon can get into a bad state:
```bash
# Kill the daemon and run fresh
dmypy stop
mypy ragzoom --ignore-missing-imports --no-error-summary --check-untyped-defs
```

### 8.3. Import Errors in Tests

Ensure you're in the virtual environment and ragzoom is installed:
```bash
source venv/bin/activate
pip install -e .
```

### 8.4. Slow Test Suite

If tests are running slowly:
1. Prefer the unified script: `./scripts/run-checks.sh` (excludes integration/benchmarks by default)
2. Verify parallel execution: `pytest tests/ -n 8`
3. Run only downstream tests for specific files: `./scripts/run-checks.sh --impacted-only path/to/file1.py path/to/file2.py`

## 9. Best Practices

1. **Write Tests First**: Practice TDD for bug fixes and new features
2. **Use Type Annotations**: All new functions must have type hints
3. **Mock External Services**: Use mocks for LLM calls and databases in unit tests
4. **Keep Tests Fast**: Target <1 second per unit test
5. **Document Complex Logic**: Add docstrings for non-obvious algorithms
6. **Check Coverage**: Aim for >80% coverage on new code

For architectural principles like "Correct-by-Construction" and system design philosophy, see [Architecture](architecture.md#key-design-principles).

## 10. Getting Help

- **Documentation**: Start with `docs/architecture.md` for system overview
- **Examples**: Look at existing tests for patterns
- **Debugging**: Use `--pdb` flag to drop into debugger
- **Performance**: Profile with `cProfile` for optimization opportunities 
