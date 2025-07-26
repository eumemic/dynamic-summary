# Developer Guide

**Last Updated**: January 2025

Welcome to the RagZoom project! This comprehensive guide covers the technology stack, development environment setup, testing strategies, and best practices for contributing to the project.

## 1. Technology Stack

-   **Core Language:** Python 3.11+
-   **API Framework:** FastAPI
-   **CLI Framework:** `click`
-   **Vector Database:** `chromadb` for storing embeddings and performing semantic search.
-   **Metadata Store:** SQLite with `sqlalchemy` for storing the node tree structure and all other metadata.
-   **LLM Interaction:** `openai` for generating summaries and embeddings.
-   **Tokenization:** `tiktoken` for accurately counting tokens.

## 2. Development Environment Setup

Getting your environment set up correctly is the most important first step.

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

- **Unit Tests**: Fast tests using `SimpleMockStore` for algorithmic logic
- **Integration Tests**: Tests that use real SQLite/ChromaDB (marked with `@pytest.mark.integration`)
- **Slow Tests**: Tests taking >5 seconds (marked with `@pytest.mark.slow`)
- **Concurrency Tests**: Thread safety and parallel operation tests

For the complete list of test markers and pytest configuration options, see [API Reference - Testing](api-reference.md#testing).

### 3.2. Running Tests

```bash
# Run fast tests only (default for pre-commit)
pytest tests/ -m "not slow and not integration" -n 8  # ~8 seconds

# Run all tests
pytest tests/ -n 8  # ~11 seconds

# Run with real store for all tests
pytest tests/ --use-real-store -n 8

# Run specific test file
pytest tests/test_integration.py

# Run single test with verbose output
pytest tests/test_integration.py::TestIntegration::test_token_budget_enforcement -vv
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

-   **Tests (`pytest`):** Runs the fast unit tests.
-   **Formatting (`black`):** Automatically reformats your code to be consistent.
-   **Linting (`ruff`):** Checks for common errors, style issues, and automatically fixes what it can.
-   **Type Checking (`mypy`):** Statically analyzes type hints to catch potential bugs.

**Important:** The hook is configured to **auto-fix** formatting and simple linting errors. After it runs, it will re-stage any files it modified. If there are still errors (e.g., a failing test or a complex `mypy` error), the commit will be aborted, and you will need to fix the issues manually.

### 4.2. Running Checks Manually

You can and should run these checks yourself as you code:
-   **Run all fast tests:** `./scripts/test_quick.sh`
-   **Run the full test suite (including slow/integration tests):** `pytest`
-   **Auto-format your code:** `black ragzoom/ tests/`
-   **Auto-fix linting issues:** `ruff check ragzoom/ tests/ --fix`
-   **Run the type checker:** `dmypy run -- ragzoom/` (11x faster after first run)
-   **Run the type checker (fresh):** `mypy ragzoom`

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

## 5. Test Markers and Categories

### Test Markers

- `@pytest.mark.integration` - Tests requiring real database/API
- `@pytest.mark.slow` - Tests taking >5 seconds
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
- State tracking (pinning)
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

## 7. Git Workflow

### 7.1. Pre-commit Hook Details

The hook runs in this order:
1. Fast tests (excluding @slow and @integration)
2. Black formatting (auto-fixes)
3. Ruff linting (auto-fixes simple issues)
4. MyPy type checking (blocking on errors)

**Timing**: ~8 seconds total

### 7.2. Bypassing Hooks (Emergency Only)

```bash
# Skip pre-commit hook
git commit --no-verify

# Disable Claude Code hooks
# Edit .claude/project_settings.json and remove hooks
```

**Warning**: Only bypass with explicit permission. The hooks prevent broken code from entering the repository.

## 8. Common Issues and Solutions

### 8.1. Segmentation Faults

If `pytest` crashes with a segmentation fault:
```bash
rm -rf chroma_db/  # ChromaDB corruption is usually the cause
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
1. Check you're using mock store: `pytest tests/ -m "not integration"`
2. Verify parallel execution: `pytest tests/ -n 8`
3. Skip slow tests: `pytest tests/ -m "not slow"`

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