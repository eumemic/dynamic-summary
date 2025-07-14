# Testing Strategy for RagZoom

## Overview

RagZoom uses a multi-layered testing approach with ~4.5 second test execution time for the full suite. Tests are automatically run at different stages of development through git hooks and Claude Code hooks.

## Test Execution Layers

### 1. Git Pre-commit Hook (Fast - ~1-2 seconds)
- **Triggers**: Before every commit
- **Scope**: Only tests affected by modified files
- **Purpose**: Catch issues early without slowing down development

Example:
```bash
# If you modify splitter.py, only test_splitter.py runs
# If you modify config.py, all tests run (since config affects everything)
```

### 2. Git Pre-push Hook (Complete - ~4.5 seconds)
- **Triggers**: Before pushing to remote
- **Scope**: Full test suite + optional type checking
- **Purpose**: Ensure all tests pass before sharing code

### 3. Claude Code Hooks (Intelligent)
- **post_edit**: Runs relevant tests after file edits
- **pre_write**: Runs linting before writing Python files
- **Purpose**: Continuous feedback during AI-assisted development

### 4. Manual Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_splitter.py -v

# Run tests matching a pattern
./test_quick.sh splitter

# Run with coverage
pytest tests/ --cov=ragzoom --cov-report=term-missing
```

## Test Organization

### Unit Tests
- **test_splitter.py**: TextSplitter functionality
  - Text chunking with token limits
  - Boundary-aware splitting
  - Adjacent context handling
  - Edge cases (empty text, overlapping)

- **test_store.py**: Storage layer
  - CRUD operations for nodes
  - Tree relationships
  - Vector similarity search
  - MMR diversity computation
  - Cache functionality
  - Dirty node marking

### Integration Tests
- **test_integration.py**: End-to-end workflows
  - Document indexing and querying
  - Multiple document handling
  - MMR diversity in practice
  - Token budget enforcement
  - Slope cap constraints
  - Node pinning
  - Eviction with freshness decay

### Concurrency Tests
- **test_concurrency.py**: Thread safety
  - Concurrent API requests
  - Service isolation per request
  - Concurrent document indexing
  - State isolation verification

## Test Coverage Gaps

Currently missing tests for:
- `cli.py` - Command-line interface
- `progress.py` - Progress tracking
- `utils.py` - Utility functions

## Testing Best Practices

### 1. Fast Feedback Loop
The git pre-commit hook runs in 1-2 seconds by only testing what changed:
```bash
# Modifying splitter.py triggers only test_splitter.py
# Modifying api.py triggers only test_concurrency.py
```

### 2. Comprehensive Pre-push
Before code reaches the remote repository, all tests must pass:
```bash
# Automatic on git push, or run manually:
pytest tests/ --tb=short
```

### 3. Test Isolation
Each test uses:
- Temporary directories for storage
- Mocked OpenAI API calls
- Fresh configuration per test

### 4. Performance Testing
For performance-sensitive changes:
```bash
# Time the test execution
time pytest tests/ -v

# Profile specific operations
python -m cProfile -o profile.stats ragzoom/index.py
```

## Continuous Improvement

### Adding New Tests
When adding features:
1. Write tests first (TDD) or immediately after
2. Place unit tests in existing files if they fit
3. Create new test files for new modules
4. Update `.git/hooks/pre-commit` to map new files

### Maintaining Test Speed
- Mock external API calls
- Use small test datasets
- Avoid file I/O when possible
- Run expensive tests only in integration suite

## Hook Configuration

### Git Hooks
Located in `.git/hooks/`:
- `pre-commit`: Smart test selection
- `pre-push`: Full validation

### Claude Code Hooks
Configured in `.claude/hooks.json`:
- Automatic test execution on file changes
- Linting before file writes

### Disabling Hooks (Temporarily)
```bash
# Skip pre-commit hook
git commit --no-verify

# Skip pre-push hook  
git push --no-verify

# Disable Claude Code hooks
# Edit .claude/hooks.json and set "enabled": false
```

## Test Utilities

### Quick Test Runner
```bash
# All tests with timing
./test_quick.sh

# Specific pattern
./test_quick.sh store
./test_quick.sh integration
```

### Coverage Analysis
```bash
# Generate coverage report
pytest tests/ --cov=ragzoom --cov-report=html

# View in browser
open htmlcov/index.html
```

### Debugging Failed Tests
```bash
# Verbose output with full traceback
pytest tests/test_integration.py::TestIntegration::test_token_budget_enforcement -vv

# Drop into debugger on failure
pytest tests/ --pdb

# Show local variables on failure
pytest tests/ -l
```

## CI/CD Integration

While not currently configured, the test suite is ready for CI/CD:
```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -e .
    pip install -r requirements-dev.txt
    pytest tests/ --cov=ragzoom
```

The ~4.5 second execution time makes it suitable for running on every PR.