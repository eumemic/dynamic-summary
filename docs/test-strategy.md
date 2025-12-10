# RagZoom Test Strategy

**Version**: 2.0  
**Last Updated**: 2025-08-20  
**Context**: Post-Issue #120 Test Suite Audit

## Overview

This document defines the comprehensive testing strategy for RagZoom, establishing standards, patterns, and practices for maintaining high-quality, reliable test coverage across the entire codebase.

## Testing Philosophy

### Core Principles
1. **Correct-by-Construction**: Tests validate design assumptions and prevent regressions
2. **Zero Duplication**: Centralized utilities prevent test code duplication
3. **Fast Feedback**: Unit tests provide immediate feedback, integration tests ensure reliability
4. **Progressive Coverage**: Prioritize business-critical paths, expand systematically

### Quality Standards
- **Minimum Coverage**: 80% for business-critical modules (index, retrieve, store)
- **Test Performance**: Unit test suite must run in <30 seconds
- **CI Optimization**: Separate fast/integration/benchmark execution
- **Documentation**: All test patterns documented with examples

## Test Architecture

### Three-Layer Testing Model

#### 1. Unit Tests (Fast - <30 seconds)
**Purpose**: Validate individual components in isolation  
**Execution**: Every commit via pre-commit hook  
**Coverage Target**: >90% for core logic modules

```python
# Example: Pure unit test with mocks
def test_text_splitter_chunks_correctly(self, mock_tokenizer):
    splitter = TextSplitter(config)
    chunks = splitter.split("test text")
    assert len(chunks) == expected_count
```

**Characteristics**:
- No external dependencies (database, API, filesystem)
- Extensive mocking of dependencies
- Fast execution (<1ms per test)
- Deterministic results

#### 2. Integration Tests (Medium - 30-60 seconds)
**Purpose**: Validate component interactions with real dependencies  
**Execution**: CI/CD pipeline, manual execution with `--use-real-store`  
**Coverage Target**: >70% for cross-component workflows

```python
@pytest.mark.integration
async def test_index_and_retrieve_workflow(self, real_store):
    # Uses real PostgreSQL, real embeddings
    runtime = IndexerRuntime(config, real_store, api_key)
    await runtime.append_text(doc_id, text, replace_existing=True)

    retriever = Retriever(config, document_store, embedding_service, budget_planner, vector_index)
    results = await retriever.retrieve_async(query, doc_id)
    assert len(results.node_ids) > 0
```

**Characteristics**:
- Real database connections (PostgreSQL + pgvector)
- Real API calls (with rate limiting considerations)
- Complex data flows
- Environment-dependent (CI provides API keys)

#### 3. Benchmark Tests (Long-running - minutes)
**Purpose**: Performance validation and regression detection  
**Execution**: Scheduled CI runs, manual performance analysis  
**Coverage Target**: Key performance paths documented

```python
@pytest.mark.benchmark
def test_indexing_performance(self, leaf_tokens, document_type):
    # Real API calls, timing measurements
    start_time = time.perf_counter()
    indexer.add_document(large_document)
    duration = time.perf_counter() - start_time
    
    assert duration < performance_threshold
```

## Test Organization

### File Structure
```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── utils.py                    # Centralized test utilities
├── mock_store.py              # Mock implementations
│
├── test_*.py                  # Unit tests (fastest)
├── test_*_integration.py      # Integration tests
└── benchmarks/                # Performance tests
    ├── test_indexing_performance.py
    └── test_query_performance.py
```

### Naming Conventions
- **Unit tests**: `test_<module>.py` (e.g., `test_splitter.py`)
- **Integration tests**: `test_<workflow>_integration.py` or `test_<module>.py` with `@pytest.mark.integration`
- **Benchmark tests**: `benchmarks/test_<area>_performance.py`
- **Test classes**: `Test<Component>` (e.g., `TestTextSplitter`)
- **Test methods**: `test_<behavior>_<context>` (e.g., `test_splits_text_at_token_boundaries`)

### Fixture Strategy

#### Centralized Fixtures (conftest.py)
```python
@pytest.fixture
def base_config():
    """Standard configuration for most tests."""
    return BackwardCompatibilityConfig(...)

@pytest.fixture
def sample_telemetry_data():
    """Consistent telemetry data across all telemetry tests."""
    return {"format_version": "4.2", ...}

@pytest.fixture  
def mock_openai_async_client():
    """Centralized OpenAI API mocking."""
    return MockOpenAIClient(...)
```

#### Specialized Fixtures (per module)
```python
@pytest.fixture
def text_splitter(base_config):
    """Configured TextSplitter for splitting tests."""
    return TextSplitter(base_config.index_config)
```

## Test Patterns and Standards

### 1. Mock Strategy

#### Centralized Mocking (Preferred)
```python
# tests/utils.py - Centralized utilities
def create_mock_openai_client():
    """Create standardized OpenAI mock."""
    client = MagicMock()
    client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )
    return client

# In tests
def test_embedding_generation(self, mock_openai_async_client):
    # Use centralized mock
    indexer.client = mock_openai_async_client
    # Test proceeds with consistent mocking
```

#### Inline Mocking (When Needed)
```python
@patch('ragzoom.indexing.runtime.openai.Embedding.create')
def test_specific_embedding_behavior(self, mock_create):
    mock_create.return_value = specific_response
    # Test specific behavior
```

### 2. Error Handling Tests

#### Exception Testing Pattern
```python
def test_handles_node_not_found_error(self):
    with pytest.raises(NodeNotFoundError) as exc_info:
        store.get_node("nonexistent_id")
    
    assert exc_info.value.node_id == "nonexistent_id"
    assert "not found" in str(exc_info.value)
```

#### Error Recovery Testing
```python
def test_retries_on_api_failure(self, mock_client):
    mock_client.side_effect = [APIError(), Mock(embedding=[0.1] * 1536)]
    
    result = indexer.get_embedding("text")
    assert result is not None
    assert mock_client.call_count == 2
```

### 3. Data-Driven Testing

#### Parameterized Tests
```python
@pytest.mark.parametrize("chunk_size,expected_chunks", [
    (100, 5),
    (200, 3),
    (500, 1),
])
def test_chunking_with_various_sizes(self, chunk_size, expected_chunks):
    config = IndexConfig(target_chunk_tokens=chunk_size)
    splitter = TextSplitter(config)
    chunks = splitter.split(standard_text)
    assert len(chunks) == expected_chunks
```

#### Property-Based Testing (Future)
```python
# Future enhancement with hypothesis
from hypothesis import given, strategies as st

@given(st.text(min_size=1))
def test_splitter_preserves_text_content(self, text):
    chunks = splitter.split(text)
    reconstructed = "".join(chunk.text for chunk in chunks)
    assert reconstructed == text
```

### 4. Performance Testing Patterns

#### Timing Validation
```python
def test_retrieval_performance_within_bounds(self):
    start = time.perf_counter()
    results = retriever.retrieve(query, budget_tokens=1000)
    duration = time.perf_counter() - start
    
    assert duration < 2.0  # Max 2 seconds
    assert len(results) > 0
```

#### Memory Usage Monitoring
```python
def test_indexing_memory_usage(self):
    initial_memory = get_memory_usage()
    indexer.add_document(large_document)
    peak_memory = get_memory_usage()
    
    memory_increase = peak_memory - initial_memory
    assert memory_increase < 100 * 1024 * 1024  # <100MB increase
```

## CI/CD Integration

### Test Execution Strategy

#### Pre-commit Hook (Impacted-only Fast Suite)
```bash
./scripts/run-checks.sh --impacted-only <files...>
# Equivalent marker selection when invoking pytest directly:
pytest tests/ -m "not benchmark and not integration" --maxfail=5
```

#### Pull Request Validation (Comprehensive)
```bash
# Fast tests (exclude benchmarks and integration)
pytest tests/ -m "not benchmark and not integration"

# Integration tests  
pytest tests/ -m "integration" --use-real-store

# Code quality
ruff check . && black --check . && mypy ragzoom/
```

#### Scheduled Performance Testing
```bash
# Weekly performance regression testing
pytest benchmarks/ -m "benchmark" --benchmark-json=results.json
```

### Test Markers

#### Standard Markers
```python
@pytest.mark.integration   # Tests requiring external dependencies
@pytest.mark.benchmark     # Performance/timing tests
@pytest.mark.skip          # Temporarily disabled tests
```

#### Custom Markers
```python
@pytest.mark.api_required   # Tests requiring OpenAI API key
@pytest.mark.postgres       # Tests requiring PostgreSQL
@pytest.mark.regression     # Regression prevention tests
```

### Environment Configuration

#### Test Environment Variables
```bash
# Required for integration tests
OPENAI_API_KEY=sk-...           # Real API key for integration tests
DATABASE_URL=postgresql://...   # Test database

# Optional for customization
PYTEST_TIMEOUT=300              # Test timeout
BENCHMARK_RUNS=3                # Performance test repetitions
```

## Coverage Standards

### Module-Level Targets

#### Critical Modules (90%+ Coverage Required)
- `ragzoom/index.py` - Core indexing logic
- `ragzoom/retrieve.py` - Query processing
- `ragzoom/greedy_tiling.py` - Tiling algorithm
- `ragzoom/store.py` - Data persistence

#### Important Modules (80%+ Coverage Required)
- `ragzoom/splitter.py` - Text processing
- `ragzoom/assemble.py` - Result assembly
- `ragzoom/config.py` - Configuration
- `ragzoom/exceptions.py` - Error handling

#### Utility Modules (70%+ Coverage Target)
- `ragzoom/cli.py` - Command-line interface
- `ragzoom/api.py` - HTTP API
- `ragzoom/telemetry_*.py` - Monitoring

### Coverage Monitoring
```bash
# Generate coverage report
pytest --cov=ragzoom --cov-report=html tests/

# Coverage threshold enforcement
pytest --cov=ragzoom --cov-fail-under=65 tests/
```

## Quality Assurance

### Code Quality Integration
```bash
# Pre-commit hook runs all checks
pre-commit run --all-files

# Individual tools
ruff check .                    # Linting
black --check .                 # Formatting  
mypy ragzoom/                   # Type checking
pytest --jscpd tests/           # Duplication detection
```

### Test Quality Standards

#### Test Readability
- Clear, descriptive test names
- Arrange-Act-Assert pattern
- Minimal setup, focused assertions
- Comprehensive docstrings for complex tests

#### Test Reliability
- No flaky tests (deterministic results)
- Proper cleanup (fixtures, temporary files)
- Isolated tests (no shared state)
- Consistent test data

#### Test Maintainability
- DRY principle (centralized utilities)
- Consistent patterns across test files
- Regular test refactoring with code changes
- Documentation for complex test scenarios

## Development Workflow

### Test-Driven Development
1. **Write failing test** for new functionality
2. **Implement minimum code** to make test pass
3. **Refactor** while maintaining test coverage
4. **Add edge case tests** for robustness

### Continuous Testing
1. **Pre-commit hook** runs fast tests automatically
2. **CI pipeline** validates all test categories
3. **Coverage reports** identify gaps
4. **Performance monitoring** prevents regressions

### Test Maintenance
1. **Regular review** of test effectiveness
2. **Refactoring** to eliminate duplication
3. **Performance optimization** for fast feedback
4. **Documentation updates** with strategy changes

## Future Enhancements

### Planned Improvements
1. **Property-based testing** with Hypothesis
2. **Performance regression detection** with baselines
3. **Parallel test execution** with pytest-xdist
4. **Visual regression testing** for UI components
5. **Contract testing** for API endpoints

### Research Areas
1. **Mutation testing** for test quality validation
2. **Chaos engineering** for reliability testing
3. **Load testing** for scalability validation
4. **Security testing** for vulnerability detection

---

This strategy provides a comprehensive framework for maintaining high-quality test coverage while supporting rapid, reliable development of the RagZoom system.
