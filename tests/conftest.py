"""Pytest configuration and fixtures for RagZoom tests."""

import os
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.db_utils import create_temp_database, drop_temp_database, get_temp_db_name
from ragzoom.store import StoreManager
from tests.mock_store import SimpleMockStore
from tests.test_builders import DocumentBuilder, TreeNodeBuilder


class BackwardCompatibilityConfig:
    """Test configuration that combines the three config types for compatibility."""

    def __init__(
        self,
        index_config: IndexConfig,
        query_config: QueryConfig,
        operational_config: OperationalConfig,
    ):
        self.index_config = index_config
        self.query_config = query_config
        self.operational_config = operational_config

    # Backward compatibility properties
    @property
    def openai_api_key(self) -> str:
        return self.operational_config.openai_api_key

    @property
    def database_url(self) -> str:
        return self.operational_config.database_url

    @property
    def target_chunk_tokens(self) -> int:
        return self.index_config.target_chunk_tokens

    @property
    def prev_context_tokens(self) -> int:
        return self.index_config.preceding_context_tokens

    @property
    def budget_tokens(self) -> int:
        return self.query_config.budget_tokens


# Set default API key for tests if not already set
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "test-key-for-tests"


def pytest_addoption(parser):
    """Add command-line options for test configuration."""
    parser.addoption(
        "--use-real-store",
        action="store_true",
        default=False,
        help="Use real Store instead of mock for all tests",
    )
    parser.addoption(
        "--integration-only",
        action="store_true",
        default=False,
        help="Run only integration tests with real Store",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring real Store"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture(scope="session", autouse=True)
def ensure_api_key():
    """Ensure API key is set for all tests."""
    # Set test API key for tests
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key-for-tests"
    yield
    # Don't clean up - let other tests use it


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command-line options."""
    if config.getoption("--integration-only"):
        # Skip tests that are NOT marked as integration
        skip_unit = pytest.mark.skip(reason="Running integration tests only")
        for item in items:
            if "integration" not in item.keywords:
                item.add_marker(skip_unit)
    else:
        # Skip integration tests by default unless --use-real-store is specified
        # Exception: In CI, run integration tests by default
        should_run_integration = (
            config.getoption("--use-real-store")
            or os.getenv("CI")
            or os.getenv("GITHUB_ACTIONS")
        )

        if not should_run_integration:
            skip_integration = pytest.mark.skip(
                reason="Integration test - use --use-real-store to run or run in CI"
            )
            for item in items:
                if "integration" in item.keywords:
                    item.add_marker(skip_integration)


@pytest.fixture
def base_config() -> BackwardCompatibilityConfig:
    """Create base configuration for tests."""
    index_config = IndexConfig.load(
        target_chunk_tokens=50,
        preceding_context_tokens=25,
    )
    query_config = QueryConfig(
        budget_tokens=1000,
    )
    operational_config = OperationalConfig(
        openai_api_key="test-key",
        database_url=os.getenv(
            "RAGZOOM_DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/ragzoom_test",
        ),
    )
    return BackwardCompatibilityConfig(index_config, query_config, operational_config)


@pytest.fixture
def mock_store(base_config) -> Generator[SimpleMockStore, None, None]:
    """Create a mock store for fast testing."""
    store = SimpleMockStore(base_config)
    yield store
    store.close()


@pytest.fixture
def real_store(base_config) -> Generator[StoreManager | None, None, None]:
    """Create a real store for integration testing (lazy loading)."""
    # Only attempt to create real store when fixture is actually requested
    real_store_instance = _create_real_store(base_config)
    if real_store_instance is None:
        # In CI, this should fail for integration tests
        # In local dev, tests can handle None gracefully
        yield None
    else:
        try:
            yield real_store_instance
        finally:
            real_store_instance.close()


@pytest.fixture
def store(request, base_config, mock_store):
    """Provide either mock or real store based on test requirements.

    This fixture automatically selects the appropriate store:
    - For tests marked with @pytest.mark.integration: uses real_store (if available)
    - For tests run with --use-real-store flag: uses real_store (if available)
    - Otherwise: uses mock_store for speed
    """
    # Check if test is marked as integration
    if hasattr(request.node, "get_closest_marker"):
        if request.node.get_closest_marker("integration"):
            # Only create real_store when actually needed for integration tests
            real_store = _create_real_store(base_config)
            if real_store is None:
                pytest.skip("PostgreSQL not available for integration test")
            try:
                yield real_store
            finally:
                # Cleanup test database if needed
                if hasattr(real_store, "_test_db_cleanup"):
                    cleanup_info = real_store._test_db_cleanup
                    try:
                        drop_temp_database(
                            cleanup_info["db_name"], cleanup_info["admin_url"]
                        )
                    except Exception:
                        pass  # Ignore cleanup errors

                real_store.close()
            return

    # Check command-line option
    if request.config.getoption("--use-real-store"):
        # Only create real_store when explicitly requested
        real_store = _create_real_store(base_config)
        if real_store is None:
            pytest.skip("PostgreSQL not available for real store test")
        try:
            yield real_store
        finally:
            # Cleanup test database if needed
            if hasattr(real_store, "_test_db_cleanup"):
                cleanup_info = real_store._test_db_cleanup
                try:
                    drop_temp_database(
                        cleanup_info["db_name"], cleanup_info["admin_url"]
                    )
                except Exception:
                    pass  # Ignore cleanup errors

            real_store.close()
        return

    # Default to mock for speed
    yield mock_store


def _create_real_store(base_config) -> StoreManager | None:
    """Create a real store for integration testing, or return None if unavailable."""
    try:
        # Use test-specific database URL or create unique one
        base_db_url = base_config.database_url

        if "ragzoom_test" in base_db_url:
            # Always create unique database name for each test to ensure isolation
            test_db_name = get_temp_db_name("ragzoom_test")
            # Extract base URL and construct new URL
            base_url_parts = base_db_url.split("/")
            base_url_parts[-1] = test_db_name
            test_db_url = "/".join(base_url_parts)

            # Create the test database
            admin_url = "/".join(base_url_parts[:-1]) + "/postgres"
            create_temp_database(test_db_name, admin_url)
        else:
            # Use base URL for custom URLs (non-test scenarios)
            test_db_url = base_db_url
            test_db_name = None

        # Create operational config with the unique database URL
        # Temporarily remove environment override to ensure our unique URL is used
        original_env = os.environ.get("RAGZOOM_DATABASE_URL")
        if "RAGZOOM_DATABASE_URL" in os.environ:
            del os.environ["RAGZOOM_DATABASE_URL"]

        try:
            operational_config = OperationalConfig(
                openai_api_key=base_config.openai_api_key,
                database_url=test_db_url,  # Use the unique database URL
            )
        finally:
            # Restore environment variable
            if original_env is not None:
                os.environ["RAGZOOM_DATABASE_URL"] = original_env

        # If we get here, PostgreSQL is available
        store = StoreManager(
            operational_config,
            embedding_model=base_config.index_config.embedding_model,
        )

        # Store cleanup info for later
        if test_db_name:
            store._test_db_cleanup = {
                "db_name": test_db_name,
                "admin_url": admin_url,
            }

        return store
    except Exception:
        # Return None - let individual tests decide how to handle unavailable PostgreSQL
        # Only integration tests should fail hard when PostgreSQL is not available
        return None


@pytest.fixture
def tree_node_builder():
    """Provide a TreeNodeBuilder for creating test nodes."""
    return TreeNodeBuilder()


@pytest.fixture
def document_builder():
    """Provide a DocumentBuilder for creating test documents."""
    return DocumentBuilder()


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    mock_client = MagicMock()

    # Mock embeddings
    async def mock_embeddings_create(**kwargs):
        input_texts = kwargs.get("input", [])
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        # Return one embedding for each input text
        return MagicMock(data=[MagicMock(embedding=[0.1] * 1536) for _ in input_texts])

    mock_client.embeddings.create = mock_embeddings_create

    # Mock chat completions for summarization
    mock_summary_response = MagicMock()
    mock_summary_response.choices = [MagicMock()]
    mock_summary_response.choices[0].message = MagicMock()
    mock_summary_response.choices[0].message.content = (
        "Summary of left and right content"
    )

    mock_client.chat.completions.create = MagicMock(return_value=mock_summary_response)

    return mock_client


@pytest.fixture
def mock_openai_async_client():
    """Create a mock AsyncOpenAI client for testing."""
    from unittest.mock import AsyncMock

    mock_client = AsyncMock()

    # Mock embeddings
    async def mock_embeddings_create(**kwargs):
        input_texts = kwargs.get("input", [])
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        # Return one embedding for each input text
        return MagicMock(data=[MagicMock(embedding=[0.1] * 1536) for _ in input_texts])

    mock_client.embeddings.create = mock_embeddings_create

    # Mock chat completions for summarization
    mock_summary_response = MagicMock()
    mock_summary_response.choices = [MagicMock()]
    mock_summary_response.choices[0].message = MagicMock()
    mock_summary_response.choices[0].message.content = (
        "Summary of left and right content"
    )

    mock_client.chat.completions.create = AsyncMock(return_value=mock_summary_response)

    return mock_client


@pytest.fixture
def sample_telemetry_data():
    """Centralized telemetry data fixture for all telemetry tests.

    This fixture provides consistent test data across all telemetry test files,
    replacing the duplicate fixtures previously scattered across multiple files.
    """
    return {
        "format_version": "4.2",
        "document_id": "test_doc",
        "source_document_tokens": 1000,
        "indexed_at": 1234567890.0,
        "config": {
            "target_chunk_tokens": 200,
            "summary_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
        },
        "model_metadata": {},
        "system_prompts": {},
        "runtime_info": {},
        "nodes": [
            {
                "node_id": "leaf-1",
                "height": 0,
                "created_at": 1234567890.0,
                "embedding": {
                    "create": {
                        "input_tokens": 50,
                        "total_tokens": 50,
                        "cost_usd": 0.001,
                    }
                },
                "summary": None,
            },
            {
                "node_id": "leaf-2",
                "height": 0,
                "created_at": 1234567890.1,
                "embedding": {
                    "create": {
                        "input_tokens": 60,
                        "total_tokens": 60,
                        "cost_usd": 0.0012,
                    }
                },
                "summary": None,
            },
            {
                "node_id": "internal-1",
                "height": 1,
                "created_at": 1234567890.2,
                "embedding": {
                    "create": {
                        "input_tokens": 30,
                        "total_tokens": 30,
                        "cost_usd": 0.0006,
                    }
                },
                "summary": {
                    "create": {
                        "input_tokens": 100,
                        "output_tokens": 25,
                        "total_tokens": 125,
                        "cost_usd": 0.005,
                    }
                },
            },
        ],
    }


@pytest.fixture
def empty_telemetry_data():
    """Shared empty telemetry data fixture for testing edge cases.

    Provides consistent empty telemetry structure for testing how analysis
    functions handle empty data scenarios.
    """
    return {
        "format_version": "4.2",
        "document_id": "test_doc",
        "source_document_tokens": 0,
        "indexed_at": 1234567890.0,
        "config": {
            "target_chunk_tokens": 100,
            "summary_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
        },
        "model_metadata": {},
        "system_prompts": {},
        "runtime_info": {},
        "nodes": [],
    }
