"""Pytest configuration and fixtures for RagZoom tests."""

import os
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.store import Store
from tests.mock_store import SimpleMockStore


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
            "RAGZOOM_DATABASE_URL", "postgresql+psycopg://localhost/ragzoom_test"
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
def real_store(base_config) -> Generator[Store | None, None, None]:
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
                        from sqlalchemy import create_engine, text

                        admin_engine = create_engine(
                            cleanup_info["admin_url"], isolation_level="AUTOCOMMIT"
                        )
                        with admin_engine.connect() as conn:
                            # Terminate connections and drop database
                            conn.execute(
                                text(
                                    "SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = :db_name AND pid <> pg_backend_pid()"
                                ),
                                {"db_name": cleanup_info["db_name"]},
                            )
                            conn.execute(
                                text(
                                    f"DROP DATABASE IF EXISTS {cleanup_info['db_name']}"
                                )
                            )  # nosec B608
                        admin_engine.dispose()
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
                    from sqlalchemy import create_engine, text

                    admin_engine = create_engine(
                        cleanup_info["admin_url"], isolation_level="AUTOCOMMIT"
                    )
                    with admin_engine.connect() as conn:
                        # Terminate connections and drop database
                        conn.execute(
                            text(
                                "SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = :db_name AND pid <> pg_backend_pid()"
                            ),
                            {"db_name": cleanup_info["db_name"]},
                        )
                        conn.execute(
                            text(f"DROP DATABASE IF EXISTS {cleanup_info['db_name']}")
                        )  # nosec B608
                    admin_engine.dispose()
                except Exception:
                    pass  # Ignore cleanup errors

            real_store.close()
        return

    # Default to mock for speed
    yield mock_store


def _create_real_store(base_config) -> Store | None:
    """Create a real store for integration testing, or return None if unavailable."""
    try:
        # Create unique database name for test isolation
        import uuid

        # Use test-specific database URL or create unique one
        base_db_url = base_config.database_url
        if "ragzoom_test" in base_db_url:
            # Always create unique database name for each test to ensure isolation
            unique_suffix = uuid.uuid4().hex[:8]
            test_db_url = base_db_url.replace(
                "ragzoom_test", f"ragzoom_test_{unique_suffix}"
            )
        else:
            # Use base URL for custom URLs (non-test scenarios)
            test_db_url = base_db_url

        # Try to create engine first to test connection
        from sqlalchemy import create_engine, text

        # If using unique database, create it first
        if test_db_url != base_db_url:
            # Extract database name from URL
            test_db_name = test_db_url.split("/")[-1]
            base_engine_url = "/".join(test_db_url.split("/")[:-1]) + "/postgres"

            # Create the test database using autocommit to avoid transaction block issues
            admin_engine = create_engine(base_engine_url, isolation_level="AUTOCOMMIT")
            with admin_engine.connect() as conn:
                # Safe database name validation
                import re

                if re.match(r"^[a-zA-Z0-9_]+$", test_db_name):
                    conn.execute(text(f"CREATE DATABASE {test_db_name}"))  # nosec B608
                else:
                    raise ValueError(f"Invalid test database name: {test_db_name}")
            admin_engine.dispose()

        # Test connection to the target database using the unique URL
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            # Create vector extension if needed
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        engine.dispose()

        # Create operational config with the unique database URL
        operational_config = OperationalConfig(
            openai_api_key=base_config.openai_api_key,
            database_url=test_db_url,  # Use the unique database URL
        )

        # If we get here, PostgreSQL is available
        store = Store(
            operational_config,
            embedding_model=base_config.index_config.embedding_model,
        )

        # Store cleanup info for later
        if test_db_url != base_db_url:
            store._test_db_cleanup = {
                "db_name": test_db_name,
                "admin_url": base_engine_url,
            }

        return store
    except Exception:
        # Return None - let individual tests decide how to handle unavailable PostgreSQL
        # Only integration tests should fail hard when PostgreSQL is not available
        return None


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
