"""Pytest configuration and fixtures for RagZoom tests."""

import os
import tempfile
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
    def sqlite_database_url(self) -> str:
        return self.operational_config.sqlite_database_url

    @property
    def chroma_persist_directory(self) -> str:
        return self.operational_config.chroma_persist_directory

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
        if not config.getoption("--use-real-store"):
            skip_integration = pytest.mark.skip(
                reason="Integration test - use --use-real-store to run"
            )
            for item in items:
                if "integration" in item.keywords:
                    item.add_marker(skip_integration)


@pytest.fixture
def base_config() -> BackwardCompatibilityConfig:
    """Create base configuration for tests."""
    index_config = IndexConfig(
        target_chunk_tokens=50,
        preceding_context_tokens=25,
    )
    query_config = QueryConfig(
        budget_tokens=1000,
    )
    operational_config = OperationalConfig(
        openai_api_key="test-key",
        sqlite_database_url="sqlite:///:memory:",
        chroma_persist_directory=":memory:",  # Will be overridden for real store
    )
    return BackwardCompatibilityConfig(index_config, query_config, operational_config)


@pytest.fixture
def mock_store(base_config) -> Generator[SimpleMockStore, None, None]:
    """Create a mock store for fast testing."""
    store = SimpleMockStore(base_config)
    yield store
    store.close()


@pytest.fixture
def real_store(base_config) -> Generator[Store, None, None]:
    """Create a real store for integration testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create operational config with real directory for ChromaDB
        operational_config = OperationalConfig(
            openai_api_key=base_config.openai_api_key,
            sqlite_database_url=base_config.sqlite_database_url,
            chroma_persist_directory=temp_dir,
        )
        store = Store(
            operational_config, embedding_model=base_config.index_config.embedding_model
        )
        yield store
        store.close()


@pytest.fixture
def store(request, base_config, mock_store, real_store):
    """Provide either mock or real store based on test requirements.

    This fixture automatically selects the appropriate store:
    - For tests marked with @pytest.mark.integration: uses real_store
    - For tests run with --use-real-store flag: uses real_store
    - Otherwise: uses mock_store for speed
    """
    # Check if test is marked as integration
    if hasattr(request.node, "get_closest_marker"):
        if request.node.get_closest_marker("integration"):
            return real_store

    # Check command-line option
    if request.config.getoption("--use-real-store"):
        return real_store

    # Default to mock for speed
    return mock_store


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
