"""Pytest configuration and fixtures for RagZoom tests."""

import os
import tempfile
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.store import Store
from tests.mock_store import SimpleMockStore

# Set default API keys for tests if not already set
# RagZoomConfig expects RAGZOOM_OPENAI_API_KEY due to env_prefix="RAGZOOM_"
if "RAGZOOM_OPENAI_API_KEY" not in os.environ:
    os.environ["RAGZOOM_OPENAI_API_KEY"] = "test-key-for-tests"
# Also set OPENAI_API_KEY for any code that uses it directly
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
    """Ensure API keys are set for all tests."""
    # RagZoomConfig expects RAGZOOM_OPENAI_API_KEY due to env_prefix="RAGZOOM_"
    if "RAGZOOM_OPENAI_API_KEY" not in os.environ:
        os.environ["RAGZOOM_OPENAI_API_KEY"] = "test-key-for-tests"
    # Also set OPENAI_API_KEY for any code that uses it directly
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
def base_config() -> RagZoomConfig:
    """Create base configuration for tests."""
    return RagZoomConfig(
        openai_api_key="test-key",
        sqlite_database_url="sqlite:///:memory:",
        chroma_persist_directory=":memory:",  # Will be overridden for real store
        leaf_tokens=50,
        adjacent_context_tokens=25,
        budget_tokens=1000,
        embedding_dimensions=1536,
    )


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
        # Update config with real directory for ChromaDB
        config = RagZoomConfig(
            openai_api_key=base_config.openai_api_key,
            sqlite_database_url=base_config.sqlite_database_url,
            chroma_persist_directory=temp_dir,
            leaf_tokens=base_config.leaf_tokens,
            adjacent_context_tokens=base_config.adjacent_context_tokens,
            budget_tokens=base_config.budget_tokens,
            embedding_dimensions=base_config.embedding_dimensions,
        )
        store = Store(config)
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


def create_tree_builder(config, store, **kwargs):
    """Helper function to create TreeBuilder."""
    return TreeBuilder(config, store, **kwargs)


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
