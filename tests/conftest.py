"""Pytest configuration and fixtures for RagZoom tests."""

import os
from collections.abc import Callable, Generator
import math
import signal
from unittest.mock import MagicMock

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend as _StorageBackendProtocol
from ragzoom.db_utils import create_temp_database, get_temp_db_name
from ragzoom.document_store import DocumentStore
from ragzoom.store import create_store
from ragzoom.telemetry_types import TelemetryDataDict
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


def pytest_addoption(parser: pytest.Parser) -> None:
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
    parser.addoption(
        "--max-test-duration",
        type=float,
        default=float(os.getenv("RZ_MAX_TEST_DURATION", "1.0")),
        help=(
            "Fail any test whose call phase exceeds N seconds (default 1.0). "
            "Override via env RZ_MAX_TEST_DURATION or this option."
        ),
    )
    # Backend selection is not exposed; tests default to SQLite backend for speed


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring real Store"
    )
    config.addinivalue_line(
        "markers",
        "slow_threshold(seconds): optional per-test duration threshold override",
    )
    # 'slow' marker deprecated; full suite runs by default on explicit invocation


@pytest.fixture(scope="session", autouse=True)
def ensure_api_key() -> Generator[None, None, None]:
    """Ensure API key is set for all tests."""
    # Set test API key for tests
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key-for-tests"
    yield
    # Don't clean up - let other tests use it


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
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

    # No dynamic slow marking; explicit invocation should run everything (except benchmarks)


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
        openai_api_key=SecretStr("test-key"),
        database_url=os.getenv(
            "RAGZOOM_DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/ragzoom_test",
        ),
    )
    return BackwardCompatibilityConfig(index_config, query_config, operational_config)


@pytest.fixture
def config_factory() -> (
    Callable[[int, int, int, str, str | None], BackwardCompatibilityConfig]
):
    """Factory fixture for creating custom test configurations.

    Returns a function that can create BackwardCompatibilityConfig with custom parameters.

    Usage:
        def test_something(config_factory):
            config = config_factory(target_chunk_tokens=200, budget_tokens=1500)
    """

    def _create_config(
        target_chunk_tokens: int = 50,
        preceding_context_tokens: int = 25,
        budget_tokens: int = 1000,
        openai_api_key: str = "test-key",
        database_url: str | None = None,
    ) -> BackwardCompatibilityConfig:
        if database_url is None:
            database_url = os.getenv(
                "RAGZOOM_DATABASE_URL",
                "postgresql+psycopg://postgres:postgres@localhost:5432/ragzoom_test",
            )

        index_config = IndexConfig.load(
            target_chunk_tokens=target_chunk_tokens,
            preceding_context_tokens=preceding_context_tokens,
        )
        query_config = QueryConfig(
            budget_tokens=budget_tokens,
        )
        operational_config = OperationalConfig(
            openai_api_key=(
                SecretStr(openai_api_key)
                if isinstance(openai_api_key, str)
                else openai_api_key
            ),
            database_url=database_url,
        )
        return BackwardCompatibilityConfig(
            index_config, query_config, operational_config
        )

    return _create_config


@pytest.fixture
def real_store(
    base_config: BackwardCompatibilityConfig,
) -> Generator[_StorageBackendProtocol | None, None, None]:
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


# --- SQLite in-memory backend fixtures (for migrating off mocks) ---


@pytest.fixture
def sqlite_backend() -> Generator[SQLiteStorageBackend, None, None]:
    """Real in-memory SQLite backend for high-fidelity testing.

    Provides a true database (no Docker) and pairs with the pure-Python vector
    index. Use together with `sqlite_store_factory` to obtain document-scoped
    stores for tests.
    """
    backend = SQLiteStorageBackend("sqlite:///:memory:")
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture
def sqlite_store_factory(
    sqlite_backend: SQLiteStorageBackend,
) -> Callable[[str | None], DocumentStore]:
    """Factory to create a document-scoped store from the SQLite backend.

    Example usage in a test:
        doc_store = sqlite_store_factory("doc1")
        # use doc_store.nodes.add_batch(...), etc.
    """

    def _make(doc_id: str | None = None) -> DocumentStore:
        return sqlite_backend.for_document(doc_id)

    return _make


# Generic, backend-agnostic fixtures


@pytest.fixture
def storage_backend() -> Generator[_StorageBackendProtocol, None, None]:
    """Default StorageBackend for tests: SQLite in-memory."""
    backend = SQLiteStorageBackend("sqlite:///:memory:")
    try:
        yield backend
    finally:
        backend.close()


from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol


@pytest.fixture
def vector_index() -> Generator[_VectorIndexProtocol, None, None]:
    """Backend-agnostic VectorIndex fixture built via the factory.

    Uses environment overrides when provided:
      - RAGZOOM_VECTOR_BACKEND (python|chroma|pgvector)
      - RAGZOOM_DATABASE_URL (to co-locate vectors for sqlite tests)

    Default embedding model mirrors test defaults: text-embedding-3-small.
    """
    import os

    from ragzoom.vector_factory import create_vector_index

    backend = os.environ.get("RAGZOOM_VECTOR_BACKEND", "python")
    database_url = os.environ.get("RAGZOOM_DATABASE_URL", "sqlite:///:memory:")
    embedding_model = os.environ.get(
        "RAGZOOM_TEST_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    idx = create_vector_index(backend, database_url, embedding_model)
    try:
        yield idx
    finally:
        # No standard close API across vector backends; rely on GC
        pass


def _create_real_store(
    base_config: BackwardCompatibilityConfig,
) -> _StorageBackendProtocol | None:
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
                openai_api_key=SecretStr(base_config.openai_api_key),
                database_url=test_db_url,  # Use the unique database URL
            )
        finally:
            # Restore environment variable
            if original_env is not None:
                os.environ["RAGZOOM_DATABASE_URL"] = original_env

        # If we get here, PostgreSQL is available
        store = create_store(
            operational_config, embedding_model=base_config.index_config.embedding_model
        )
        return store
    except Exception:
        # Return None - let individual tests decide how to handle unavailable PostgreSQL
        # Only integration tests should fail hard when PostgreSQL is not available
        return None


@pytest.fixture
def tree_node_builder() -> TreeNodeBuilder:
    """Provide a TreeNodeBuilder for creating test nodes."""
    return TreeNodeBuilder()


@pytest.fixture
def document_builder() -> DocumentBuilder:
    """Provide a DocumentBuilder for creating test documents."""
    return DocumentBuilder()


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client for testing."""
    mock_client = MagicMock()

    # Mock embeddings
    async def mock_embeddings_create(**kwargs: object) -> MagicMock:
        input_texts = kwargs.get("input", [])
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        # Return one embedding for each input text
        input_list = input_texts if isinstance(input_texts, list) else [input_texts]
        return MagicMock(data=[MagicMock(embedding=[0.1] * 1536) for _ in input_list])

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
def mock_openai_async_client() -> MagicMock:
    """Create a mock AsyncOpenAI client for testing."""
    from unittest.mock import AsyncMock

    mock_client = AsyncMock()

    # Mock embeddings
    async def mock_async_embeddings_create(**kwargs: object) -> MagicMock:
        input_texts = kwargs.get("input", [])
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        # Return one embedding for each input text
        input_list = input_texts if isinstance(input_texts, list) else [input_texts]
        return MagicMock(data=[MagicMock(embedding=[0.1] * 1536) for _ in input_list])

    mock_client.embeddings.create = mock_async_embeddings_create

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
def sample_telemetry_data() -> TelemetryDataDict:
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
        "runtime_info": {
            "python_version": "3.11.0",
            "platform": "linux",
            "ragzoom_version": "1.0.0",
        },
        "nodes": [
            {
                "node_id": "leaf-1",
                "height": 0,
                "created_at": 1234567890.0,
                "embedding": {
                    "text_tokens": 50,
                    "batch_size": 1,
                    "batch_position": 0,
                    "model": "text-embedding-3-small",
                    "start_time": 1234567890.0,
                    "end_time": 1234567890.1,
                },
            },
            {
                "node_id": "leaf-2",
                "height": 0,
                "created_at": 1234567890.1,
                "embedding": {
                    "text_tokens": 60,
                    "batch_size": 2,
                    "batch_position": 0,
                    "model": "text-embedding-3-small",
                    "start_time": 1234567890.1,
                    "end_time": 1234567890.2,
                },
            },
            {
                "node_id": "internal-1",
                "height": 1,
                "created_at": 1234567890.2,
                "embedding": {
                    "text_tokens": 30,
                    "batch_size": 1,
                    "batch_position": 0,
                    "model": "text-embedding-3-small",
                    "start_time": 1234567890.2,
                    "end_time": 1234567890.3,
                },
                "summary_attempts": [
                    {
                        "target_tokens": 100,
                        "prompt_tokens": 100,
                        "completion_tokens": 25,
                        "actual_tokens": 25,
                        "model": "gpt-4o-mini",
                        "start_time": 1234567890.3,
                        "end_time": 1234567890.4,
                    }
                ],
                "accepted_attempt": 0,
            },
        ],
    }


@pytest.fixture
def empty_telemetry_data() -> TelemetryDataDict:
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
        "runtime_info": {
            "python_version": "3.11.0",
            "platform": "test",
            "ragzoom_version": "1.0.0",
        },
        "nodes": [],
    }


# Enforce per-test time budget (call phase) with default 1.0s
# This fails tests that exceed the threshold, keeping suites fast.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[object]
) -> Generator[None, None, None]:
    outcome = yield
    rep = outcome.get_result()
    if rep.when != "call":
        return
    # Skip if test already failed/skipped
    if getattr(rep, "failed", False) or getattr(rep, "skipped", False):
        return
    # Determine threshold: marker override or global option
    marker = item.get_closest_marker("slow_threshold")
    if marker and marker.args:
        try:
            threshold = float(marker.args[0])
        except Exception:
            threshold = item.config.getoption("--max-test-duration")
    else:
        threshold = item.config.getoption("--max-test-duration")

    duration = getattr(rep, "duration", None)
    if duration is None:
        return
    if duration > threshold:
        rep.outcome = "failed"
        rep.longrepr = (
            f"Test exceeded time budget: {duration:.3f}s > {threshold:.3f}s"
        )


# Hard per-test timeout: interrupt the test call phase using POSIX timers.
# This enforces the threshold even if the test blocks (best-effort within process).
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    threshold = float(item.config.getoption("--max-test-duration"))
    if threshold <= 0:
        yield
        return

    def _on_timeout(signum: int, frame: object) -> None:
        try:
            import faulthandler  # type: ignore

            faulthandler.dump_traceback()
        except Exception:
            pass
        raise TimeoutError(f"Per-test timeout exceeded: {threshold:.3f}s")

    prev_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, threshold)
        else:
            signal.alarm(int(math.ceil(threshold)))
        outcome = yield
    finally:
        try:
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, 0)
            else:
                signal.alarm(0)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGALRM, prev_handler)
        except Exception:
            pass
