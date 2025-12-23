"""Pytest configuration and fixtures for RagZoom tests."""

import asyncio
import logging
import math
import os
import signal
from collections.abc import AsyncIterator, Callable, Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import grpc
import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend as _StorageBackendProtocol
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol
from ragzoom.db_utils import create_temp_database, get_temp_db_name
from ragzoom.document_store import DocumentStore
from ragzoom.indexing.runtime import (
    ClearedDocumentResult,
    IndexerRuntime,
    TruncateResult,
)
from ragzoom.progress import configure_progress, get_progress_config
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc
from ragzoom.server.append_executor import AppendExecutor
from ragzoom.server.indexing_engine import IndexingEngine
from ragzoom.server.run_manager import TelemetryRunManager
from ragzoom.server.servicers import (
    GrpcServerProto,
    IndexerServicer,
    RetrievalServicer,
    WorkerServicer,
    shutdown_gracefully,
)
from ragzoom.server.state import ServerState
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.services.llm_service import LLMService
from ragzoom.store import create_store
from ragzoom.telemetry_types import TelemetryDataDict
from ragzoom.vector_factory import create_vector_index
from tests.test_builders import DocumentBuilder, TreeNodeBuilder

logger = logging.getLogger(__name__)


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
    def preceding_context_budget(self) -> int:
        return self.index_config.preceding_context_budget

    @property
    def budget_tokens(self) -> int | None:
        return self.query_config.budget_tokens


@dataclass
class IndexerRuntimeHarness:
    """Convenience wrapper exposing async helpers around IndexerRuntime."""

    runtime: IndexerRuntime
    indexing_engine: IndexingEngine
    llm_service: LLMService
    telemetry_manager: TelemetryRunManager

    async def append(
        self,
        document_id: str,
        text: str,
        *,
        replace_existing: bool = False,
        collect_telemetry: bool = False,
        file_path: str | None = None,
        await_idle: bool = True,
    ) -> IndexingResult:
        session = self.runtime.get_session(document_id, file_path=file_path)
        result = await session.append_text(
            text,
            replace_existing=replace_existing,
            collect_telemetry=collect_telemetry,
        )
        if await_idle:
            await self.indexing_engine.wait_until_idle(document_id)
        return result

    async def clear(self, document_id: str) -> ClearedDocumentResult:
        session = self.runtime.get_session(document_id)
        result = await session.clear()
        await self.indexing_engine.wait_until_idle(document_id)
        return result

    async def truncate(self, document_id: str, span_start: int) -> TruncateResult:
        session = self.runtime.get_session(document_id)
        result = await session.truncate_from_span(span_start)
        await self.indexing_engine.wait_until_idle(document_id)
        return result

    async def wait_for_idle(self, document_id: str | None = None) -> None:
        await self.indexing_engine.wait_until_idle(document_id)
        # Complete any active telemetry runs for this document now that indexing is done
        if document_id is not None:
            run_context = await self.telemetry_manager.latest_for_document(document_id)
            if run_context is not None and run_context.status == "in_progress":
                await self.telemetry_manager.complete_run(
                    run_context.run_id, error=None
                )


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
        default=float(os.getenv("RZ_MAX_TEST_DURATION", "2.0")),
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
    """Ensure API key and test-safe defaults are set for all tests.

    - Provides a dummy OpenAI API key to avoid network.
    - Forces the in-memory Python vector index backend by default to keep
      tests fast and deterministic. Integration tests can override via env.
    """
    # Set test API key for tests
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key-for-tests"
    # Default vector backend to python for unit tests unless explicitly overridden
    os.environ.setdefault("RAGZOOM_VECTOR_BACKEND", "python")
    yield
    # Don't clean up - let other tests use it


@pytest.fixture(scope="session", autouse=True)
def enable_strict_errors() -> Generator[None, None, None]:
    """Enable strict error mode for all tests.

    In strict mode, handle_graceful_error() raises exceptions instead of
    logging and returning defaults. This ensures tests catch any issues that
    would be silently handled in production.
    """
    os.environ["RAGZOOM_STRICT_ERRORS"] = "1"
    yield
    # Don't clean up - let other tests use it


# Globally disable the tqdm monitor thread in tests (no background thread)
@pytest.fixture(scope="session", autouse=True)
def suppress_progress_globally() -> Generator[None, None, None]:
    cfg = get_progress_config()
    prev_disable_monitor = cfg.disable_monitor_thread
    try:
        # Disable monitor thread to avoid background thread interference
        configure_progress(disable_monitor_thread=True)
        yield
    finally:
        # Restore previous settings
        configure_progress(disable_monitor_thread=prev_disable_monitor)


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
    Callable[[int, int, str, str | None], BackwardCompatibilityConfig]
):
    """Factory fixture for creating custom test configurations.

    Returns a function that can create BackwardCompatibilityConfig with custom parameters.

    Usage:
        def test_something(config_factory):
            config = config_factory(target_chunk_tokens=200, budget_tokens=1500)
    """

    def _create_config(
        target_chunk_tokens: int = 50,
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
    """Default StorageBackend for tests.

    Honors environment overrides so benchmarks can reuse the same database and
    vector index as upstream indexing jobs.
    """

    db_url = os.getenv("RAGZOOM_DATABASE_URL", "sqlite:///:memory:")
    vector_backend = os.getenv("RAGZOOM_VECTOR_BACKEND", "python")
    vector_dir = os.getenv("RAGZOOM_VECTOR_PERSIST_DIR")

    backend = SQLiteStorageBackend(
        db_url,
        vector_backend=vector_backend,
        vector_persist_dir=vector_dir,
    )
    try:
        yield backend
    finally:
        backend.close()


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


@pytest.fixture
async def indexer_runtime_harness(
    storage_backend: _StorageBackendProtocol,
    base_config: BackwardCompatibilityConfig,
) -> AsyncIterator[IndexerRuntimeHarness]:
    """Provide a fully wired IndexerRuntime backed by real components."""

    index_config = base_config.index_config
    operational_template = base_config.operational_config

    llm_service = LLMService(index_config, api_key=operational_template.openai_api_key)
    append_executor = AppendExecutor(index_config, llm_service)
    telemetry_manager = TelemetryRunManager(index_config)

    default_vector_backend = os.environ.get(
        "RAGZOOM_VECTOR_BACKEND", operational_template.vector_backend
    )
    store_db_url = getattr(getattr(storage_backend, "db", None), "url", None)
    default_database_url = os.environ.get(
        "RAGZOOM_DATABASE_URL",
        (
            str(store_db_url)
            if store_db_url is not None
            else operational_template.database_url
        ),
    )

    def _index_for_model(model_id: str) -> _VectorIndexProtocol:
        backend = os.environ.get("RAGZOOM_VECTOR_BACKEND", default_vector_backend)
        db_url = os.environ.get("RAGZOOM_DATABASE_URL", default_database_url)
        return create_vector_index(backend, db_url, model_id)

    def _index_for_document(document_id: str) -> _VectorIndexProtocol:
        record = storage_backend.get_document_by_id(document_id)
        model = (
            getattr(record, "embedding_model", None) if record is not None else None
        ) or index_config.embedding_model
        return _index_for_model(model)

    from openai import OpenAI

    openai_client = OpenAI(
        api_key=operational_template.openai_api_key.get_secret_value()
    )
    # Use max_parallelism=1 for SQLite to avoid connection contention.
    # SQLite with StaticPool shares a single connection across all sessions,
    # so concurrent async jobs can block each other.
    indexing_engine = IndexingEngine(
        store=storage_backend,
        llm_service=llm_service,
        index_config=index_config,
        openai_client=openai_client,
        vector_index_factory=_index_for_model,
        max_parallelism=1,
    )

    runtime = IndexerRuntime(
        store=storage_backend,
        index_config=index_config,
        append_executor=append_executor,
        indexing_engine=indexing_engine,
        telemetry_manager=telemetry_manager,
        vector_index_factory=_index_for_model,
    )

    harness = IndexerRuntimeHarness(
        runtime=runtime,
        indexing_engine=indexing_engine,
        llm_service=llm_service,
        telemetry_manager=telemetry_manager,
    )

    try:
        yield harness
    finally:
        try:
            await asyncio.wait_for(
                asyncio.shield(indexing_engine.wait_until_idle()),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "IndexerRuntimeHarness teardown: wait_until_idle timed out; forcing shutdown"
            )
        except asyncio.CancelledError:
            logger.warning(
                "IndexerRuntimeHarness teardown interrupted by cancellation; forcing shutdown"
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "IndexerRuntimeHarness teardown: wait_until_idle failed", exc_info=exc
            )
        try:
            await asyncio.shield(indexing_engine.shutdown())
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "IndexerRuntimeHarness teardown: shutdown failed", exc_info=exc
            )
        await telemetry_manager.prune_expired()


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
    def mock_embeddings_create(**kwargs: object) -> MagicMock:
        input_texts = kwargs.get("input", [])
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        # Return one embedding for each input text
        input_list = input_texts if isinstance(input_texts, list) else [input_texts]
        from types import SimpleNamespace

        return MagicMock(
            data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in input_list]
        )

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
        from types import SimpleNamespace

        return MagicMock(
            data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in input_list]
        )

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
        "format_version": "4.3",
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


@pytest.fixture()
async def grpc_test_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[str, ServerState]]:
    """Spin up a lightweight gRPC server backed by in-memory components."""

    database_path = tmp_path / "integration.db"
    os.environ["PYTEST_CURRENT_TEST"] = "grpc-server-integration"

    # 8-dim test vector used consistently across sync and async stubs
    test_vector = [1.0] + [0.0] * 7

    class _StubEmbeddings:
        def __init__(self) -> None:
            self._vector = test_vector

        def create(self, *, model: str, input: object, **_: object) -> object:
            if isinstance(input, str):
                texts = [input]
            else:
                texts = list(cast(Sequence[str], input))

            class _Item:
                def __init__(self, embedding: list[float]) -> None:
                    self.embedding = embedding

            class _Resp:
                def __init__(self, items: list[_Item]) -> None:
                    self.data = items

            return _Resp([_Item(list(self._vector)) for _ in texts])

    class _StubOpenAI:
        def __init__(self, **_: object) -> None:
            self.embeddings = _StubEmbeddings()

    # Async stub for LLMService (matches sync stub dimensions)
    class _AsyncStubEmbeddings:
        def __init__(self) -> None:
            self._vector = test_vector

        async def create(self, *, input: object, **_: object) -> object:
            if isinstance(input, str):
                texts = [input]
            else:
                texts = list(cast(Sequence[str], input))

            class _Item:
                def __init__(self, embedding: list[float]) -> None:
                    self.embedding = embedding

            class _Resp:
                def __init__(self, items: list[_Item]) -> None:
                    self.data = items

            return _Resp([_Item(list(self._vector)) for _ in texts])

    class _AsyncStubCompletions:
        async def create(self, **_: object) -> object:
            class _Msg:
                content = "summary"

            class _Choice:
                message = _Msg()

            class _Usage:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                prompt_tokens_details = {"cached_tokens": 0}

            class _Resp:
                choices = [_Choice()]
                usage = _Usage()

            return _Resp()

    class _AsyncStubChat:
        completions = _AsyncStubCompletions()

    class _AsyncStubClient:
        embeddings = _AsyncStubEmbeddings()
        chat = _AsyncStubChat()

    def _stub_build_test_openai_client(_model_id: str) -> object:
        return _AsyncStubClient()

    monkeypatch.setattr("openai.OpenAI", _StubOpenAI, raising=False)
    monkeypatch.setattr("ragzoom.server.servicers.OpenAI", _StubOpenAI, raising=False)
    monkeypatch.setattr("ragzoom.server.state.OpenAI", _StubOpenAI, raising=False)
    monkeypatch.setattr(
        "ragzoom.retrieval.embedding_service.OpenAI", _StubOpenAI, raising=False
    )
    # Patch LLMService test stub builder to use consistent 8-dim vectors
    monkeypatch.setattr(
        "ragzoom.services.llm_service._build_test_openai_client",
        _stub_build_test_openai_client,
        raising=True,
    )

    operational_cfg = OperationalConfig(
        openai_api_key=SecretStr("test-key"),
        backend="sqlite",
        database_url=f"sqlite:///{database_path}",
        vector_backend="python",
    )
    state = ServerState.create(
        index_config=IndexConfig.load(),
        query_config=QueryConfig(),
        operational_config=operational_cfg,
        collect_telemetry=True,
    )

    server = grpc.aio.server()
    pb2_grpc.add_IndexerServiceServicer_to_server(IndexerServicer(state), server)
    pb2_grpc.add_RetrievalServiceServicer_to_server(RetrievalServicer(state), server)
    pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(state), server)

    port = server.add_insecure_port("127.0.0.1:0")
    # IndexingEngine doesn't require explicit start
    await server.start()

    address = f"127.0.0.1:{port}"

    try:
        yield address, state
    finally:
        try:
            await asyncio.wait_for(state.indexing_engine.wait_until_idle(), timeout=5)
        except Exception:
            pass
        await shutdown_gracefully(cast(GrpcServerProto, server))
        await state.indexing_engine.shutdown()
        state.store.close()


@pytest.fixture
def empty_telemetry_data() -> TelemetryDataDict:
    """Shared empty telemetry data fixture for testing edge cases.

    Provides consistent empty telemetry structure for testing how analysis
    functions handle empty data scenarios.
    """
    return {
        "format_version": "4.3",
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
    from typing import cast

    outcome_obj: object = yield
    get_result = getattr(outcome_obj, "get_result", None)
    if get_result is None:
        return
    rep = cast(pytest.TestReport, get_result())
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
        rep.longrepr = f"Test exceeded time budget: {duration:.3f}s > {threshold:.3f}s"


# Hard per-test timeout: interrupt the test call phase using POSIX timers.
# This enforces the threshold even if the test blocks (best-effort within process).
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    # Determine threshold: allow per-test override via slow_threshold marker
    marker = item.get_closest_marker("slow_threshold")
    if marker and marker.args:
        try:
            threshold = float(marker.args[0])
        except Exception:
            threshold = float(item.config.getoption("--max-test-duration"))
    else:
        threshold = float(item.config.getoption("--max-test-duration"))
    if threshold <= 0:
        yield
        return

    def _on_timeout(signum: int, frame: object) -> None:
        try:
            import faulthandler

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
        _outcome = yield
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


# ============================================================================
# Transcript Sync Test Fixtures
# ============================================================================


@dataclass
class FakeAppendResult:
    """Fake append result that mimics IndexingResult for transcript sync tests."""

    span_start: int
    span_end: int
    chunks_created: int = 1


class FakeTranscriptClient:
    """Fake client for transcript sync tests that tracks appends and truncations."""

    def __init__(self) -> None:
        self.appends: list[tuple[str, str]] = []
        self.truncates: list[tuple[str, int]] = []
        self._current_span: int = 0

    def append(self, document_id: str, text: str) -> FakeAppendResult:
        """Append text and return span positions."""
        self.appends.append((document_id, text))
        span_start = self._current_span
        span_end = self._current_span + len(text)
        self._current_span = span_end
        return FakeAppendResult(
            span_start=span_start,
            span_end=span_end,
            chunks_created=1,
        )

    def truncate(self, document_id: str, span_start: int) -> None:
        """Truncate document to span."""
        self.truncates.append((document_id, span_start))
        self._current_span = span_start


@pytest.fixture
def fake_transcript_client() -> FakeTranscriptClient:
    """Create a fake client for transcript sync tests."""
    return FakeTranscriptClient()
