"""Shared test utilities and mock setups."""

from collections.abc import Generator
from types import SimpleNamespace
from typing import TypeGuard, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from openai import OpenAI

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.retrieval.budget_planner import BudgetPlanner
from ragzoom.retrieval.embedding_service import EmbeddingService
from ragzoom.retrieve import Retriever


def create_mock_openai_clients() -> tuple[Mock, Mock, Mock]:
    """Create standard mock OpenAI clients for testing.

    Returns a tuple of (mock_index_client, mock_retrieve_client, mock_assemble_client)
    with standard embeddings and chat completion responses.
    """

    # Standard embedding response
    async def mock_embeddings_create_async(*args: object, **kwargs: object) -> Mock:
        input_data = kwargs.get("input", args[0] if args else "")
        if isinstance(input_data, list):
            from types import SimpleNamespace

            return Mock(
                data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in input_data],
                usage=Mock(total_tokens=len(input_data) * 100),
            )
        else:
            from types import SimpleNamespace

            return Mock(
                data=[SimpleNamespace(embedding=[0.1] * 1536)],
                usage=Mock(total_tokens=100),
            )

    def mock_embeddings_create_sync(*args: object, **kwargs: object) -> Mock:
        input_data = kwargs.get("input", args[0] if args else "")
        if isinstance(input_data, list):
            from types import SimpleNamespace

            return Mock(
                data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in input_data],
                usage=Mock(total_tokens=len(input_data) * 100),
            )
        else:
            from types import SimpleNamespace

            return Mock(
                data=[SimpleNamespace(embedding=[0.1] * 1536)],
                usage=Mock(total_tokens=100),
            )

    # Standard chat completion response
    async def mock_chat_create_async(*args: object, **kwargs: object) -> Mock:
        return Mock(
            choices=[Mock(message=Mock(content="Summary of left and right content"))]
        )

    def mock_chat_create_sync(*args: object, **kwargs: object) -> Mock:
        return Mock(
            choices=[Mock(message=Mock(content="Summary of left and right content"))]
        )

    # Create mock clients
    mock_index_client = Mock()
    mock_index_client.embeddings.create = AsyncMock(
        side_effect=mock_embeddings_create_async
    )
    mock_index_client.chat.completions.create = AsyncMock(
        side_effect=mock_chat_create_async
    )

    mock_retrieve_client = Mock()
    mock_retrieve_client.embeddings.create = Mock(
        side_effect=mock_embeddings_create_sync
    )
    mock_retrieve_client.chat.completions.create = Mock(
        side_effect=mock_chat_create_sync
    )

    mock_assemble_client = Mock()
    mock_assemble_client.embeddings.create = Mock(
        side_effect=mock_embeddings_create_sync
    )
    mock_assemble_client.chat.completions.create = Mock(
        side_effect=mock_chat_create_sync
    )

    return mock_index_client, mock_retrieve_client, mock_assemble_client


def create_test_documents() -> dict[str, str]:
    """Create standard test documents for testing.

    Returns a dict with different document types and sizes.
    """
    return {
        "simple": "This is a simple test document.",
        "medium": "This is a test document. " * 50,
        "large": "Test content. " * 500,
        "multi_paragraph": """First paragraph with some content.

Second paragraph with different content.

Third paragraph with yet more content.""",
        "code": """def hello_world():
    print("Hello, world!")
    return True""",
    }


class MockOpenAIContext:
    """Context manager that provides OpenAI mocking for tests."""

    def __init__(self, embedding_rules: dict[str, list[float]] | None = None) -> None:
        """Initialize with optional specialized embedding rules."""
        self.embedding_rules = embedding_rules

    def __enter__(self) -> tuple[Mock, Mock, Mock]:
        """Enter context and set up mocks."""
        self.index_patcher = patch("ragzoom.services.llm_service.AsyncOpenAI")
        self.retrieve_patcher = patch("openai.OpenAI")

        mock_index_class = self.index_patcher.start()
        mock_retrieve_class = self.retrieve_patcher.start()

        if self.embedding_rules:
            mock_index_client, mock_retrieve_client, mock_assemble_client = (
                create_specialized_openai_mocks(self.embedding_rules)
            )
        else:
            mock_index_client, mock_retrieve_client, mock_assemble_client = (
                create_mock_openai_clients()
            )

        mock_index_class.return_value = mock_index_client
        mock_retrieve_class.return_value = mock_retrieve_client

        return mock_index_client, mock_retrieve_client, mock_assemble_client

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context and clean up mocks."""
        self.index_patcher.stop()
        self.retrieve_patcher.stop()


def mock_openai_context(
    embedding_rules: dict[str, list[float]] | None = None,
) -> MockOpenAIContext:
    """Context manager that mocks all OpenAI clients.

    Args:
        embedding_rules: Optional dict mapping text patterns to embedding values
                        for specialized embedding behavior

    Returns a context manager that yields (mock_index_client, mock_retrieve_client, mock_assemble_client).
    Use this in tests that need OpenAI mocking.

    Usage:
        with mock_openai_context() as (mock_index, mock_retrieve, mock_assemble):
            # Your test code here

        # For specialized embeddings:
        rules = {"dragon": [0.9] * 1536, "wizard": [0.8] * 1536}
        with mock_openai_context(rules) as (mock_index, mock_retrieve, mock_assemble):
            # Your test code here
    """
    return MockOpenAIContext(embedding_rules)


def mock_openai_fixture(
    embedding_rules: dict[str, list[float]] | None = None,
) -> Generator[tuple[Mock, Mock, Mock], None, None]:
    """Pytest fixture that mocks all OpenAI clients.

    Args:
        embedding_rules: Optional dict for specialized embedding behavior

    Usage:
        @pytest.fixture
        def mock_openai(self):
            return mock_openai_fixture()
    """
    with (
        patch("ragzoom.services.llm_service.AsyncOpenAI") as mock_index,
        patch("openai.OpenAI") as mock_retrieve,
    ):
        if embedding_rules:
            mock_index_client, mock_retrieve_client, mock_assemble_client = (
                create_specialized_openai_mocks(embedding_rules)
            )
        else:
            mock_index_client, mock_retrieve_client, mock_assemble_client = (
                create_mock_openai_clients()
            )

        mock_index.return_value = mock_index_client
        mock_retrieve.return_value = mock_retrieve_client
        # Assemble doesn't use OpenAI so we don't need to mock it

        yield mock_index_client, mock_retrieve_client, mock_assemble_client


@pytest.fixture
def openai_mocks() -> Generator[tuple[Mock, Mock, Mock], None, None]:
    """Centralized pytest fixture for OpenAI mocking.

    This fixture provides consistent OpenAI mocking across all tests.
    Use this instead of creating custom mock fixtures in test files.

    Returns:
        tuple: (mock_index_client, mock_retrieve_client, mock_assemble_client)
    """
    yield from mock_openai_fixture()


def create_mock_embedding_response(
    texts: str | list[str], embedding_dim: int = 1536
) -> Mock:
    """Create a mock embedding response for given texts.

    Args:
        texts: Single text or list of texts
        embedding_dim: Dimension of embeddings (default 1536)

    Returns:
        Mock response object with embeddings
    """
    if isinstance(texts, str):
        texts = [texts]

    from types import SimpleNamespace

    return Mock(data=[SimpleNamespace(embedding=[0.1] * embedding_dim) for _ in texts])


def create_mock_chat_response(content: str) -> Mock:
    """Create a mock chat completion response.

    Args:
        content: The content to return in the response

    Returns:
        Mock response object
    """
    return Mock(choices=[Mock(message=Mock(content=content))])


def create_predictable_summary_mock() -> tuple[object, object]:
    """Create a mock that returns predictable summaries based on content.

    This is useful for tests that need consistent, deterministic summaries
    based on the input content patterns.
    """

    def mock_chat_create(*args: object, **kwargs: object) -> Mock:
        messages = kwargs.get("messages", [])
        from typing import cast

        messages_list = (
            cast(list[dict[str, str]], messages) if isinstance(messages, list) else []
        )
        content = messages_list[-1]["content"] if messages_list else ""

        # Return shorter summary if prompt asks for specific token count
        if "approximately 50 tokens" in content:
            return Mock(
                choices=[Mock(message=Mock(content="Short summary of both children."))]
            )
        elif "First chunk" in content and "Second chunk" in content:
            return Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content="Summary of first two chunks. Combined content of chunks 1 and 2."
                        )
                    )
                ]
            )
        elif "Third chunk" in content and "Fourth chunk" in content:
            return Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content="Summary of last two chunks. Combined content of chunks 3 and 4."
                        )
                    )
                ]
            )
        elif "Summary of first" in content and "Summary of last" in content:
            return Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content="Overall document summary. Complete document overview."
                        )
                    )
                ]
            )
        else:
            return Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content="This is the combined summary text for both children."
                        )
                    )
                ]
            )

    async def mock_chat_create_async(*args: object, **kwargs: object) -> Mock:
        return mock_chat_create(*args, **kwargs)

    return mock_chat_create, mock_chat_create_async


def create_hash_based_embedding_mock() -> tuple[object, object]:
    """Create an embedding mock that returns deterministic hash-based embeddings.

    This is useful for tests that need consistent, deterministic embeddings
    based on text content.
    """

    def calculate_hash_embedding(text: str) -> list[float]:
        hash_val = sum(ord(c) for c in text) % 100
        return [hash_val / 100.0] * 1536

    async def hash_embeddings_create_async(*args: object, **kwargs: object) -> Mock:
        texts = kwargs.get("input")
        if texts is None and len(args) > 0:
            texts = args[0]
        if not isinstance(texts, list):
            texts = [texts]
        embeddings = []
        for text in texts:
            text_str = str(text) if not isinstance(text, str) else text
            embedding = calculate_hash_embedding(text_str)
            from types import SimpleNamespace

            embeddings.append(SimpleNamespace(embedding=embedding))
        return Mock(data=embeddings, usage=Mock(total_tokens=len(texts) * 100))

    def hash_embeddings_create_sync(*args: object, **kwargs: object) -> Mock:
        texts = kwargs.get("input")
        if texts is None and len(args) > 0:
            texts = args[0]
        if not isinstance(texts, list):
            texts = [texts]
        embeddings = []
        for text in texts:
            text_str = str(text) if not isinstance(text, str) else text
            embedding = calculate_hash_embedding(text_str)
            from types import SimpleNamespace

            embeddings.append(SimpleNamespace(embedding=embedding))
        return Mock(data=embeddings, usage=Mock(total_tokens=len(texts) * 100))

    return hash_embeddings_create_sync, hash_embeddings_create_async


def create_telemetry_summary_mock() -> tuple[object, object]:
    """Create a mock for telemetry tests that includes usage data.

    This mock returns summaries with token usage information needed for telemetry collection.
    """

    async def mock_chat_completion_with_usage(
        *args: object, **kwargs: object
    ) -> MagicMock:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        # Return a summary that's close to the target token count to avoid retries
        response.choices[0].message.content = " ".join(
            ["Summary", "word"] * 50
        )  # ~100 tokens
        # Add usage data for telemetry
        response.usage = MagicMock()
        response.usage.prompt_tokens = 250
        response.usage.completion_tokens = 50
        response.usage.total_tokens = 300
        return response

    def mock_chat_completion_with_usage_sync(
        *args: object, **kwargs: object
    ) -> MagicMock:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = " ".join(
            ["Summary", "word"] * 50
        )  # ~100 tokens
        response.usage = MagicMock()
        response.usage.prompt_tokens = 250
        response.usage.completion_tokens = 50
        response.usage.total_tokens = 300
        return response

    return mock_chat_completion_with_usage_sync, mock_chat_completion_with_usage


def _calculate_embedding_from_rules(
    text: str, embedding_rules: dict[str, list[float]]
) -> list[float]:
    """Calculate embedding vector based on text patterns and rules.

    Args:
        text: Input text to analyze
        embedding_rules: Dict mapping text patterns to embedding values

    Returns:
        List of float values representing the embedding
    """
    embedding = [0.5] * 1536  # default
    for pattern, values in embedding_rules.items():
        if pattern.lower() in text.lower():
            embedding = values
            break
    return embedding


def create_specialized_openai_mocks(
    embedding_rules: dict[str, list[float]] | None = None,
) -> tuple[Mock, Mock, Mock]:
    """Create OpenAI mocks with specialized embedding behavior.

    Args:
        embedding_rules: Dict mapping text patterns to embedding values
                        e.g., {"dragon": [0.9] * 1536, "wizard": [0.8] * 1536}

    Returns:
        tuple: (mock_index_client, mock_retrieve_client, mock_assemble_client)
    """
    if embedding_rules is None:
        return create_mock_openai_clients()

    # Create specialized embedding functions
    async def specialized_embeddings_create_async(
        *args: object, **kwargs: object
    ) -> Mock:
        input_data = kwargs.get("input", args[0] if args else "")
        if isinstance(input_data, list):
            embeddings = []
            for text in input_data:
                text_str = str(text) if not isinstance(text, str) else text
                embedding = _calculate_embedding_from_rules(text_str, embedding_rules)
                from types import SimpleNamespace

                embeddings.append(SimpleNamespace(embedding=embedding))
            return Mock(data=embeddings, usage=Mock(total_tokens=len(input_data) * 100))
        else:
            text_str = (
                str(input_data) if not isinstance(input_data, str) else input_data
            )
            embedding = _calculate_embedding_from_rules(text_str, embedding_rules)
            from types import SimpleNamespace

            return Mock(
                data=[SimpleNamespace(embedding=embedding)],
                usage=Mock(total_tokens=100),
            )

    def specialized_embeddings_create_sync(*args: object, **kwargs: object) -> Mock:
        input_data = kwargs.get("input", args[0] if args else "")
        text_str = str(input_data) if not isinstance(input_data, str) else input_data
        embedding = _calculate_embedding_from_rules(text_str, embedding_rules)
        return Mock(data=[Mock(embedding=embedding)], usage=Mock(total_tokens=100))

    # Standard chat completion
    async def mock_chat_create_async(*args: object, **kwargs: object) -> Mock:
        return Mock(
            choices=[Mock(message=Mock(content="Summary of left and right content"))]
        )

    def mock_chat_create_sync(*args: object, **kwargs: object) -> Mock:
        return Mock(
            choices=[Mock(message=Mock(content="Summary of left and right content"))]
        )

    # Create mock clients
    mock_index_client = Mock()
    mock_index_client.embeddings.create = AsyncMock(
        side_effect=specialized_embeddings_create_async
    )
    mock_index_client.chat.completions.create = AsyncMock(
        side_effect=mock_chat_create_async
    )

    mock_retrieve_client = Mock()
    mock_retrieve_client.embeddings.create = Mock(
        side_effect=specialized_embeddings_create_sync
    )
    mock_retrieve_client.chat.completions.create = Mock(
        side_effect=mock_chat_create_sync
    )

    mock_assemble_client = Mock()
    mock_assemble_client.embeddings.create = Mock(
        side_effect=specialized_embeddings_create_sync
    )
    mock_assemble_client.chat.completions.create = Mock(
        side_effect=mock_chat_create_sync
    )

    return mock_index_client, mock_retrieve_client, mock_assemble_client


def create_retriever(
    query_config: QueryConfig,
    store: StorageBackend | DocumentStore,
    document_id: str | None = None,
    api_key: str = "test-key",
    embedding_model: str | None = None,
    target_chunk_tokens: int | None = None,
    client: object | None = None,  # Accept any client type for testing
    *,
    vector_index: _VectorIndex,  # REQUIRED VectorIndex to reuse
) -> Retriever:
    """Create a Retriever instance with proper service dependencies.

    Args:
        query_config: QueryConfig instance
        store: StorageBackend or DocumentStore instance
        document_id: Optional document ID to scope retriever to
        api_key: OpenAI API key (defaults to test key)
        embedding_model: Optional embedding model override
        target_chunk_tokens: Optional chunk size override
        client: Optional OpenAI client instance (for testing with mocks)

    Returns:
        Retriever instance configured with proper dependencies
    """
    # Create OpenAI client if not provided
    if client is None:
        client = OpenAI(api_key=api_key)

    # Get document store - handle both StorageBackend and DocumentStore
    if isinstance(store, DocumentStore):
        # This is already a DocumentStore
        doc_store = store
    elif hasattr(store, "for_document") and callable(getattr(store, "for_document")):
        # Support pluggable backends that expose for_document method
        doc_store = store.for_document(document_id)
    else:
        # Fallback for test mocks
        doc_store = cast(DocumentStore, store)

    # Create services with DocumentStore
    # Cast client to OpenAI for type checker - in tests this may be a Mock
    embedding_service = EmbeddingService(
        cast(OpenAI, client), doc_store, embedding_model or query_config.embedding_model
    )

    # Get chunk tokens from IndexConfig if not provided
    if target_chunk_tokens is None:
        index_cfg = IndexConfig.load()
        target_chunk_tokens = index_cfg.target_chunk_tokens

    budget_planner = BudgetPlanner(doc_store, target_chunk_tokens)

    # Require caller to pass the VectorIndex explicitly (no implicit creation)
    # vector_index is required; no implicit creation here
    return Retriever(
        query_config,
        doc_store,
        embedding_service,
        budget_planner,
        vector_index,
    )


# Type conversion utilities for common test patterns


def _safe_int(val: object, default: int = 0) -> int:
    """Convert common scalar types to int safely for typing.

    Accepts int, float, bool, and numeric strings; returns default otherwise.
    """
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        try:
            return int(val)
        except Exception:
            return default
    return default


def ensure_document_store(
    store: StorageBackend | DocumentStore | None,
) -> DocumentStore:
    """Safely convert StorageBackend to DocumentStore for tests.

    Args:
        store: Store instance that might be StorageBackend, DocumentStore, or None

    Returns:
        DocumentStore instance

    Raises:
        TypeError: If store cannot be converted to DocumentStore
    """
    if store is None:
        raise TypeError("Store cannot be None")

    # If it's already a DocumentStore, return it
    if isinstance(store, DocumentStore):
        return store

    # If it's a StorageBackend with for_document method, create document store
    if hasattr(store, "for_document") and callable(getattr(store, "for_document")):
        return store.for_document(None)

    # Otherwise, assume it implements DocumentStore interface
    return cast(DocumentStore, store)


def ensure_storage_backend(
    store: DocumentStore | StorageBackend | None,
) -> StorageBackend:
    """Safely convert DocumentStore to StorageBackend for tests.

    Args:
        store: Store instance that might be DocumentStore, StorageBackend, or None

    Returns:
        StorageBackend instance

    Raises:
        TypeError: If store cannot be converted to StorageBackend
    """
    if store is None:
        raise TypeError("Store cannot be None")

    # If it's already a StorageBackend, return it
    if hasattr(store, "for_document") and not isinstance(store, DocumentStore):
        # Type checker knows this is StorageBackend due to the hasattr check
        return store

    # Otherwise, this is likely a test mock or similar
    # Cast without redundancy warning by being more specific
    if hasattr(store, "clear_document"):
        return cast(StorageBackend, store)

    # Fallback for test scenarios
    raise TypeError(f"Cannot convert {type(store)} to StorageBackend")


def extract_single_config(
    config: (
        tuple[IndexConfig, QueryConfig, OperationalConfig]
        | IndexConfig
        | QueryConfig
        | OperationalConfig
        | None
    ),
    config_type: (
        type[IndexConfig] | type[QueryConfig] | type[OperationalConfig] | None
    ) = None,
) -> IndexConfig | QueryConfig | OperationalConfig | None:
    """Extract single config from tuple or pass through single config.

    Args:
        config: Either a tuple of configs or a single config
        config_type: Preferred config type to extract from tuple (optional)

    Returns:
        Single config instance or None
    """
    if config is None:
        return None

    if isinstance(config, tuple):
        # If specific type requested, find it in the tuple
        if config_type is not None:
            for cfg in config:
                if isinstance(cfg, config_type):
                    return cfg
        # Otherwise return first config
        return config[0] if config else None

    # Already a single config
    return config


def ensure_secret_str_value(
    value: str | SecretStr | None, default: str = "test-key"
) -> str:
    """Safely extract value from SecretStr or regular string.

    Args:
        value: String or SecretStr value
        default: Default value if input is None or empty

    Returns:
        Actual string value
    """
    if value is None:
        return default

    if hasattr(value, "get_secret_value"):
        return value.get_secret_value() or default

    return str(value) or default


def is_tree_node(value: object) -> TypeGuard[TreeNode]:
    """Type guard to check if value is a TreeNode.

    Args:
        value: Value to check

    Returns:
        True if value is a TreeNode
    """
    # Check for TreeNode attributes
    return (
        hasattr(value, "id")
        and hasattr(value, "text")
        and hasattr(value, "span_start")
        and hasattr(value, "span_end")
    )


def ensure_tree_node(value: object, node_id: str = "unknown") -> TreeNode:
    """Safely convert object to TreeNode or raise informative error.

    Args:
        value: Object that should be a TreeNode
        node_id: ID for error messages

    Returns:
        TreeNode instance

    Raises:
        TypeError: If value is not a TreeNode
    """
    if value is None:
        raise TypeError(f"Expected TreeNode for {node_id}, got None")

    if not is_tree_node(value):
        raise TypeError(f"Expected TreeNode for {node_id}, got {type(value)}")

    return value  # Type guard ensures this is TreeNode


def safe_tree_node_access(value: object) -> TreeNode | None:
    """Safely access TreeNode attributes, returning None if not a TreeNode.

    Args:
        value: Object that might be a TreeNode

    Returns:
        TreeNode if valid, None otherwise
    """
    if is_tree_node(value):
        return value  # Type guard ensures this is TreeNode
    return None


def create_mock_config_tuple() -> tuple[IndexConfig, QueryConfig, OperationalConfig]:
    """Create a standard tuple of mock configs for testing.

    Returns:
        Tuple of (IndexConfig, QueryConfig, OperationalConfig)
    """
    index_config = IndexConfig.load()
    query_config = QueryConfig()
    operational_config = OperationalConfig()

    return (index_config, query_config, operational_config)


def cast_simple_namespace_to_dict(ns: SimpleNamespace) -> dict[str, object]:
    """Convert SimpleNamespace to dict for type compatibility.

    Args:
        ns: SimpleNamespace object

    Returns:
        Dictionary representation
    """
    return ns.__dict__


def assert_compatible_store_types(
    store1: StorageBackend | DocumentStore, store2: StorageBackend | DocumentStore
) -> None:
    """Assert that two stores are type-compatible for testing.

    Args:
        store1: First store
        store2: Second store

    Raises:
        AssertionError: If stores are not compatible
    """
    # Both should be either StorageBackend or DocumentStore compatible
    # Check for core storage operations
    if hasattr(store1, "for_document"):
        # StorageBackend - check for backend methods
        assert hasattr(
            store1, "clear_document"
        ), f"Store1 missing clear_document: {type(store1)}"
    else:
        # DocumentStore - check for document store methods
        assert hasattr(store1, "nodes"), f"Store1 missing nodes: {type(store1)}"

    if hasattr(store2, "for_document"):
        # StorageBackend - check for backend methods
        assert hasattr(
            store2, "clear_document"
        ), f"Store2 missing clear_document: {type(store2)}"
    else:
        # DocumentStore - check for document store methods
        assert hasattr(store2, "nodes"), f"Store2 missing nodes: {type(store2)}"
