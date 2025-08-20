"""Shared test utilities and mock setups."""

from unittest.mock import AsyncMock, Mock, patch

import pytest


def create_mock_openai_clients():
    """Create standard mock OpenAI clients for testing.

    Returns a tuple of (mock_index_client, mock_retrieve_client, mock_assemble_client)
    with standard embeddings and chat completion responses.
    """

    # Standard embedding response
    async def mock_embeddings_create_async(*args, **kwargs):
        input_data = kwargs.get("input", args[0] if args else "")
        if isinstance(input_data, list):
            return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in input_data])
        else:
            return Mock(data=[Mock(embedding=[0.1] * 1536)])

    def mock_embeddings_create_sync(*args, **kwargs):
        input_data = kwargs.get("input", args[0] if args else "")
        if isinstance(input_data, list):
            return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in input_data])
        else:
            return Mock(data=[Mock(embedding=[0.1] * 1536)])

    # Standard chat completion response
    async def mock_chat_create_async(*args, **kwargs):
        return Mock(
            choices=[Mock(message=Mock(content="Summary of left and right content"))]
        )

    def mock_chat_create_sync(*args, **kwargs):
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


def create_test_documents():
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

    def __init__(self, embedding_rules=None):
        """Initialize with optional specialized embedding rules."""
        self.embedding_rules = embedding_rules

    def __enter__(self):
        """Enter context and set up mocks."""
        self.index_patcher = patch("ragzoom.index.AsyncOpenAI")
        self.retrieve_patcher = patch("ragzoom.retrieve.OpenAI")

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and clean up mocks."""
        self.index_patcher.stop()
        self.retrieve_patcher.stop()


def mock_openai_context(embedding_rules=None):
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


def mock_openai_fixture():
    """Pytest fixture that mocks all OpenAI clients.

    Usage:
        @pytest.fixture
        def mock_openai(self):
            return mock_openai_fixture()
    """
    with (
        patch("ragzoom.index.AsyncOpenAI") as mock_index,
        patch("ragzoom.retrieve.OpenAI") as mock_retrieve,
    ):
        mock_index_client, mock_retrieve_client, mock_assemble_client = (
            create_mock_openai_clients()
        )

        mock_index.return_value = mock_index_client
        mock_retrieve.return_value = mock_retrieve_client
        # Assemble doesn't use OpenAI so we don't need to mock it

        yield mock_index_client, mock_retrieve_client, mock_assemble_client


@pytest.fixture
def openai_mocks():
    """Centralized pytest fixture for OpenAI mocking.

    This fixture provides consistent OpenAI mocking across all tests.
    Use this instead of creating custom mock fixtures in test files.

    Returns:
        tuple: (mock_index_client, mock_retrieve_client, mock_assemble_client)
    """
    with mock_openai_fixture() as mocks:
        yield mocks


def create_mock_embedding_response(texts, embedding_dim=1536):
    """Create a mock embedding response for given texts.

    Args:
        texts: Single text or list of texts
        embedding_dim: Dimension of embeddings (default 1536)

    Returns:
        Mock response object with embeddings
    """
    if isinstance(texts, str):
        texts = [texts]

    return Mock(data=[Mock(embedding=[0.1] * embedding_dim) for _ in texts])


def create_mock_chat_response(content):
    """Create a mock chat completion response.

    Args:
        content: The content to return in the response

    Returns:
        Mock response object
    """
    return Mock(choices=[Mock(message=Mock(content=content))])


def create_specialized_openai_mocks(embedding_rules=None):
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
    async def specialized_embeddings_create_async(*args, **kwargs):
        input_data = kwargs.get("input", args[0] if args else "")
        if isinstance(input_data, list):
            embeddings = []
            for text in input_data:
                embedding = [0.5] * 1536  # default
                for pattern, values in embedding_rules.items():
                    if pattern.lower() in text.lower():
                        embedding = values
                        break
                embeddings.append(Mock(embedding=embedding))
            return Mock(data=embeddings)
        else:
            embedding = [0.5] * 1536  # default
            for pattern, values in embedding_rules.items():
                if pattern.lower() in input_data.lower():
                    embedding = values
                    break
            return Mock(data=[Mock(embedding=embedding)])

    def specialized_embeddings_create_sync(*args, **kwargs):
        input_data = kwargs.get("input", args[0] if args else "")
        embedding = [0.5] * 1536  # default
        for pattern, values in embedding_rules.items():
            if pattern.lower() in input_data.lower():
                embedding = values
                break
        return Mock(data=[Mock(embedding=embedding)])

    # Standard chat completion
    async def mock_chat_create_async(*args, **kwargs):
        return Mock(
            choices=[Mock(message=Mock(content="Summary of left and right content"))]
        )

    def mock_chat_create_sync(*args, **kwargs):
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
