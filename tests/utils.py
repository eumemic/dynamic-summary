"""Shared test utilities and mock setups."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
        self.index_patcher = patch("ragzoom.services.llm_service.AsyncOpenAI")
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


def mock_openai_fixture(embedding_rules=None):
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
        patch("ragzoom.retrieve.OpenAI") as mock_retrieve,
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


def create_predictable_summary_mock():
    """Create a mock that returns predictable summaries based on content.

    This is useful for tests that need consistent, deterministic summaries
    based on the input content patterns.
    """

    def mock_chat_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        content = messages[-1]["content"] if messages else ""

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

    async def mock_chat_create_async(*args, **kwargs):
        return mock_chat_create(*args, **kwargs)

    return mock_chat_create, mock_chat_create_async


def create_hash_based_embedding_mock():
    """Create an embedding mock that returns deterministic hash-based embeddings.

    This is useful for tests that need consistent, deterministic embeddings
    based on text content.
    """

    def calculate_hash_embedding(text):
        hash_val = sum(ord(c) for c in text) % 100
        return [hash_val / 100.0] * 1536

    async def hash_embeddings_create_async(*args, **kwargs):
        texts = kwargs.get("input")
        if texts is None and len(args) > 0:
            texts = args[0]
        if not isinstance(texts, list):
            texts = [texts]
        embeddings = []
        for text in texts:
            embedding = calculate_hash_embedding(text)
            embeddings.append(Mock(embedding=embedding))
        return Mock(data=embeddings)

    def hash_embeddings_create_sync(*args, **kwargs):
        texts = kwargs.get("input")
        if texts is None and len(args) > 0:
            texts = args[0]
        if not isinstance(texts, list):
            texts = [texts]
        embeddings = []
        for text in texts:
            embedding = calculate_hash_embedding(text)
            embeddings.append(Mock(embedding=embedding))
        return Mock(data=embeddings)

    return hash_embeddings_create_sync, hash_embeddings_create_async


def create_telemetry_summary_mock():
    """Create a mock for telemetry tests that includes usage data.

    This mock returns summaries with token usage information needed for telemetry collection.
    """

    async def mock_chat_completion_with_usage(*args, **kwargs):
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
        return response

    def mock_chat_completion_with_usage_sync(*args, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = " ".join(
            ["Summary", "word"] * 50
        )  # ~100 tokens
        response.usage = MagicMock()
        response.usage.prompt_tokens = 250
        response.usage.completion_tokens = 50
        return response

    return mock_chat_completion_with_usage_sync, mock_chat_completion_with_usage


def _calculate_embedding_from_rules(text, embedding_rules):
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
                embedding = _calculate_embedding_from_rules(text, embedding_rules)
                embeddings.append(Mock(embedding=embedding))
            return Mock(data=embeddings)
        else:
            embedding = _calculate_embedding_from_rules(input_data, embedding_rules)
            return Mock(data=[Mock(embedding=embedding)])

    def specialized_embeddings_create_sync(*args, **kwargs):
        input_data = kwargs.get("input", args[0] if args else "")
        embedding = _calculate_embedding_from_rules(input_data, embedding_rules)
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
