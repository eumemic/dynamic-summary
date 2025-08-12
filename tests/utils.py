"""Shared test utilities and mock setups."""

from unittest.mock import AsyncMock, Mock, patch


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
