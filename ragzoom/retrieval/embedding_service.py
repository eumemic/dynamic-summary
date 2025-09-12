"""Service for generating query embeddings with auto-detection support."""

import logging
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from ragzoom.document_store import DocumentStore

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles query embedding generation with model auto-detection."""

    def __init__(
        self, client: OpenAI, document_store: "DocumentStore | None", default_model: str
    ):
        """Initialize embedding service.

        Args:
            client: OpenAI client for API calls
            document_store: Optional document store for model detection
            default_model: Default embedding model from config
        """
        self.client = client
        self.document_store = document_store
        self.default_model = default_model
        self._warned_missing_model: set[str] = set()

    def get_query_embedding(
        self, query: str, document_id: str | None = None
    ) -> list[float]:
        """Get embedding for query text.

        Args:
            query: Query text to embed
            document_id: Optional document ID to auto-detect embedding model

        Returns:
            Query embedding vector

        If document_id is provided, uses the embedding model from that document.
        Otherwise falls back to default_model.
        """
        embedding_model = self._detect_embedding_model(document_id)

        try:
            response = self.client.embeddings.create(
                model=embedding_model,
                input=query,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(
                f"Error getting query embedding with model {embedding_model}: {e}"
            )
            raise

    def _detect_embedding_model(self, document_id: str | None) -> str:
        """Auto-detect embedding model from document if provided."""
        if not document_id or not self.document_store:
            return self.default_model

        # Verify document store matches the requested document
        if self.document_store.document_id != document_id:
            logger.warning(
                f"Document store is for document {self.document_store.document_id} "
                f"but query is for document {document_id}. Using default model."
            )
            return self.default_model

        doc_embedding_model = self.document_store.get_embedding_model()
        if doc_embedding_model:
            logger.debug(
                f"Auto-detected embedding model '{doc_embedding_model}' for document {document_id}"
            )
            return doc_embedding_model
        else:
            # Warn once per document to avoid log spam in tight loops
            if document_id not in self._warned_missing_model:
                self._warned_missing_model.add(document_id)
                logger.warning(
                    f"No embedding model found for document {document_id}, using config default: {self.default_model}. "
                    f"This may indicate the document was indexed before model tracking was implemented."
                )
            return self.default_model
