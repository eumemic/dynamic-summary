"""Service for generating query embeddings with auto-detection support."""

import logging
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from ragzoom.document_store import DocumentStore

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles query embedding generation with model auto-detection."""

    def __init__(self, client: OpenAI, store: "DocumentStore", default_model: str):
        """Initialize embedding service.

        Args:
            client: OpenAI client for API calls
            store: DocumentStore for model detection
            default_model: Default embedding model from config
        """
        self.client = client
        self.store = store
        self.default_model = default_model

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
        if not document_id:
            return self.default_model

        documents_repo = getattr(self.store, "documents", None)
        if documents_repo is None:
            # DocumentStore doesn't expose documents; fall back to default
            return self.default_model

        doc_embedding_model: str | None = documents_repo.get_document_embedding_model(
            document_id
        )
        if isinstance(doc_embedding_model, str) and doc_embedding_model:
            logger.debug(
                f"Auto-detected embedding model '{doc_embedding_model}' for document {document_id}"
            )
            return doc_embedding_model
        else:
            logger.warning(
                f"No embedding model found for document {document_id}, using config default: {self.default_model}. "
                f"This may indicate the document was indexed before model tracking was implemented."
            )
            return self.default_model
