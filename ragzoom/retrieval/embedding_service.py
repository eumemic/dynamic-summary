"""Service for generating query embeddings with auto-detection support."""

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from openai import AsyncOpenAI, OpenAI

if TYPE_CHECKING:
    from ragzoom.document_store import DocumentStore

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles query embedding generation with model auto-detection."""

    def __init__(
        self,
        client: OpenAI,
        document_store: "DocumentStore | None",
        default_model: str,
        async_client: AsyncOpenAI | None = None,
    ):
        """Initialize embedding service.

        Args:
            client: Sync OpenAI client for API calls (used by sync methods)
            document_store: Optional document store for model detection
            default_model: Default embedding model from config
            async_client: Optional async OpenAI client for async methods
        """
        self.client = client
        self.async_client = async_client
        self.document_store = document_store
        self.default_model = default_model
        self._warned_missing_model: set[str] = set()
        self._cache: dict[tuple[str, str], list[float]] = {}

    def _store_and_return(
        self, cache_key: tuple[str, str], embedding: list[float]
    ) -> list[float]:
        """Store embedding in cache and return a copy."""
        self._cache[cache_key] = embedding
        return list(embedding)

    def get_query_embedding(
        self, query: str, document_id: str | None = None
    ) -> list[float]:
        """Get embedding for query text (sync version).

        Args:
            query: Query text to embed
            document_id: Optional document ID to auto-detect embedding model

        Returns:
            Query embedding vector

        If document_id is provided, uses the embedding model from that document.
        Otherwise falls back to default_model.
        """
        embedding_model = self._detect_embedding_model(document_id)

        cache_key = (embedding_model, query)
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        try:
            response = self.client.embeddings.create(
                model=embedding_model,
                input=query,
            )
            return self._store_and_return(cache_key, list(response.data[0].embedding))
        except Exception as e:
            logger.error(
                f"Error getting query embedding with model {embedding_model}: {e}"
            )
            raise

    async def get_query_embedding_async(
        self, query: str, document_id: str | None = None
    ) -> list[float]:
        """Get embedding for query text (async version).

        Falls back to sync version in a thread if no async client is configured.

        Args:
            query: Query text to embed
            document_id: Optional document ID to auto-detect embedding model

        Returns:
            Query embedding vector
        """
        # Fall back to sync version in thread pool if no async client
        if self.async_client is None:
            return await asyncio.to_thread(self.get_query_embedding, query, document_id)

        embedding_model = self._detect_embedding_model(document_id)

        cache_key = (embedding_model, query)
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        try:
            response = await self.async_client.embeddings.create(
                model=embedding_model,
                input=query,
            )
            return self._store_and_return(cache_key, list(response.data[0].embedding))
        except Exception as e:
            logger.error(
                f"Error getting query embedding with model {embedding_model}: {e}"
            )
            raise

    def embed_texts(
        self, texts: Sequence[str], document_id: str | None = None
    ) -> list[list[float]]:
        """Embed multiple texts using the detected model with caching."""

        if not texts:
            return []

        embedding_model = self._detect_embedding_model(document_id)

        missing: list[str] = []
        for text in texts:
            cache_key = (embedding_model, text)
            if cache_key not in self._cache:
                missing.append(text)

        if missing:
            try:
                response = self.client.embeddings.create(
                    model=embedding_model,
                    input=list(missing),
                )
            except Exception as e:  # pragma: no cover - network failures
                logger.error(
                    "Error getting batch embeddings with model %s: %s",
                    embedding_model,
                    e,
                )
                raise

            if len(response.data) != len(missing):
                raise RuntimeError(
                    "Embedding batch response size mismatch "
                    f"({len(response.data)} vs {len(missing)})"
                )

            for text, item in zip(missing, response.data):
                cache_key = (embedding_model, text)
                self._cache[cache_key] = list(item.embedding)

        return [
            list(self._cache[(embedding_model, text)])  # guaranteed after fill
            for text in texts
        ]

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
            if document_id not in self._warned_missing_model:
                self._warned_missing_model.add(document_id)
                logger.warning(
                    f"No embedding model found for document {document_id}, using config default: {self.default_model}. "
                    f"This may indicate the document was indexed before model tracking was implemented."
                )
            return self.default_model
