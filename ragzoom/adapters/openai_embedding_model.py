"""OpenAI adapter implementing EmbeddingModel using AsyncOpenAI."""

from __future__ import annotations

from collections.abc import Sequence

from openai import AsyncOpenAI

from ragzoom.contracts.embedding_model import (
    EmbeddingModel,
    EmbeddingResult,
    EmbeddingUsageInfo,
)


# jscpd:ignore-start - Class boilerplate mirrors chat adapter by design
class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, client: AsyncOpenAI, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    # jscpd:ignore-end

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Pass through to OpenAI embeddings.create with list input
        resp = await self._client.embeddings.create(
            model=self._model_id, input=list(texts)
        )
        return [d.embedding for d in resp.data]

    async def embed_with_usage(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed texts and return usage information from the API response."""
        resp = await self._client.embeddings.create(
            model=self._model_id, input=list(texts)
        )
        usage: EmbeddingUsageInfo = {
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            "model": self._model_id,
        }
        return {
            "embeddings": [d.embedding for d in resp.data],
            "usage": usage,
        }
