"""OpenAI adapter implementing EmbeddingModel using AsyncOpenAI."""

from __future__ import annotations

from collections.abc import Sequence

from openai import AsyncOpenAI

from ragzoom.contracts.embedding_model import (
    EmbeddingModel,
    EmbeddingResult,
    EmbeddingUsageInfo,
)

# OpenAI's embeddings endpoint rejects any request whose ``input`` array exceeds
# 2048 items. Callers that embed an arbitrary number of chunks (e.g. flat-RAG
# over a large transcript) rely on the adapter to split into compliant batches.
MAX_EMBEDDING_INPUTS_PER_REQUEST = 2048


def _batches(texts: Sequence[str], size: int) -> list[list[str]]:
    return [list(texts[i : i + size]) for i in range(0, len(texts), size)]


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
        """Embed texts, batching to respect the API's per-request input limit."""
        embeddings: list[list[float]] = []
        for batch in _batches(texts, MAX_EMBEDDING_INPUTS_PER_REQUEST):
            resp = await self._client.embeddings.create(
                model=self._model_id, input=batch
            )
            embeddings.extend(d.embedding for d in resp.data)
        return embeddings

    async def embed_with_usage(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed texts and return summed usage, batching to respect the input limit."""
        embeddings: list[list[float]] = []
        total_tokens = 0
        for batch in _batches(texts, MAX_EMBEDDING_INPUTS_PER_REQUEST):
            resp = await self._client.embeddings.create(
                model=self._model_id, input=batch
            )
            embeddings.extend(d.embedding for d in resp.data)
            total_tokens += resp.usage.total_tokens if resp.usage else 0
        usage: EmbeddingUsageInfo = {
            "total_tokens": total_tokens,
            "model": self._model_id,
        }
        return {"embeddings": embeddings, "usage": usage}
