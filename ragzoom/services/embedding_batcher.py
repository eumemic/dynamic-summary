"""Batch orchestration for embeddings using an EmbeddingModel.

Centralizes validation and batch splitting while delegating provider calls to
the injected EmbeddingModel. Keeps provider adapters thin and business logic
in one place.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from ragzoom.contracts.embedding_model import EmbeddingModel
from ragzoom.error_utils import preserve_exception_chain
from ragzoom.exceptions import LLMError


class EmbeddingBatcher:
    def __init__(self, model: EmbeddingModel, *, max_batch_size: int = 1000) -> None:
        self._model = model
        self._max_batch: Final[int] = int(max_batch_size)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        # Validate inputs early to avoid provider errors
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(
                    f"Empty text at index {i} in embedding batch. This should be filtered by the caller."
                )

        # Split if needed
        if len(texts) > self._max_batch:
            out: list[list[float]] = []
            for i in range(0, len(texts), self._max_batch):
                bat = texts[i : i + self._max_batch]
                part = await self.embed(bat)
                out.extend(part)
            return out

        try:
            return await self._model.embed(texts)
        except Exception as e:
            llm_error = LLMError(
                operation="get_batch_embeddings",
                model=self._model.model_id,
                message=f"Failed to get batch embeddings: {e}",
                batch_size=len(texts),
            )
            raise preserve_exception_chain(llm_error, e)
