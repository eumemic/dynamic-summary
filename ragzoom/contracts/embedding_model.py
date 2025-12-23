"""Embedding model protocols.

Defines minimal, backend-agnostic interfaces for embedding providers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from typing_extensions import TypedDict


class EmbeddingUsageInfo(TypedDict, total=False):
    """Usage information from an embedding API call."""

    total_tokens: int
    model: str


class EmbeddingResult(TypedDict):
    """Result from embedding API call with usage info."""

    embeddings: list[list[float]]
    usage: EmbeddingUsageInfo


class EmbeddingProvider(Protocol):
    """Protocol for async embedding providers used in indexing and telemetry."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding providers.

    Implementations accept a batch of texts and return one embedding per input
    in the same order. Implementations should not perform batch splitting or
    retries; those concerns live in orchestration logic.
    """

    @property
    def model_id(self) -> str:  # pragma: no cover - protocol surface
        ...

    async def embed(
        self, texts: Sequence[str]
    ) -> list[list[float]]:  # pragma: no cover - protocol surface
        ...
