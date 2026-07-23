"""The OpenAI embedding adapter must batch large inputs.

OpenAI's embeddings endpoint rejects any request whose ``input`` array exceeds
2048 items. Oolong transcripts chunk into far more than that, so ``embed`` must
split into batches, call the API once per batch, and concatenate the results in
the original order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from openai import AsyncOpenAI

from ragzoom.adapters.openai_embedding_model import (
    MAX_EMBEDDING_INPUTS_PER_REQUEST,
    OpenAIEmbeddingModel,
)


@dataclass
class _FakeDatum:
    embedding: list[float]


@dataclass
class _FakeUsage:
    total_tokens: int


class _FakeResp:
    def __init__(self, data: list[_FakeDatum], total_tokens: int) -> None:
        self.data = data
        self.usage = _FakeUsage(total_tokens)


class _RecordingEmbeddings:
    """Records each create() call's input length and echoes deterministic vectors."""

    def __init__(self) -> None:
        self.call_input_lengths: list[int] = []

    async def create(self, *, model: str, input: list[str]) -> _FakeResp:
        self.call_input_lengths.append(len(input))
        # Encode each text's index-in-batch into a 1-D vector so order is checkable.
        data = [_FakeDatum([float(len(t))]) for t in input]
        return _FakeResp(data, total_tokens=len(input))


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _RecordingEmbeddings()


def _model() -> tuple[OpenAIEmbeddingModel, _RecordingEmbeddings]:
    client = _FakeClient()
    model = OpenAIEmbeddingModel(cast(AsyncOpenAI, client), "text-embedding-3-small")
    return model, client.embeddings


@pytest.mark.asyncio
async def test_embed_batches_inputs_over_the_limit() -> None:
    model, rec = _model()
    n = MAX_EMBEDDING_INPUTS_PER_REQUEST * 2 + 7  # forces 3 batches
    texts = [f"t{i}" for i in range(n)]

    out = await model.embed(texts)

    # One embedding per input, in order.
    assert len(out) == n
    assert out == [[float(len(t))] for t in texts]
    # Split into >=3 batches, none exceeding the API limit.
    assert len(rec.call_input_lengths) >= 3
    assert max(rec.call_input_lengths) <= MAX_EMBEDDING_INPUTS_PER_REQUEST
    assert sum(rec.call_input_lengths) == n


@pytest.mark.asyncio
async def test_embed_single_batch_when_small() -> None:
    model, rec = _model()
    out = await model.embed(["a", "b", "c"])
    assert len(out) == 3
    assert rec.call_input_lengths == [3]  # exactly one call


@pytest.mark.asyncio
async def test_embed_with_usage_batches_and_sums_tokens() -> None:
    model, rec = _model()
    n = MAX_EMBEDDING_INPUTS_PER_REQUEST + 5
    texts = [f"x{i}" for i in range(n)]

    result = await model.embed_with_usage(texts)

    assert len(result["embeddings"]) == n
    assert len(rec.call_input_lengths) >= 2
    assert max(rec.call_input_lengths) <= MAX_EMBEDDING_INPUTS_PER_REQUEST
    # Usage is summed across batches (fake returns total_tokens == batch size).
    assert result["usage"]["total_tokens"] == n
