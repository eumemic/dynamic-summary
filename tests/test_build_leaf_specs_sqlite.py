"""Tests for _build_leaf_specs with per-chunk timestamps."""

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig
from ragzoom.contracts.embedding_model import EmbeddingProvider
from ragzoom.server.append_executor import AppendExecutor


class StubEmbedder(EmbeddingProvider):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1)] * 4 for i, _ in enumerate(texts)]


class TestBuildLeafSpecsWithTimestamps:
    """Test _build_leaf_specs() with per-chunk timestamps parameter."""

    def test_build_leaf_specs_with_timestamps_per_chunk(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Each chunk can have its own timestamp via the timestamps sequence."""
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())

        # Three chunks, each with its own timestamp
        chunks = ["Hello", "World", "Test"]
        timestamps: list[tuple[float, float] | None] = [
            (1705847400.0, 1705847405.0),  # First chunk: 5 second span
            (1705847405.0, 1705847410.0),  # Second chunk: starts where first ends
            (1705847410.0, 1705847415.0),  # Third chunk: continues the sequence
        ]

        specs = executor._build_leaf_specs(
            chunks,
            span_start=0,
            preceding_neighbor_id=None,
            start_level_index=0,
            timestamps=timestamps,
        )

        assert len(specs) == 3

        # Each spec should have its corresponding timestamp
        assert specs[0].time_start == 1705847400.0
        assert specs[0].time_end == 1705847405.0

        assert specs[1].time_start == 1705847405.0
        assert specs[1].time_end == 1705847410.0

        assert specs[2].time_start == 1705847410.0
        assert specs[2].time_end == 1705847415.0

    def test_build_leaf_specs_with_none_timestamps(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """When timestamps is None, all chunks get None timestamps."""
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())

        chunks = ["Hello", "World"]

        specs = executor._build_leaf_specs(
            chunks,
            span_start=0,
            preceding_neighbor_id=None,
            start_level_index=0,
            timestamps=None,
        )

        assert len(specs) == 2
        assert specs[0].time_start is None
        assert specs[0].time_end is None
        assert specs[1].time_start is None
        assert specs[1].time_end is None

    def test_build_leaf_specs_with_mixed_timestamps(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Some chunks can have timestamps while others have None."""
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())

        chunks = ["Hello", "World", "Test"]
        timestamps: list[tuple[float, float] | None] = [
            (1705847400.0, 1705847405.0),  # First chunk has timestamp
            None,  # Second chunk has no timestamp
            (1705847410.0, 1705847415.0),  # Third chunk has timestamp
        ]

        specs = executor._build_leaf_specs(
            chunks,
            span_start=0,
            preceding_neighbor_id=None,
            start_level_index=0,
            timestamps=timestamps,
        )

        assert len(specs) == 3

        assert specs[0].time_start == 1705847400.0
        assert specs[0].time_end == 1705847405.0

        assert specs[1].time_start is None
        assert specs[1].time_end is None

        assert specs[2].time_start == 1705847410.0
        assert specs[2].time_end == 1705847415.0

    def test_build_leaf_specs_timestamps_length_mismatch_raises(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """If timestamps provided, its length must match chunks length."""
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())

        chunks = ["Hello", "World", "Test"]
        timestamps: list[tuple[float, float] | None] = [
            (1705847400.0, 1705847405.0),
            (1705847405.0, 1705847410.0),
            # Missing third timestamp!
        ]

        with pytest.raises(ValueError, match="timestamps length.*must match.*chunks"):
            executor._build_leaf_specs(
                chunks,
                span_start=0,
                preceding_neighbor_id=None,
                start_level_index=0,
                timestamps=timestamps,
            )

    def test_build_leaf_specs_preserves_other_fields(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Timestamps parameter doesn't affect span, neighbor links, etc."""
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())

        chunks = ["Hello", "World"]
        timestamps: list[tuple[float, float] | None] = [
            (1705847400.0, 1705847405.0),
            (1705847405.0, 1705847410.0),
        ]

        specs = executor._build_leaf_specs(
            chunks,
            span_start=100,
            preceding_neighbor_id="prev-node-id",
            start_level_index=5,
            timestamps=timestamps,
        )

        assert len(specs) == 2

        # First spec
        assert specs[0].text == "Hello"
        assert specs[0].span_start == 100
        assert specs[0].span_end == 105  # 100 + len("Hello")
        assert specs[0].preceding_neighbor_id == "prev-node-id"
        assert specs[0].following_neighbor_id == specs[1].node_id
        assert specs[0].level_index == 5

        # Second spec
        assert specs[1].text == "World"
        assert specs[1].span_start == 105
        assert specs[1].span_end == 110  # 105 + len("World")
        assert specs[1].preceding_neighbor_id == specs[0].node_id
        assert specs[1].following_neighbor_id is None
        assert specs[1].level_index == 6
