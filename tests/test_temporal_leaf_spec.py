"""Tests for temporal metadata on LeafSpec dataclass."""

from ragzoom.server.append_executor import LeafSpec


def test_leaf_spec_accepts_timestamps() -> None:
    """LeafSpec should accept optional time_start and time_end fields."""
    spec = LeafSpec(
        node_id="leaf-1",
        text="Hello world",
        span_start=0,
        span_end=11,
        token_count=2,
        preceding_neighbor_id=None,
        following_neighbor_id=None,
        level_index=0,
        time_start=1705848600.0,
        time_end=1705848612.0,
    )

    assert spec.time_start == 1705848600.0
    assert spec.time_end == 1705848612.0


def test_leaf_spec_timestamps_default_to_none() -> None:
    """LeafSpec should default timestamps to None when not provided."""
    spec = LeafSpec(
        node_id="leaf-2",
        text="No timestamps",
        span_start=0,
        span_end=13,
        token_count=2,
        preceding_neighbor_id=None,
        following_neighbor_id=None,
        level_index=0,
    )

    assert spec.time_start is None
    assert spec.time_end is None
