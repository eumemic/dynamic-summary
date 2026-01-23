"""Tests for AppendUnit dataclass."""

import pytest

from ragzoom import AppendUnit


class TestAppendUnitDataclass:
    """Test AppendUnit dataclass fields and initialization."""

    def test_text_only(self) -> None:
        """Create AppendUnit with text only (non-temporal)."""
        unit = AppendUnit(text="Hello world")
        assert unit.text == "Hello world"
        assert unit.time_start is None
        assert unit.time_end is None

    def test_with_timestamps(self) -> None:
        """Create AppendUnit with both timestamps (temporal)."""
        unit = AppendUnit(
            text="Hello world",
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:35:00Z",
        )
        assert unit.text == "Hello world"
        assert unit.time_start == "2024-01-21T14:30:00Z"
        assert unit.time_end == "2024-01-21T14:35:00Z"

    def test_with_same_timestamp(self) -> None:
        """Create AppendUnit with same start and end timestamp (point-in-time)."""
        unit = AppendUnit(
            text="Event occurred",
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:00Z",
        )
        assert unit.time_start == unit.time_end


class TestAppendUnitValidation:
    """Test AppendUnit validation rules."""

    def test_reject_time_start_only(self) -> None:
        """Reject AppendUnit with only time_start (must provide both or neither)."""
        with pytest.raises(ValueError) as exc_info:
            AppendUnit(text="Hello", time_start="2024-01-21T14:30:00Z")
        assert "time_start" in str(exc_info.value)
        assert "time_end" in str(exc_info.value)
        assert "both" in str(exc_info.value).lower()

    def test_reject_time_end_only(self) -> None:
        """Reject AppendUnit with only time_end (must provide both or neither)."""
        with pytest.raises(ValueError) as exc_info:
            AppendUnit(text="Hello", time_end="2024-01-21T14:30:00Z")
        assert "time_start" in str(exc_info.value)
        assert "time_end" in str(exc_info.value)
        assert "both" in str(exc_info.value).lower()

    def test_empty_text_allowed(self) -> None:
        """Allow empty text (validation happens at indexing time)."""
        # AppendUnit itself doesn't validate text content
        # The server will reject empty text during indexing
        unit = AppendUnit(text="")
        assert unit.text == ""


class TestAppendUnitIsTemporal:
    """Test AppendUnit.is_temporal property."""

    def test_non_temporal(self) -> None:
        """Non-temporal unit has is_temporal=False."""
        unit = AppendUnit(text="Hello")
        assert unit.is_temporal is False

    def test_temporal(self) -> None:
        """Temporal unit has is_temporal=True."""
        unit = AppendUnit(
            text="Hello",
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:35:00Z",
        )
        assert unit.is_temporal is True
