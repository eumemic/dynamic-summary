"""Tests for ISO 8601 timestamp parsing."""

import pytest

from ragzoom.server.append_executor import parse_timestamp


class TestParseTimestampFormats:
    """Test parse_timestamp with various ISO 8601 formats."""

    def test_utc_z_suffix(self) -> None:
        """Parse timestamp with Z suffix (UTC)."""
        result = parse_timestamp("2024-01-21T14:30:00Z")
        # 2024-01-21T14:30:00Z = 1705847400.0 in Unix time
        assert result == 1705847400.0

    def test_utc_offset_plus_zero(self) -> None:
        """Parse timestamp with +00:00 offset (UTC)."""
        result = parse_timestamp("2024-01-21T14:30:00+00:00")
        assert result == 1705847400.0

    def test_positive_offset(self) -> None:
        """Parse timestamp with positive timezone offset."""
        # 2024-01-21T14:30:00+05:00 is 5 hours ahead of UTC
        # So UTC time is 09:30:00, which is 1705829400.0
        result = parse_timestamp("2024-01-21T14:30:00+05:00")
        assert result == 1705829400.0

    def test_negative_offset(self) -> None:
        """Parse timestamp with negative timezone offset."""
        # 2024-01-21T14:30:00-05:00 is 5 hours behind UTC
        # So UTC time is 19:30:00, which is 1705865400.0
        result = parse_timestamp("2024-01-21T14:30:00-05:00")
        assert result == 1705865400.0

    def test_with_microseconds(self) -> None:
        """Parse timestamp with microseconds."""
        result = parse_timestamp("2024-01-21T14:30:00.123456Z")
        assert abs(result - 1705847400.123456) < 0.000001

    def test_with_milliseconds(self) -> None:
        """Parse timestamp with milliseconds (3 decimal places)."""
        result = parse_timestamp("2024-01-21T14:30:00.123Z")
        assert abs(result - 1705847400.123) < 0.0001

    def test_microseconds_with_offset(self) -> None:
        """Parse timestamp with microseconds and timezone offset."""
        result = parse_timestamp("2024-01-21T14:30:00.123456-05:00")
        # 14:30:00.123456-05:00 = 19:30:00.123456 UTC
        expected = 1705865400.123456
        assert abs(result - expected) < 0.000001


class TestRejectTimestampWithoutTimezone:
    """Test that timestamps without timezone info are rejected."""

    def test_reject_no_timezone(self) -> None:
        """Reject timestamp without any timezone info."""
        with pytest.raises(ValueError) as exc_info:
            parse_timestamp("2024-01-21T14:30:00")
        assert "timezone" in str(exc_info.value).lower()

    def test_reject_date_only(self) -> None:
        """Reject date-only string."""
        with pytest.raises(ValueError) as exc_info:
            parse_timestamp("2024-01-21")
        # fromisoformat will fail or we check for timezone
        assert (
            "timezone" in str(exc_info.value).lower()
            or "format" in str(exc_info.value).lower()
        )

    def test_reject_no_timezone_with_microseconds(self) -> None:
        """Reject timestamp with microseconds but no timezone."""
        with pytest.raises(ValueError) as exc_info:
            parse_timestamp("2024-01-21T14:30:00.123456")
        assert "timezone" in str(exc_info.value).lower()


class TestParseTimestampInvalidInput:
    """Test parse_timestamp with invalid input."""

    def test_reject_empty_string(self) -> None:
        """Reject empty string."""
        with pytest.raises(ValueError):
            parse_timestamp("")

    def test_reject_invalid_format(self) -> None:
        """Reject non-ISO 8601 format."""
        with pytest.raises(ValueError):
            parse_timestamp("Jan 21, 2024 2:30 PM")

    def test_reject_unix_timestamp_string(self) -> None:
        """Reject Unix timestamp as string (not ISO 8601)."""
        with pytest.raises(ValueError):
            parse_timestamp("1705847400")

    def test_reject_none_type(self) -> None:
        """Reject None input."""
        with pytest.raises(AttributeError):
            parse_timestamp(None)  # type: ignore[arg-type]
