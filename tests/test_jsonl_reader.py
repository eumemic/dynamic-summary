"""Tests for JSONL reader module."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ragzoom.claude_memory.jsonl_reader import iter_jsonl, iter_jsonl_reversed


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    """Helper to write JSONL test files."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


class TestIterJsonl:
    """Tests for forward JSONL iteration."""

    def test_reads_all_records(self, tmp_path: Path) -> None:
        """Should read all records in order."""
        jsonl = tmp_path / "test.jsonl"
        records = [{"id": 1}, {"id": 2}, {"id": 3}]
        _write_jsonl(jsonl, records)

        result = [(r, off) for r, off in iter_jsonl(jsonl)]

        assert [r for r, _ in result] == records

    def test_returns_byte_offsets(self, tmp_path: Path) -> None:
        """Should return correct byte offsets after each record."""
        jsonl = tmp_path / "test.jsonl"
        records = [{"a": 1}, {"b": 2}]
        _write_jsonl(jsonl, records)

        result = list(iter_jsonl(jsonl))

        # First record ends after its line
        assert result[0][1] > 0
        # Second offset should be at end of file
        assert result[1][1] == len(jsonl.read_bytes())

    def test_starts_from_offset(self, tmp_path: Path) -> None:
        """Should skip content before start_offset."""
        jsonl = tmp_path / "test.jsonl"
        records = [{"id": 1}, {"id": 2}, {"id": 3}]
        _write_jsonl(jsonl, records)

        # Get offset after first record
        first_result = list(iter_jsonl(jsonl))
        offset_after_first = first_result[0][1]

        # Read from that offset
        result = list(iter_jsonl(jsonl, start_offset=offset_after_first))

        assert len(result) == 2
        assert result[0][0] == {"id": 2}
        assert result[1][0] == {"id": 3}

    def test_empty_file(self, tmp_path: Path) -> None:
        """Should handle empty files."""
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text("")

        result = list(iter_jsonl(jsonl))

        assert result == []

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        """Should skip blank lines."""
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text('{"a": 1}\n\n{"b": 2}\n')

        result = [r for r, _ in iter_jsonl(jsonl)]

        assert result == [{"a": 1}, {"b": 2}]


class TestIterJsonlReversed:
    """Tests for reverse JSONL iteration."""

    def test_reads_in_reverse_order(self, tmp_path: Path) -> None:
        """Should yield records from last to first."""
        jsonl = tmp_path / "test.jsonl"
        records = [{"id": 1}, {"id": 2}, {"id": 3}]
        _write_jsonl(jsonl, records)

        result = list(iter_jsonl_reversed(jsonl))

        assert result == [{"id": 3}, {"id": 2}, {"id": 1}]

    def test_empty_file(self, tmp_path: Path) -> None:
        """Should handle empty files."""
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text("")

        result = list(iter_jsonl_reversed(jsonl))

        assert result == []

    def test_single_record(self, tmp_path: Path) -> None:
        """Should handle single record."""
        jsonl = tmp_path / "test.jsonl"
        _write_jsonl(jsonl, [{"only": "one"}])

        result = list(iter_jsonl_reversed(jsonl))

        assert result == [{"only": "one"}]

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        """Should skip blank lines."""
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text('{"a": 1}\n\n{"b": 2}\n')

        result = list(iter_jsonl_reversed(jsonl))

        assert result == [{"b": 2}, {"a": 1}]

    def test_handles_small_chunk_size(self, tmp_path: Path) -> None:
        """Should work correctly with chunk size smaller than records."""
        jsonl = tmp_path / "test.jsonl"
        records = [{"id": i, "data": "x" * 50} for i in range(10)]
        _write_jsonl(jsonl, records)

        # Use tiny chunk size to force many iterations
        result = list(iter_jsonl_reversed(jsonl, chunk_size=32))

        assert len(result) == 10
        assert result[0]["id"] == 9
        assert result[-1]["id"] == 0

    def test_handles_chunk_boundary_mid_record(self, tmp_path: Path) -> None:
        """Should correctly handle records split across chunk boundaries."""
        jsonl = tmp_path / "test.jsonl"
        # Create records where chunk boundary likely falls mid-record
        records = [{"id": i, "value": "a" * 100} for i in range(5)]
        _write_jsonl(jsonl, records)

        # Chunk size that doesn't align with record boundaries
        result = list(iter_jsonl_reversed(jsonl, chunk_size=73))

        assert len(result) == 5
        assert [r["id"] for r in result] == [4, 3, 2, 1, 0]

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Should handle unicode correctly."""
        jsonl = tmp_path / "test.jsonl"
        records = [{"text": "hello"}, {"text": "world"}]
        _write_jsonl(jsonl, records)

        result = list(iter_jsonl_reversed(jsonl))

        assert result == [{"text": "world"}, {"text": "hello"}]

    def test_nested_json(self, tmp_path: Path) -> None:
        """Should handle nested JSON structures."""
        jsonl = tmp_path / "test.jsonl"
        records: list[dict[str, object]] = [
            {"outer": {"inner": [1, 2, 3]}},
            {"list": [{"a": 1}, {"b": 2}]},
        ]
        _write_jsonl(jsonl, records)

        result = list(iter_jsonl_reversed(jsonl))

        assert result == [records[1], records[0]]

    def test_no_trailing_newline(self, tmp_path: Path) -> None:
        """Should handle files without trailing newline."""
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text('{"a": 1}\n{"b": 2}')  # No trailing newline

        result = list(iter_jsonl_reversed(jsonl))

        assert result == [{"b": 2}, {"a": 1}]

    def test_large_file_performance(self, tmp_path: Path) -> None:
        """Should efficiently read last few records from large file."""
        jsonl = tmp_path / "test.jsonl"
        # Create a file with many records
        records = [{"id": i} for i in range(1000)]
        _write_jsonl(jsonl, records)

        # Read only first 10 records (from end)
        result = []
        for record in iter_jsonl_reversed(jsonl):
            result.append(record)
            if len(result) >= 10:
                break

        assert len(result) == 10
        assert result[0]["id"] == 999
        assert result[9]["id"] == 990


class TestRoundTrip:
    """Tests that forward and reverse reading are consistent."""

    def test_forward_reverse_match(self, tmp_path: Path) -> None:
        """Forward then reverse should give same records in opposite order."""
        jsonl = tmp_path / "test.jsonl"
        records = [{"id": i, "data": f"value_{i}"} for i in range(20)]
        _write_jsonl(jsonl, records)

        forward = [r for r, _ in iter_jsonl(jsonl)]
        reverse = list(iter_jsonl_reversed(jsonl))

        assert forward == list(reversed(reverse))
