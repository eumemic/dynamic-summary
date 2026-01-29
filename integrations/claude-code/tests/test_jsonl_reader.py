"""Tests for JSONL reader error tolerance."""

from __future__ import annotations

import json
from pathlib import Path

from ragzoom_claude_code.jsonl_reader import iter_jsonl, iter_jsonl_reversed


class TestIterJsonlErrorTolerance:
    """iter_jsonl should skip corrupt lines without crashing."""

    def test_skips_corrupt_line(self, tmp_path: Path) -> None:
        """Valid records are yielded around a corrupt line."""
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text(
            json.dumps({"a": 1}) + "\n{CORRUPT\n" + json.dumps({"b": 2}) + "\n"
        )

        records = [rec for rec, _ in iter_jsonl(jsonl)]

        assert records == [{"a": 1}, {"b": 2}]

    def test_all_corrupt_yields_nothing(self, tmp_path: Path) -> None:
        """All-corrupt file yields nothing without crashing."""
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text("{BAD1\n{BAD2\n")

        records = [rec for rec, _ in iter_jsonl(jsonl)]

        assert records == []


class TestIterJsonlReversedErrorTolerance:
    """iter_jsonl_reversed should skip corrupt lines without crashing."""

    def test_skips_corrupt_line(self, tmp_path: Path) -> None:
        """Valid records are yielded (in reverse) around a corrupt line."""
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text(
            json.dumps({"a": 1}) + "\n{CORRUPT\n" + json.dumps({"b": 2}) + "\n"
        )

        records = list(iter_jsonl_reversed(jsonl))

        assert records == [{"b": 2}, {"a": 1}]

    def test_all_corrupt_yields_nothing(self, tmp_path: Path) -> None:
        """All-corrupt file yields nothing without crashing."""
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text("{BAD1\n{BAD2\n")

        records = list(iter_jsonl_reversed(jsonl))

        assert records == []
