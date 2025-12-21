"""Transcript sync with revert detection."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ragzoom.jsonl_reader import iter_jsonl


@dataclass
class AppendEntry:
    """A single entry in the append log."""

    last_uuid: str
    """UUID of the last message in this append."""

    span_end: int
    """Document span position after this append."""

    def to_json(self) -> dict[str, object]:
        """Serialize to JSON-compatible dict."""
        return {"last_uuid": self.last_uuid, "span_end": self.span_end}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> AppendEntry:
        """Deserialize from JSON dict."""
        last_uuid = data["last_uuid"]
        span_end = data["span_end"]
        if not isinstance(last_uuid, str):
            raise TypeError(f"last_uuid must be str, got {type(last_uuid)}")
        if not isinstance(span_end, int):
            raise TypeError(f"span_end must be int, got {type(span_end)}")
        return cls(last_uuid=last_uuid, span_end=span_end)


class AppendLog:
    """JSONL-based log of append operations."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, entry: AppendEntry) -> None:
        """Append an entry to the log."""
        with open(self._path, "a") as f:
            f.write(json.dumps(entry.to_json()) + "\n")

    def last_entry(self) -> AppendEntry | None:
        """Get the last entry, or None if empty."""
        if not self._path.exists():
            return None

        # Read last line efficiently
        content = self._path.read_text().rstrip("\n")
        if not content:
            return None

        last_line = content.rsplit("\n", 1)[-1]
        return AppendEntry.from_json(json.loads(last_line))

    def truncate_to(self, last_uuid: str) -> None:
        """Remove all entries after the one with the given uuid."""
        entries = list(self)
        found_idx = None
        for i, entry in enumerate(entries):
            if entry.last_uuid == last_uuid:
                found_idx = i
                break

        if found_idx is None:
            raise ValueError(f"uuid {last_uuid!r} not found in append log")

        # Rewrite file with only entries up to and including found_idx
        with open(self._path, "w") as f:
            for entry in entries[: found_idx + 1]:
                f.write(json.dumps(entry.to_json()) + "\n")

    def find_valid_prefix(
        self, current_head: str, parent_map: dict[str, str | None]
    ) -> AppendEntry | None:
        """Find the last entry whose uuid is an ancestor of current_head.

        Walks backwards through the append log, checking each entry's last_uuid
        against the ancestor chain of current_head.

        Returns None if the log is empty, signaling the caller should transcribe
        the entire ancestor chain from root to current_head.
        """
        entries = list(self)
        if not entries:
            return None

        # Get the last indexed uuid
        last_indexed = entries[-1].last_uuid

        # Find common ancestor between last_indexed and current_head
        common = find_common_ancestor(last_indexed, current_head, parent_map)

        # No common ancestor means completely disjoint branches - user reverted
        # to before anything we indexed, so start fresh
        if common is None:
            return None

        # Walk backwards through entries to find the one containing the common ancestor
        # An entry is valid if its last_uuid is the common ancestor or an ancestor of it
        common_ancestors = _get_ancestors(common, parent_map)
        common_ancestors.add(common)

        for entry in reversed(entries):
            if entry.last_uuid in common_ancestors:
                return entry

        return None

    def __iter__(self) -> Iterator[AppendEntry]:
        """Iterate over all entries."""
        if not self._path.exists():
            return

        for record, _ in iter_jsonl(self._path):
            yield AppendEntry.from_json(record)


def build_parent_map(transcript_path: Path) -> dict[str, str | None]:
    """Build a uuid -> parentUuid map from a transcript file."""
    parent_map: dict[str, str | None] = {}

    for record, _ in iter_jsonl(transcript_path):
        uuid = record.get("uuid")
        if isinstance(uuid, str):
            parent_uuid = record.get("parentUuid")
            if parent_uuid is None or isinstance(parent_uuid, str):
                parent_map[uuid] = parent_uuid
            else:
                # parentUuid exists but is not a string - skip this record
                continue

    return parent_map


def find_common_ancestor(
    x: str, y: str, parent_map: dict[str, str | None]
) -> str | None:
    """Find the most recent common ancestor of x and y.

    Traces both ancestor chains until they intersect.
    Returns None if x and y have no common ancestor (completely disjoint branches).
    Raises KeyError if either x or y is not in the parent map.
    """
    if x not in parent_map:
        raise KeyError(x)
    if y not in parent_map:
        raise KeyError(y)

    # Get all ancestors of x (including x itself)
    x_ancestors = _get_ancestors(x, parent_map)
    x_ancestors.add(x)

    # Walk up y's chain until we find a common ancestor
    current: str | None = y
    while current is not None:
        if current in x_ancestors:
            return current
        current = parent_map.get(current)

    # No intersection - completely disjoint branches (e.g., user reverted to
    # before the first message and started a new conversation)
    return None


def _get_ancestors(uuid: str, parent_map: dict[str, str | None]) -> set[str]:
    """Get all ancestors of a uuid (not including the uuid itself)."""
    ancestors: set[str] = set()
    current = parent_map.get(uuid)
    while current is not None:
        ancestors.add(current)
        current = parent_map.get(current)
    return ancestors
