"""JSONL reader with support for reverse streaming from files and bytes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def iter_jsonl(
    path: Path, start_offset: int = 0
) -> Iterator[tuple[dict[str, object], int]]:
    """Iterate over JSONL records from a file.

    Args:
        path: Path to the JSONL file.
        start_offset: Byte offset to start reading from.

    Yields:
        Tuples of (record, end_offset) where end_offset is the byte position
        after the newline following this record.
    """
    with open(path, "rb") as f:
        if start_offset > 0:
            f.seek(start_offset)

        for line in f:
            line_str = line.decode("utf-8").strip()
            if line_str:
                yield json.loads(line_str), f.tell()


def iter_jsonl_reversed(
    path: Path, chunk_size: int = 65536
) -> Iterator[dict[str, object]]:
    """Iterate over JSONL records from a file in reverse order.

    Reads the file backwards in chunks, yielding parsed JSON objects
    from last line to first. Memory efficient - only holds one chunk
    plus partial line buffer at a time.

    Args:
        path: Path to the JSONL file.
        chunk_size: Size of chunks to read (default 64KB).

    Yields:
        Parsed JSON objects, from last record to first.
    """
    with open(path, "rb") as f:
        # Seek to end to get file size
        f.seek(0, 2)
        position = f.tell()

        if position == 0:
            return

        buffer = b""

        while position > 0:
            # Read a chunk backwards
            read_size = min(chunk_size, position)
            position -= read_size
            f.seek(position)
            chunk = f.read(read_size)
            buffer = chunk + buffer

            # Split into lines and yield complete ones
            lines = buffer.split(b"\n")
            # First element may be incomplete (split mid-line), keep it
            buffer = lines[0]

            # Yield complete lines in reverse order
            for line in reversed(lines[1:]):
                line_str = line.decode("utf-8").strip()
                if line_str:
                    yield json.loads(line_str)

        # Yield the first line (was kept in buffer)
        if buffer:
            line_str = buffer.decode("utf-8").strip()
            if line_str:
                yield json.loads(line_str)


def iter_jsonl_bytes_reversed(
    content: bytes,
) -> Iterator[dict[str, object]]:
    """Iterate over JSONL records from bytes in reverse order.

    Parses the bytes and yields JSON objects from last line to first.
    This is the in-memory equivalent of iter_jsonl_reversed for file paths.

    Args:
        content: JSONL content as bytes (newline-delimited JSON records).

    Yields:
        Parsed JSON objects, from last record to first.
    """
    if not content:
        return

    # Split into lines and iterate in reverse
    lines = content.split(b"\n")

    for line in reversed(lines):
        line_str = line.decode("utf-8").strip()
        if line_str:
            yield json.loads(line_str)
