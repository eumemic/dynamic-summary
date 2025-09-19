#!/usr/bin/env python3
"""Recreate incremental Hobbit chunk fixtures from the full text."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split test_data/the_hobbit.txt into evenly sized chunk files."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("test_data/the_hobbit.txt"),
        help="Path to the source Hobbit text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data/the_hobbit_incremental"),
        help="Directory where chunk files will be written.",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=10,
        help="Number of chunks to produce (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.chunks <= 0:
        raise ValueError("--chunks must be positive")

    source = args.source
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing chunk files to avoid mixing stale content.
    for existing in output_dir.glob("the_hobbit_chunk_*.txt"):
        existing.unlink()

    content = source.read_text(encoding="utf-8")
    total_len = len(content)
    if total_len == 0:
        raise ValueError(f"Source file {source} is empty")

    chunk_size = math.ceil(total_len / args.chunks)

    for index in range(args.chunks):
        start = index * chunk_size
        end = min(total_len, (index + 1) * chunk_size)
        if start >= total_len:
            break

        chunk_path = output_dir / f"the_hobbit_chunk_{index + 1}.txt"
        chunk_path.write_text(content[start:end], encoding="utf-8")


if __name__ == "__main__":
    main()
