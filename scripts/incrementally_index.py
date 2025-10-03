#!/usr/bin/env python3
"""Utility to split a document into chunks and run incremental indexing."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Split a document into evenly sized chunks and drive incremental "
            "indexing through the RagZoom CLI. Any additional arguments after "
            "the known options are forwarded to the `ragzoom index` command."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the source document to split.",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=10,
        help="Number of chunks to generate (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory to write chunk files into. Defaults to a temporary "
            "directory that will be reported to stdout."
        ),
    )
    parser.add_argument(
        "--document-id",
        type=str,
        help=(
            "RagZoom document ID to use. Defaults to the source file name "
            "(without extension)."
        ),
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke the RagZoom CLI (default: current interpreter).",
    )

    # Capture additional arguments to forward to `ragzoom index`
    known_args, forward_args = parser.parse_known_args()
    return known_args, forward_args


def chunk_document(source: Path, chunks: int, output_dir: Path) -> Sequence[Path]:
    if chunks <= 0:
        raise ValueError("--chunks must be positive")
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    text = source.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"Source file {source} is empty")

    output_dir.mkdir(parents=True, exist_ok=True)

    for existing in output_dir.glob("*_chunk_*.txt"):
        existing.unlink()

    chunk_size = max(1, math.ceil(len(text) / chunks))
    written: list[Path] = []
    base_name = source.stem

    for index in range(chunks):
        start = index * chunk_size
        end = min(len(text), (index + 1) * chunk_size)
        if start >= len(text):
            break

        chunk_path = output_dir / f"{base_name}_chunk_{index + 1}.txt"
        chunk_path.write_text(text[start:end], encoding="utf-8")
        written.append(chunk_path)

    return written


def run_cli(python_exec: str, args: list[str]) -> None:
    subprocess.run([python_exec, "-m", "ragzoom.cli", *args], check=True)


def sanitize_forward_args(forward_args: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    for arg in forward_args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            continue
        if arg == "--append":
            continue
        if arg == "--document-id":
            skip_next = True
            continue
        filtered.append(arg)
    return filtered


def should_append_no_await(forward_args: list[str]) -> bool:
    """Return True when we should add --no-await-workers by default."""

    explicit_flags = {"--await-workers", "--no-await-workers"}

    for arg in forward_args:
        if arg in explicit_flags:
            return False
        if arg == "--telemetry" or arg.startswith("--telemetry="):
            return False

    return True


def main() -> None:
    args, forward_args = parse_args()

    if args.output_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="ragzoom_chunks_")
        output_dir = Path(tmp_dir)
        print(f"[info] Writing chunks to temporary directory: {output_dir}")
    else:
        output_dir = args.output_dir

    chunk_paths = chunk_document(args.source, args.chunks, output_dir)
    if not chunk_paths:
        raise RuntimeError("Chunking produced no files")

    document_id = args.document_id or args.source.stem

    print(f"[info] Clearing document '{document_id}' before indexing")
    run_cli(
        args.python,
        ["clear", "--document-id", document_id, "--confirm"],
    )

    sanitized_args = sanitize_forward_args(forward_args)
    add_no_await = should_append_no_await(forward_args)
    collect_requested = "--collect-telemetry" in forward_args
    if add_no_await:
        print(
            "[info] Using --no-await-workers; summarization will continue asynchronously."
        )
        if collect_requested:
            print(
                "[info] Telemetry will continue accumulating in the background. "
                "Export with `ragzoom telemetry-export --document-id "
                f"{document_id}` once workers are idle."
            )

    for chunk_path in chunk_paths:
        cmd = [
            "index",
            str(chunk_path),
            "--append",
            "--document-id",
            document_id,
        ]

        if add_no_await:
            cmd.append("--no-await-workers")

        cmd.extend(sanitized_args)
        run_cli(args.python, cmd)

    print("[done] Incremental indexing complete")


if __name__ == "__main__":
    main()
