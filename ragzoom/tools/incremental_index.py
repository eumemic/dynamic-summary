"""Helpers for chunking documents and driving incremental indexing."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import grpc

from ragzoom.client.grpc_client import GrpcRagzoomClient
from ragzoom.constants import DEFAULT_GRPC_ADDRESS


@dataclass(frozen=True)
class IncrementalIndexResult:
    """Outcome information returned by incremental indexing."""

    document_id: str
    chunk_directory: Path
    chunk_paths: tuple[Path, ...]
    sanitized_args: tuple[str, ...]
    applied_no_await: bool
    telemetry_path: Path | None = None


def chunk_document(source: Path, chunks: int, output_dir: Path) -> tuple[Path, ...]:
    """Split *source* into *chunks* roughly-equal text files under *output_dir*."""

    if chunks <= 0:
        raise ValueError("chunks must be positive")
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

    return tuple(written)


def sanitize_forward_args(forward_args: Sequence[str]) -> list[str]:
    """Strip flags that will be re-specified by the helper."""

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


def should_append_no_await(forward_args: Iterable[str]) -> bool:
    """Return True when --no-await-workers should be injected."""

    explicit_flags = {"--await-workers", "--no-await-workers"}
    for arg in forward_args:
        if arg in explicit_flags:
            return False
        if arg == "--telemetry" or arg.startswith("--telemetry="):
            return False
    return True


def _run_cli(python_exec: str, args: Sequence[str]) -> None:
    subprocess.run([python_exec, "-m", "ragzoom.cli", *args], check=True)


def _wait_for_workers(
    address: str, document_id: str, echo: Callable[[str], None]
) -> None:
    """Block until all workers for *document_id* are idle."""

    echo(f"[info] Waiting for summarization workers to finish for '{document_id}'...")
    client = GrpcRagzoomClient(address, stream_timeout=None)
    try:
        for snapshot in client.iter_worker_snapshots():
            progress = snapshot.documents.get(document_id)
            if progress is None:
                if snapshot.idle:
                    break
                continue
            if progress.pending == 0 and progress.inflight == 0:
                break
    finally:
        client.close()


def run_incremental_indexing(
    *,
    source: Path,
    chunk_count: int,
    document_id: str | None = None,
    python_exec: str | None = None,
    forward_args: Sequence[str] | None = None,
    output_dir: Path | None = None,
    echo: Callable[[str], None] = print,
    validate: bool = False,
    telemetry: Path | None = None,
) -> IncrementalIndexResult:
    """Drive `ragzoom index --append` for an entire document.

    When ``validate`` is ``True`` the helper waits for worker idle, exports
    telemetry (to ``telemetry`` if provided), and runs
    ``ragzoom validate --complete`` before returning. When ``telemetry`` is
    provided without validation the helper still waits for completion and
    exports telemetry but skips the validation step.
    """

    python_exec = python_exec or sys.executable
    forward_args = list(forward_args or [])

    server_address = _resolve_server_address(forward_args)
    ensure_server_running(server_address)

    if output_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="ragzoom_chunks_"))
        echo(f"[info] Writing chunks to temporary directory: {tmp_dir}")
        chunk_root = tmp_dir
    else:
        chunk_root = output_dir

    chunk_paths = chunk_document(source, chunk_count, chunk_root)
    if not chunk_paths:
        raise RuntimeError("Chunking produced no files")

    doc_id = document_id or source.stem
    echo(f"[info] Clearing document '{doc_id}' before indexing")
    _run_cli(python_exec, ["clear", "--document-id", doc_id, "--confirm"])

    sanitized_args = sanitize_forward_args(forward_args)
    add_no_await = should_append_no_await(forward_args)

    telemetry_requested = telemetry is not None
    needs_completion = validate or telemetry_requested

    def _has_flag(flag: str, args: Sequence[str]) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)

    if needs_completion and not (
        _has_flag("--collect-telemetry", forward_args)
        or _has_flag("--collect-telemetry", sanitized_args)
        or _has_flag("--no-collect-telemetry", forward_args)
    ):
        echo("[info] Enabling telemetry collection for completion tracking")
        sanitized_args.append("--collect-telemetry")

    if add_no_await:
        echo(
            "[info] Using --no-await-workers; summarization will continue asynchronously."
        )
        if "--collect-telemetry" in forward_args:
            echo(
                "[info] Telemetry will continue accumulating in the background. "
                f"Export with `ragzoom telemetry-export --document-id {doc_id}` "
                "once workers are idle."
            )

    for chunk_path in chunk_paths:
        echo(f"[info] Queueing {chunk_path.name} for document '{doc_id}'")
        cmd = [
            "index",
            str(chunk_path),
            "--append",
            "--document-id",
            doc_id,
        ]
        if add_no_await:
            cmd.append("--no-await-workers")
        cmd.extend(sanitized_args)
        _run_cli(python_exec, cmd)

    telemetry_path: Path | None = None

    if needs_completion:
        _wait_for_workers(server_address, doc_id, echo)
        export_path: Path | None
        if telemetry_requested:
            assert telemetry is not None
            export_path = telemetry.resolve()
        elif validate:
            export_path = (chunk_root / f"{doc_id}_telemetry.json").resolve()
        else:
            export_path = None

        if export_path is not None:
            telemetry_path = export_path
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)

            echo(f"[info] Exporting telemetry to {telemetry_path}")
            _run_cli(
                python_exec,
                [
                    "telemetry-export",
                    "--document-id",
                    doc_id,
                    "--output",
                    str(telemetry_path),
                ],
            )

    if validate:
        validate_cmd: list[str] = ["validate", "--complete"]
        if telemetry_path is not None:
            validate_cmd.extend(["--telemetry-file", str(telemetry_path)])
        validate_cmd.append(doc_id)

        echo("[info] Running document validation")
        _run_cli(python_exec, validate_cmd)

    echo("[done] Incremental indexing complete")
    return IncrementalIndexResult(
        document_id=doc_id,
        chunk_directory=chunk_root,
        chunk_paths=chunk_paths,
        sanitized_args=tuple(sanitized_args),
        applied_no_await=add_no_await,
        telemetry_path=telemetry_path,
    )


def _resolve_server_address(forward_args: Sequence[str]) -> str:
    for index, arg in enumerate(forward_args):
        if arg == "--server-address" and index + 1 < len(forward_args):
            return forward_args[index + 1]
        if arg.startswith("--server-address="):
            return arg.split("=", 1)[1]
    env_address = os.environ.get("RAGZOOM_SERVER_ADDRESS")
    if env_address:
        return env_address
    return DEFAULT_GRPC_ADDRESS


def ensure_server_running(address: str, *, timeout: float = 2.0) -> None:
    """Raise RuntimeError if a RagZoom server is not reachable at *address*."""

    channel = grpc.insecure_channel(address)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "No RagZoom gRPC server detected at "
            f"{address}. Start the server before running incremental indexing."
        ) from exc
    finally:
        channel.close()


def cli_main(argv: Sequence[str] | None = None) -> None:
    """CLI entry-point used by scripts and tooling."""

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Split a document into evenly sized chunks and drive incremental "
            "indexing through the RagZoom CLI. Any additional arguments after "
            "the known options are forwarded to the `ragzoom index` command."
        )
    )
    parser.add_argument("source", type=Path, help="Path to the source document.")
    parser.add_argument(
        "--chunks",
        type=int,
        default=10,
        help="Number of chunks to generate (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write chunk files into.",
    )
    parser.add_argument(
        "--document-id",
        type=str,
        help="RagZoom document ID (defaults to the source file name).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter used to invoke the RagZoom CLI.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Wait for workers, export telemetry, and run document validation after indexing.",
    )
    parser.add_argument(
        "--telemetry",
        nargs="?",
        const=Path("telemetry.json"),
        type=Path,
        help=(
            "Export merged telemetry to FILE (default: telemetry.json) and "
            "wait for completion even without --validate."
        ),
    )

    args, forward_args = parser.parse_known_args(argv)
    try:
        run_incremental_indexing(
            source=args.source,
            chunk_count=args.chunks,
            document_id=args.document_id,
            python_exec=args.python,
            forward_args=forward_args,
            output_dir=args.output_dir,
            validate=args.validate,
            telemetry=args.telemetry,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "No RagZoom gRPC server detected" in message:
            print(f"[error] {message}", file=sys.stderr)
            sys.exit(1)
        raise


if __name__ == "__main__":
    cli_main()
