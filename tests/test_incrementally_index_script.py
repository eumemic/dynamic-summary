"""Unit tests for the incremental indexing helper utilities."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pytest import CaptureFixture

from ragzoom.constants import DEFAULT_GRPC_ADDRESS
from ragzoom.server.state import ServerState
from ragzoom.tools import incremental_index as incremental


def test_sanitize_forward_args_strips_append_and_document_id() -> None:
    args = ["--append", "--document-id", "doc", "--debug"]
    sanitized = incremental.sanitize_forward_args(args)
    assert sanitized == ["--debug"]


def test_should_append_no_await_defaults_to_true() -> None:
    assert incremental.should_append_no_await([])


def test_should_append_no_await_respects_explicit_flags() -> None:
    assert not incremental.should_append_no_await(
        ["--await-workers"]
    )  # user wants to wait
    assert not incremental.should_append_no_await(
        ["--no-await-workers"]
    )  # already present


def test_should_append_no_await_avoids_telemetry_conflict() -> None:
    assert not incremental.should_append_no_await(["--telemetry", "out.json"])
    assert not incremental.should_append_no_await(["--telemetry=out.json"])


def test_run_incremental_indexing_does_not_modify_vector_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[list[str]] = []

    def fake_run_cli(_python: str, args: Sequence[str]) -> None:
        recorded.append(list(args))

    def fake_chunk_document(_source: Path, _count: int, _dir: Path) -> tuple[Path, ...]:
        return (Path("chunk_a.txt"),)

    monkeypatch.setattr(incremental, "_run_cli", fake_run_cli)
    monkeypatch.setattr(incremental, "chunk_document", fake_chunk_document)
    monkeypatch.setattr(incremental, "ensure_server_running", lambda address: None)
    monkeypatch.delenv("RAGZOOM_VECTOR_BACKEND", raising=False)

    incremental.run_incremental_indexing(
        source=Path("/tmp/doc.txt"),
        chunk_count=1,
        python_exec="python",
        forward_args=[],
        echo=lambda *_: None,
    )

    # The first call clears the document; the second performs indexing with injected flag.
    index_cmd = recorded[1]
    assert "--vector-backend" not in index_cmd
    assert "RAGZOOM_VECTOR_BACKEND" not in os.environ


def test_run_incremental_indexing_respects_server_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[str] = []

    def fake_run_cli(_python: str, args: Sequence[str]) -> None:
        pass

    def fake_chunk_document(_source: Path, _count: int, _dir: Path) -> tuple[Path, ...]:
        return (Path("chunk_a.txt"),)

    def fake_ensure(address: str, *, timeout: float = 2.0) -> None:
        recorded.append(address)

    monkeypatch.setattr(incremental, "_run_cli", fake_run_cli)
    monkeypatch.setattr(incremental, "chunk_document", fake_chunk_document)
    monkeypatch.setattr(incremental, "ensure_server_running", fake_ensure)

    incremental.run_incremental_indexing(
        source=Path("/tmp/doc.txt"),
        chunk_count=1,
        python_exec="python",
        forward_args=["--server-address", "0.0.0.0:5555"],
        echo=lambda *_: None,
    )

    assert recorded == ["0.0.0.0:5555"]

    recorded.clear()
    monkeypatch.delenv("RAGZOOM_SERVER_ADDRESS", raising=False)
    incremental.run_incremental_indexing(
        source=Path("/tmp/doc.txt"),
        chunk_count=1,
        python_exec="python",
        forward_args=[],
        echo=lambda *_: None,
    )

    assert recorded == [DEFAULT_GRPC_ADDRESS]


def test_run_incremental_indexing_runs_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list[tuple[str, ...]] = []

    def fake_run_cli(_python: str, args: Sequence[str]) -> None:
        recorded.append(tuple(str(arg) for arg in args))

    def fake_chunk_document(
        source: Path, _count: int, output_dir: Path
    ) -> tuple[Path, ...]:
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk = output_dir / "chunk_1.txt"
        chunk.write_text("chunk", encoding="utf-8")
        return (chunk,)

    wait_calls: list[tuple[str, str]] = []

    def fake_wait(address: str, document_id: str, _echo: Callable[[str], None]) -> None:
        wait_calls.append((address, document_id))

    doc = tmp_path / "doc.txt"
    doc.write_text("hello world", encoding="utf-8")
    monkeypatch.setattr(incremental, "_run_cli", fake_run_cli)
    monkeypatch.setattr(incremental, "chunk_document", fake_chunk_document)
    monkeypatch.setattr(incremental, "_wait_for_workers", fake_wait)
    monkeypatch.setattr(incremental, "ensure_server_running", lambda *_: None)

    telemetry_path = tmp_path / "telemetry.json"
    result = incremental.run_incremental_indexing(
        source=doc,
        chunk_count=1,
        python_exec=sys.executable,
        forward_args=["--server-address", "127.0.0.1:5555", "--collect-telemetry"],
        output_dir=tmp_path / "chunks",
        echo=lambda *_: None,
        validate=True,
        telemetry=telemetry_path,
    )

    assert wait_calls == [("127.0.0.1:5555", "doc")]
    telemetry_cmd = (
        "telemetry-export",
        "--document-id",
        "doc",
        "--output",
        str(telemetry_path),
    )
    validate_cmd = (
        "validate",
        "--complete",
        "--telemetry-file",
        str(telemetry_path),
        "doc",
    )
    index_cmd = next(cmd for cmd in recorded if cmd[0] == "index")
    assert telemetry_cmd in recorded
    assert validate_cmd in recorded
    assert "--collect-telemetry" in index_cmd
    assert result.telemetry_path == telemetry_path


def test_run_incremental_indexing_exports_without_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list[tuple[str, ...]] = []

    def fake_run_cli(_python: str, args: Sequence[str]) -> None:
        recorded.append(tuple(str(arg) for arg in args))

    def fake_chunk_document(
        source: Path, _count: int, output_dir: Path
    ) -> tuple[Path, ...]:
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk = output_dir / "chunk_1.txt"
        chunk.write_text("chunk", encoding="utf-8")
        return (chunk,)

    wait_calls: list[tuple[str, str]] = []

    def fake_wait(address: str, document_id: str, _echo: Callable[[str], None]) -> None:
        wait_calls.append((address, document_id))

    doc = tmp_path / "doc.txt"
    doc.write_text("hello world", encoding="utf-8")
    monkeypatch.setattr(incremental, "_run_cli", fake_run_cli)
    monkeypatch.setattr(incremental, "chunk_document", fake_chunk_document)
    monkeypatch.setattr(incremental, "_wait_for_workers", fake_wait)
    monkeypatch.setattr(incremental, "ensure_server_running", lambda *_: None)

    telemetry_path = tmp_path / "telemetry.json"
    result = incremental.run_incremental_indexing(
        source=doc,
        chunk_count=1,
        python_exec=sys.executable,
        forward_args=["--server-address", "127.0.0.1:5555"],
        output_dir=tmp_path / "chunks",
        echo=lambda *_: None,
        validate=False,
        telemetry=telemetry_path,
    )

    assert wait_calls == [("127.0.0.1:5555", "doc")]
    telemetry_cmd = (
        "telemetry-export",
        "--document-id",
        "doc",
        "--output",
        str(telemetry_path),
    )
    assert telemetry_cmd in recorded
    assert all(cmd[0] != "validate" for cmd in recorded)
    assert result.telemetry_path == telemetry_path


def test_cli_main_reports_missing_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("hello", encoding="utf-8")

    def fake_run_incremental_indexing(**_kwargs: object) -> None:
        raise RuntimeError("No RagZoom gRPC server detected at fake")

    monkeypatch.setattr(
        incremental,
        "run_incremental_indexing",
        fake_run_incremental_indexing,
    )

    with pytest.raises(SystemExit) as excinfo:
        incremental.cli_main([str(doc)])

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "No RagZoom gRPC server detected" in err


def test_cli_main_defaults_telemetry_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("hello", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_incremental_indexing(
        **kwargs: object,
    ) -> incremental.IncrementalIndexResult:
        captured.update(kwargs)
        return incremental.IncrementalIndexResult(
            document_id="doc",
            chunk_directory=tmp_path,
            chunk_paths=tuple(),
            sanitized_args=tuple(),
            applied_no_await=False,
        )

    monkeypatch.setattr(
        incremental,
        "run_incremental_indexing",
        fake_run_incremental_indexing,
    )

    incremental.cli_main([str(doc), "--telemetry"])

    assert Path("telemetry.json") == captured.get("telemetry")


@pytest.mark.asyncio
@pytest.mark.slow_threshold(30.0)
async def test_incremental_index_cli_validates_document(
    grpc_test_environment: tuple[str, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    address, state = grpc_test_environment
    monkeypatch.setenv("RAGZOOM_SERVER_ADDRESS", address)
    monkeypatch.setenv("RAGZOOM_DATABASE_URL", state.operational_config.database_url)
    monkeypatch.setenv(
        "RAGZOOM_VECTOR_BACKEND",
        state.operational_config.vector_backend or "python",
    )

    telemetry_path = tmp_path / "telemetry.json"
    chunk_root = tmp_path / "chunks"
    doc_path = Path("test_data/smoke_test_larger.txt")

    result = await asyncio.to_thread(
        incremental.run_incremental_indexing,
        source=doc_path,
        chunk_count=2,
        python_exec=sys.executable,
        forward_args=[
            "--server-address",
            address,
            "--collect-telemetry",
            "--no-progress",
        ],
        output_dir=chunk_root,
        echo=lambda *_: None,
        validate=True,
        telemetry=telemetry_path,
    )

    assert telemetry_path.exists()
    assert result.document_id == "smoke_test_larger"
