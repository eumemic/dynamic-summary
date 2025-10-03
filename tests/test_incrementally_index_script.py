"""Unit tests for the incremental indexing helper utilities."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from pytest import CaptureFixture

from ragzoom.constants import DEFAULT_GRPC_ADDRESS
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
