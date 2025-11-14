from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from ragzoom.config import OperationalConfig
from ragzoom.runtime_metadata import apply_runtime_metadata


def _write_metadata(data_dir: Path, payload: Mapping[str, object]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = data_dir / ".ragzoom"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    target = runtime_dir / "runtime.json"
    target.write_text(json.dumps(payload), encoding="utf-8")


def test_apply_runtime_metadata_translates_container_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = data_dir / "sqlite.db"
    sqlite_path.touch()

    payload = {
        "backend": "sqlite",
        "database_url": "sqlite:////data/sqlite.db",
        "vector_backend": "chroma",
        "sqlite_path": "sqlite.db",
    }
    _write_metadata(data_dir, payload)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGZOOM_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("RAGZOOM_BACKEND", raising=False)
    monkeypatch.delenv("RAGZOOM_DATABASE_URL", raising=False)
    applied = apply_runtime_metadata(OperationalConfig())

    assert applied.vector_backend == "chroma"
    assert applied.database_url == f"sqlite:///{sqlite_path.resolve().as_posix()}"


def test_apply_runtime_metadata_handles_missing_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = data_dir / "sqlite.db"
    sqlite_path.touch()

    payload = {
        "backend": "sqlite",
        "database_url": "sqlite:////data/sqlite.db",
        "vector_backend": "chroma",
    }
    _write_metadata(data_dir, payload)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGZOOM_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("RAGZOOM_BACKEND", raising=False)
    monkeypatch.delenv("RAGZOOM_DATABASE_URL", raising=False)
    applied = apply_runtime_metadata(OperationalConfig())

    assert applied.database_url == f"sqlite:///{sqlite_path.resolve().as_posix()}"
