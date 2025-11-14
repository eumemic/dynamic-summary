"""Persisted runtime metadata shared between server and CLI tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from ragzoom.config import OperationalConfig
from ragzoom.worktree_utils import DEFAULT_DATA_DIR_NAME

_METADATA_DIR = ".ragzoom"
_METADATA_FILE = "runtime.json"


def _sqlite_base_dir(database_url: str | None) -> Path | None:
    if not database_url:
        return None
    url = database_url.strip()
    if not url.lower().startswith("sqlite"):
        return None

    parsed = urlparse(url)
    raw_path = parsed.path or ""
    if url.startswith("sqlite:////"):
        sqlite_path = Path(raw_path)
    else:
        sqlite_path = Path(raw_path.lstrip("/"))
        if not sqlite_path.is_absolute():
            sqlite_path = Path.cwd() / sqlite_path

    if sqlite_path.suffix:
        return sqlite_path.parent
    return sqlite_path


def _resolve_runtime_paths(
    config: OperationalConfig | None, *, ensure: bool
) -> tuple[Path, Path]:
    env_dir = os.environ.get("RAGZOOM_DATA_DIR")
    base = Path(env_dir) if env_dir else Path.cwd() / DEFAULT_DATA_DIR_NAME

    sqlite_dir = _sqlite_base_dir(config.database_url if config else None)
    if sqlite_dir is not None:
        base = sqlite_dir

    runtime_dir = base / _METADATA_DIR
    if ensure:
        runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir, base


def get_runtime_metadata_path(
    config: OperationalConfig | None, *, ensure: bool
) -> Path:
    directory, _ = _resolve_runtime_paths(config, ensure=ensure)
    return directory / _METADATA_FILE


def _relative_sqlite_path(database_url: str, base_dir: Path) -> str | None:
    if not database_url.lower().startswith("sqlite"):
        return None
    parsed = urlparse(database_url)
    raw_path = parsed.path or ""
    sqlite_path = Path(raw_path)
    if sqlite_path.suffix == "":
        return None
    try:
        rel = sqlite_path.relative_to(base_dir)
    except ValueError:
        return None
    return rel.as_posix()


def write_runtime_metadata(config: OperationalConfig) -> None:
    runtime_dir, base_dir = _resolve_runtime_paths(config, ensure=True)
    path = runtime_dir / _METADATA_FILE
    data = {
        "backend": config.backend,
        "database_url": config.database_url,
        "vector_backend": config.vector_backend,
    }
    sqlite_rel = _relative_sqlite_path(config.database_url, base_dir)
    if sqlite_rel:
        data["sqlite_path"] = sqlite_rel
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp_path, path)


def load_runtime_metadata(
    config: OperationalConfig | None = None,
) -> tuple[dict[str, object], Path] | None:
    runtime_dir, base_dir = _resolve_runtime_paths(config, ensure=False)
    path = runtime_dir / _METADATA_FILE
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return data, base_dir
    return None


def _resolve_database_url(
    metadata_url: str, metadata: dict[str, object], base_dir: Path
) -> str | None:
    url = metadata_url.strip()
    if not url:
        return None
    if not url.lower().startswith("sqlite"):
        return url

    sqlite_rel = metadata.get("sqlite_path")
    if isinstance(sqlite_rel, str):
        host_path = (base_dir / sqlite_rel).resolve()
        return f"sqlite:///{host_path.as_posix()}"

    parsed = urlparse(url)
    raw_path = parsed.path or ""
    sqlite_path = Path(raw_path)
    if sqlite_path.exists():
        return url

    container_data_root = Path("/data")
    try:
        rel = sqlite_path.relative_to(container_data_root)
    except ValueError:
        return None
    host_path = (base_dir / rel).resolve()
    return f"sqlite:///{host_path.as_posix()}"


def apply_runtime_metadata(config: OperationalConfig) -> OperationalConfig:
    loaded = load_runtime_metadata(config)
    if not loaded:
        return config
    metadata, base_dir = loaded

    vector_backend = metadata.get("vector_backend")
    if vector_backend and not os.environ.get("RAGZOOM_VECTOR_BACKEND"):
        config.vector_backend = str(vector_backend)

    backend = metadata.get("backend")
    if backend and not os.environ.get("RAGZOOM_BACKEND"):
        config.backend = str(backend)

    database_url = metadata.get("database_url")
    if database_url and not os.environ.get("RAGZOOM_DATABASE_URL"):
        resolved = _resolve_database_url(str(database_url), metadata, base_dir)
        if resolved:
            config.database_url = resolved

    return config
