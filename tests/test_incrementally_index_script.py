"""Unit tests for the incremental indexing helper script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    scripts_dir = (
        Path(__file__).resolve().parent.parent / "scripts" / "incrementally_index.py"
    )
    spec = importlib.util.spec_from_file_location(
        "incrementally_index_module", scripts_dir
    )
    if spec is None:
        raise RuntimeError("Failed to load incrementally_index.py for testing")
    loader = spec.loader
    if loader is None:
        raise RuntimeError("Failed to load incrementally_index.py for testing")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    loader.exec_module(module)
    return module


incremental = _load_script_module()


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
