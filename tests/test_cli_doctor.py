"""Tests for the `ragzoom doctor` CLI command."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from importlib.machinery import ModuleSpec

from pytest import CaptureFixture, MonkeyPatch

from ragzoom import cli as cli_module


def _run_doctor(*, capfd: CaptureFixture[str]) -> str:
    callback: Callable[[], None] | None = cli_module.doctor.callback
    assert callback is not None
    callback()
    captured = capfd.readouterr()
    return captured.out


def test_doctor_reports_chroma_available(
    monkeypatch: MonkeyPatch, capfd: CaptureFixture[str]
) -> None:
    """Doctor should confirm Chroma availability when installed."""
    monkeypatch.delenv("RAGZOOM_VECTOR_BACKEND", raising=False)
    output = _run_doctor(capfd=capfd)
    assert "Vector index: Chroma available" in output


def test_doctor_handles_missing_chroma(
    monkeypatch: MonkeyPatch, capfd: CaptureFixture[str]
) -> None:
    """Doctor should surface a helpful warning when chromadb is absent."""
    monkeypatch.delenv("RAGZOOM_VECTOR_BACKEND", raising=False)

    original_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "chromadb":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    output = _run_doctor(capfd=capfd)
    assert "Chroma not installed" in output
    assert "Install with" in output
