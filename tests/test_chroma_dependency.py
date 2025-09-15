import importlib.util
from importlib.machinery import ModuleSpec

import pytest


def test_operational_config_requires_chromadb_when_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure we fail loudly if chroma is selected but chromadb is missing.

    The OperationalConfig.__post_init__ enforces this policy for the sqlite backend.
    """
    # Simulate chromadb not being installed regardless of environment
    original_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "chromadb":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    # Ensure env overrides force the selection we want
    monkeypatch.setenv("RAGZOOM_BACKEND", "sqlite")
    monkeypatch.setenv("RAGZOOM_VECTOR_BACKEND", "chroma")

    from ragzoom.config import OperationalConfig, SecretStr

    with pytest.raises(ImportError):
        OperationalConfig(
            openai_api_key=SecretStr("test-key"),
        )


def test_operational_config_defaults_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Base configuration should default to the lightweight Python vector index."""
    monkeypatch.delenv("RAGZOOM_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("RAGZOOM_BACKEND", raising=False)

    from ragzoom.config import OperationalConfig

    config = OperationalConfig()
    assert config.vector_backend == "python"
