"""Tests for summary-model endpoint config plumbing.

The LiteLLM refactor adds two saved IndexConfig fields (summary_api_base,
summary_api_key) and finally honors three env knobs (RAGZOOM_SUMMARY_MODEL,
RAGZOOM_SUMMARY_API_BASE, RAGZOOM_SUMMARY_API_KEY). Precedence is
CLI > env > config file > default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragzoom.config import IndexConfig


def test_summary_endpoint_fields_default_none() -> None:
    cfg = IndexConfig.load()
    assert cfg.summary_api_base is None
    assert cfg.summary_api_key is None


def test_load_from_file_populates_summary_endpoint(tmp_path: Path) -> None:
    """A config file round-trips through from_dict into the new fields."""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "summary_model": "anthropic/claude-opus-4-8",
                "summary_api_base": "https://litellm-proxy.example.com",
                "summary_api_key": "sk-ap-secret",
            }
        )
    )
    built = IndexConfig.load(config_path=config_file)
    assert built.summary_model == "anthropic/claude-opus-4-8"
    assert built.summary_api_base == "https://litellm-proxy.example.com"
    assert built.summary_api_key is not None
    assert built.summary_api_key.get_secret_value() == "sk-ap-secret"
    # Secret must redact in string form.
    assert "sk-ap-secret" not in str(built.summary_api_key)


def test_env_summary_model_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: RAGZOOM_SUMMARY_MODEL used to be dead documentation."""
    monkeypatch.setenv("RAGZOOM_SUMMARY_MODEL", "gpt-5.5")
    cfg = IndexConfig.load()
    assert cfg.summary_model == "gpt-5.5"


def test_env_summary_api_base_and_key_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGZOOM_SUMMARY_API_BASE", "https://litellm-proxy.example.com")
    monkeypatch.setenv("RAGZOOM_SUMMARY_API_KEY", "sk-ap-fromenv")
    cfg = IndexConfig.load()
    assert cfg.summary_api_base == "https://litellm-proxy.example.com"
    assert cfg.summary_api_key is not None
    assert cfg.summary_api_key.get_secret_value() == "sk-ap-fromenv"


def test_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI option must win over env var."""
    monkeypatch.setenv("RAGZOOM_SUMMARY_MODEL", "gpt-5.5")
    cfg = IndexConfig.load(summary_model="claude-opus-4-8")
    assert cfg.summary_model == "claude-opus-4-8"


def test_env_overrides_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Env var must win over a config file value."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"summary_model": "gpt-4o"}))
    monkeypatch.setenv("RAGZOOM_SUMMARY_MODEL", "gpt-5.5")
    cfg = IndexConfig.load(config_path=config_file)
    assert cfg.summary_model == "gpt-5.5"


def test_file_overrides_default(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"summary_model": "gpt-4o"}))
    cfg = IndexConfig.load(config_path=config_file)
    assert cfg.summary_model == "gpt-4o"


def test_replace_preserves_summary_endpoint() -> None:
    cfg = IndexConfig.load(
        summary_api_base="https://litellm-proxy.example.com",
        summary_api_key="sk-ap-x",
    )
    replaced = cfg.replace(max_parallelism=8)
    assert replaced.summary_api_base == "https://litellm-proxy.example.com"
    assert replaced.summary_api_key is not None
    assert replaced.summary_api_key.get_secret_value() == "sk-ap-x"
