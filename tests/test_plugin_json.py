"""Tests for ragzoom-memory plugin.json manifest."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def plugin_json_path() -> Path:
    """Return the path to plugin.json."""
    return (
        Path(__file__).parent.parent
        / "integrations"
        / "claude-code"
        / "plugin"
        / ".claude-plugin"
        / "plugin.json"
    )


@pytest.fixture
def plugin_data(plugin_json_path: Path) -> dict[str, object]:
    """Load and return plugin.json data."""
    with open(plugin_json_path) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    return data


def test_plugin_json_exists(plugin_json_path: Path) -> None:
    """Verify plugin.json exists at expected location."""
    assert plugin_json_path.exists(), f"plugin.json not found at {plugin_json_path}"


def test_plugin_json_valid_json(plugin_data: dict[str, object]) -> None:
    """Verify plugin.json is valid JSON."""
    assert isinstance(plugin_data, dict)


def test_plugin_json_has_required_fields(plugin_data: dict[str, object]) -> None:
    """Verify plugin.json has all required fields."""
    assert "name" in plugin_data, "plugin.json missing 'name' field"
    assert "version" in plugin_data, "plugin.json missing 'version' field"
    assert "description" in plugin_data, "plugin.json missing 'description' field"
    assert "author" in plugin_data, "plugin.json missing 'author' field"
    author = plugin_data["author"]
    assert isinstance(author, dict)
    assert "name" in author, "plugin.json author missing 'name' field"


def test_plugin_json_name_is_ragzoom_memory(plugin_data: dict[str, object]) -> None:
    """Verify plugin.json has correct name."""
    name = plugin_data["name"]
    assert name == "ragzoom-memory", f"Expected name 'ragzoom-memory', got '{name}'"


def test_plugin_json_has_version(plugin_data: dict[str, object]) -> None:
    """Verify plugin.json has version field."""
    assert "version" in plugin_data, "plugin.json missing 'version' field"
    version = plugin_data["version"]
    assert isinstance(version, str)
    parts = version.split(".")
    assert len(parts) >= 2, f"Version '{version}' should be in semver format (X.Y.Z)"
