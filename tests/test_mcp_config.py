"""Tests for MCP configuration at repo root."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import pytest

REPO_ROOT = Path(__file__).parent.parent


class McpServerConfig(TypedDict):
    """Type for an MCP server configuration entry."""

    command: str
    args: list[str]


class McpConfig(TypedDict):
    """Type for the root MCP configuration."""

    mcpServers: dict[str, McpServerConfig]


@pytest.fixture
def mcp_config() -> McpConfig:
    """Load and return the MCP configuration."""
    mcp_config_path = REPO_ROOT / ".mcp.json"
    assert mcp_config_path.exists(), f".mcp.json not found at {mcp_config_path}"

    with open(mcp_config_path) as f:
        config: McpConfig = json.load(f)
    return config


def test_mcp_config_structure(mcp_config: McpConfig) -> None:
    """Verify .mcp.json has valid structure with ragzoom-memory server configured."""
    assert "mcpServers" in mcp_config, "Missing 'mcpServers' key"
    assert isinstance(mcp_config["mcpServers"], dict), "'mcpServers' should be a dict"

    assert (
        "ragzoom-memory" in mcp_config["mcpServers"]
    ), "Missing 'ragzoom-memory' server"

    server_config = mcp_config["mcpServers"]["ragzoom-memory"]
    assert (
        server_config["command"] == "bash"
    ), "Should use bash for proper environment handling"
    assert "args" in server_config, "Missing 'args' in server config"
    assert any(
        "start-mcp-server" in arg for arg in server_config["args"]
    ), "Should reference start-mcp-server script"


def test_mcp_server_script_exists() -> None:
    """Verify the start-mcp-server script referenced in config exists and is executable."""
    script_path = REPO_ROOT / "scripts" / "start-mcp-server"

    assert script_path.exists(), f"Script not found at {script_path}"
    assert script_path.stat().st_mode & 0o111, "Script should be executable"
