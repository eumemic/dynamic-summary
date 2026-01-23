"""Tests for start-mcp-server script."""

import subprocess
from pathlib import Path


def test_start_mcp_server_uses_correct_module() -> None:
    """Verify start-mcp-server calls ragzoom.claude_memory.mcp_server.

    The script should use `ragzoom.claude_memory.mcp_server` module rather than
    the incorrect `memory_service.ingestion.claude.mcp_server` module path.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "start-mcp-server"
    assert script_path.exists(), f"MCP server script not found at {script_path}"

    content = script_path.read_text()

    # Should NOT contain the old incorrect module path
    assert "memory_service.ingestion.claude.mcp_server" not in content, (
        "Script still uses incorrect module path 'memory_service.ingestion.claude.mcp_server'. "
        "Should use 'ragzoom.claude_memory.mcp_server' instead."
    )

    # Should use the correct ragzoom module
    assert (
        "ragzoom.claude_memory.mcp_server" in content
    ), "Script should call 'python -m ragzoom.claude_memory.mcp_server'"


def test_start_mcp_server_is_executable() -> None:
    """Verify the MCP server script is executable."""
    script_path = Path(__file__).parent.parent / "scripts" / "start-mcp-server"
    assert script_path.exists(), f"MCP server script not found at {script_path}"

    # Check if file has executable permission
    assert script_path.stat().st_mode & 0o111, "MCP server script is not executable"


def test_start_mcp_server_syntax_valid() -> None:
    """Verify the MCP server script has valid bash syntax."""
    script_path = Path(__file__).parent.parent / "scripts" / "start-mcp-server"
    assert script_path.exists(), f"MCP server script not found at {script_path}"

    # Check bash syntax without executing
    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"MCP server script has syntax errors: {result.stderr}"
