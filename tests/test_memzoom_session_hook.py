"""Tests for memzoom session-start hook."""

import subprocess
from pathlib import Path


def test_session_start_hook_uses_pid_temp_file() -> None:
    """Verify session-start.sh writes PID temp file, not using deprecated CLI commands.

    The hook should write directly to /tmp/ragzoom-session-$PPID for MCP server
    lookup, not use the old CLI commands that wrote to state files.
    """
    hook_path = Path(__file__).parent.parent / ".claude" / "hooks" / "session-start.sh"
    assert hook_path.exists(), f"Hook script not found at {hook_path}"

    content = hook_path.read_text()

    # Should NOT contain the old incorrect module path
    assert "memory_service.ingestion.claude" not in content, (
        "Hook still uses incorrect module path 'memory_service.ingestion.claude'. "
        "Should write to PID temp file directly."
    )

    # Should NOT use the old ragzoom core CLI (moved to integration package)
    assert "ragzoom set-session-pid" not in content, (
        "Hook uses old 'ragzoom set-session-pid' command. "
        "Should write to PID temp file directly."
    )

    # Should NOT use the deprecated set-pid CLI command (removed in Phase 63)
    assert "ragzoom-claude-code set-pid" not in content, (
        "Hook uses deprecated 'ragzoom-claude-code set-pid' command. "
        "Should write to PID temp file directly."
    )

    # Should write session ID to PID-keyed temp file directly
    assert (
        '"/tmp/ragzoom-session-$PPID"' in content
    ), "Hook should write to /tmp/ragzoom-session-$PPID for MCP server lookup"


def test_session_start_hook_is_executable() -> None:
    """Verify the hook script is executable."""
    hook_path = Path(__file__).parent.parent / ".claude" / "hooks" / "session-start.sh"
    assert hook_path.exists(), f"Hook script not found at {hook_path}"

    # Check if file has executable permission
    assert hook_path.stat().st_mode & 0o111, "Hook script is not executable"


def test_session_start_hook_syntax_valid() -> None:
    """Verify the hook script has valid bash syntax."""
    hook_path = Path(__file__).parent.parent / ".claude" / "hooks" / "session-start.sh"
    assert hook_path.exists(), f"Hook script not found at {hook_path}"

    # Check bash syntax without executing
    result = subprocess.run(
        ["bash", "-n", str(hook_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Hook script has syntax errors: {result.stderr}"
