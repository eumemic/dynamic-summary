"""Tests for memzoom session-start hook."""

import subprocess
from pathlib import Path


def test_session_start_hook_uses_correct_module() -> None:
    """Verify session-start.sh calls ragzoom CLI, not memory_service module.

    The hook should use `ragzoom set-session-pid` CLI command rather than
    the incorrect `python -m memory_service.ingestion.claude` module path.
    """
    hook_path = Path(__file__).parent.parent / ".claude" / "hooks" / "session-start.sh"
    assert hook_path.exists(), f"Hook script not found at {hook_path}"

    content = hook_path.read_text()

    # Should NOT contain the old incorrect module path
    assert "memory_service.ingestion.claude" not in content, (
        "Hook still uses incorrect module path 'memory_service.ingestion.claude'. "
        "Should use 'ragzoom set-session-pid' CLI command instead."
    )

    # Should use the ragzoom CLI command
    assert (
        "ragzoom set-session-pid" in content
    ), "Hook should call 'ragzoom set-session-pid' CLI command"


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
