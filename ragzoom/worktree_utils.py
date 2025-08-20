"""Utilities for worktree detection and isolation."""

import re
from pathlib import Path


def get_worktree_id() -> str | None:
    """Detect the current worktree ID from the working directory.

    Returns:
        Worktree ID (e.g., "worktree-3") or None if not in a worktree
    """
    cwd = Path.cwd()

    # Check if we're in a worktree directory structure
    # Pattern: .../worktrees/worktree-N/...
    for parent in [cwd] + list(cwd.parents):
        if parent.name.startswith("worktree-"):
            # Validate it matches the expected pattern
            if re.match(r"^worktree-\d+$", parent.name):
                return parent.name

    return None


def get_worktree_database_name(base_name: str = "ragzoom") -> str:
    """Get the database name for the current worktree.

    Args:
        base_name: Base database name to use

    Returns:
        Database name, either base_name or base_name_worktree_N
    """
    worktree_id = get_worktree_id()
    if worktree_id:
        # Replace hyphens with underscores for PostgreSQL compatibility
        worktree_suffix = worktree_id.replace("-", "_")
        return f"{base_name}_{worktree_suffix}"
    return base_name


def get_worktree_database_url(base_url: str) -> str:
    """Get the database URL for the current worktree.

    Args:
        base_url: Base database URL

    Returns:
        Database URL with worktree-specific database name
    """
    worktree_id = get_worktree_id()
    if not worktree_id:
        return base_url

    # Parse the URL and replace the database name
    # Handle both postgresql+psycopg://localhost/ragzoom and postgresql://...
    if "/ragzoom" in base_url:
        worktree_db_name = get_worktree_database_name()
        return base_url.replace("/ragzoom", f"/{worktree_db_name}")

    return base_url
