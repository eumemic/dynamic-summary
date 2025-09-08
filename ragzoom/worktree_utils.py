"""Utilities for worktree detection and isolation."""

import logging
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

# Default configuration - can be overridden for different projects
DEFAULT_DATABASE_NAME = "ragzoom"
DEFAULT_DATABASE_URL_TEMPLATE = "postgresql+psycopg://localhost/{database_name}"
DEFAULT_CONTAINER_NAME = "ragzoom-postgres"
DEFAULT_DATA_DIR_NAME = "data"
DEFAULT_VECTOR_DIR_NAME = "vectors"


def _ensure_str_path(p: Path) -> str:
    """Return POSIX-style absolute path string for URLs."""
    try:
        return p.resolve().as_posix()
    except Exception:
        return str(p)


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
                logger.debug(f"Detected worktree: {parent.name}")
                return parent.name

    logger.debug("Not in a worktree environment")
    return None


def get_worktree_database_name(base_name: str = DEFAULT_DATABASE_NAME) -> str:
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

    Uses proper URL parsing to safely replace only the database component
    of the URL, avoiding false matches in other URL components.

    Args:
        base_url: Base database URL

    Returns:
        Database URL with worktree-specific database name
    """
    worktree_id = get_worktree_id()
    if not worktree_id:
        return base_url

    # Parse the URL to safely modify only the database component
    parsed = urlparse(base_url)

    # Only transform if the database path matches the default database name
    default_path = f"/{DEFAULT_DATABASE_NAME}"
    if parsed.path == default_path:
        worktree_db_name = get_worktree_database_name()
        # Replace the database name in the path
        new_parsed = parsed._replace(path=f"/{worktree_db_name}")
        transformed_url = urlunparse(new_parsed)
        logger.debug(f"Transformed URL: {base_url} -> {transformed_url}")
        return transformed_url

    # For paths like '/ragzoom?params', replace database but preserve query params
    if parsed.path.startswith(default_path) and (
        parsed.path == default_path or parsed.path.startswith(f"{default_path}?")
    ):
        worktree_db_name = get_worktree_database_name()
        new_path = parsed.path.replace(default_path, f"/{worktree_db_name}", 1)
        new_parsed = parsed._replace(path=new_path)
        transformed_url = urlunparse(new_parsed)
        logger.debug(f"Transformed URL with params: {base_url} -> {transformed_url}")
        return transformed_url

    logger.debug(f"URL unchanged (not {DEFAULT_DATABASE_NAME} database): {base_url}")
    return base_url


def get_default_database_url() -> str:
    """Get the default database URL for the current configuration.

    Returns:
        Default database URL, potentially worktree-specific
    """
    return DEFAULT_DATABASE_URL_TEMPLATE.format(
        database_name=get_worktree_database_name()
    )


def get_default_sqlite_path(base_dir: Path | None = None) -> Path:
    """Get a default file-backed SQLite path.

    Defaults to ./storage/ when base_dir is not provided. If base_dir is
    provided (e.g., via RAGZOOM_DATA_DIR), use it directly as the storage
    directory. The filename includes the worktree id when present.
    """
    if base_dir is None:
        data_dir = Path.cwd() / DEFAULT_DATA_DIR_NAME
    else:
        data_dir = Path(base_dir) / DEFAULT_DATA_DIR_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    wt = get_worktree_id()
    suffix = f"_{wt.replace('-', '_')}" if wt else ""
    return data_dir / f"{DEFAULT_DATABASE_NAME}{suffix}.db"


def get_default_sqlite_url(base_dir: Path | None = None) -> str:
    """Build a sqlite:/// URL for the default SQLite database file."""
    path = get_default_sqlite_path(base_dir)
    return f"sqlite:///{_ensure_str_path(path)}"


def get_default_vector_dir(base_dir: Path | None = None) -> Path:
    """Get default vector index directory under the data directory."""
    if base_dir is None:
        base = Path.cwd()
    else:
        base = Path(base_dir)
    return base / DEFAULT_DATA_DIR_NAME / DEFAULT_VECTOR_DIR_NAME
