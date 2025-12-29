"""Store factory for selecting the configured backend.

This module exposes only backend-agnostic factory functions that return a
StorageBackend. The legacy PostgreSQL manager is internal to the postgres
backend implementation and not exported.
"""

import logging
import os

from ragzoom.config import OperationalConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.worktree_utils import DEFAULT_VECTOR_DIR_NAME

logger = logging.getLogger(__name__)


def create_store(
    config: OperationalConfig, embedding_model: str = "text-embedding-3-small"
) -> StorageBackend:
    """Create a store based on OperationalConfig backend.

    - sqlite: uses SQLiteStorageBackend with a persistent Python/Chroma vector index
    - postgres: uses StoreManager (pgvector) with optional Docker auto-start
    """
    if config.backend == "sqlite" or config.database_url.lower().startswith("sqlite"):
        try:
            from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
        except Exception as e:  # pragma: no cover - import failures are surfaced
            raise OSError(f"SQLite backend unavailable: {e}")

        # Derive a vector persistence directory near the sqlite db file
        vec_dir: str | None = None
        url = config.database_url
        if url.startswith("sqlite:") and ":memory:" not in url:
            # naive parse: sqlite:////abs or sqlite:///rel
            path_part = url.split("sqlite:///")[-1]
            db_dir = os.path.dirname(path_part)
            vec_dir = os.path.join(db_dir, DEFAULT_VECTOR_DIR_NAME)

        return SQLiteStorageBackend(
            url=config.database_url,
            vector_backend=config.vector_backend,
            vector_persist_dir=vec_dir,
        )

    # Postgres path (with optional Docker auto-start)
    database_url = config.database_url

    from ragzoom.worktree_utils import (
        DEFAULT_DATABASE_NAME,
        DEFAULT_DATABASE_URL_TEMPLATE,
        get_worktree_database_name,
    )

    expected_base_url = DEFAULT_DATABASE_URL_TEMPLATE.format(
        database_name=DEFAULT_DATABASE_NAME
    )
    expected_worktree_db_name = get_worktree_database_name()
    expected_worktree_url = DEFAULT_DATABASE_URL_TEMPLATE.format(
        database_name=expected_worktree_db_name
    )

    should_auto_start = (
        (database_url == expected_base_url or database_url == expected_worktree_url)
        and not os.getenv("RAGZOOM_DATABASE_URL")
        and not os.getenv("RAGZOOM_NO_DOCKER")
    )

    if should_auto_start:
        try:
            from ragzoom.docker_postgres import DockerPostgres

            docker_pg = DockerPostgres()
            if database_url == expected_worktree_url:
                database_url = docker_pg.ensure_database_exists(
                    expected_worktree_db_name
                )
                logger.info(
                    f"✅ PostgreSQL ready with worktree database: {expected_worktree_db_name}"
                )
            else:
                database_url = docker_pg.ensure_running()
                logger.info("✅ PostgreSQL ready in Docker container")

            config = OperationalConfig(
                openai_api_key=config.openai_api_key,
                backend="postgres",
                database_url=database_url,
                vector_backend=config.vector_backend,
                cache_size=config.cache_size,
            )
        except ImportError:
            logger.debug("Docker PostgreSQL management not available")
        except OSError:
            raise
        except Exception as e:
            logger.debug(f"Auto-start failed: {e}")
            raise OSError(
                f"\n❌ Failed to start PostgreSQL automatically.\n\n"
                f"Run 'ragzoom doctor' to diagnose the issue.\n"
                f"Error: {str(e)}"
            )

    from ragzoom.backends.postgres_backend import PostgresStorageBackend

    return PostgresStorageBackend(config, embedding_model)


def create_store_with_docker(
    config: OperationalConfig, embedding_model: str = "text-embedding-3-small"
) -> StorageBackend:
    """Legacy factory retained for CLI/test compatibility.

    Delegates to create_store(). Docker will only be used for postgres.
    """
    return create_store(config, embedding_model)
