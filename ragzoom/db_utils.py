"""Database utility functions for PostgreSQL operations."""

import logging
import re

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


def create_temp_database(
    db_name: str,
    admin_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
) -> str:
    """Create a temporary PostgreSQL database.

    Args:
        db_name: Name of the database to create (must be alphanumeric + underscore)
        admin_url: PostgreSQL admin connection URL

    Returns:
        Database URL for the created database

    Raises:
        ValueError: If database name is invalid
        OSError: If database creation fails
    """
    # Validate database name (security: prevent SQL injection)
    if not re.match(r"^[a-zA-Z0-9_]+$", db_name):
        raise ValueError(f"Invalid database name: {db_name}")

    # Extract base URL for the new database
    base_url = "/".join(admin_url.split("/")[:-1])
    new_db_url = f"{base_url}/{db_name}"

    # Create database using admin connection
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            # Create database using psycopg's sql.Identifier for safe quoting
            from psycopg import Connection as PsycopgConnection
            from psycopg import sql

            dbapi_conn = conn.connection.dbapi_connection
            assert isinstance(dbapi_conn, PsycopgConnection)
            with dbapi_conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                )
            logger.debug(f"Created temporary database: {db_name}")

            # Create vector extension in the new database
            db_engine = create_engine(new_db_url)
            try:
                with db_engine.connect() as db_conn:
                    db_conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    db_conn.commit()
                    logger.debug(f"Created vector extension in database: {db_name}")
            finally:
                db_engine.dispose()

        return new_db_url
    except Exception as e:
        raise OSError(f"Failed to create database {db_name}: {e}")
    finally:
        admin_engine.dispose()


def drop_temp_database(
    db_name: str,
    admin_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
    force: bool = True,
) -> None:
    """Drop a temporary PostgreSQL database.

    Args:
        db_name: Name of the database to drop
        admin_url: PostgreSQL admin connection URL
        force: If True, terminate existing connections before dropping

    Raises:
        ValueError: If database name is invalid
    """
    # Validate database name (security: prevent SQL injection)
    if not re.match(r"^[a-zA-Z0-9_]+$", db_name):
        logger.warning(f"Invalid database name for cleanup: {db_name}")
        return

    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            if force:
                # Terminate existing connections
                conn.execute(
                    text(
                        "SELECT pg_terminate_backend(pg_stat_activity.pid) "
                        "FROM pg_stat_activity "
                        "WHERE pg_stat_activity.datname = :db_name "
                        "AND pid <> pg_backend_pid()"
                    ),
                    {"db_name": db_name},
                )

            # Drop database using psycopg's sql.Identifier for safe quoting
            from psycopg import Connection as PsycopgConnection
            from psycopg import sql

            dbapi_conn = conn.connection.dbapi_connection
            assert isinstance(dbapi_conn, PsycopgConnection)
            with dbapi_conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {}").format(
                        sql.Identifier(db_name)
                    )
                )
            logger.debug(f"Dropped temporary database: {db_name}")
    except Exception as e:
        logger.warning(f"Failed to drop database {db_name}: {e}")
    finally:
        admin_engine.dispose()


def get_temp_db_name(prefix: str = "ragzoom_temp") -> str:
    """Generate a unique temporary database name.

    Args:
        prefix: Prefix for the database name

    Returns:
        Unique database name
    """
    import uuid

    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{suffix}"
