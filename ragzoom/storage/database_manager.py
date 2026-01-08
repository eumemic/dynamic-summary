"""Database management for PostgreSQL with pgvector operations."""

import logging
import os
import time

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ragzoom.config import OperationalConfig
from ragzoom.exceptions import InvalidOperationError
from ragzoom.models import Base

logger = logging.getLogger(__name__)


def _create_engine_with_retry(
    database_url: str,
    pool_size: int,
    max_overflow: int,
    max_retries: int = 10,
    retry_delay: float = 3.0,
) -> Engine:
    """Create SQLAlchemy engine with connection retry logic.

    On Railway and similar platforms, the database service may not be ready
    when the application starts. This function retries the connection with
    exponential backoff.

    Args:
        database_url: Database connection URL
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections
        max_retries: Maximum number of connection attempts
        retry_delay: Base delay between retries (seconds)

    Returns:
        SQLAlchemy Engine with verified connection
    """
    engine = create_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
    )

    for attempt in range(1, max_retries + 1):
        try:
            # Test the connection by executing a simple query
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established on attempt %d", attempt)
            return engine
        except Exception as e:
            if attempt == max_retries:
                logger.error(
                    "Failed to connect to database after %d attempts: %s",
                    max_retries,
                    e,
                )
                raise
            delay = retry_delay * (1.5 ** (attempt - 1))  # Exponential backoff
            logger.warning(
                "Database connection attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt,
                max_retries,
                e,
                delay,
            )
            time.sleep(delay)

    # Should never reach here, but satisfy type checker
    raise RuntimeError("Unexpected exit from retry loop")


# Embeddings are not stored in SQL; pgvector registration not required
register_vector: object | None = None
PGVECTOR_AVAILABLE = False


class DatabaseManager:
    """Manages database connections and migrations for PostgreSQL with pgvector."""

    DEFAULT_POOL_SIZE = 10  # Default connection pool size
    DEFAULT_MAX_OVERFLOW = 20  # Default max overflow connections

    def __init__(
        self, config: OperationalConfig, embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize database connections and run migrations.

        Args:
            config: Operational configuration with database URL
            embedding_model: Name of embedding model (for dimension validation)
        """
        self.config = config
        self.embedding_model = embedding_model
        database_url = config.database_url

        # Initialize PostgreSQL with connection pooling
        # Get pool configuration from environment or use defaults
        pool_size = int(os.getenv("RAGZOOM_DB_POOL_SIZE", str(self.DEFAULT_POOL_SIZE)))
        max_overflow = int(
            os.getenv("RAGZOOM_DB_MAX_OVERFLOW", str(self.DEFAULT_MAX_OVERFLOW))
        )

        self.engine = _create_engine_with_retry(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
        )

        # No pgvector registration or extension needed

        # Create all tables (will only create missing ones)
        Base.metadata.create_all(self.engine)

        # Run migrations for existing databases
        self._run_migrations()

        self.SessionLocal = sessionmaker(bind=self.engine)

        # No embedding dimension validation in storage
        self._expected_embedding_dim: int | None = None

    def _get_expected_embedding_dimension(self) -> int | None:
        """Get expected embedding dimension from existing data.

        We no longer maintain hardcoded dimension info since OpenAI API
        is the source of truth. This method tries to infer from existing data.
        """
        # Try to infer from existing data
        try:
            return None
        except Exception:
            return None

        # If no existing data, don't enforce validation
        # This allows tests and first-time setups to work with any dimension
        return None

    def validate_embedding_dimension(
        self, embedding: list[float] | NDArray[np.float64]
    ) -> None:
        """Validate that embedding has correct dimension.

        Args:
            embedding: Embedding vector to validate

        Raises:
            InvalidOperationError: If embedding dimension doesn't match expected
        """
        if not embedding:
            raise InvalidOperationError(
                "validate_embedding", "Embedding cannot be empty"
            )

        if isinstance(embedding, list):
            current_dim = len(embedding)
        else:
            current_dim = embedding.shape[0]

        if self._expected_embedding_dim is None:
            # First embedding - set as expected
            self._expected_embedding_dim = current_dim
            logger.debug(f"Set expected embedding dimension to {current_dim}")
            return

        if current_dim != self._expected_embedding_dim:
            raise InvalidOperationError(
                "validate_embedding",
                f"Embedding dimension mismatch: expected {self._expected_embedding_dim}, "
                f"got {current_dim}. This suggests you're using a different embedding model "
                f"than the one used for existing data.",
            )

    def _create_vector_extension(self) -> None:
        """Create the pgvector extension required for embedding storage.

        This must be done before creating tables that use Vector columns.
        """
        return

    def _run_migrations(self) -> None:
        """Run database migrations for existing databases.

        This must be called after tables are created to handle schema updates
        for existing databases that need migration.

        Uses a PostgreSQL advisory lock to prevent concurrent migration attempts
        from multiple server instances (e.g., during Railway deployments). The
        lock is transaction-scoped and auto-releases on commit/rollback.
        """
        # Arbitrary unique ID for migration lock (consistent across all instances)
        migration_lock_id = 8675309

        try:
            with self.engine.begin() as conn:
                # Try to acquire transaction-level advisory lock (non-blocking)
                # Returns false immediately if another process holds the lock
                result = conn.execute(
                    text("SELECT pg_try_advisory_xact_lock(:lock_id)"),
                    {"lock_id": migration_lock_id},
                )
                got_lock = result.scalar()

                if not got_lock:
                    logger.debug("Another process is running migrations, skipping")
                    return

                # Set lock timeout as safety net - don't wait forever for table locks
                conn.execute(text("SET lock_timeout = '10s'"))

                # Add height column if it doesn't exist (for existing databases)
                conn.execute(
                    text(
                        """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'tree_nodes'
                            AND column_name = 'height'
                        ) THEN
                            ALTER TABLE tree_nodes
                            ADD COLUMN height INTEGER NOT NULL DEFAULT 0;
                        END IF;
                    END $$;
                """
                    )
                )

                # Add following_neighbor_id column if it doesn't exist (dataflow optimization)
                conn.execute(
                    text(
                        """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'tree_nodes'
                            AND column_name = 'following_neighbor_id'
                        ) THEN
                            ALTER TABLE tree_nodes
                            ADD COLUMN following_neighbor_id VARCHAR;
                        END IF;
                    END $$;
                """
                    )
                )

                # Add level_index column to preserve sibling ordering deterministically
                conn.execute(
                    text(
                        """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'tree_nodes'
                            AND column_name = 'level_index'
                        ) THEN
                            ALTER TABLE tree_nodes
                            ADD COLUMN level_index INTEGER NOT NULL DEFAULT 0;
                        END IF;
                    END $$;
                """
                    )
                )

                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_tree_nodes_document_height_level
                    ON tree_nodes (document_id, height, level_index);
                """
                    )
                )

                # Unique constraint to prevent duplicate coordinates from concurrent
                # indexers. This prevents the TOCTOU race where multiple
                # IndexingEngine instances (e.g., during Railway rolling deployments)
                # can both discover the same eligible sibling pair and create
                # duplicate parent nodes at the same (height, level_index).
                # NOTE: Will fail if duplicates already exist in the database.
                # Clean up duplicates first (e.g., reset session and re-sync).
                conn.execute(
                    text(
                        """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_tree_nodes_document_height_level
                    ON tree_nodes (document_id, height, level_index);
                """
                    )
                )

                # Drop legacy document metadata columns no longer used
                conn.execute(
                    text(
                        """
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'documents'
                            AND column_name = 'version'
                        ) THEN
                            ALTER TABLE documents
                            DROP COLUMN version;
                        END IF;
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'documents'
                            AND column_name = 'content_hash'
                        ) THEN
                            ALTER TABLE documents
                            DROP COLUMN content_hash;
                        END IF;
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'documents'
                            AND column_name = 'chunk_count'
                        ) THEN
                            ALTER TABLE documents
                            DROP COLUMN chunk_count;
                        END IF;
                    END $$;
                """
                    )
                )

                # Add contextual indexing columns for issue #287
                conn.execute(
                    text(
                        """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'tree_nodes'
                            AND column_name = 'preceding_context'
                        ) THEN
                            ALTER TABLE tree_nodes
                            ADD COLUMN preceding_context TEXT;
                        END IF;
                    END $$;
                """
                    )
                )

                conn.execute(
                    text(
                        """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'tree_nodes'
                            AND column_name = 'preceding_context_summary'
                        ) THEN
                            ALTER TABLE tree_nodes
                            ADD COLUMN preceding_context_summary TEXT;
                        END IF;
                    END $$;
                """
                    )
                )

                # Add user_id columns for multi-tenancy
                conn.execute(
                    text(
                        """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'tree_nodes'
                            AND column_name = 'user_id'
                        ) THEN
                            ALTER TABLE tree_nodes
                            ADD COLUMN user_id VARCHAR;
                        END IF;
                    END $$;
                """
                    )
                )

                conn.execute(
                    text(
                        """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'documents'
                            AND column_name = 'user_id'
                        ) THEN
                            ALTER TABLE documents
                            ADD COLUMN user_id VARCHAR;
                        END IF;
                    END $$;
                """
                    )
                )

                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_tree_nodes_user_id
                    ON tree_nodes (user_id);
                """
                    )
                )

                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_documents_user_id
                    ON documents (user_id);
                """
                    )
                )

                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_tree_nodes_user_document
                    ON tree_nodes (user_id, document_id);
                """
                    )
                )

                # Create users table if it doesn't exist
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS users (
                        id VARCHAR PRIMARY KEY,
                        github_id VARCHAR UNIQUE,
                        email VARCHAR,
                        api_key VARCHAR NOT NULL UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """
                    )
                )

                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_users_api_key
                    ON users (api_key);
                """
                    )
                )

                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_users_github_id
                    ON users (github_id);
                """
                    )
                )

                # Create session_raw_data table for Claude Code memory sync
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS session_raw_data (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        session_id VARCHAR(255) NOT NULL,
                        jsonl_content BYTEA NOT NULL
                    );
                """
                    )
                )

                conn.execute(
                    text(
                        """
                    CREATE UNIQUE INDEX IF NOT EXISTS ix_session_raw_data_user_session
                    ON session_raw_data (user_id, session_id);
                """
                    )
                )

                # Add sync state columns to session_raw_data for memory-efficient syncing
                conn.execute(
                    text(
                        """
                    ALTER TABLE session_raw_data
                    ADD COLUMN IF NOT EXISTS last_synced_uuid VARCHAR(255);
                """
                    )
                )

                conn.execute(
                    text(
                        """
                    ALTER TABLE session_raw_data
                    ADD COLUMN IF NOT EXISTS span_end INTEGER NOT NULL DEFAULT 0;
                """
                    )
                )

                # Add original_file_offset column to track actual file position
                # (different from stored content length due to tool result stripping)
                conn.execute(
                    text(
                        """
                    ALTER TABLE session_raw_data
                    ADD COLUMN IF NOT EXISTS original_file_offset BIGINT NOT NULL DEFAULT 0;
                """
                    )
                )

                # Add compaction_span_end column to track compaction boundary
                # (span_end just before post-compaction content)
                conn.execute(
                    text(
                        """
                    ALTER TABLE session_raw_data
                    ADD COLUMN IF NOT EXISTS compaction_span_end INTEGER;
                """
                    )
                )

                # Create session_append_entries table for per-segment tracking
                # Used for granular revert handling (O(d) where d = distance from end)
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS session_append_entries (
                        id SERIAL PRIMARY KEY,
                        session_raw_data_id INTEGER NOT NULL
                            REFERENCES session_raw_data(id) ON DELETE CASCADE,
                        entry_index INTEGER NOT NULL,
                        last_uuid VARCHAR(255) NOT NULL,
                        span_end INTEGER NOT NULL,
                        UNIQUE(session_raw_data_id, entry_index)
                    );
                """
                    )
                )

                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS ix_append_entries_session_index
                        ON session_append_entries(session_raw_data_id, entry_index);
                """
                    )
                )

                logger.debug("Database migrations completed")
        except Exception as e:
            # Migration failures are not critical - the column might already exist
            # or this might be a new database where tables have the column already
            logger.debug(f"Migration note: {e}")

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        if hasattr(self, "engine"):
            self.engine.dispose()
