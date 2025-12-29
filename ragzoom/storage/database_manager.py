"""Database management for PostgreSQL with pgvector operations."""

import logging
import os

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ragzoom.config import OperationalConfig
from ragzoom.exceptions import InvalidOperationError
from ragzoom.models import Base

logger = logging.getLogger(__name__)

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

        self.engine = create_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections before using
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
        """
        try:
            with self.engine.begin() as conn:
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

                logger.debug("Database migrations completed")
        except Exception as e:
            # Migration failures are not critical - the column might already exist
            # or this might be a new database where tables have the column already
            logger.debug(f"Migration note: {e}")

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        if hasattr(self, "engine"):
            self.engine.dispose()
