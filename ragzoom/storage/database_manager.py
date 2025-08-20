"""Database management for PostgreSQL with pgvector operations."""

import logging
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import sessionmaker

from ragzoom.config import OperationalConfig
from ragzoom.exceptions import InvalidOperationError
from ragzoom.models import Base, TreeNode

logger = logging.getLogger(__name__)

# Import pgvector registration function
try:
    from pgvector.psycopg import register_vector

    PGVECTOR_AVAILABLE = True
except ImportError:
    try:
        from pgvector.psycopg2 import register_vector

        PGVECTOR_AVAILABLE = True
    except ImportError:
        PGVECTOR_AVAILABLE = False
        register_vector = None


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

        # Register pgvector extension only for PostgreSQL connections
        if database_url.startswith("postgresql") and PGVECTOR_AVAILABLE:

            @event.listens_for(self.engine, "connect")
            def register_vector_extension(
                dbapi_conn: Any, connection_record: Any
            ) -> None:
                try:
                    # Get the underlying psycopg connection for pgvector registration
                    raw_conn = (
                        dbapi_conn.connection
                        if hasattr(dbapi_conn, "connection")
                        else dbapi_conn
                    )
                    if register_vector:
                        register_vector(raw_conn)
                except Exception as e:
                    logger.debug(f"pgvector registration note: {e}")
                    # Don't fail the connection - tables can still be created

        # Create vector extension first (required for Vector columns)
        self._create_vector_extension()

        # Create all tables (will only create missing ones)
        Base.metadata.create_all(self.engine)

        # Run migrations for existing databases
        self._run_migrations()

        self.SessionLocal = sessionmaker(bind=self.engine)

        # Cache expected embedding dimension for validation
        self._expected_embedding_dim = self._get_expected_embedding_dimension()

    def _get_expected_embedding_dimension(self) -> int | None:
        """Get expected embedding dimension from existing data.

        We no longer maintain hardcoded dimension info since OpenAI API
        is the source of truth. This method tries to infer from existing data.
        """
        # Try to infer from existing data
        try:
            with self.SessionLocal() as session:
                result = session.execute(select(TreeNode.embedding).limit(1)).first()
                if result and result[0] is not None:
                    return len(result[0])
        except Exception as e:
            logger.debug(f"Could not infer embedding dimension: {e}")

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
            raise InvalidOperationError("Embedding cannot be empty")

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
                f"Embedding dimension mismatch: expected {self._expected_embedding_dim}, "
                f"got {current_dim}. This suggests you're using a different embedding model "
                f"than the one used for existing data."
            )

    def _create_vector_extension(self) -> None:
        """Create the pgvector extension required for embedding storage.

        This must be done before creating tables that use Vector columns.
        """
        try:
            with self.engine.begin() as conn:
                # Create vector extension - this is required for pgvector Vector() columns
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.debug("Vector extension created successfully")
        except Exception as e:
            # This is a critical error - we can't create Vector columns without the extension
            error_msg = str(e).lower()
            if "permission denied" in error_msg:
                raise OSError(
                    f"\n❌ Unable to create vector extension: permission denied.\n\n"
                    f"This usually happens when the database user lacks superuser privileges.\n"
                    f"Try running 'ragzoom doctor' to check your setup.\n\n"
                    f"Technical error: {e}"
                )
            elif "could not access file" in error_msg or "no such file" in error_msg:
                raise OSError(
                    f"\n❌ Vector extension not available in this PostgreSQL installation.\n\n"
                    f"Make sure you're using the pgvector/pgvector Docker image.\n"
                    f"Run 'ragzoom doctor' to check your setup.\n\n"
                    f"Technical error: {e}"
                )
            else:
                # Extension might already exist, which is fine
                logger.debug(f"Vector extension note: {e}")

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

                # Add embedding column if it doesn't exist (for migration from old SQLite)
                conn.execute(
                    text(
                        """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'tree_nodes'
                            AND column_name = 'embedding'
                        ) THEN
                            ALTER TABLE tree_nodes
                            ADD COLUMN embedding vector;
                        END IF;
                    END $$;
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
