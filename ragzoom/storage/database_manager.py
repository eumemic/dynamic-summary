"""Database management for SQLite and ChromaDB operations."""

import logging
from typing import Any, cast

import chromadb
import numpy as np
from chromadb.config import Settings
from numpy.typing import NDArray
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ragzoom.config import OperationalConfig
from ragzoom.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and migrations for SQLite and ChromaDB."""

    def __init__(
        self, config: OperationalConfig, embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize database connections and run migrations.

        Args:
            config: Operational configuration with storage paths
            embedding_model: Name of embedding model (for dimension validation)
        """
        self.config = config
        self.embedding_model = embedding_model

        # Initialize SQLite
        self.engine = create_engine(
            config.sqlite_database_url, connect_args={"check_same_thread": False}
        )

        # Handle migration before creating tables with new schema
        self._run_migrations()

        # Create all tables (will only create missing ones)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Initialize Chroma
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="ragzoom_nodes",
            metadata={"hnsw:space": "cosine"},
        )

        # Cache expected embedding dimension for validation
        self._expected_embedding_dim = self._get_expected_embedding_dimension()

    def _get_expected_embedding_dimension(self) -> int | None:
        """Get expected embedding dimension from existing embeddings."""
        # Query ChromaDB for any existing embedding to determine dimension
        try:
            result = self.collection.peek(limit=1)
            embeddings = result.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                return len(embeddings[0])
        except Exception as e:
            logger.debug(f"Could not determine embedding dimension from ChromaDB: {e}")

        # If no embeddings exist, we can't determine dimension yet
        return None

    def validate_embedding_dimension(
        self, embedding: list[float] | NDArray[np.float64]
    ) -> None:
        """Validate that embedding has correct dimension.

        Args:
            embedding: Embedding vector to validate

        Raises:
            ValueError: If embedding dimension doesn't match expected
        """
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
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._expected_embedding_dim}, "
                f"got {current_dim}. This suggests you're using a different embedding model "
                f"than was used to create this database. Expected model: {self.embedding_model}"
            )

    def _run_migrations(self) -> None:
        """Run database migrations to update schema."""
        with self.engine.connect() as conn:
            # Check if documents table exists
            result = conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
                )
            )
            documents_table_exists = result.fetchone() is not None

            if not documents_table_exists:
                logger.info("Documents table does not exist, creating fresh schema")
                # Fresh database, no migrations needed
                return

            # Check and apply migrations for TreeNode table
            self._drop_column_migration(conn, "tree_nodes", "similarity_threshold")
            self._add_preceding_neighbor_column_migration(conn)
            self._add_height_column_migration(conn)
            self._backfill_node_heights(conn)
            self._add_document_model_columns_migration(conn)

            # Clean ChromaDB metadata
            self._clean_chromadb_metadata()

            # Commit all changes
            conn.commit()

    def _drop_column_migration(
        self, conn: Any, table_name: str, column_name: str
    ) -> None:
        """Drop a column from a table if it exists."""
        # Check if column exists
        result = conn.execute(text(f"PRAGMA table_info({table_name})"))
        columns = [row[1] for row in result.fetchall()]

        if column_name not in columns:
            return

        logger.info(f"Dropping column {column_name} from {table_name}")

        # Get all columns except the one to drop
        remaining_columns = [col for col in columns if col != column_name]

        # Create the column list for the new table
        if table_name == "tree_nodes":
            # Define the new schema for tree_nodes without similarity_threshold
            new_schema = """
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                left_child_id TEXT,
                right_child_id TEXT,
                span_start INTEGER NOT NULL,
                span_end INTEGER NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER DEFAULT 0 NOT NULL,
                is_pinned INTEGER DEFAULT 0,
                last_accessed DATETIME,
                access_count INTEGER DEFAULT 0,
                created_at DATETIME,
                document_id TEXT,
                FOREIGN KEY(parent_id) REFERENCES tree_nodes(id),
                FOREIGN KEY(document_id) REFERENCES documents(id)
            """
        else:
            # For other tables, we'd need to define their schema
            logger.warning(f"Unknown table {table_name} for migration")
            return

        # Step 1: Create new table with correct schema
        conn.execute(text(f"CREATE TABLE {table_name}_new ({new_schema})"))

        # Step 2: Copy data from old table to new table
        select_columns = ", ".join(remaining_columns)
        conn.execute(
            text(
                f"INSERT INTO {table_name}_new ({select_columns}) "
                f"SELECT {select_columns} FROM {table_name}"
            )
        )

        # Step 3: Drop old table and rename new table
        conn.execute(text(f"DROP TABLE {table_name}"))
        conn.execute(text(f"ALTER TABLE {table_name}_new RENAME TO {table_name}"))

        logger.info(f"Successfully dropped column {column_name} from {table_name}")

    def _add_preceding_neighbor_column_migration(self, conn: Any) -> None:
        """Add preceding_neighbor_id column to tree_nodes table if it doesn't exist."""
        # Check if column exists
        result = conn.execute(text("PRAGMA table_info(tree_nodes)"))
        columns = [row[1] for row in result.fetchall()]

        if "preceding_neighbor_id" not in columns:
            logger.info("Adding preceding_neighbor_id column to tree_nodes")
            conn.execute(
                text("ALTER TABLE tree_nodes ADD COLUMN preceding_neighbor_id TEXT")
            )

    def _add_height_column_migration(self, conn: Any) -> None:
        """Add height column to tree_nodes table if it doesn't exist."""
        # Check if column exists
        result = conn.execute(text("PRAGMA table_info(tree_nodes)"))
        columns = [row[1] for row in result.fetchall()]

        if "height" not in columns:
            logger.info("Adding height column to tree_nodes")
            conn.execute(
                text(
                    "ALTER TABLE tree_nodes ADD COLUMN height INTEGER NOT NULL DEFAULT 0"
                )
            )

    def _backfill_node_heights(self, conn: Any) -> None:
        """Backfill height values for existing nodes."""
        # Check if we need to backfill (any nodes with height = 0 that should have height > 0)
        result = conn.execute(
            text(
                "SELECT COUNT(*) FROM tree_nodes WHERE height = 0 AND (left_child_id IS NOT NULL OR right_child_id IS NOT NULL)"
            )
        )
        nodes_to_backfill = result.fetchone()[0]

        if nodes_to_backfill == 0:
            return

        logger.info(f"Backfilling height for {nodes_to_backfill} nodes")

        # Algorithm: repeatedly update parent heights based on child heights
        # until no more updates are needed
        max_iterations = 100
        for iteration in range(max_iterations):
            # Update heights of internal nodes based on their children
            result = conn.execute(
                text(
                    """
                    UPDATE tree_nodes
                    SET height = (
                        SELECT MAX(COALESCE(c.height, 0)) + 1
                        FROM tree_nodes c
                        WHERE c.parent_id = tree_nodes.id
                    )
                    WHERE (left_child_id IS NOT NULL OR right_child_id IS NOT NULL)
                    AND height < (
                        SELECT MAX(COALESCE(c.height, 0)) + 1
                        FROM tree_nodes c
                        WHERE c.parent_id = tree_nodes.id
                    )
                """
                )
            )

            rows_updated = result.rowcount
            logger.debug(f"Iteration {iteration + 1}: Updated {rows_updated} nodes")

            if rows_updated == 0:
                break

        logger.info("Height backfill completed")

    def _add_document_model_columns_migration(self, conn: Any) -> None:
        """Add embedding_model and summary_model columns to documents table if they don't exist."""
        # Check if columns exist
        result = conn.execute(text("PRAGMA table_info(documents)"))
        columns = [row[1] for row in result.fetchall()]

        if "embedding_model" not in columns:
            logger.info("Adding embedding_model column to documents")
            conn.execute(
                text(
                    f"ALTER TABLE documents ADD COLUMN embedding_model TEXT DEFAULT '{self.embedding_model}'"
                )
            )
            # Update existing rows
            conn.execute(
                text(
                    f"UPDATE documents SET embedding_model = '{self.embedding_model}' WHERE embedding_model IS NULL"
                )
            )

        if "summary_model" not in columns:
            logger.info("Adding summary_model column to documents")
            conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN summary_model TEXT DEFAULT 'gpt-4o-mini'"
                )
            )
            # Update existing rows with a reasonable default
            conn.execute(
                text(
                    "UPDATE documents SET summary_model = 'gpt-4o-mini' WHERE summary_model IS NULL"
                )
            )

    def _clean_chromadb_metadata(self) -> None:
        """Clean up ChromaDB metadata to remove None values."""
        try:
            # Get all embeddings
            all_data = self.collection.get()
            if not all_data.get("ids"):
                return

            ids = all_data["ids"]
            metadatas = all_data.get("metadatas", [])

            # Guard against None metadatas
            if metadatas is None:
                return

            # Clean metadata - replace None with empty string
            cleaned_metadatas = []
            for metadata in metadatas:
                if metadata is None:
                    metadata = {}
                cleaned_metadata = {}
                for key, value in metadata.items():
                    cleaned_metadata[key] = value if value is not None else ""
                cleaned_metadatas.append(cleaned_metadata)

            # Update all at once if we have any changes
            if cleaned_metadatas != metadatas:
                logger.info("Cleaning ChromaDB metadata...")
                # Delete and re-add with clean metadata
                embeddings = all_data.get("embeddings", [])
                documents = all_data.get("documents", [])

                # ChromaDB doesn't have bulk update, so we need to delete and re-add
                self.collection.delete(ids=ids)
                if embeddings and documents:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=cast(Any, cleaned_metadatas),
                        documents=documents,
                    )

        except Exception as e:
            logger.warning(f"Could not clean ChromaDB metadata: {e}")

    def close(self) -> None:
        """Close database connections."""
        if hasattr(self, "engine"):
            self.engine.dispose()
