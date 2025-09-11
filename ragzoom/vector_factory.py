"""Factory for constructing VectorIndex implementations from config."""

from __future__ import annotations

import uuid
from pathlib import Path

from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.worktree_utils import (
    DEFAULT_VECTOR_DIR_NAME,
    get_default_vector_dir,
)


def create_vector_index(
    backend: str, database_url: str, embedding_model: str
) -> VectorIndex:
    """Create a VectorIndex instance based on backend hints.

    Args:
        backend: vector backend key ("python", "chroma", "pgvector")
        database_url: storage database URL (used to derive default vector dir)
        embedding_model: embedding model identifier used to produce vectors
    """
    b = backend.strip().lower()
    if b not in {"python", "chroma", "pgvector"}:
        raise NotImplementedError(f"Vector backend '{backend}' is not supported")

    # Derive a vector persistence directory near the sqlite database when possible
    persist_dir: str | None = None
    if database_url.startswith("sqlite:") and ":memory:" not in database_url:
        path_part = database_url.split("sqlite:///")[-1]
        base_dir = Path(path_part).parent
        persist_dir = str(base_dir / DEFAULT_VECTOR_DIR_NAME)
    else:
        # For ephemeral/in-memory or non-sqlite URLs, isolate each index to avoid cross-test contamination
        base = get_default_vector_dir(None)
        persist_dir = str(Path(base) / f"idx-{uuid.uuid4().hex}")

    if b == "python":
        from ragzoom.backends.vector_index_python import PythonVectorIndexAdapter

        return PythonVectorIndexAdapter(persist_dir, embedding_model)
    elif b == "chroma":
        try:
            from ragzoom.backends.vector_index_chroma import ChromaVectorIndexAdapter
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "chromadb is not installed but vector backend 'chroma' was selected. "
                "Install with `pip install chromadb` or set RAGZOOM_VECTOR_BACKEND=python."
            ) from e
        # Chroma requires directory path
        return ChromaVectorIndexAdapter(persist_dir, embedding_model)
    else:
        # Use Postgres database URL to create pgvector-backed index
        from ragzoom.backends.vector_index_pgvector import PgVectorIndexAdapter

        return PgVectorIndexAdapter(database_url, embedding_model)
