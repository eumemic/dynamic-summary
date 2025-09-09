"""Factory for constructing VectorIndex v2 implementations from config."""

from __future__ import annotations

from pathlib import Path

from ragzoom.contracts.vector_index_v2 import VectorIndex
from ragzoom.worktree_utils import (
    DEFAULT_VECTOR_DIR_NAME,
    get_default_vector_dir,
)


def create_vector_index(
    backend: str, database_url: str, embedding_model: str
) -> VectorIndex:
    """Create a VectorIndex v2 instance based on backend hints.

    Args:
        backend: vector backend key ("python", "chroma", "pgvector")
        database_url: storage database URL (used to derive default vector dir)
        embedding_model: embedding model identifier used to produce vectors
    """
    b = backend.strip().lower()
    if b not in {"python", "chroma", "pgvector"}:
        raise NotImplementedError(
            f"Vector backend '{backend}' is not supported yet in v2"
        )

    # Derive a vector persistence directory near the sqlite database when possible
    persist_dir: str | None = None
    if database_url.startswith("sqlite:") and ":memory:" not in database_url:
        path_part = database_url.split("sqlite:///")[-1]
        base_dir = Path(path_part).parent
        persist_dir = str(base_dir / DEFAULT_VECTOR_DIR_NAME)
    else:
        # Fallback to repository default location (worktree-aware)
        persist_dir = str(get_default_vector_dir(None))

    if b == "python":
        from ragzoom.backends.vector_index_v2_python import PythonVectorIndexV2

        return PythonVectorIndexV2(persist_dir, embedding_model)
    elif b == "chroma":
        try:
            from ragzoom.backends.vector_index_v2_chroma import ChromaVectorIndexV2
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "chromadb is not installed but vector backend 'chroma' was selected. "
                "Install with `pip install chromadb` or set RAGZOOM_VECTOR_BACKEND=python."
            ) from e
        # Chroma requires directory path
        return ChromaVectorIndexV2(persist_dir, embedding_model)  # type: ignore[arg-type]
    else:
        # pgvector backend placeholder for now
        raise NotImplementedError(
            "Vector backend 'pgvector' is planned but not implemented in this step."
        )
