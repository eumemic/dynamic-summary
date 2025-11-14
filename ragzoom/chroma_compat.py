from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def chroma_panic_guard() -> Iterator[None]:
    """Provide a per-process temporary runtime directory for chroma."""

    tmp_root = Path(tempfile.gettempdir()) / "chroma-runtime"
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CHROMADB_PERSIST_DIRECTORY", tmp_root.as_posix())

    try:
        yield
    except Exception as exc:
        logger.error("Chroma runtime error: %s", exc)
        raise
