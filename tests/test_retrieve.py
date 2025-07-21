"""Tests for the Retriever class and its methods."""

from unittest.mock import MagicMock

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.retrieve import Retriever
from ragzoom.store import Store


class TestRetriever:
    """Test suite for the Retriever class."""

    @pytest.fixture
    def setup_retriever(self, tmp_path, monkeypatch):
        """Set up a Retriever instance with a mock store."""
        monkeypatch.setenv(
            "RAGZOOM_SQLITE_DATABASE_URL", f"sqlite:///{tmp_path}/test.db"
        )
        monkeypatch.setenv("RAGZOOM_CHROMA_DB_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("RAGZOOM_OPENAI_API_KEY", "test-key")

        config = RagZoomConfig()
        store = MagicMock(spec=Store)
        retriever = Retriever(config, store)
        return retriever, store

    # Legacy test removed - was testing _extract_frontier method that no longer exists
    # The DP algorithm now handles tiling generation
