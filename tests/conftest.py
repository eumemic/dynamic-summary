"""Pytest configuration and fixtures for RagZoom tests."""

import shutil
import tempfile

import pytest

from ragzoom.api import RagZoomService
from ragzoom.config import RagZoomConfig
from ragzoom.store import Store


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration with temporary directories."""
    config = RagZoomConfig(
        sqlite_database_url=f"sqlite:///{temp_dir}/test.db",
        chroma_persist_directory=temp_dir,
    )
    return config


@pytest.fixture
def store(test_config):
    """Create a Store instance that will be properly closed after test."""
    store = Store(test_config)
    yield store
    # Ensure proper cleanup
    store.close()


@pytest.fixture
def ragzoom_service(test_config):
    """Create a RagZoomService instance that will be properly closed after test."""
    # Temporarily override the config
    import ragzoom.api
    original_config_class = ragzoom.api.RagZoomConfig

    # Monkey patch to use test config
    ragzoom.api.RagZoomConfig = lambda: test_config

    service = RagZoomService()
    yield service

    # Cleanup
    service.close()

    # Restore original config class
    ragzoom.api.RagZoomConfig = original_config_class
