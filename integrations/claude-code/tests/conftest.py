"""Pytest configuration for Claude Code integration tests.

Re-exports shared fixtures and pytest hooks from the main test suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Add repo root and tests directory to path to access shared fixtures
_repo_root = Path(__file__).parent.parent.parent.parent
_tests_dir = _repo_root / "tests"
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

# Import shared fixtures using importlib to avoid name collision
_conftest_path = _tests_dir / "conftest.py"
_spec = importlib.util.spec_from_file_location("main_conftest", _conftest_path)
assert _spec is not None and _spec.loader is not None
_main_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main_conftest)

# Re-export fixtures
FakeTranscriptClient = _main_conftest.FakeTranscriptClient
fake_transcript_client = _main_conftest.fake_transcript_client
sqlite_backend = _main_conftest.sqlite_backend
sqlite_store_factory = _main_conftest.sqlite_store_factory
storage_backend = _main_conftest.storage_backend
base_config = _main_conftest.base_config
indexer_runtime_harness = _main_conftest.indexer_runtime_harness

# Re-export pytest hooks for CLI options (--max-test-duration, --use-real-store)
# The main conftest uses _safe_addoption so duplicates are silently ignored
pytest_addoption = _main_conftest.pytest_addoption
