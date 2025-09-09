"""Tests for worktree database isolation functionality."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragzoom.config import OperationalConfig
from ragzoom.docker_postgres import DockerPostgres
from ragzoom.worktree_utils import (
    get_default_sqlite_url,
    get_worktree_database_name,
    get_worktree_database_url,
    get_worktree_id,
)


class TestWorktreeDetection:
    """Test worktree ID detection logic."""

    def test_detects_worktree_from_current_directory(self) -> None:
        """Test detection when CWD is directly a worktree directory."""
        with patch("ragzoom.worktree_utils.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/path/to/worktrees/worktree-5")
            result = get_worktree_id()
            assert result == "worktree-5"

    def test_detects_worktree_from_nested_directory(self) -> None:
        """Test detection when CWD is nested inside a worktree."""
        with patch("ragzoom.worktree_utils.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/path/to/worktrees/worktree-3/src/module")
            result = get_worktree_id()
            assert result == "worktree-3"

    def test_returns_none_for_non_worktree_directory(self) -> None:
        """Test that non-worktree directories return None."""
        with patch("ragzoom.worktree_utils.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/home/user/project")
            result = get_worktree_id()
            assert result is None

    def test_returns_none_for_invalid_worktree_name(self) -> None:
        """Test that invalid worktree names are rejected."""
        test_cases = [
            Path("/path/worktree-"),  # Missing number
            Path("/path/worktree-abc"),  # Non-numeric suffix
            Path("/path/worktree-5-extra"),  # Extra suffix
            Path("/path/my-worktree-5"),  # Different prefix
        ]

        for test_path in test_cases:
            with patch("ragzoom.worktree_utils.Path.cwd") as mock_cwd:
                mock_cwd.return_value = test_path
                result = get_worktree_id()
                assert result is None, f"Should reject invalid path: {test_path}"

    def test_finds_worktree_in_parent_hierarchy(self) -> None:
        """Test that it searches up the directory hierarchy."""
        with patch("ragzoom.worktree_utils.Path.cwd") as mock_cwd:
            # Create a mock path object with proper parent hierarchy
            mock_path = MagicMock()
            mock_path.name = "components"

            # Set up the hierarchy: components -> src -> app -> worktree-2
            mock_parents = [
                MagicMock(name="src"),
                MagicMock(name="app"),
                MagicMock(name="worktree-2"),
                MagicMock(name="worktrees"),
                MagicMock(name="projects"),
                MagicMock(name=""),
            ]

            # Set the name attributes for the hierarchy
            mock_parents[0].name = "src"
            mock_parents[1].name = "app"
            mock_parents[2].name = "worktree-2"
            mock_parents[3].name = "worktrees"
            mock_parents[4].name = "projects"
            mock_parents[5].name = ""

            mock_path.parents = mock_parents
            mock_cwd.return_value = mock_path

            result = get_worktree_id()
            assert result == "worktree-2"


class TestDatabaseNameGeneration:
    """Test database name generation logic."""

    def test_generates_worktree_database_name(self) -> None:
        """Test database name generation for worktree."""
        with patch("ragzoom.worktree_utils.get_worktree_id", return_value="worktree-3"):
            result = get_worktree_database_name()
            assert result == "ragzoom_worktree_3"

    def test_uses_custom_base_name(self) -> None:
        """Test database name generation with custom base name."""
        with patch("ragzoom.worktree_utils.get_worktree_id", return_value="worktree-7"):
            result = get_worktree_database_name("custom_db")
            assert result == "custom_db_worktree_7"

    def test_returns_base_name_when_not_in_worktree(self) -> None:
        """Test that base name is returned when not in a worktree."""
        with patch("ragzoom.worktree_utils.get_worktree_id", return_value=None):
            result = get_worktree_database_name()
            assert result == "ragzoom"

    def test_replaces_hyphens_with_underscores(self) -> None:
        """Test that hyphens are replaced with underscores for PostgreSQL compatibility."""
        with patch(
            "ragzoom.worktree_utils.get_worktree_id", return_value="worktree-10"
        ):
            result = get_worktree_database_name()
            assert result == "ragzoom_worktree_10"
            assert "-" not in result


class TestDatabaseURLGeneration:
    """Test database URL generation logic."""

    def test_transforms_ragzoom_database_url(self) -> None:
        """Test URL transformation for ragzoom database."""
        with patch("ragzoom.worktree_utils.get_worktree_id", return_value="worktree-3"):
            base_url = "postgresql+psycopg://localhost/ragzoom"
            result = get_worktree_database_url(base_url)
            assert result == "postgresql+psycopg://localhost/ragzoom_worktree_3"

    def test_transforms_ragzoom_with_credentials(self) -> None:
        """Test URL transformation with user credentials."""
        with patch("ragzoom.worktree_utils.get_worktree_id", return_value="worktree-5"):
            base_url = "postgresql+psycopg://user:pass@localhost:5432/ragzoom"
            result = get_worktree_database_url(base_url)
            expected = (
                "postgresql+psycopg://user:pass@localhost:5432/ragzoom_worktree_5"
            )
            assert result == expected

    def test_leaves_non_ragzoom_urls_unchanged(self) -> None:
        """Test that non-ragzoom URLs are left unchanged."""
        with patch("ragzoom.worktree_utils.get_worktree_id", return_value="worktree-3"):
            base_url = "postgresql+psycopg://localhost/other_db"
            result = get_worktree_database_url(base_url)
            assert result == base_url

    def test_returns_unchanged_when_not_in_worktree(self) -> None:
        """Test URL is unchanged when not in a worktree."""
        with patch("ragzoom.worktree_utils.get_worktree_id", return_value=None):
            base_url = "postgresql+psycopg://localhost/ragzoom"
            result = get_worktree_database_url(base_url)
            assert result == base_url

    def test_handles_edge_case_urls(self) -> None:
        """Test handling of edge case URLs."""
        with patch("ragzoom.worktree_utils.get_worktree_id", return_value="worktree-1"):
            # Test URL with ragzoom in host (should not be modified)
            base_url = "postgresql+psycopg://ragzoom.example.com/mydb"
            result = get_worktree_database_url(base_url)
            assert result == base_url

            # Test URL with multiple ragzoom occurrences
            base_url = (
                "postgresql+psycopg://localhost/ragzoom?application_name=ragzoom_client"
            )
            result = get_worktree_database_url(base_url)
            # Should only replace the database name part
            expected = "postgresql+psycopg://localhost/ragzoom_worktree_1?application_name=ragzoom_client"
            assert result == expected


class TestOperationalConfigIntegration:
    """Test OperationalConfig integration with worktree isolation."""

    def test_applies_worktree_url_automatically(self) -> None:
        """Test that OperationalConfig automatically applies worktree URL."""
        with patch("ragzoom.worktree_utils.get_worktree_id", return_value="worktree-2"):
            # Ensure no environment override
            env_backup = os.environ.get("RAGZOOM_DATABASE_URL")
            if "RAGZOOM_DATABASE_URL" in os.environ:
                del os.environ["RAGZOOM_DATABASE_URL"]

            try:
                config = OperationalConfig()
                if config.backend == "postgres":
                    expected = "postgresql+psycopg://localhost/ragzoom_worktree_2"
                    assert config.database_url == expected
                else:
                    base_dir_env = os.environ.get("RAGZOOM_DATA_DIR")
                    expected_sqlite = (
                        get_default_sqlite_url(Path(base_dir_env))
                        if base_dir_env
                        else get_default_sqlite_url(None)
                    )
                    assert config.database_url == expected_sqlite
            finally:
                # Restore environment
                if env_backup:
                    os.environ["RAGZOOM_DATABASE_URL"] = env_backup

    def test_respects_environment_override(self) -> None:
        """Test that environment variable still takes precedence."""
        with patch("ragzoom.worktree_utils.get_worktree_id", return_value="worktree-3"):
            custom_url = "postgresql://custom.server/custom_db"
            os.environ["RAGZOOM_DATABASE_URL"] = custom_url

            try:
                config = OperationalConfig()
                assert config.database_url == custom_url
            finally:
                del os.environ["RAGZOOM_DATABASE_URL"]

    def test_no_change_when_not_in_worktree(self) -> None:
        """Test that config is unchanged when not in a worktree."""
        with patch("ragzoom.worktree_utils.get_worktree_id", return_value=None):
            # Ensure no environment override
            env_backup = os.environ.get("RAGZOOM_DATABASE_URL")
            if "RAGZOOM_DATABASE_URL" in os.environ:
                del os.environ["RAGZOOM_DATABASE_URL"]

            try:
                config = OperationalConfig()
                if config.backend == "postgres":
                    assert (
                        config.database_url == "postgresql+psycopg://localhost/ragzoom"
                    )
                else:
                    base_dir_env = os.environ.get("RAGZOOM_DATA_DIR")
                    expected_sqlite = (
                        get_default_sqlite_url(Path(base_dir_env))
                        if base_dir_env
                        else get_default_sqlite_url(None)
                    )
                    assert config.database_url == expected_sqlite
            finally:
                # Restore environment
                if env_backup:
                    os.environ["RAGZOOM_DATABASE_URL"] = env_backup


class TestDockerPostgresIntegration:
    """Test DockerPostgres integration with worktree databases."""

    def test_create_database_success(self) -> None:
        """Test successful database creation."""
        docker_pg = DockerPostgres()

        with patch.object(docker_pg, "container_running", return_value=True):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""

            with patch(
                "ragzoom.docker_postgres.subprocess.run", return_value=mock_result
            ):
                result = docker_pg.create_database("test_worktree_db")
                assert result is True

    def test_create_database_already_exists(self) -> None:
        """Test handling when database already exists - with IF NOT EXISTS, should succeed."""
        docker_pg = DockerPostgres()

        with patch.object(docker_pg, "container_running", return_value=True):
            # With IF NOT EXISTS, the command should succeed (return code 0)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""

            with patch(
                "ragzoom.docker_postgres.subprocess.run", return_value=mock_result
            ):
                result = docker_pg.create_database("existing_db")
                assert result is True

    def test_create_database_container_not_running(self) -> None:
        """Test handling when container is not running."""
        docker_pg = DockerPostgres()

        with patch.object(docker_pg, "container_running", return_value=False):
            result = docker_pg.create_database("test_db")
            assert result is False

    def test_create_database_failure(self) -> None:
        """Test handling of database creation failure."""
        docker_pg = DockerPostgres()

        with patch.object(docker_pg, "container_running", return_value=True):
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "permission denied"

            with patch(
                "ragzoom.docker_postgres.subprocess.run", return_value=mock_result
            ):
                result = docker_pg.create_database("test_db")
                assert result is False

    def test_ensure_database_exists(self) -> None:
        """Test ensure_database_exists method."""
        docker_pg = DockerPostgres()

        with patch.object(docker_pg, "ensure_running", return_value="base_url"):
            with patch.object(docker_pg, "create_database", return_value=True):
                with patch.object(
                    docker_pg, "get_connection_url", return_value="test_url"
                ) as mock_get_url:
                    result = docker_pg.ensure_database_exists("test_db")
                    assert result == "test_url"
                    mock_get_url.assert_called_once_with("test_db")

    def test_ensure_database_exists_creation_failure(self) -> None:
        """Test ensure_database_exists when creation fails."""
        docker_pg = DockerPostgres()

        with patch.object(docker_pg, "ensure_running"):
            with patch.object(docker_pg, "create_database", return_value=False):
                with pytest.raises(RuntimeError, match="Failed to create database"):
                    docker_pg.ensure_database_exists("test_db")

    def test_get_connection_url_with_database_name(self) -> None:
        """Test connection URL generation with custom database name."""
        docker_pg = DockerPostgres()
        result = docker_pg.get_connection_url("custom_db")
        expected = f"postgresql+psycopg://{docker_pg.USER}:{docker_pg.PASSWORD}@localhost:{docker_pg.DEFAULT_PORT}/custom_db"
        assert result == expected

    def test_get_connection_url_default(self) -> None:
        """Test connection URL generation with default database name."""
        docker_pg = DockerPostgres()
        result = docker_pg.get_connection_url()
        expected = f"postgresql+psycopg://{docker_pg.USER}:{docker_pg.PASSWORD}@localhost:{docker_pg.DEFAULT_PORT}/{docker_pg.DATABASE}"
        assert result == expected


class TestWorktreeIsolationEndToEnd:
    """End-to-end tests for worktree isolation."""

    def test_worktree_isolation_workflow(self) -> None:
        """Test complete worktree isolation workflow."""
        with patch(
            "ragzoom.worktree_utils.get_worktree_id", return_value="worktree-4"
        ) as mock_get_id:
            # Test 1: Worktree detection
            assert mock_get_id.return_value == "worktree-4"

            # Test 2: Database name generation
            db_name = get_worktree_database_name()
            assert db_name == "ragzoom_worktree_4"

            # Test 3: URL transformation
            base_url = "postgresql+psycopg://localhost/ragzoom"
            worktree_url = get_worktree_database_url(base_url)
            assert worktree_url == "postgresql+psycopg://localhost/ragzoom_worktree_4"

            # Test 4: Config integration (no env override)
            env_backup = os.environ.get("RAGZOOM_DATABASE_URL")
            if "RAGZOOM_DATABASE_URL" in os.environ:
                del os.environ["RAGZOOM_DATABASE_URL"]

            try:
                config = OperationalConfig()
                if config.backend == "postgres":
                    assert config.database_url == worktree_url
                else:
                    # SQLite backend ignores worktree for default URL
                    base_dir_env = os.environ.get("RAGZOOM_DATA_DIR")
                    expected_sqlite = (
                        get_default_sqlite_url(Path(base_dir_env))
                        if base_dir_env
                        else get_default_sqlite_url(None)
                    )
                    assert config.database_url == expected_sqlite
            finally:
                if env_backup:
                    os.environ["RAGZOOM_DATABASE_URL"] = env_backup

    def test_non_worktree_isolation_workflow(self) -> None:
        """Test workflow when not in a worktree environment."""
        with patch(
            "ragzoom.worktree_utils.get_worktree_id", return_value=None
        ) as mock_get_id:
            # Test 1: No worktree detection
            assert mock_get_id.return_value is None

            # Test 2: Default database name
            db_name = get_worktree_database_name()
            assert db_name == "ragzoom"

            # Test 3: Unchanged URL
            base_url = "postgresql+psycopg://localhost/ragzoom"
            result_url = get_worktree_database_url(base_url)
            assert result_url == base_url

            # Test 4: Default config
            env_backup = os.environ.get("RAGZOOM_DATABASE_URL")
            if "RAGZOOM_DATABASE_URL" in os.environ:
                del os.environ["RAGZOOM_DATABASE_URL"]

            try:
                config = OperationalConfig()
                if config.backend == "postgres":
                    assert config.database_url == base_url
                else:
                    base_dir_env = os.environ.get("RAGZOOM_DATA_DIR")
                    expected_sqlite = (
                        get_default_sqlite_url(Path(base_dir_env))
                        if base_dir_env
                        else get_default_sqlite_url(None)
                    )
                    assert config.database_url == expected_sqlite
            finally:
                if env_backup:
                    os.environ["RAGZOOM_DATABASE_URL"] = env_backup
