"""Automatic Docker PostgreSQL management for RagZoom development."""

import logging
import subprocess
import time

from ragzoom.worktree_utils import DEFAULT_CONTAINER_NAME, DEFAULT_DATABASE_NAME

logger = logging.getLogger(__name__)


class DockerPostgres:
    """Automatically manage PostgreSQL container for development."""

    CONTAINER_NAME = DEFAULT_CONTAINER_NAME
    IMAGE = "pgvector/pgvector:pg16"
    DEFAULT_PORT = 5432
    DATABASE = DEFAULT_DATABASE_NAME
    USER = "postgres"
    PASSWORD = "postgres"

    def __init__(self, container_name: str | None = None):
        """Initialize Docker PostgreSQL manager.

        Args:
            container_name: Custom container name (defaults to ragzoom-postgres)
        """
        self.container_name = container_name or self.CONTAINER_NAME

    def docker_installed(self) -> bool:
        """Check if Docker is installed (regardless of daemon status)."""
        try:
            subprocess.run(
                ["docker", "--version"], capture_output=True, check=True, timeout=5
            )
            return True
        except FileNotFoundError:
            return False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Docker is installed but may have issues
            return True

    def docker_daemon_running(self) -> bool:
        """Check if Docker daemon is running."""
        if not self.docker_installed():
            return False
        try:
            subprocess.run(
                ["docker", "info"], capture_output=True, check=True, timeout=10
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def docker_available(self) -> bool:
        """Check if Docker is available and running (legacy compatibility)."""
        return self.docker_installed() and self.docker_daemon_running()

    def _check_container_status(self, include_stopped: bool = False) -> bool:
        """Helper method to check container status."""
        try:
            cmd = ["docker", "ps"]
            if include_stopped:
                cmd.append("-a")
            cmd.extend(
                [
                    "--filter",
                    f"name={self.container_name}",
                    "--format",
                    "{{.Names}}",
                ]
            )

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return self.container_name in result.stdout.strip().split("\n")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def container_exists(self) -> bool:
        """Check if the PostgreSQL container exists."""
        return self._check_container_status(include_stopped=True)

    def container_running(self) -> bool:
        """Check if the PostgreSQL container is currently running."""
        return self._check_container_status(include_stopped=False)

    def create_container(self) -> bool:
        """Create the PostgreSQL container."""
        try:
            logger.info(f"Creating PostgreSQL container: {self.container_name}")
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    self.container_name,
                    "-e",
                    f"POSTGRES_PASSWORD={self.PASSWORD}",
                    "-e",
                    f"POSTGRES_DB={self.DATABASE}",
                    "-p",
                    f"{self.DEFAULT_PORT}:5432",
                    self.IMAGE,
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )

            logger.info("PostgreSQL container created successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create PostgreSQL container: {e}")
            if e.stderr:
                logger.debug(f"Docker error details: {e.stderr.decode()}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Timeout while creating PostgreSQL container")
            return False

    def start_container(self) -> bool:
        """Start the PostgreSQL container."""
        try:
            logger.info(f"Starting PostgreSQL container: {self.container_name}")
            subprocess.run(
                ["docker", "start", self.container_name],
                check=True,
                capture_output=True,
                timeout=30,
            )

            logger.info("PostgreSQL container started successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start PostgreSQL container: {e}")
            if e.stderr:
                logger.debug(f"Docker error details: {e.stderr.decode()}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Timeout while starting PostgreSQL container")
            return False

    def wait_for_ready(self, timeout: int = 30) -> bool:
        """Wait for PostgreSQL to be ready to accept connections.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if PostgreSQL is ready, False if timeout reached
        """
        logger.info("Waiting for PostgreSQL to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Test PostgreSQL readiness using pg_isready inside the container
                result = subprocess.run(
                    [
                        "docker",
                        "exec",
                        self.container_name,
                        "pg_isready",
                        "-U",
                        self.USER,
                        "-d",
                        self.DATABASE,
                    ],
                    capture_output=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    logger.info("PostgreSQL is ready")
                    return True

            except subprocess.TimeoutExpired:
                pass

            time.sleep(1)

        logger.warning(f"PostgreSQL not ready after {timeout} seconds")
        return False

    def get_connection_url(self, database_name: str | None = None) -> str:
        """Get the connection URL for the PostgreSQL container.

        Args:
            database_name: Optional database name to use instead of default
        """
        db_name = database_name or self.DATABASE
        return f"postgresql+psycopg://{self.USER}:{self.PASSWORD}@localhost:{self.DEFAULT_PORT}/{db_name}"

    def ensure_running(self) -> str:
        """Ensure PostgreSQL container is running and return connection URL.

        Returns:
            Connection URL string

        Raises:
            EnvironmentError: If Docker is not available or not running
            RuntimeError: If container cannot be started or is not ready
        """
        if not self.docker_installed():
            raise OSError(
                "\n❌ Docker is not installed.\n\n"
                "To use RagZoom, you need Docker:\n"
                "  1. Install Docker Desktop from https://docker.com\n"
                "  2. Start Docker Desktop\n"
                "  3. Try again\n\n"
                "Alternative: Set RAGZOOM_DATABASE_URL to use existing PostgreSQL"
            )

        if not self.docker_daemon_running():
            raise OSError(
                "\n❌ Docker is installed but not running.\n\n"
                "Please start Docker:\n"
                "  • macOS: Start Docker Desktop from Applications\n"
                "  • Linux: sudo systemctl start docker\n"
                "  • Windows: Start Docker Desktop\n\n"
                "Then try again."
            )

        # Track whether we need to wait for readiness
        needs_readiness_check = False

        # Check if container exists
        if not self.container_exists():
            logger.info("PostgreSQL container does not exist, creating...")
            if not self.create_container():
                raise RuntimeError(
                    f"Failed to create PostgreSQL container: {self.container_name}"
                )
            needs_readiness_check = True

        # Check if container is running
        if not self.container_running():
            logger.info("PostgreSQL container is not running, starting...")
            if not self.start_container():
                raise RuntimeError(
                    f"Failed to start PostgreSQL container: {self.container_name}"
                )
            needs_readiness_check = True

        # Only wait for PostgreSQL readiness if we just started/created the container
        if needs_readiness_check:
            if not self.wait_for_ready():
                raise RuntimeError(
                    "PostgreSQL container started but is not accepting connections"
                )

        return self.get_connection_url()

    def stop_container(self) -> bool:
        """Stop the PostgreSQL container."""
        try:
            logger.info(f"Stopping PostgreSQL container: {self.container_name}")
            subprocess.run(
                ["docker", "stop", self.container_name],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def remove_container(self, force: bool = False) -> bool:
        """Remove the PostgreSQL container.

        Args:
            force: Force remove even if running
        """
        try:
            cmd = ["docker", "rm"]
            if force:
                cmd.append("-f")
            cmd.append(self.container_name)

            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            logger.info(f"Removed PostgreSQL container: {self.container_name}")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def get_status(self) -> dict[str, bool | str | None]:
        """Get detailed status of the PostgreSQL setup.

        Returns:
            Dictionary with status information
        """
        status: dict[str, bool | str | None] = {
            "docker_available": self.docker_available(),
            "container_exists": False,
            "container_running": False,
            "postgres_ready": False,
            "connection_url": None,
        }

        if status["docker_available"]:
            status["container_exists"] = self.container_exists()
            if status["container_exists"]:
                status["container_running"] = self.container_running()
                if status["container_running"]:
                    status["postgres_ready"] = self.wait_for_ready(timeout=5)
                    if status["postgres_ready"]:
                        status["connection_url"] = self.get_connection_url()

        return status

    def create_database(self, database_name: str) -> bool:
        """Create an additional database in the PostgreSQL container.

        Checks for database existence first, then creates if needed.

        Args:
            database_name: Name of the database to create

        Returns:
            True if database was created or already exists, False on error
        """
        if not self.container_running():
            logger.error("PostgreSQL container is not running")
            return False

        try:
            # First check if database exists
            logger.debug(f"Checking if database exists: {database_name}")
            check_result = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container_name,
                    "psql",
                    "-U",
                    self.USER,
                    "-d",
                    self.DATABASE,
                    "-tAc",
                    f"SELECT 1 FROM pg_database WHERE datname='{database_name}';",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # If database exists, return True
            if check_result.returncode == 0 and check_result.stdout.strip() == "1":
                logger.debug(f"Database {database_name} already exists")
                return True

            # Database doesn't exist, create it
            logger.debug(f"Creating database: {database_name}")
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container_name,
                    "psql",
                    "-U",
                    self.USER,
                    "-d",
                    self.DATABASE,
                    "-c",
                    f'CREATE DATABASE "{database_name}";',
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Check only for success return code
            if result.returncode == 0:
                logger.debug(f"Database {database_name} created successfully")
                return True
            else:
                # Log the actual error for debugging
                logger.error(
                    f"Failed to create database {database_name}: {result.stderr.strip()}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while creating database {database_name}")
            return False
        except Exception as e:
            logger.error(f"Error creating database {database_name}: {e}")
            return False

    def ensure_database_exists(self, database_name: str) -> str:
        """Ensure a specific database exists and return its connection URL.

        Args:
            database_name: Name of the database to ensure exists

        Returns:
            Connection URL for the specified database

        Raises:
            RuntimeError: If database cannot be created
        """
        # First ensure the PostgreSQL container is running
        self.ensure_running()

        # Then ensure the specific database exists
        if not self.create_database(database_name):
            raise RuntimeError(f"Failed to create database: {database_name}")

        return self.get_connection_url(database_name)
