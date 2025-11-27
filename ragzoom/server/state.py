"""Shared server state for the gRPC runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.indexing import IndexerRuntime
from ragzoom.query_log import QueryLog
from ragzoom.server.append_executor import AppendExecutor
from ragzoom.server.run_manager import TelemetryRunManager
from ragzoom.server.worker_coordinator import WorkerCoordinator
from ragzoom.services.llm_service import LLMService
from ragzoom.services.query_service import QueryService
from ragzoom.store import create_store_with_docker
from ragzoom.telemetry_log import DocumentTelemetryLog
from ragzoom.vector_factory import create_vector_index
from ragzoom.worktree_utils import DEFAULT_DATA_DIR_NAME


@dataclass(slots=True)
class ServerState:
    """Container for long-lived resources owned by the gRPC server."""

    index_config: IndexConfig
    query_config: QueryConfig
    operational_config: OperationalConfig
    store: StorageBackend
    query_log: QueryLog
    query_service: QueryService
    llm_service: LLMService
    telemetry_run_manager: TelemetryRunManager
    telemetry_log: DocumentTelemetryLog | None
    append_executor: AppendExecutor
    worker_coordinator: WorkerCoordinator
    index_runtime: IndexerRuntime

    @classmethod
    def create(
        cls,
        *,
        index_config: IndexConfig | None = None,
        query_config: QueryConfig | None = None,
        operational_config: OperationalConfig | None = None,
        collect_telemetry: bool = False,
        telemetry_dir: Path | None = None,
    ) -> ServerState:
        """Instantiate server state using the provided or default configs."""

        index_cfg = index_config or IndexConfig.load()
        query_cfg = query_config or QueryConfig()
        operational_cfg = operational_config or OperationalConfig()

        store = create_store_with_docker(
            operational_cfg, embedding_model=index_cfg.embedding_model
        )
        query_log = QueryLog(_resolve_query_log_path(operational_cfg))
        query_service = QueryService(store, query_cfg, operational_cfg, query_log)
        llm_service = LLMService(
            index_cfg,
            api_key=operational_cfg.openai_api_key,
        )
        telemetry_log = None
        if collect_telemetry:
            telemetry_path = _resolve_telemetry_dir(
                operational_cfg,
                telemetry_dir,
            )
            telemetry_log = DocumentTelemetryLog(telemetry_path)
        telemetry_run_manager = TelemetryRunManager(
            index_cfg,
            telemetry_log=telemetry_log,
        )
        append_executor = AppendExecutor(index_cfg, llm_service)
        worker_coordinator = WorkerCoordinator(
            store=store,
            index_config=index_cfg,
            operational_config=operational_cfg,
            llm_service=llm_service,
            run_manager=telemetry_run_manager,
        )
        vector_factory = lambda model: create_vector_index(  # noqa: E731
            operational_cfg.vector_backend,
            operational_cfg.database_url,
            model,
        )
        index_runtime = IndexerRuntime(
            store=store,
            index_config=index_cfg,
            append_executor=append_executor,
            worker_coordinator=worker_coordinator,
            telemetry_manager=telemetry_run_manager,
            vector_index_factory=vector_factory,
        )

        return cls(
            index_config=index_cfg,
            query_config=query_cfg,
            operational_config=operational_cfg,
            store=store,
            query_log=query_log,
            query_service=query_service,
            llm_service=llm_service,
            telemetry_run_manager=telemetry_run_manager,
            telemetry_log=telemetry_log,
            append_executor=append_executor,
            worker_coordinator=worker_coordinator,
            index_runtime=index_runtime,
        )


def _sqlite_base_dir(database_url: str | None) -> Path | None:
    """Extract the base directory from a file-based SQLite URL.

    Returns None if the URL is not a file-based SQLite URL (e.g., in-memory or non-SQLite).
    """
    url = (database_url or "").strip()
    if not url.startswith("sqlite") or ":memory:" in url:
        return None
    parsed = urlparse(url)
    raw_path = parsed.path or ""
    if url.startswith("sqlite:////"):
        sqlite_path = Path(raw_path)
    else:
        sqlite_path = Path(raw_path.lstrip("/"))
        if not sqlite_path.is_absolute():
            sqlite_path = Path.cwd() / sqlite_path
    return sqlite_path.parent if sqlite_path.suffix else sqlite_path


def _resolve_telemetry_dir(
    operational_config: OperationalConfig,
    override: Path | None,
) -> Path:
    if override is not None:
        return override

    sqlite_base = _sqlite_base_dir(operational_config.database_url)
    if sqlite_base is not None:
        return sqlite_base / "telemetry"

    data_root = os.environ.get("RAGZOOM_DATA_DIR")
    if data_root:
        return Path(data_root) / DEFAULT_DATA_DIR_NAME / "telemetry"

    return Path.cwd() / DEFAULT_DATA_DIR_NAME / "telemetry"


def _resolve_query_log_path(operational_config: OperationalConfig) -> Path:
    override = os.environ.get("RAGZOOM_QUERY_LOG_PATH")
    if override:
        return Path(override)

    sqlite_base = _sqlite_base_dir(operational_config.database_url)
    if sqlite_base is not None:
        return sqlite_base / "query-log.db"

    data_root = os.environ.get("RAGZOOM_DATA_DIR")
    if data_root:
        return QueryLog.default_path(Path(data_root))

    return QueryLog.default_path()
