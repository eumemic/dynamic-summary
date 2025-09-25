"""Shared server state for the gRPC runtime."""

from __future__ import annotations

from dataclasses import dataclass

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.server.append_executor import AppendExecutor
from ragzoom.server.worker_coordinator import WorkerCoordinator
from ragzoom.services.indexing_service import IndexingService
from ragzoom.services.llm_service import LLMService
from ragzoom.services.query_service import QueryService
from ragzoom.store import create_store_with_docker


@dataclass(slots=True)
class ServerState:
    """Container for long-lived resources owned by the gRPC server."""

    index_config: IndexConfig
    query_config: QueryConfig
    operational_config: OperationalConfig
    store: StorageBackend
    indexing_service: IndexingService
    query_service: QueryService
    llm_service: LLMService
    append_executor: AppendExecutor
    worker_coordinator: WorkerCoordinator

    @classmethod
    def create(
        cls,
        *,
        index_config: IndexConfig | None = None,
        query_config: QueryConfig | None = None,
        operational_config: OperationalConfig | None = None,
    ) -> ServerState:
        """Instantiate server state using the provided or default configs."""

        index_cfg = index_config or IndexConfig.load()
        query_cfg = query_config or QueryConfig()
        operational_cfg = operational_config or OperationalConfig()

        store = create_store_with_docker(
            operational_cfg, embedding_model=index_cfg.embedding_model
        )
        indexing_service = IndexingService(store, index_cfg, operational_cfg)
        query_service = QueryService(store, query_cfg, operational_cfg)
        llm_service = LLMService(
            index_cfg,
            api_key=operational_cfg.openai_api_key,
        )
        append_executor = AppendExecutor(index_cfg, llm_service)
        worker_coordinator = WorkerCoordinator(
            store=store,
            index_config=index_cfg,
            operational_config=operational_cfg,
            llm_service=llm_service,
        )

        return cls(
            index_config=index_cfg,
            query_config=query_cfg,
            operational_config=operational_cfg,
            store=store,
            indexing_service=indexing_service,
            query_service=query_service,
            llm_service=llm_service,
            append_executor=append_executor,
            worker_coordinator=worker_coordinator,
        )
