"""Shared server state for the gRPC runtime."""

from __future__ import annotations

from dataclasses import dataclass

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.services.indexing_service import IndexingService
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

        return cls(
            index_config=index_cfg,
            query_config=query_cfg,
            operational_config=operational_cfg,
            store=store,
            indexing_service=indexing_service,
            query_service=query_service,
        )
