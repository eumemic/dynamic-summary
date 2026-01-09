"""Entry points for launching the RagZoom gRPC server."""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from ragzoom.config import (
    IndexConfig,
    OperationalConfig,
    PrecedingContextConfig,
    PrecedingContextSettings,
    QueryConfig,
)
from ragzoom.constants import DEFAULT_GRPC_HOST, DEFAULT_GRPC_PORT
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.server.lease import LeaseConfig
from ragzoom.server.servicers import serve
from ragzoom.server.state import ServerState
from ragzoom.store import create_store

logger = logging.getLogger(__name__)


@dataclass
class ServerOptions:
    """Command-level options for starting the gRPC server."""

    host: str = DEFAULT_GRPC_HOST
    port: int = DEFAULT_GRPC_PORT
    config_path: str | None = None
    collect_telemetry: bool = False
    telemetry_dir: str | None = None
    max_parallelism: int | None = None
    # Per-node-type config overrides
    preceding_context_leaf_num_seeds: int | None = None
    preceding_context_leaf_verbatim_tokens: int | None = None
    preceding_context_leaf_min_forest_completeness: float | None = None
    preceding_context_leaf_token_cap: int | None = None
    preceding_context_inner_num_seeds: int | None = None
    preceding_context_inner_verbatim_tokens: int | None = None
    preceding_context_inner_min_forest_completeness: float | None = None
    preceding_context_inner_token_cap: int | None = None


def _apply_config_overrides(
    base: PrecedingContextConfig,
    num_seeds: int | None,
    verbatim_tokens: int | None,
    min_forest_completeness: float | None,
    token_cap: int | None,
) -> PrecedingContextConfig:
    """Apply CLI overrides to a PrecedingContextConfig."""
    return PrecedingContextConfig(
        num_seeds=num_seeds if num_seeds is not None else base.num_seeds,
        verbatim_tokens=(
            verbatim_tokens if verbatim_tokens is not None else base.verbatim_tokens
        ),
        min_forest_completeness=(
            min_forest_completeness
            if min_forest_completeness is not None
            else base.min_forest_completeness
        ),
        max_forest_height_differential=base.max_forest_height_differential,
        token_cap=(token_cap if token_cap is not None else base.token_cap),
    )


def build_state(
    options: ServerOptions,
    *,
    store: StorageBackend | None = None,
    operational_cfg: OperationalConfig | None = None,
) -> ServerState:
    """Create ServerState using the supplied options.

    Args:
        options: Server command-line options.
        store: Pre-created storage backend. If None, one will be created.
        operational_cfg: Pre-created operational config. If None, one will be created.
    """

    config_path = Path(options.config_path) if options.config_path else None
    index_cfg = IndexConfig.load(config_path=config_path)

    # Apply per-node-type CLI overrides
    has_leaf_overrides = any(
        x is not None
        for x in [
            options.preceding_context_leaf_num_seeds,
            options.preceding_context_leaf_verbatim_tokens,
            options.preceding_context_leaf_min_forest_completeness,
            options.preceding_context_leaf_token_cap,
        ]
    )
    has_inner_overrides = any(
        x is not None
        for x in [
            options.preceding_context_inner_num_seeds,
            options.preceding_context_inner_verbatim_tokens,
            options.preceding_context_inner_min_forest_completeness,
            options.preceding_context_inner_token_cap,
        ]
    )

    if has_leaf_overrides or has_inner_overrides:
        leaf_cfg = _apply_config_overrides(
            index_cfg.preceding_context.leaf,
            options.preceding_context_leaf_num_seeds,
            options.preceding_context_leaf_verbatim_tokens,
            options.preceding_context_leaf_min_forest_completeness,
            options.preceding_context_leaf_token_cap,
        )
        inner_cfg = _apply_config_overrides(
            index_cfg.preceding_context.inner,
            options.preceding_context_inner_num_seeds,
            options.preceding_context_inner_verbatim_tokens,
            options.preceding_context_inner_min_forest_completeness,
            options.preceding_context_inner_token_cap,
        )
        index_cfg = index_cfg.replace(
            preceding_context=PrecedingContextSettings(leaf=leaf_cfg, inner=inner_cfg)
        )

    query_cfg = QueryConfig()
    if operational_cfg is None:
        operational_cfg = OperationalConfig()

    # Allow env var override for max_parallelism
    max_parallelism = options.max_parallelism
    if max_parallelism is None:
        import os

        env_parallelism = os.environ.get("RAGZOOM_MAX_PARALLELISM")
        if env_parallelism is not None:
            max_parallelism = int(env_parallelism)

    return ServerState.create(
        index_config=index_cfg,
        query_config=query_cfg,
        operational_config=operational_cfg,
        collect_telemetry=options.collect_telemetry,
        telemetry_dir=(
            Path(options.telemetry_dir) if options.telemetry_dir is not None else None
        ),
        max_parallelism=max_parallelism,
        store=store,
    )


async def _serve_async(state: ServerState, options: ServerOptions) -> None:
    await serve(state, host=options.host, port=options.port)


async def _run_with_lease(
    options: ServerOptions,
    store: StorageBackend,
    operational_cfg: OperationalConfig,
) -> None:
    """Run server with lease acquisition for single-writer coordination.

    The lease mechanism ensures only one IndexingEngine writes to the database
    at a time, preventing corruption during deployments where multiple server
    instances may briefly run simultaneously.
    """
    import os

    # Load lease config from environment
    lease_config = LeaseConfig(
        ttl_seconds=float(os.environ.get("RAGZOOM_LEASE_TTL", "60")),
        heartbeat_interval=float(os.environ.get("RAGZOOM_LEASE_HEARTBEAT", "15")),
        acquire_timeout=float(os.environ.get("RAGZOOM_LEASE_TIMEOUT", "90")),
    )

    # Get lease from the storage backend
    lease = store.create_lease()
    lease._config = lease_config  # Apply config overrides

    if not await lease.acquire():
        logger.critical("Failed to acquire indexer lease - exiting")
        sys.exit(1)  # Container orchestrator will restart us

    try:
        state = build_state(options, store=store, operational_cfg=operational_cfg)
        await _serve_async(state, options)
    finally:
        await lease.release()


def run_server(options: ServerOptions) -> None:
    """Blocking helper used by CLI to start the server.

    Acquires a global lease before starting to ensure only one IndexingEngine
    writes to the database at a time. This prevents corruption during
    deployments where old and new containers briefly run simultaneously.
    """
    operational_cfg = OperationalConfig()

    # Create store first so we can get a lease from it
    index_cfg = IndexConfig.load(
        config_path=Path(options.config_path) if options.config_path else None
    )
    store = create_store(operational_cfg, embedding_model=index_cfg.embedding_model)

    logger.info("Acquiring global indexer lease")
    try:
        asyncio.run(_run_with_lease(options, store, operational_cfg))
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        logger.info("Shutting down RagZoom gRPC server")
    finally:
        store.close()
