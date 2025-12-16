"""Entry points for launching the RagZoom gRPC server."""

from __future__ import annotations

import asyncio
import logging
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
from ragzoom.server.servicers import serve
from ragzoom.server.state import ServerState

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
        token_cap=(token_cap if token_cap is not None else base.token_cap),
    )


def build_state(options: ServerOptions) -> ServerState:
    """Create ServerState using the supplied options."""

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
    operational_cfg = OperationalConfig()

    return ServerState.create(
        index_config=index_cfg,
        query_config=query_cfg,
        operational_config=operational_cfg,
        collect_telemetry=options.collect_telemetry,
        telemetry_dir=(
            Path(options.telemetry_dir) if options.telemetry_dir is not None else None
        ),
        max_parallelism=options.max_parallelism,
    )


async def _serve_async(state: ServerState, options: ServerOptions) -> None:
    await serve(state, host=options.host, port=options.port)


def run_server(options: ServerOptions) -> None:
    """Blocking helper used by CLI to start the server."""

    state = build_state(options)

    try:
        asyncio.run(_serve_async(state, options))
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        logger.info("Shutting down RagZoom gRPC server")
