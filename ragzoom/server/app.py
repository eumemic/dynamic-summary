"""Entry points for launching the RagZoom gRPC server."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.constants import DEFAULT_GRPC_HOST, DEFAULT_GRPC_PORT
from ragzoom.server.servicers import serve
from ragzoom.server.state import ServerState

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ServerOptions:
    """Command-level options for starting the gRPC server."""

    host: str = DEFAULT_GRPC_HOST
    port: int = DEFAULT_GRPC_PORT
    config_path: str | None = None
    collect_telemetry: bool = False
    telemetry_dir: str | None = None
    # Config overrides (take precedence over config file)
    preceding_summary_budget_tokens: int | None = None
    context_lag_tokens: int | None = None
    max_parallelism: int | None = None
    preceding_context_max_extraneous_detail: int | None = None


def build_state(options: ServerOptions) -> ServerState:
    """Create ServerState using the supplied options."""

    config_path = Path(options.config_path) if options.config_path else None
    index_cfg = IndexConfig.load(config_path=config_path)

    # Apply CLI overrides if provided
    if (
        options.preceding_summary_budget_tokens is not None
        or options.context_lag_tokens is not None
        or options.preceding_context_max_extraneous_detail is not None
    ):
        index_cfg = index_cfg.replace(
            preceding_summary_budget_tokens=options.preceding_summary_budget_tokens,
            context_lag_tokens=options.context_lag_tokens,
            preceding_context_max_extraneous_detail=options.preceding_context_max_extraneous_detail,
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
