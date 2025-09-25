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


def build_state(options: ServerOptions) -> ServerState:
    """Create ServerState using the supplied options."""

    config_path = Path(options.config_path) if options.config_path else None
    index_cfg = IndexConfig.load(config_path=config_path)
    query_cfg = QueryConfig()
    operational_cfg = OperationalConfig()

    return ServerState.create(
        index_config=index_cfg,
        query_config=query_cfg,
        operational_config=operational_cfg,
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
