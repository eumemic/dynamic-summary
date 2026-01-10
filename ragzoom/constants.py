"""Shared constants for RagZoom configuration."""

DEFAULT_GRPC_HOST = "127.0.0.1"
DEFAULT_GRPC_PORT = 50051
DEFAULT_GRPC_ADDRESS = f"{DEFAULT_GRPC_HOST}:{DEFAULT_GRPC_PORT}"

# gRPC client timeouts (seconds)
DEFAULT_GRPC_TIMEOUT = 30.0
DEFAULT_GRPC_STREAM_TIMEOUT: float | None = None
# Session ingestion may trigger full re-index requiring tree building + embeddings
DEFAULT_SESSION_INGEST_TIMEOUT = 300.0

__all__ = [
    "DEFAULT_GRPC_HOST",
    "DEFAULT_GRPC_PORT",
    "DEFAULT_GRPC_ADDRESS",
    "DEFAULT_GRPC_TIMEOUT",
    "DEFAULT_GRPC_STREAM_TIMEOUT",
    "DEFAULT_SESSION_INGEST_TIMEOUT",
]
