"""Shared constants for RagZoom configuration."""

DEFAULT_GRPC_HOST = "127.0.0.1"
DEFAULT_GRPC_PORT = 50051
DEV_GRPC_PORT = 50052
DEFAULT_GRPC_ADDRESS = f"{DEFAULT_GRPC_HOST}:{DEFAULT_GRPC_PORT}"

# gRPC client timeouts (seconds)
# None = no timeout for unary RPCs (batch_append, clear, etc.)
# These are local calls to a colocated service; let them complete.
DEFAULT_GRPC_TIMEOUT: float | None = None
DEFAULT_GRPC_STREAM_TIMEOUT: float | None = 120.0
# Session ingestion may trigger full re-index requiring tree building + embeddings
DEFAULT_SESSION_INGEST_TIMEOUT = 300.0

# Default system prompt for summary generation
DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "You are a text compressor. You compress sections of documents while "
    "preserving their meaning. You output ONLY the compressed text, nothing else."
)

__all__ = [
    "DEFAULT_GRPC_HOST",
    "DEFAULT_GRPC_PORT",
    "DEV_GRPC_PORT",
    "DEFAULT_GRPC_ADDRESS",
    "DEFAULT_GRPC_TIMEOUT",
    "DEFAULT_GRPC_STREAM_TIMEOUT",
    "DEFAULT_SESSION_INGEST_TIMEOUT",
    "DEFAULT_SUMMARY_SYSTEM_PROMPT",
]
