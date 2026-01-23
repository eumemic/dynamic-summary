"""RagZoom: Incremental, hierarchical RAG memory system."""

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("ragzoom")
except Exception:
    __version__ = "0.1.0"

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.exceptions import (
    DocumentNotFoundError,
    InvalidOperationError,
    NodeNotFoundError,
    StorageError,
)
from ragzoom.retrieve import Retriever
from ragzoom.store import create_store, create_store_with_docker
from ragzoom.wrapper import AppendUnit, AsyncRagZoom, QueryResponse, RagZoom

__all__ = [
    "IndexConfig",
    "QueryConfig",
    "OperationalConfig",
    "Retriever",
    "Assembler",
    "create_store",
    "create_store_with_docker",
    "RagZoom",
    "AsyncRagZoom",
    "AppendUnit",
    "QueryResponse",
    "NodeNotFoundError",
    "DocumentNotFoundError",
    "InvalidOperationError",
    "StorageError",
]
