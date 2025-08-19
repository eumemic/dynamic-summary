"""RagZoom: Incremental, hierarchical RAG memory system."""

__version__ = "0.1.0"

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.exceptions import (
    DocumentNotFoundError,
    InvalidOperationError,
    NodeNotFoundError,
    StorageError,
)
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever

__all__ = [
    "IndexConfig",
    "QueryConfig",
    "OperationalConfig",
    "TreeBuilder",
    "Retriever",
    "Assembler",
    "NodeNotFoundError",
    "DocumentNotFoundError",
    "InvalidOperationError",
    "StorageError",
]
