"""Domain-specific exceptions for RagZoom."""


class NodeNotFoundError(Exception):
    """Raised when a requested node cannot be found in the store."""

    def __init__(self, node_id: str):
        super().__init__(f"Node {node_id} not found")
        self.node_id = node_id


class DocumentNotFoundError(Exception):
    """Raised when a requested document cannot be found in the store."""

    def __init__(self, document_id: str):
        super().__init__(f"Document {document_id} not found")
        self.document_id = document_id


class InvalidOperationError(Exception):
    """Raised when an operation cannot be performed due to invalid state or parameters."""

    pass


class StorageError(Exception):
    """Raised when storage operations encounter internal errors."""

    pass
