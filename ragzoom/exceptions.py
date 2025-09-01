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

    def __init__(self, operation: str, reason: str, **context: object) -> None:
        super().__init__(f"Invalid operation '{operation}': {reason}")
        self.operation = operation
        self.reason = reason
        self.context = context


class StorageError(Exception):
    """Raised when storage operations encounter internal errors."""

    def __init__(self, operation: str, message: str, **context: object) -> None:
        super().__init__(f"Storage error during {operation}: {message}")
        self.operation = operation
        self.context = context


class ValidationError(Exception):
    """Raised when data validation fails."""

    def __init__(self, field: str, value: str, reason: str):
        super().__init__(f"Validation failed for {field}='{value}': {reason}")
        self.field = field
        self.value = value
        self.reason = reason


class DatabaseError(Exception):
    """Raised when database operations fail."""

    def __init__(self, operation: str, message: str, **context: object) -> None:
        super().__init__(f"Database error during {operation}: {message}")
        self.operation = operation
        self.context = context


class LLMError(Exception):
    """Raised when LLM service operations fail."""

    def __init__(
        self, operation: str, model: str, message: str, **context: object
    ) -> None:
        super().__init__(f"LLM error during {operation} with {model}: {message}")
        self.operation = operation
        self.model = model
        self.context = context


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    def __init__(self, setting: str, expected: str, actual: str | None = None) -> None:
        msg = f"Configuration error for '{setting}': expected {expected}"
        if actual is not None:
            msg += f", got '{actual}'"
        super().__init__(msg)
        self.setting = setting
        self.expected = expected
        self.actual = actual


class ResourceError(Exception):
    """Raised when resource allocation or management fails."""

    def __init__(
        self, resource: str, operation: str, reason: str, **context: object
    ) -> None:
        super().__init__(f"Resource error for {resource} during {operation}: {reason}")
        self.resource = resource
        self.operation = operation
        self.reason = reason
        self.context = context
