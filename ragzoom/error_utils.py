"""Utilities for error handling and context management."""

import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class ErrorContext:
    """Builder for rich error context information."""

    def __init__(self, operation: str):
        self.operation = operation
        self.context: dict[str, Any] = {}
        self.request_id = str(uuid4())[:8]

    def add(self, key: str, value: Any) -> "ErrorContext":
        """Add contextual information."""
        self.context[key] = value
        return self

    def build_exception(
        self, exc_class: type[Exception], message: str, **kwargs: Any
    ) -> Exception:
        """Build exception with full context."""
        # Create exception with its specific constructor parameters
        exc = exc_class(**kwargs)

        # Add our context as attributes
        exc.operation = self.operation  # type: ignore[attr-defined]
        exc.request_id = self.request_id  # type: ignore[attr-defined]
        for key, value in self.context.items():
            setattr(exc, key, value)

        return exc


def categorize_exception(exc: Exception) -> str:
    """Categorize exception by type for structured error handling."""
    from ragzoom.exceptions import (
        ConfigurationError,
        DatabaseError,
        DocumentNotFoundError,
        InvalidOperationError,
        LLMError,
        NodeNotFoundError,
        ResourceError,
        StorageError,
        ValidationError,
    )

    if isinstance(exc, DatabaseError | StorageError):
        return "storage"
    elif isinstance(exc, ValidationError):
        return "validation"
    elif isinstance(exc, LLMError):
        return "llm"
    elif isinstance(exc, ConfigurationError):
        return "configuration"
    elif isinstance(exc, ResourceError):
        return "resource"
    elif isinstance(exc, NodeNotFoundError | DocumentNotFoundError):
        return "not_found"
    elif isinstance(exc, InvalidOperationError):
        return "invalid_operation"
    else:
        return "unknown"


def format_structured_error(
    exc: Exception, include_traceback: bool = False
) -> dict[str, Any]:
    """Format exception as structured error response."""
    error_data = {
        "type": type(exc).__name__,
        "category": categorize_exception(exc),
        "message": str(exc),
    }

    # Add structured context from custom exceptions
    if hasattr(exc, "operation"):
        error_data["operation"] = exc.operation
    if hasattr(exc, "context"):
        error_data["context"] = exc.context
    if hasattr(exc, "request_id"):
        error_data["request_id"] = exc.request_id

    # Add specific fields for domain exceptions
    if hasattr(exc, "node_id"):
        error_data["node_id"] = exc.node_id
    if hasattr(exc, "document_id"):
        error_data["document_id"] = exc.document_id
    if hasattr(exc, "field"):
        error_data["field"] = exc.field  # type: ignore[attr-defined]
        error_data["value"] = exc.value  # type: ignore[attr-defined]
        error_data["reason"] = exc.reason  # type: ignore[attr-defined]

    if include_traceback:
        import traceback

        error_data["traceback"] = traceback.format_exc()

    return error_data


def preserve_exception_chain(new_exc: Exception, original_exc: Exception) -> Exception:
    """Preserve exception chain when translating exceptions."""
    new_exc.__cause__ = original_exc
    return new_exc


def log_error_with_context(
    logger: logging.Logger, exc: Exception, operation: str, **context: Any
) -> None:
    """Log error with full context for debugging."""
    error_info = format_structured_error(exc)
    error_info.update(context)

    logger.error(
        f"Error during {operation}: {exc}",
        extra={
            "operation": operation,
            "error_type": type(exc).__name__,
            "error_category": categorize_exception(exc),
            **error_info,
        },
        exc_info=True,
    )
