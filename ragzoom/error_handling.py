"""Environment-aware error handling utilities.

This module provides utilities for handling exceptions in a way that varies
based on the runtime environment:

- In development/test (RAGZOOM_STRICT_ERRORS=1): Exceptions are re-raised
  immediately to fail fast and reveal bugs early.
- In production: Exceptions are logged with context and a fallback value
  is returned to allow graceful degradation.

Usage:
    from ragzoom.error_handling import handle_graceful_error

    try:
        result = risky_operation()
    except SomeError as exc:
        result = handle_graceful_error(
            exc, "Failed to perform risky operation", default=fallback_value
        )
"""

import logging
import os
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_strict_mode() -> bool:
    """Check if strict error mode is enabled (for dev/test environments).

    Returns:
        True if RAGZOOM_STRICT_ERRORS environment variable is set to a truthy value.
    """
    return os.environ.get("RAGZOOM_STRICT_ERRORS", "").lower() in ("1", "true", "yes")


def handle_graceful_error(
    exc: BaseException,
    context: str,
    *,
    default: T,
) -> T:
    """Handle an exception gracefully based on environment.

    In strict mode (dev/test): re-raises the exception immediately.
    In production mode: logs the error with context and returns the default value.

    Args:
        exc: The caught exception
        context: Description of what operation failed (used in log message)
        default: Value to return in production mode

    Returns:
        The default value (production mode only)

    Raises:
        The original exception (strict mode only)

    Example:
        try:
            value = int(user_input)
        except ValueError as exc:
            value = handle_graceful_error(exc, "Invalid user input", default=0)
    """
    if is_strict_mode():
        raise exc
    logger.warning("%s: %s", context, exc)
    return default
