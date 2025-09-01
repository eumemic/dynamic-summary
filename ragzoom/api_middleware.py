"""FastAPI middleware for centralized error handling."""

import logging
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from ragzoom.error_utils import categorize_exception, format_structured_error

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling with structured responses."""

    def __init__(self, app: ASGIApp, include_traceback: bool = False) -> None:
        super().__init__(app)
        self.include_traceback = include_traceback

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = str(uuid4())[:8]

        try:
            response = await call_next(request)
            return response
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as exc:
            # Log the error with full context
            logger.error(
                f"API error [{request_id}]: {exc}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "error_type": type(exc).__name__,
                    "error_category": categorize_exception(exc),
                },
                exc_info=True,
            )

            # Convert to appropriate HTTP exception
            http_exc = self._convert_to_http_exception(exc, request_id)
            raise http_exc

    def _convert_to_http_exception(
        self, exc: Exception, request_id: str
    ) -> HTTPException:
        """Convert domain exceptions to appropriate HTTP status codes."""
        error_data = format_structured_error(
            exc, include_traceback=self.include_traceback
        )
        error_data["request_id"] = request_id

        # Use categorization to determine HTTP status
        category = categorize_exception(exc)

        status_code_map = {
            "not_found": 404,
            "validation": 400,
            "invalid_operation": 422,
            "configuration": 500,
            "storage": 503,
            "llm": 502,
            "resource": 507,
            "unknown": 500,
        }

        message_map = {
            "configuration": "Server configuration error",
            "storage": "Service temporarily unavailable",
            "llm": "External AI service error",
            "resource": "Resource allocation failed",
        }

        status_code = status_code_map.get(category, 500)

        if category in message_map:
            error_data["message"] = message_map[category]
        elif category == "unknown":
            return HTTPException(
                status_code=500,
                detail={
                    "type": "InternalServerError",
                    "category": "unknown",
                    "message": "An internal server error occurred",
                    "request_id": request_id,
                },
            )

        return HTTPException(status_code=status_code, detail=error_data)


def create_error_response(
    status_code: int,
    message: str,
    error_type: str = "APIError",
    **context: object,
) -> dict[str, object]:
    """Create standardized error response structure."""
    return {
        "error": {
            "type": error_type,
            "message": message,
            "status_code": status_code,
            **context,
        }
    }


# Removed complex TypedDict definitions to avoid mypy internal errors


def create_error_handling_middleware(
    include_traceback: bool = False,
) -> type[ErrorHandlingMiddleware]:
    """Factory function to create ErrorHandlingMiddleware with custom parameters."""

    class ConfiguredErrorHandlingMiddleware(ErrorHandlingMiddleware):
        """Pre-configured ErrorHandlingMiddleware."""

        def __init__(self, app: ASGIApp) -> None:
            super().__init__(app, include_traceback=include_traceback)

    return ConfiguredErrorHandlingMiddleware
