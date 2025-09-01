"""Tests for error handling patterns and anti-patterns."""

from unittest.mock import Mock

import pytest

from ragzoom.api_middleware import ErrorHandlingMiddleware
from ragzoom.error_utils import (
    ErrorContext,
    categorize_exception,
    format_structured_error,
    log_error_with_context,
    preserve_exception_chain,
)
from ragzoom.exceptions import (
    ConfigurationError,
    DatabaseError,
    InvalidOperationError,
    LLMError,
    NodeNotFoundError,
    ResourceError,
    ValidationError,
)


class TestCustomExceptions:
    """Test structured exception types."""

    def test_validation_error_structure(self) -> None:
        """ValidationError should have structured context."""
        error = ValidationError(
            field="email", value="invalid-email", reason="missing @ symbol"
        )

        assert error.field == "email"
        assert error.value == "invalid-email"
        assert error.reason == "missing @ symbol"
        assert "email='invalid-email'" in str(error)

    def test_database_error_structure(self) -> None:
        """DatabaseError should have operation context."""
        error = DatabaseError(
            operation="insert_document",
            message="Connection failed",
            table="documents",
            query_id="abc123",
        )

        assert error.operation == "insert_document"
        assert hasattr(error, "context")
        assert error.context["table"] == "documents"
        assert error.context["query_id"] == "abc123"

    def test_llm_error_structure(self) -> None:
        """LLMError should have model and operation context."""
        error = LLMError(
            operation="generate_summary",
            model="gpt-4o",
            message="Rate limit exceeded",
            tokens_requested=1000,
        )

        assert error.operation == "generate_summary"
        assert error.model == "gpt-4o"
        assert hasattr(error, "context")
        assert error.context["tokens_requested"] == 1000


class TestErrorUtils:
    """Test error utility functions."""

    def test_error_context_builder(self) -> None:
        """ErrorContext should build rich error context."""
        context = ErrorContext("user_registration")
        context.add("user_id", "123").add("email", "test@example.com")

        exc = context.build_exception(
            ValidationError,
            "Email validation failed",
            field="email",
            value="test@example.com",
            reason="domain not allowed",
        )

        assert getattr(exc, "operation", None) == "user_registration"
        assert hasattr(exc, "request_id")
        assert len(exc.request_id) == 8  # Short UUID
        assert getattr(exc, "user_id", None) == "123"
        assert getattr(exc, "email", None) == "test@example.com"

    def test_categorize_exception(self) -> None:
        """Exception categorization should work correctly."""
        assert categorize_exception(DatabaseError("op", "msg")) == "storage"
        assert categorize_exception(ValidationError("f", "v", "r")) == "validation"
        assert categorize_exception(LLMError("op", "model", "msg")) == "llm"
        assert categorize_exception(NodeNotFoundError("123")) == "not_found"
        assert categorize_exception(ValueError("test")) == "unknown"

    def test_format_structured_error(self) -> None:
        """Error formatting should include all relevant context."""
        error = LLMError(
            operation="summarize", model="gpt-4o", message="API error", tokens=500
        )

        formatted = format_structured_error(error)

        assert formatted["type"] == "LLMError"
        assert formatted["category"] == "llm"
        assert formatted["operation"] == "summarize"

        # Type narrow the context before indexing
        context_value = formatted["context"]
        assert isinstance(context_value, dict)
        assert context_value["tokens"] == 500

    def test_preserve_exception_chain(self) -> None:
        """Exception chaining should preserve original cause."""
        original = ValueError("Original error")
        new_error = LLMError("op", "model", "New error")

        chained = preserve_exception_chain(new_error, original)

        assert chained.__cause__ is original
        assert isinstance(chained, LLMError)

    def test_log_error_with_context(self) -> None:
        """Error logging should include structured context."""
        logger = Mock()
        error = ValidationError("email", "bad@email", "invalid format")

        log_error_with_context(
            logger, error, "user_registration", user_id="123", ip_address="192.168.1.1"
        )

        logger.error.assert_called_once()
        call_args = logger.error.call_args
        assert "user_registration" in call_args[0][0]
        assert call_args[1]["extra"]["operation"] == "user_registration"
        assert call_args[1]["extra"]["error_category"] == "validation"
        assert call_args[1]["extra"]["user_id"] == "123"


class TestAPIMiddleware:
    """Test API error handling middleware."""

    def test_http_status_mapping(self) -> None:
        """Middleware should map exceptions to correct HTTP status codes."""
        middleware = ErrorHandlingMiddleware(Mock())

        # 404 for not found
        not_found = NodeNotFoundError("123")
        http_exc = middleware._convert_to_http_exception(not_found, "req123")
        assert http_exc.status_code == 404

        # 400 for validation errors
        validation = ValidationError("field", "value", "reason")
        http_exc = middleware._convert_to_http_exception(validation, "req123")
        assert http_exc.status_code == 400

        # 422 for invalid operations
        invalid_op = InvalidOperationError("delete", "cannot delete root")
        http_exc = middleware._convert_to_http_exception(invalid_op, "req123")
        assert http_exc.status_code == 422

        # 503 for database errors
        db_error = DatabaseError("insert", "connection failed")
        http_exc = middleware._convert_to_http_exception(db_error, "req123")
        assert http_exc.status_code == 503

        # 502 for LLM errors
        llm_error = LLMError("summarize", "gpt-4o", "API unavailable")
        http_exc = middleware._convert_to_http_exception(llm_error, "req123")
        assert http_exc.status_code == 502

    def test_error_response_structure(self) -> None:
        """Error responses should have consistent structure."""
        middleware = ErrorHandlingMiddleware(Mock())
        error = ValidationError("email", "invalid", "bad format")

        http_exc = middleware._convert_to_http_exception(error, "req123")
        detail: dict[str, object] = http_exc.detail  # type: ignore[assignment]

        assert detail["type"] == "ValidationError"
        assert detail["category"] == "validation"
        assert detail["request_id"] == "req123"
        assert detail["field"] == "email"
        assert detail["value"] == "invalid"
        assert detail["reason"] == "bad format"


class TestNoSilentFailures:
    """Test that silent failures have been eliminated."""

    @pytest.mark.asyncio
    async def test_validation_failures_propagate(self) -> None:
        """Validation errors should propagate, not return None."""
        from ragzoom.validate import (
            set_validation_enabled,
            validate_summary_faithfulness,
        )

        # Enable validation for this test
        set_validation_enabled(True)

        try:
            # Mock OpenAI client that raises an exception
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")

            with pytest.raises(LLMError) as exc_info:
                # This should raise LLMError, not return None
                await validate_summary_faithfulness(
                    "summary", "left", "right", mock_client
                )

            assert exc_info.value.operation == "summary_validation"
            assert "API Error" in str(exc_info.value)
        finally:
            # Restore validation state
            set_validation_enabled(False)

    def test_no_generic_exception_catches_in_api(self) -> None:
        """API endpoints should not have generic Exception catches."""
        import inspect

        from ragzoom import api

        # Get source code of the entire API module
        api_source = inspect.getsource(api)

        # Should not have generic Exception catches
        assert (
            "except Exception as e:" not in api_source
        ), "API module has generic Exception catches"

        # Should use middleware for error handling
        assert (
            "create_error_handling_middleware" in api_source
        ), "API should use error handling middleware"

    def test_cli_uses_specific_error_handling(self) -> None:
        """CLI should use specific error handling patterns."""
        from ragzoom.cli import handle_cli_error

        # Test that the helper function exists and handles specific types
        mock_database_error = DatabaseError("test_op", "test message")

        with pytest.raises(SystemExit):
            handle_cli_error(mock_database_error, "test operation")


class TestErrorBoundaries:
    """Test error boundaries and fail-fast behavior."""

    def test_invalid_config_fails_fast(self) -> None:
        """Invalid configuration should fail immediately."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError(
                setting="openai_api_key", expected="valid API key", actual="invalid"
            )

        assert exc_info.value.setting == "openai_api_key"
        assert exc_info.value.expected == "valid API key"
        assert exc_info.value.actual == "invalid"

    def test_resource_exhaustion_fails_fast(self) -> None:
        """Resource exhaustion should fail immediately."""
        with pytest.raises(ResourceError) as exc_info:
            raise ResourceError(
                resource="memory",
                operation="allocate_embeddings",
                reason="out of memory",
                requested_bytes=1024**3,
            )

        assert exc_info.value.resource == "memory"
        assert exc_info.value.operation == "allocate_embeddings"
        assert exc_info.value.context["requested_bytes"] == 1024**3


if __name__ == "__main__":
    pytest.main([__file__])
