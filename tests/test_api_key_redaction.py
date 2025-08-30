"""Tests for API key redaction functionality."""

import json
import logging
from unittest.mock import Mock, patch

from ragzoom.config import OperationalConfig, SecretStr
from ragzoom.error_utils import (
    RedactionFilter,
    format_structured_error,
    sanitize_dict,
    sanitize_list,
    sanitize_message,
)
from ragzoom.exceptions import ConfigurationError


class TestSecretStr:
    """Test SecretStr class behavior."""

    def test_secret_str_redacts_repr(self) -> None:
        """SecretStr should redact value in repr()."""
        secret = SecretStr("sk-1234567890123456789012345678901234567890123456789")
        assert repr(secret) == "***REDACTED***"

    def test_secret_str_redacts_str(self) -> None:
        """SecretStr should redact value in str()."""
        secret = SecretStr("sk-1234567890123456789012345678901234567890123456789")
        assert str(secret) == "***REDACTED***"

    def test_secret_str_get_secret_value(self) -> None:
        """SecretStr should return actual value via get_secret_value()."""
        api_key = "sk-1234567890123456789012345678901234567890123456789"
        secret = SecretStr(api_key)
        assert secret.get_secret_value() == api_key

    def test_secret_str_in_f_strings(self) -> None:
        """SecretStr should be redacted when used in f-strings."""
        secret = SecretStr("sk-1234567890123456789012345678901234567890123456789")
        message = f"Using API key: {secret}"
        assert message == "Using API key: ***REDACTED***"

    def test_secret_str_in_error_messages(self) -> None:
        """SecretStr should be redacted when used in exception messages."""
        secret = SecretStr("sk-1234567890123456789012345678901234567890123456789")

        try:
            raise ValueError(f"Invalid API key: {secret}")
        except ValueError as e:
            assert "***REDACTED***" in str(e)
            assert "sk-" not in str(e)


class TestOperationalConfigSecretStr:
    """Test OperationalConfig usage of SecretStr."""

    def test_operational_config_uses_secret_str(self) -> None:
        """OperationalConfig should use SecretStr for API key."""
        api_key = "sk-1234567890123456789012345678901234567890123456789"
        config = OperationalConfig(openai_api_key=SecretStr(api_key))

        assert isinstance(config.openai_api_key, SecretStr)
        assert str(config.openai_api_key) == "***REDACTED***"
        assert config.openai_api_key.get_secret_value() == api_key

    def test_operational_config_loads_from_env(self) -> None:
        """OperationalConfig should wrap environment API key in SecretStr."""
        api_key = "sk-1234567890123456789012345678901234567890123456789"

        with patch.dict("os.environ", {"OPENAI_API_KEY": api_key}):
            config = OperationalConfig()

            assert isinstance(config.openai_api_key, SecretStr)
            assert config.openai_api_key.get_secret_value() == api_key

    def test_operational_config_in_repr(self) -> None:
        """OperationalConfig should not expose API key in repr."""
        api_key = "sk-1234567890123456789012345678901234567890123456789"
        config = OperationalConfig(openai_api_key=SecretStr(api_key))

        config_repr = repr(config)
        assert "***REDACTED***" in config_repr
        assert "sk-" not in config_repr


class TestSanitizationFunctions:
    """Test message and data sanitization functions."""

    def test_sanitize_message_basic(self) -> None:
        """sanitize_message should redact OpenAI API keys."""
        message = "Using API key sk-1234567890123456789012345678901234567890123456789 for request"
        result = sanitize_message(message)

        assert "***REDACTED***" in result
        assert "sk-1234567890123456789012345678901234567890123456789" not in result
        assert "Using API key ***REDACTED*** for request" == result

    def test_sanitize_message_multiple_keys(self) -> None:
        """sanitize_message should redact multiple API keys."""
        message = "Key1: sk-1111111111111111111111111111111111111111111111111, Key2: sk-2222222222222222222222222222222222222222222222222"
        result = sanitize_message(message)

        assert result.count("***REDACTED***") == 2
        assert "sk-" not in result

    def test_sanitize_message_no_keys(self) -> None:
        """sanitize_message should leave message unchanged if no keys."""
        message = "This is a normal log message without keys"
        result = sanitize_message(message)

        assert result == message

    def test_sanitize_dict_nested(self) -> None:
        """sanitize_dict should recursively sanitize nested structures."""
        data = {
            "config": {
                "api_key": "sk-1234567890123456789012345678901234567890123456789",
                "model": "gpt-4",
                "nested": {
                    "auth": "Bearer sk-2222222222222222222222222222222222222222222222222"
                },
            },
            "safe_field": "normal value",
        }

        result = sanitize_dict(data)

        assert result["config"]["api_key"] == "***REDACTED***"
        assert result["config"]["model"] == "gpt-4"
        assert result["config"]["nested"]["auth"] == "Bearer ***REDACTED***"
        assert result["safe_field"] == "normal value"

    def test_sanitize_list_mixed_types(self) -> None:
        """sanitize_list should handle mixed types in lists."""
        data = [
            "sk-1234567890123456789012345678901234567890123456789",
            {"key": "sk-2222222222222222222222222222222222222222222222222"},
            ["nested", "sk-3333333333333333333333333333333333333333333333333"],
            42,
            None,
        ]

        result = sanitize_list(data)

        assert result[0] == "***REDACTED***"
        assert result[1]["key"] == "***REDACTED***"
        assert result[2][0] == "nested"
        assert result[2][1] == "***REDACTED***"
        assert result[3] == 42
        assert result[4] is None


class TestLoggingRedaction:
    """Test logging redaction filter."""

    def test_redaction_filter_message(self) -> None:
        """RedactionFilter should redact API keys in log messages."""
        filter_obj = RedactionFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="API key: sk-1234567890123456789012345678901234567890123456789",
            args=(),
            exc_info=None,
        )

        filter_obj.filter(record)

        assert record.msg == "API key: ***REDACTED***"

    def test_redaction_filter_args(self) -> None:
        """RedactionFilter should redact API keys in log record args."""
        filter_obj = RedactionFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Using key %s for %s",
            args=("sk-1234567890123456789012345678901234567890123456789", "testing"),
            exc_info=None,
        )

        filter_obj.filter(record)

        args_tuple = record.args
        assert args_tuple is not None
        # Cast to list then to tuple to handle type checker
        args_list = list(args_tuple)
        assert args_list[0] == "***REDACTED***"
        assert args_list[1] == "testing"

    def test_logging_integration(self) -> None:
        """Test full logging integration with redaction filter."""
        # Create a logger with the redaction filter
        logger = logging.getLogger("test_redaction")
        logger.setLevel(logging.INFO)

        # Create a handler that captures log output
        import io

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.addFilter(RedactionFilter())
        logger.addHandler(handler)

        try:
            # Log a message with an API key
            api_key = "sk-1234567890123456789012345678901234567890123456789"
            logger.info("Initializing with API key: %s", api_key)

            # Check that the logged output is redacted
            log_output = log_stream.getvalue()
            assert "***REDACTED***" in log_output
            assert (
                "sk-1234567890123456789012345678901234567890123456789" not in log_output
            )
        finally:
            logger.removeHandler(handler)


class TestErrorHandlingRedaction:
    """Test error handling and formatting redaction."""

    def test_format_structured_error_redacts_message(self) -> None:
        """format_structured_error should redact API keys in exception messages."""
        api_key = "sk-1234567890123456789012345678901234567890123456789"
        exc = ConfigurationError(
            "openai_api_key", "valid API key", f"Invalid API key: {api_key}"
        )

        result = format_structured_error(exc)

        assert "***REDACTED***" in result["message"]
        assert api_key not in result["message"]

    def test_format_structured_error_redacts_context(self) -> None:
        """format_structured_error should redact API keys in exception context."""
        api_key = "sk-1234567890123456789012345678901234567890123456789"
        exc = ConfigurationError("config", "valid configuration")
        exc.context = {"api_key": api_key, "model": "gpt-4"}  # type: ignore

        result = format_structured_error(exc)

        assert result["context"]["api_key"] == "***REDACTED***"
        assert result["context"]["model"] == "gpt-4"

    def test_format_structured_error_redacts_traceback(self) -> None:
        """format_structured_error should redact API keys in tracebacks."""
        api_key = "sk-1234567890123456789012345678901234567890123456789"

        try:
            raise ValueError(f"API key {api_key} is invalid")
        except ValueError as exc:
            result = format_structured_error(exc, include_traceback=True)

            assert "***REDACTED***" in result["traceback"]
            assert api_key not in result["traceback"]


class TestEndToEndScenarios:
    """Test end-to-end scenarios for API key protection."""

    def test_api_key_not_exposed_in_service_errors(self) -> None:
        """Ensure API keys aren't exposed when services fail."""
        from ragzoom.config import QueryConfig
        from ragzoom.services.query_service import QueryService
        from tests.conftest import SimpleMockStore  # type: ignore

        # Create config with API key
        api_key = "sk-1234567890123456789012345678901234567890123456789"
        operational_config = OperationalConfig(openai_api_key=SecretStr(api_key))
        query_config = QueryConfig()
        store = Mock(spec=SimpleMockStore)

        # This should not expose the API key even if construction fails
        try:
            service = QueryService(store, query_config, operational_config)
            # The service should be created successfully
            assert service is not None
        except Exception as e:
            # If there's an error, the API key should not be exposed
            error_str = str(e)
            assert api_key not in error_str
            if "sk-" in error_str:
                assert "***REDACTED***" in error_str

    def test_json_serialization_safety(self) -> None:
        """Test that SecretStr can be safely serialized for logging/debugging."""
        # This test shows that when we want to safely serialize config containing
        # SecretStr objects, we need to be explicit about the serialization
        api_key = SecretStr("sk-1234567890123456789012345678901234567890123456789")
        config_dict = {
            "api_key": api_key,
            "model": "gpt-4",
        }

        # Safe serialization approach: convert SecretStr to string first
        safe_dict = {
            key: str(value) if isinstance(value, SecretStr) else value
            for key, value in config_dict.items()
        }

        json_str = json.dumps(safe_dict)

        assert "***REDACTED***" in json_str
        assert "sk-" not in json_str

    def test_actual_api_functionality_preserved(self) -> None:
        """Test that actual API functionality still works with SecretStr."""
        api_key = "test-key-for-functionality"
        secret = SecretStr(api_key)

        # The actual value should be available for API calls
        assert secret.get_secret_value() == api_key

        # Should work with OpenAI client initialization (mocked)
        # Patch where OpenAI is imported in utils, not the original module
        with patch("tests.utils.OpenAI") as mock_openai:
            from ragzoom.config import QueryConfig
            from tests.conftest import SimpleMockStore  # type: ignore
            from tests.utils import create_retriever  # type: ignore

            query_config = QueryConfig()
            store = Mock(spec=SimpleMockStore)
            # Add for_document method to the mock
            store.for_document = Mock(return_value=Mock())

            # This should pass the actual API key to the OpenAI client
            # Note: Don't pass a client so create_retriever creates one
            create_retriever(
                query_config, store, api_key=secret.get_secret_value(), client=None
            )

            # Verify the OpenAI client was initialized with the actual key
            mock_openai.assert_called_once_with(api_key=api_key)

    def test_environment_variable_protection(self) -> None:
        """Test that environment variables are properly protected."""
        api_key = "sk-1234567890123456789012345678901234567890123456789"

        with patch.dict("os.environ", {"OPENAI_API_KEY": api_key}):
            config = OperationalConfig()

            # Environment variable should be wrapped in SecretStr
            assert isinstance(config.openai_api_key, SecretStr)
            assert config.openai_api_key.get_secret_value() == api_key

            # String representation should be redacted
            config_str = str(config)
            assert "***REDACTED***" in config_str
            assert api_key not in config_str
