"""Centralized tokenization utilities for RagZoom.

This module provides a singleton tokenizer to eliminate duplication of
tiktoken initialization across the codebase and improve performance.

In restricted network environments (e.g., Claude Code web), tiktoken may
fail to download its encoding data. This module provides a fallback
approximation tokenizer that estimates ~4 characters per token.
"""

import logging
import threading
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class TokenEncoder(Protocol):
    """Protocol for tokenizer encoders."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...


class ApproximateEncoder:
    """Fallback encoder that approximates tiktoken behavior.

    Uses ~4 characters per token, which is a reasonable approximation
    for English text with cl100k_base encoding. This allows tests to
    run in network-restricted environments.
    """

    CHARS_PER_TOKEN = 4

    def encode(self, text: str) -> list[int]:
        """Approximate encoding: ~4 chars per token."""
        if not text:
            return []
        # Return sequential token IDs for each chunk
        num_tokens = max(1, len(text) // self.CHARS_PER_TOKEN)
        return list(range(num_tokens))

    def decode(self, tokens: list[int]) -> str:
        """Decode is lossy with approximation - return placeholder."""
        # We can't truly decode without the real encoder
        # Return a placeholder of approximate length
        return "x" * (len(tokens) * self.CHARS_PER_TOKEN)


# Try to load tiktoken, fall back to approximation if network is unavailable
_DEFAULT_ENCODER: TokenEncoder
_USING_FALLBACK = False

try:
    import tiktoken

    # Pre-initialize encoder at import time to avoid per-test slow starts
    _DEFAULT_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception as _tok_err:
    # Network unavailable or other tiktoken error - use fallback
    _DEFAULT_ENCODER = ApproximateEncoder()
    _USING_FALLBACK = True
    logger.warning(
        "tiktoken unavailable (network restricted?), using approximate tokenizer. "
        "Token counts will be estimates. Error: %s",
        _tok_err,
    )


def is_using_fallback_tokenizer() -> bool:
    """Check if the fallback approximate tokenizer is in use.

    Returns:
        True if tiktoken couldn't be loaded and we're using approximation.
    """
    return _USING_FALLBACK


class TokenizerUtil:
    """Thread-safe singleton tokenizer utility.

    Provides centralized access to tiktoken encoding operations with lazy
    initialization and caching to eliminate duplicate encoder creation.
    """

    _instance: Optional["TokenizerUtil"] = None
    _lock = threading.Lock()
    _encoder: TokenEncoder | None = _DEFAULT_ENCODER

    def __new__(cls) -> "TokenizerUtil":
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def encoder(self) -> TokenEncoder:
        """Get the tokenizer encoder, initializing if needed."""
        if TokenizerUtil._encoder is None:
            with TokenizerUtil._lock:
                if TokenizerUtil._encoder is None:
                    TokenizerUtil._encoder = _DEFAULT_ENCODER
        return TokenizerUtil._encoder

    @property
    def is_approximate(self) -> bool:
        """Check if using the approximate fallback encoder."""
        return _USING_FALLBACK

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to tokenize and count

        Returns:
            Number of tokens in the text
        """
        return len(self.encoder.encode(text))

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        return self.encoder.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs to decode

        Returns:
            Decoded text
        """
        return self.encoder.decode(tokens)

    def truncate_to_token_limit(
        self, text: str, max_tokens: int, from_end: bool = True
    ) -> str:
        """Truncate text to fit within a token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens allowed
            from_end: If True, keep the end of the text (truncate from start).
                     If False, keep the start of the text (truncate from end).

        Returns:
            Truncated text that fits within the token limit
        """
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text

        if from_end:
            # Keep the last max_tokens tokens (truncate from start)
            truncated_tokens = tokens[-max_tokens:]
        else:
            # Keep the first max_tokens tokens (truncate from end)
            truncated_tokens = tokens[:max_tokens]

        return self.decode(truncated_tokens)


# Global instance for convenience
tokenizer = TokenizerUtil()


def count_tokens(text: str) -> int:
    """Convenience function to count tokens in text.

    Args:
        text: Text to tokenize and count

    Returns:
        Number of tokens in the text
    """
    return tokenizer.count_tokens(text)


def encode_text(text: str) -> list[int]:
    """Convenience function to encode text to token IDs.

    Args:
        text: Text to encode

    Returns:
        List of token IDs
    """
    return tokenizer.encode(text)


def decode_tokens(tokens: list[int]) -> str:
    """Convenience function to decode token IDs to text.

    Args:
        tokens: List of token IDs to decode

    Returns:
        Decoded text
    """
    return tokenizer.decode(tokens)
