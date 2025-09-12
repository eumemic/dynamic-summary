"""Centralized tokenization utilities for RagZoom.

This module provides a singleton tokenizer to eliminate duplication of
tiktoken initialization across the codebase and improve performance.
"""

import threading
from typing import Optional

import tiktoken

try:
    # Pre-initialize encoder at import time to avoid per-test slow starts
    _DEFAULT_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception as _tok_err:  # pragma: no cover - environment specific
    raise


class TokenizerUtil:
    """Thread-safe singleton tokenizer utility.

    Provides centralized access to tiktoken encoding operations with lazy
    initialization and caching to eliminate duplicate encoder creation.
    """

    _instance: Optional["TokenizerUtil"] = None
    _lock = threading.Lock()
    _encoder: tiktoken.Encoding | None = _DEFAULT_ENCODER

    def __new__(cls) -> "TokenizerUtil":
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def encoder(self) -> tiktoken.Encoding:
        """Get the tiktoken encoder, initializing if needed."""
        if TokenizerUtil._encoder is None:
            with TokenizerUtil._lock:
                if TokenizerUtil._encoder is None:
                    TokenizerUtil._encoder = _DEFAULT_ENCODER
        return TokenizerUtil._encoder

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
