"""Unit tests for tokenization utilities."""

import threading
import time
from collections.abc import Generator

import pytest

from ragzoom.utils.tokenization import (
    TokenizerUtil,
    count_tokens,
    decode_tokens,
    encode_text,
    tokenizer,
)


class TestTokenizerUtil:
    """Test the TokenizerUtil singleton class."""

    def test_singleton_pattern(self) -> None:
        """Test that TokenizerUtil follows singleton pattern."""
        t1 = TokenizerUtil()
        t2 = TokenizerUtil()
        assert t1 is t2, "TokenizerUtil should return same instance"

    def test_singleton_with_global_instance(self) -> None:
        """Test that global tokenizer instance is same as class instance."""
        t1 = TokenizerUtil()
        assert tokenizer is t1, "Global tokenizer should be same as class instance"

    def test_thread_safety(self) -> None:
        """Test that TokenizerUtil is thread-safe."""
        instances = []
        results = []

        def create_tokenizer() -> None:
            """Worker function to create tokenizer instances."""
            instances.append(TokenizerUtil())
            # Also test encoding in parallel
            result = TokenizerUtil().count_tokens("test text")
            results.append(result)

        # Create multiple threads accessing tokenizer concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_tokenizer)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All instances should be the same object
        assert (
            len(set(id(inst) for inst in instances)) == 1
        ), "All instances should be same object"

        # All results should be identical
        assert len(set(results)) == 1, "All tokenization results should be identical"
        assert results[0] == 2, "Token count should be correct"

    def test_lazy_initialization(self) -> None:
        """Test that encoder is only created when first accessed."""
        # Note: Can't fully test lazy initialization in isolation since
        # other tests may have already initialized the encoder.
        # This test verifies the caching behavior instead.

        t = TokenizerUtil()

        # Access encoder multiple times
        encoder1 = t.encoder
        encoder2 = t.encoder

        # Should return same cached encoder
        assert encoder1 is encoder2, "Should return same cached encoder"

        # After initialization, class variable should be set
        # (though it may have been set by other tests already)
        assert TokenizerUtil._encoder is not None, "Class encoder should be initialized"

    def test_encoder_initialization_thread_safety(self) -> None:
        """Test that encoder initialization is thread-safe."""
        encoders = []

        def access_encoder() -> None:
            """Worker function to access encoder."""
            t = TokenizerUtil()
            encoders.append(t.encoder)

        # Create multiple threads accessing encoder concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=access_encoder)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All encoders should be the same object
        assert (
            len(set(id(enc) for enc in encoders)) == 1
        ), "All encoders should be same object"

    def test_count_tokens(self) -> None:
        """Test token counting functionality."""
        t = TokenizerUtil()

        # Test basic counting
        assert t.count_tokens("hello world") == 2
        assert t.count_tokens("") == 0
        assert t.count_tokens("a") == 1

        # Test with longer text
        text = "This is a longer text with multiple words and punctuation!"
        count = t.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_encode_decode_roundtrip(self) -> None:
        """Test encoding and decoding roundtrip."""
        t = TokenizerUtil()

        test_text = "Hello, world! This is a test."
        tokens = t.encode(test_text)
        decoded = t.decode(tokens)

        assert isinstance(tokens, list)
        assert all(isinstance(token, int) for token in tokens)
        assert decoded == test_text

    def test_edge_cases(self) -> None:
        """Test edge cases for tokenization."""
        t = TokenizerUtil()

        # Empty string
        assert t.count_tokens("") == 0
        assert t.encode("") == []
        assert t.decode([]) == ""

        # Unicode characters
        assert t.count_tokens("🚀 emoji test") > 0

        # Very long text
        long_text = "test " * 1000
        count = t.count_tokens(long_text)
        assert count > 1000  # Should be more than 1000 tokens


class TestConvenienceFunctions:
    """Test the convenience functions."""

    def test_count_tokens_function(self) -> None:
        """Test count_tokens convenience function."""
        count = count_tokens("hello world")
        assert count == 2

        # Should match class method
        t = TokenizerUtil()
        assert count == t.count_tokens("hello world")

    def test_encode_text_function(self) -> None:
        """Test encode_text convenience function."""
        tokens = encode_text("hello world")
        assert isinstance(tokens, list)
        assert len(tokens) == 2

        # Should match class method
        t = TokenizerUtil()
        assert tokens == t.encode("hello world")

    def test_decode_tokens_function(self) -> None:
        """Test decode_tokens convenience function."""
        text = "hello world"
        tokens = encode_text(text)
        decoded = decode_tokens(tokens)

        assert decoded == text

        # Should match class method
        t = TokenizerUtil()
        assert decoded == t.decode(tokens)

    def test_consistency_across_functions(self) -> None:
        """Test that all functions give consistent results."""
        test_text = "This is a test of consistency across different APIs."

        # Count via different methods
        count1 = count_tokens(test_text)
        count2 = TokenizerUtil().count_tokens(test_text)
        count3 = len(encode_text(test_text))

        assert count1 == count2 == count3, "All counting methods should agree"

        # Encode/decode roundtrip via different methods
        tokens1 = encode_text(test_text)
        tokens2 = TokenizerUtil().encode(test_text)

        assert tokens1 == tokens2, "Encoding methods should agree"

        decoded1 = decode_tokens(tokens1)
        decoded2 = TokenizerUtil().decode(tokens2)

        assert decoded1 == decoded2 == test_text, "Decoding methods should agree"


class TestPerformance:
    """Test performance characteristics."""

    def test_singleton_performance(self) -> None:
        """Test that singleton creation is consistently fast."""
        # Measure creation time multiple times
        times = []
        for _ in range(100):
            start = time.time()
            TokenizerUtil()
            times.append(time.time() - start)

        # All creations should be very fast. Allow CI headroom (<20ms)
        max_time = max(times)
        assert (
            max_time < 0.02
        ), f"Singleton creation should be fast, got {max_time:.6f}s"

    def test_encoder_caching_performance(self) -> None:
        """Test that encoder access is consistently fast."""
        t = TokenizerUtil()

        # Measure encoder access time multiple times
        times = []
        for _ in range(100):
            start = time.time()
            t.encoder
            times.append(time.time() - start)

        # All accesses should be very fast (less than 20ms in CI environments)
        max_time = max(times)
        assert max_time < 0.02, f"Encoder access should be fast, got {max_time:.6f}s"


@pytest.fixture(autouse=True)
def reset_singleton() -> Generator[None, None, None]:
    """Reset singleton state after each test to avoid interference."""
    yield
    # Don't reset after tests as it would break other parts of the system
    # that depend on the singleton. The singleton is designed to persist.
