"""LRU cache management for hot TreeNodes."""

from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class CacheManager(Generic[T]):
    """LRU cache manager for storing frequently accessed objects."""

    def __init__(self, cache_size: int = 1000):
        """Initialize cache manager with specified size.

        Args:
            cache_size: Maximum number of items to store in cache
        """
        self.cache_size = cache_size
        self.cache: dict[str, T] = {}
        self.cache_order: deque[str] = deque(maxlen=cache_size)

    def get(self, key: str) -> T | None:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item if found, None otherwise
        """
        if key not in self.cache:
            return None

        # Move to end (most recently used)
        self._move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: T) -> None:
        """Put item in cache.

        Args:
            key: Cache key
            value: Item to cache
        """
        if key in self.cache:
            # Update existing item
            self.cache[key] = value
            self._move_to_end(key)
        else:
            # Add new item
            if len(self.cache) >= self.cache_size:
                # Remove least recently used item
                oldest_key = self.cache_order[0]
                self.remove(oldest_key)

            self.cache[key] = value
            self.cache_order.append(key)

    def remove(self, key: str) -> None:
        """Remove item from cache.

        Args:
            key: Cache key to remove
        """
        if key in self.cache:
            del self.cache[key]
            if key in self.cache_order:
                self.cache_order.remove(key)

    def clear(self) -> None:
        """Clear all items from cache."""
        self.cache.clear()
        self.cache_order.clear()

    def _move_to_end(self, key: str) -> None:
        """Move key to end of cache order (most recently used)."""
        if key in self.cache_order:
            self.cache_order.remove(key)
        self.cache_order.append(key)

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def contains(self, key: str) -> bool:
        """Check if key is in cache."""
        return key in self.cache

    def invalidate(self, key: str) -> None:
        """Invalidate (remove) item from cache.

        Args:
            key: Cache key to invalidate
        """
        self.remove(key)
