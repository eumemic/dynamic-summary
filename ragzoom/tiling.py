"""Tiling data structure for representing optimal node selections."""

from dataclasses import dataclass


@dataclass
class Tiling:
    """Represents a tiling with its nodes and token-weighted relevance score."""

    node_ids: list[str]
    relevance_tokens: float  # Total relevance score weighted by token count

    def __add__(self, other: "Tiling") -> "Tiling":
        """Concatenate two tilings."""
        return Tiling(
            node_ids=self.node_ids + other.node_ids,
            relevance_tokens=self.relevance_tokens + other.relevance_tokens,
        )

    @classmethod
    def empty(cls) -> "Tiling":
        """Create an empty tiling."""
        return cls(node_ids=[], relevance_tokens=0.0)
