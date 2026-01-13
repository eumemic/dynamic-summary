"""Service for calculating conservative seed counts based on token budgets."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragzoom.document_store import DocumentStore

logger = logging.getLogger(__name__)


class BudgetPlanner:
    """Plans conservative seed counts to ensure budget compliance."""

    def __init__(
        self, document_store: "DocumentStore | None", default_chunk_tokens: int
    ):
        """Initialize budget planner.

        Args:
            document_store: Optional document store for statistics
            default_chunk_tokens: Default chunk size from config

        Raises:
            ValueError: If default_chunk_tokens is not positive
        """
        if default_chunk_tokens <= 0:
            raise ValueError(
                f"default_chunk_tokens must be positive, got {default_chunk_tokens}"
            )
        self.document_store = document_store
        self.default_chunk_tokens = default_chunk_tokens

    def calculate_conservative_num_seeds(
        self, budget_tokens: int, document_id: str | None = None
    ) -> int:
        """Calculate num_seeds based on budget and average leaf token size.

        Args:
            budget_tokens: Token budget for the summary
            document_id: Optional document ID for better estimation

        Returns:
            Number of seeds that should fit in budget
        """
        chunk_tokens = self._get_effective_chunk_tokens(document_id)
        return max(1, budget_tokens // chunk_tokens)

    def _get_effective_chunk_tokens(self, document_id: str | None) -> int:
        """Determine effective chunk token size for calculations.

        Tries to use actual document statistics, falls back to defaults.

        Args:
            document_id: Optional document ID for better estimation

        Returns:
            Effective chunk token size to use for calculations
        """
        # No document store or ID available - use default
        if not document_id or not self.document_store:
            logger.info(
                f"Cross-document query: using estimated chunk size "
                f"{self.default_chunk_tokens} for num_seeds calculation"
            )
            return self.default_chunk_tokens

        # Document store is for different document - use default
        if self.document_store.document_id != document_id:
            logger.warning(
                f"Document store is for document {self.document_store.document_id} "
                f"but query is for document {document_id}. Using default estimation."
            )
            return self.default_chunk_tokens

        # Try to get actual statistics from document
        avg_leaf_tokens = self.document_store.get_avg_leaf_tokens()
        if avg_leaf_tokens:
            logger.debug(
                f"Using actual avg leaf tokens {avg_leaf_tokens} for document {document_id}"
            )
            return avg_leaf_tokens

        # No statistics available - use default
        logger.info(
            f"No token statistics for document {document_id}. "
            f"Using default chunk size {self.default_chunk_tokens} for estimation"
        )
        return self.default_chunk_tokens
