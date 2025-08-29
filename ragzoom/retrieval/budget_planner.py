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
        """
        self.document_store = document_store
        self.default_chunk_tokens = default_chunk_tokens

    def calculate_conservative_num_seeds(
        self, budget_tokens: int, document_id: str | None = None
    ) -> int:
        """Calculate conservative num_seeds using efficient SQL aggregation.

        Args:
            budget_tokens: Token budget for the summary
            document_id: Optional document ID for better estimation

        Returns:
            Conservative number of seeds that should fit in budget
        """
        if not document_id or not self.document_store:
            logger.info(
                f"Cross-document query: using estimated chunk size {self.default_chunk_tokens} for num_seeds calculation"
            )
            return max(1, int(budget_tokens // self.default_chunk_tokens))

        # Verify document store matches the requested document
        if self.document_store.document_id != document_id:
            logger.warning(
                f"Document store is for document {self.document_store.document_id} "
                f"but query is for document {document_id}. Using default estimation."
            )
            return max(1, int(budget_tokens // self.default_chunk_tokens))

        # Try to get actual statistics from document
        avg_leaf_tokens = self.document_store.get_avg_leaf_tokens()
        if avg_leaf_tokens:
            logger.debug(
                f"Using actual avg leaf tokens {avg_leaf_tokens} for document {document_id}"
            )
            return max(1, int(budget_tokens // avg_leaf_tokens))

        # Fallback to default if no statistics available
        logger.info(
            f"No token statistics for document {document_id}. "
            f"Using default chunk size {self.default_chunk_tokens} for estimation"
        )
        return max(1, int(budget_tokens // self.default_chunk_tokens))
