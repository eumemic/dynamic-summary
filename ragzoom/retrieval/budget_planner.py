"""Service for calculating conservative seed counts based on token budgets."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragzoom.document_store import DocumentStore
    from ragzoom.store import StoreManager

logger = logging.getLogger(__name__)


class BudgetPlanner:
    """Plans conservative seed counts to ensure budget compliance."""

    def __init__(
        self, store: "StoreManager | DocumentStore", default_chunk_tokens: int
    ):
        """Initialize budget planner.

        Args:
            store: Store instance (system-wide or document-scoped) for statistics
            default_chunk_tokens: Default chunk size from config
        """
        self.store = store
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
        if not document_id:
            logger.info(
                f"Cross-document query: using estimated chunk size {self.default_chunk_tokens} for num_seeds calculation"
            )
            return max(1, int(budget_tokens // self.default_chunk_tokens))

        # TODO: Implement document token stats in DocumentStore
        # For now, use default estimation
        logger.warning(
            f"Document token stats not implemented for {document_id}. "
            f"Using default chunk size {self.default_chunk_tokens} for estimation"
        )
        return max(1, int(budget_tokens // self.default_chunk_tokens))
