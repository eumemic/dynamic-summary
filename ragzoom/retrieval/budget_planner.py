"""Service for calculating conservative seed counts based on token budgets."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragzoom.store import Store

logger = logging.getLogger(__name__)


class BudgetPlanner:
    """Plans conservative seed counts to ensure budget compliance."""

    def __init__(self, store: "Store", default_chunk_tokens: int):
        """Initialize budget planner.

        Args:
            store: Store instance for statistics
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

        stats = self.store.get_document_token_stats(document_id)

        if not stats["node_count"] or not stats["avg_tokens"]:
            logger.warning(
                f"Document {document_id} has no token statistics. "
                f"Using default chunk size estimate: {self.default_chunk_tokens}"
            )
            return max(1, int(budget_tokens // self.default_chunk_tokens))

        safe_average_cost = stats["avg_tokens"] * 1.25
        conservative_num_seeds = max(1, int(budget_tokens // safe_average_cost))

        return conservative_num_seeds
