"""
Active strategies manifest for experiments.

Edit this file to control which strategies are tested in the next run.
The evolution happens through our conversation and curation of this list.
"""

from strategies.library.word_count import WordCountStrategy
from strategies.library.word_count_compensated import WordCountCompensatedStrategy

# Generation 3: Focus on word-based strategies only
# Dropped: absolute_char (28% error too high)
ACTIVE_STRATEGIES = [
    WordCountCompensatedStrategy(),  # 7.3% error - BEST
    WordCountStrategy(),  # 9.5% error - excellent for comparison
]
