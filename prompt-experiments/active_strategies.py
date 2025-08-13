"""
Active strategies manifest for experiments.

Edit this file to control which strategies are tested in the next run.
The evolution happens through our conversation and curation of this list.
"""

from strategies import ALL_STRATEGIES

# Generation 1: Test all available strategies
ACTIVE_STRATEGIES = ALL_STRATEGIES

# After reviewing results, you might update to something like:
# 
# from strategies.library.word_count import WordCountStrategy
# from strategies.library.word_count_compensated import WordCountCompensatedStrategy
# from strategies.library.shorten_percentage import ShortenPercentageStrategy
#
# ACTIVE_STRATEGIES = [
#     WordCountStrategy(),
#     WordCountCompensatedStrategy(compensation_factor=0.94),
#     WordCountCompensatedStrategy(compensation_factor=0.90),
#     ShortenPercentageStrategy(),
# ]