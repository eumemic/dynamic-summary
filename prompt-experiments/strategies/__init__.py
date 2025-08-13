"""Length targeting strategies for summarization experiments."""

from .absolute_token import AbsoluteTokenStrategy
from .relative_token import RelativeTokenStrategy
from .absolute_char import AbsoluteCharStrategy
from .relative_char import RelativeCharStrategy
from .percentage import PercentageStrategy
from .word_count import WordCountStrategy

# List of all available strategies
ALL_STRATEGIES = [
    AbsoluteTokenStrategy(),
    RelativeTokenStrategy(),
    AbsoluteCharStrategy(),
    RelativeCharStrategy(),
    PercentageStrategy(),
    WordCountStrategy(),
]

# Dictionary for easy lookup
STRATEGIES_BY_NAME = {s.name: s for s in ALL_STRATEGIES}