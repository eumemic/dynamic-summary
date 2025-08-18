"""Length targeting strategies for summarization experiments."""

from .library.absolute_token import AbsoluteTokenStrategy
from .library.relative_token import RelativeTokenStrategy
from .library.absolute_char import AbsoluteCharStrategy
from .library.relative_char import RelativeCharStrategy
from .library.percentage import PercentageStrategy
from .library.word_count import WordCountStrategy
from .library.shorten_percentage import ShortenPercentageStrategy
from .library.reduce_percentage import ReducePercentageStrategy
from .library.word_count_compensated import WordCountCompensatedStrategy
from .library.cut_percentage import CutPercentageStrategy

# List of all available strategies
ALL_STRATEGIES = [
    # Original strategies
    AbsoluteTokenStrategy(),
    RelativeTokenStrategy(),
    AbsoluteCharStrategy(),
    RelativeCharStrategy(),
    PercentageStrategy(),  # Known to be broken but kept for comparison
    WordCountStrategy(),
    
    # New generation 1 strategies based on discoveries
    ShortenPercentageStrategy(),  # Best performer in testing
    ReducePercentageStrategy(),
    WordCountCompensatedStrategy(),
    CutPercentageStrategy(),
]

# Dictionary for easy lookup
STRATEGIES_BY_NAME = {s.name: s for s in ALL_STRATEGIES}