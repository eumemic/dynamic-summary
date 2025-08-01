"""Shared configuration constants for telemetry visualization and analysis."""

# Visualization configuration constants
DISPLAY_DPI = 100  # Screen display resolution for development
SAVE_DPI = 300  # High resolution for production reports
DEFAULT_FONT_SIZE = 10  # Base font size optimized for readability

# Figure dimensions chosen for comprehensive dashboard layout:
# - 20 inch width: accommodates side-by-side subplots with readable labels
# - 24 inch height: allows vertical layout of 6-7 charts without cramping
FIGURE_WIDTH = 20
FIGURE_HEIGHT = 24

# API pricing constants (as of January 2025, used for visualization consistency)
# Note: These are older pricing values maintained for consistency with existing benchmarks
# To use current pricing, set these environment variables:
# - RAGZOOM_EMBEDDING_COST_PER_1K (default: 0.0001)
# - RAGZOOM_SUMMARY_INPUT_COST_PER_1K (default: 0.0025)
# - RAGZOOM_SUMMARY_OUTPUT_COST_PER_1K (default: 0.01)
EMBEDDING_COST_PER_1K = 0.0001  # text-embedding-3-small (older pricing)
SUMMARY_INPUT_COST_PER_1K = 0.0025  # gpt-4o-mini input (older pricing)
SUMMARY_OUTPUT_COST_PER_1K = 0.01  # gpt-4o-mini output (older pricing)

# Default chunk size if unable to determine from benchmark data
DEFAULT_CHUNK_SIZE = 200

# Emoji display thresholds - these control when to show warning/success indicators
# They don't trigger regression failures, just visual feedback
EMOJI_THRESHOLD_NEGLIGIBLE = 1.0  # Changes below this are not highlighted
EMOJI_THRESHOLD_COST_WARN = 10.0  # Cost increase above this shows warning
EMOJI_THRESHOLD_COST_GOOD = 5.0  # Cost decrease above this shows success
EMOJI_THRESHOLD_MINOR = 5.0  # Minor changes worth noting
EMOJI_THRESHOLD_MODERATE = 10.0  # Moderate changes that warrant attention
EMOJI_THRESHOLD_MAJOR = 20.0  # Major changes that are concerning

# Change significance threshold - changes below this are hidden in comparison reports
# This filters out noise from minor fluctuations to focus on meaningful changes
CHANGE_SIGNIFICANCE_THRESHOLD = 1.0  # Hide changes smaller than ±1%
