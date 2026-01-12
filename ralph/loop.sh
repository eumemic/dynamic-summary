#!/bin/bash
# Ralph Loop - Spec-to-Code Synchronization Engine
#
# Usage:
#   ./ralph/loop.sh plan         # Plan mode, unlimited iterations
#   ./ralph/loop.sh plan 5       # Plan mode, max 5 iterations
#   ./ralph/loop.sh build        # Build mode, unlimited iterations
#   ./ralph/loop.sh build 20     # Build mode, max 20 iterations

set -euo pipefail

RALPH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Help
show_help() {
    cat <<'EOF'
Ralph Loop - Spec-to-Code Synchronization Engine

Usage:
  ./ralph/loop.sh <mode> [max_iterations]

Modes:
  plan         Plan mode - analyzes specs/ and updates IMPLEMENTATION_PLAN.md
  build        Build mode - implements tasks from IMPLEMENTATION_PLAN.md

Options:
  -h, --help   Show this help message

Examples:
  ./ralph/loop.sh plan         # Plan mode, unlimited iterations
  ./ralph/loop.sh plan 5       # Plan mode, max 5 iterations
  ./ralph/loop.sh build        # Build mode, unlimited iterations
  ./ralph/loop.sh build 20     # Build mode, max 20 iterations

Termination:
  - Plan mode: stops when plan file unchanged after an iteration
  - Build mode: stops when all tasks are checked off (no more "- [ ]" items)
  - Both modes: stops when max_iterations reached (if specified)

Files:
  specs/                       Source of truth (human-authored specifications)
  ralph/IMPLEMENTATION_PLAN.md Generated task list (machine-authored)
  ralph/PROMPT_plan.md         Instructions for planning mode
  ralph/PROMPT_build.md        Instructions for building mode
EOF
}

# Parse arguments
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_help
    exit 0
elif [ "${1:-}" = "plan" ]; then
    MODE="plan"
    PROMPT_FILE="$RALPH_DIR/PROMPT_plan.md"
    MAX_ITERATIONS=${2:-0}
elif [ "${1:-}" = "build" ]; then
    MODE="build"
    PROMPT_FILE="$RALPH_DIR/PROMPT_build.md"
    MAX_ITERATIONS=${2:-0}
else
    echo "Error: Mode required. Use 'plan' or 'build'."
    echo ""
    show_help
    exit 1
fi

ITERATION=0
CURRENT_BRANCH=$(git branch --show-current)
PLAN_FILE="$RALPH_DIR/IMPLEMENTATION_PLAN.md"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Mode:   $MODE"
echo "Prompt: $PROMPT_FILE"
echo "Branch: $CURRENT_BRANCH"
if [ $MAX_ITERATIONS -gt 0 ]; then
    echo "Max:    $MAX_ITERATIONS iterations"
else
    echo "Max:    unlimited"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verify prompt file exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: $PROMPT_FILE not found"
    exit 1
fi

# Get hash of plan file (empty string if doesn't exist)
get_plan_hash() {
    if [ -f "$PLAN_FILE" ]; then
        md5 -q "$PLAN_FILE" 2>/dev/null || md5sum "$PLAN_FILE" | cut -d' ' -f1
    else
        echo ""
    fi
}

# Check if plan has pending tasks (unchecked markdown checkboxes)
plan_has_pending_tasks() {
    if [ -f "$PLAN_FILE" ]; then
        grep -q '^\s*- \[ \]' "$PLAN_FILE"
    else
        return 1  # No plan = no tasks
    fi
}

while true; do
    if [ $MAX_ITERATIONS -gt 0 ] && [ $ITERATION -ge $MAX_ITERATIONS ]; then
        echo "Reached max iterations: $MAX_ITERATIONS"
        break
    fi

    # Build mode: check if all tasks are complete
    if [ "$MODE" = "build" ] && ! plan_has_pending_tasks; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "All tasks complete - build finished"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        break
    fi

    ITERATION=$((ITERATION + 1))
    PLAN_HASH_BEFORE=$(get_plan_hash)

    echo ""
    echo "━━━━━━━━━━━━━━━━ Iteration $ITERATION ━━━━━━━━━━━━━━━━"
    echo ""

    # Run Ralph iteration
    # -p: Headless mode (non-interactive, reads from stdin)
    # --dangerously-skip-permissions: Auto-approve all tool calls
    # --model: Use appropriate model for the task
    # --output-format stream-json: Stream JSONL for real-time progress
    # --verbose: Required for stream-json in print mode
    cat "$PROMPT_FILE" | claude -p \
        --dangerously-skip-permissions \
        --model sonnet \
        --output-format stream-json \
        --verbose \
        | "$RALPH_DIR/stream-progress.sh"

    # Check if plan changed (for planning mode termination)
    PLAN_HASH_AFTER=$(get_plan_hash)
    if [ "$MODE" = "plan" ] && [ "$PLAN_HASH_BEFORE" = "$PLAN_HASH_AFTER" ] && [ -n "$PLAN_HASH_BEFORE" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Plan unchanged - planning complete"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        break
    fi

done
