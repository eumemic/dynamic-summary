#!/bin/bash
# Common script for running Python quality checks
# Used by both git hooks and Claude hooks
#
# Usage: run-checks.sh [--skip tests,dmypy,ruff,black,jscpd] [file_or_directory ...]
#
# Exit codes:
#   0 - All checks passed
#   2 - One or more checks failed (Claude-compatible)

set -uo pipefail  # Don't use -e, we handle errors explicitly

# Parse command line arguments
SKIP_CHECKS=""
TARGETS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip)
            SKIP_CHECKS="$2"
            shift 2
            ;;
        *)
            TARGETS="$TARGETS $1"
            shift
            ;;
    esac
done

# Get repository root (works in main repo and worktrees)
GIT_ROOT="$(git rev-parse --show-toplevel)"

# Source common Python check functions
source "$GIT_ROOT/scripts/python-checks-common.sh"

# Convert comma-separated skip list to array
if [[ -n "$SKIP_CHECKS" ]]; then
    IFS=',' read -ra SKIP_ARRAY <<< "$SKIP_CHECKS"
else
    SKIP_ARRAY=()
fi

# Function to check if a check should be skipped
should_skip() {
    local check="$1"
    [[ ${#SKIP_ARRAY[@]} -eq 0 ]] && return 1
    for skip in "${SKIP_ARRAY[@]}"; do
        [[ "$skip" == "$check" ]] && return 0
    done
    return 1
}

# Get list of modified files for git context (if available)
if git diff --cached --name-only &>/dev/null; then
    modified_files=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.py$' || true)
else
    modified_files=""
fi

# Default targets if none specified
if [[ -z "$TARGETS" ]]; then
    TARGETS="ragzoom tests"
fi

# Track overall success
OVERALL_FAILED=0

# Create temporary directory for storing results
tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Function to run a check and handle output
run_check() {
    local check_name="$1"
    local check_function="$2"
    local check_args="${3:-}"
    
    if should_skip "$check_name"; then
        return 0
    fi
    
    # Run the check, capturing both stdout and stderr
    local output_file="$tmpdir/${check_name}.output"
    local stderr_file="$tmpdir/${check_name}.stderr"
    local result_file="$tmpdir/${check_name}.result"
    
    # Execute the check
    if [[ -n "$check_args" ]]; then
        $check_function $check_args > "$output_file" 2> "$stderr_file"
        echo $? > "$result_file"
    else
        $check_function > "$output_file" 2> "$stderr_file"
        echo $? > "$result_file"
    fi
    
    local result=$(cat "$result_file")
    
    # Handle output based on result
    if [[ $result -eq 0 ]]; then
        # Success - output to stdout
        cat "$output_file"
        [[ -s "$stderr_file" ]] && cat "$stderr_file"
    else
        # Failure - output to stderr
        cat "$output_file" >&2
        [[ -s "$stderr_file" ]] && cat "$stderr_file" >&2
        OVERALL_FAILED=1
    fi
    
    return $result
}

# Run tests
if ! should_skip "tests"; then
    echo "[Tests] Starting..."
    if command -v pytest &> /dev/null; then
        tmpfile=$(mktemp)
        pytest tests/ -q --tb=short -m "not slow and not integration and not benchmark" -n 8 --no-header > "$tmpfile" 2>&1
        result=$?
        if [ $result -ne 0 ]; then
            echo "[Tests] ❌ Failed!" >&2
            cat "$tmpfile" >&2
            OVERALL_FAILED=1
        else
            echo "[Tests] ✅ Passed!"
        fi
        rm -f "$tmpfile"
    else
        echo "[Tests] Skipped (pytest not installed)"
    fi
fi

# Run dmypy (no arguments - let it figure out what changed)
if ! should_skip "dmypy"; then
    echo "[Mypy] Starting..."
    if command -v dmypy &> /dev/null; then
        # Don't pass file arguments to dmypy - it's more efficient without them
        dmypy run -- ragzoom --ignore-missing-imports --no-error-summary --check-untyped-defs > "$tmpdir/dmypy.output" 2>&1
        result=$?
        if [ $result -ne 0 ]; then
            echo "[Mypy] ❌ Failed! Type errors found" >&2
            cat "$tmpdir/dmypy.output" >&2
            OVERALL_FAILED=1
        else
            echo "[Mypy] ✅ Passed!"
        fi
    else
        echo "[Mypy] ❌ dmypy not installed - cannot run type checks" >&2
        exit 2
    fi
fi

# Run ruff
if ! should_skip "ruff"; then
    run_check "ruff" "run_ruff" "$TARGETS \"$modified_files\""
fi

# Run black
if ! should_skip "black"; then
    run_check "black" "run_black" "$TARGETS \"$modified_files\""
fi

# Run jscpd
if ! should_skip "jscpd"; then
    if command -v npx &> /dev/null && [[ -n "$modified_files" ]]; then
        echo "[JSCPD] Starting..."
        tmpfile=$(mktemp)
        npx jscpd@latest ragzoom/ --config "$GIT_ROOT/.jscpd.json" > "$tmpfile" 2>&1
        result=$?
        if [ $result -ne 0 ]; then
            echo "[JSCPD] ❌ Code duplication found!" >&2
            cat "$tmpfile" >&2
            OVERALL_FAILED=1
        else
            echo "[JSCPD] ✅ Passed!"
        fi
        rm -f "$tmpfile"
    else
        echo "[JSCPD] Skipped (no Python files or npx not available)"
    fi
fi

# Exit with code 2 if any check failed (Claude-compatible)
if [ $OVERALL_FAILED -ne 0 ]; then
    exit 2
else
    exit 0
fi