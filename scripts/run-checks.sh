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
FAIL_FAST=false
INCLUDE_SLOW_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip)
            SKIP_CHECKS="$2"
            shift 2
            ;;
        --fail-fast)
            FAIL_FAST=true
            shift
            ;;
        --include-slow-tests)
            INCLUDE_SLOW_TESTS=true
            shift
            ;;
        *)
            TARGETS="$TARGETS $1"
            shift
            ;;
    esac
done

# Check for incoherent options
if [ "$INCLUDE_SLOW_TESTS" = true ] && [[ "$SKIP_CHECKS" == *"tests"* ]]; then
    echo "Error: Cannot use --include-slow-tests with --skip tests (incoherent)" >&2
    exit 1
fi

# Get repository root (works in main repo and worktrees)
GIT_ROOT="$(git rev-parse --show-toplevel)"

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

# Store process IDs for parallel execution
declare -a pids=()

# Cleanup function
cleanup() {
    # Kill any remaining background processes
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    rm -rf "$tmpdir"
}
trap cleanup EXIT

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

# Function to run a check in background
run_check_background() {
    local check_name="$1"
    local check_cmd="$2"
    local output_file="$tmpdir/${check_name}.output"
    local result_file="$tmpdir/${check_name}.result"
    
    (
        echo "[$check_name] Starting..." > "$output_file"
        if eval "$check_cmd" >> "$output_file" 2>&1; then
            echo 0 > "$result_file"
            echo "[$check_name] ✅ Passed!" >> "$output_file"
        else
            echo 1 > "$result_file"
            echo "[$check_name] ❌ Failed!" >> "$output_file"
        fi
    ) &
    local pid=$!
    pids+=("$pid")
}

# Start all checks in parallel

# Tests
if ! should_skip "tests"; then
    if command -v pytest &> /dev/null; then
        if [ "$INCLUDE_SLOW_TESTS" = true ]; then
            # Ensure PostgreSQL is running for integration tests
            echo "[PostgreSQL] Ensuring PostgreSQL is running for integration tests..."
            python -c "from ragzoom.docker_postgres import DockerPostgres; dp = DockerPostgres(); dp.ensure_running()" 2>/dev/null || {
                echo "[PostgreSQL] Warning: Could not start PostgreSQL, integration tests may fail"
            }
            # Run all tests including slow and integration
            run_check_background "Tests" "pytest tests/ -q --tb=short -m 'not benchmark' -n 8 --no-header"
        else
            # Run only fast tests (default)
            run_check_background "Tests" "pytest tests/ -q --tb=short -m 'not slow and not integration and not benchmark' -n 8 --no-header"
        fi
    else
        echo "[Tests] Skipped (pytest not installed)"
    fi
fi

# dmypy
if ! should_skip "dmypy"; then
    if command -v dmypy &> /dev/null; then
        run_check_background "Mypy" "dmypy run -- ragzoom --ignore-missing-imports --no-error-summary --check-untyped-defs"
    else
        echo "[Mypy] ❌ dmypy not installed - cannot run type checks" >&2
        exit 2
    fi
fi

# ruff
if ! should_skip "ruff"; then
    if command -v ruff &> /dev/null; then
        if [ -z "$modified_files" ] || [ -n "$modified_files" ]; then
            run_check_background "Ruff" "ruff check $TARGETS --fix --quiet --output-format concise"
        else
            echo "[Ruff] Skipped (no Python files)"
        fi
    else
        echo "[Ruff] Skipped (not installed)"
    fi
fi

# black
if ! should_skip "black"; then
    if command -v black &> /dev/null; then
        if [ -z "$modified_files" ] || [ -n "$modified_files" ]; then
            run_check_background "Black" "black $TARGETS --quiet"
        else
            echo "[Black] Skipped (no Python files)"
        fi
    else
        echo "[Black] Skipped (not installed)"
    fi
fi

# jscpd
if ! should_skip "jscpd"; then
    if command -v npx &> /dev/null; then
        run_check_background "JSCPD" "npx jscpd@latest ragzoom/ --config $GIT_ROOT/.jscpd.json"
    else
        echo "[JSCPD] Skipped (npx not available)"
    fi
fi

# bandit (security check)
if ! should_skip "bandit"; then
    if command -v bandit &> /dev/null; then
        run_check_background "Bandit" "bandit -r ragzoom/ -ll --quiet"
    else
        echo "[Bandit] Skipped (not installed)"
    fi
fi

# Wait for all processes or fail-fast
if [ "$FAIL_FAST" = true ]; then
    # Monitor processes and exit on first failure
    while [ ${#pids[@]} -gt 0 ]; do
        new_pids=()
        for pid in "${pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process finished, check result
                for check in Tests Mypy Ruff Black JSCPD Bandit; do
                    result_file="$tmpdir/${check}.result"
                    if [ -f "$result_file" ] && [ "$(cat "$result_file")" = "1" ]; then
                        # Found a failure, kill all other processes
                        {
                            for other_pid in "${pids[@]}"; do
                                kill "$other_pid" || true
                            done
                            wait  # Wait for processes to die to avoid "Terminated" messages
                        } 2>/dev/null
                        # Output the failure
                        cat "$tmpdir/${check}.output" >&2
                        exit 2
                    fi
                done
            else
                new_pids+=("$pid")
            fi
        done
        pids=("${new_pids[@]+"${new_pids[@]}"}")
        [ ${#pids[@]} -gt 0 ] && sleep 0.05
    done
else
    # Wait for all processes to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
fi

# Display results in order
for check in Tests Mypy Ruff Black JSCPD Bandit; do
    output_file="$tmpdir/${check}.output"
    result_file="$tmpdir/${check}.result"
    if [ -f "$output_file" ]; then
        if [ -f "$result_file" ] && [ "$(cat "$result_file")" = "1" ]; then
            cat "$output_file" >&2
            OVERALL_FAILED=1
        else
            cat "$output_file"
        fi
    fi
done

# Exit with code 2 if any check failed (Claude-compatible)
if [ $OVERALL_FAILED -ne 0 ]; then
    exit 2
else
    exit 0
fi