#!/bin/bash
# Common script for running Python quality checks
# Used by both git hooks and Claude hooks
#
# Usage: 
#   run-checks.sh [OPTIONS] [file_or_directory ...]
#
# Options:
#   --skip CHECKS           Skip specific checks (comma-separated): tests,dmypy,ruff,black,jscpd,bandit
#   --fail-fast             Stop at first failure (useful for debugging)
#   --include-slow-tests    Include slow and integration tests (auto-starts PostgreSQL)
#   --ignore-lint-rules RULES  Ignore specific lint rules (comma-separated): F401,E402,etc.
#   --fail-on-autofix       Exit with failure if any auto-fixes were applied
#   --help                  Show this help message
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
IGNORE_LINT_RULES=""
FAIL_ON_AUTOFIX=false
TEST_SCOPE="fast"  # fast (default), smoke, or all

show_help() {
    sed -n '2,/^$/p' "$0" | sed 's/^# *//'
}

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
        --ignore-lint-rules)
            IGNORE_LINT_RULES="$2"
            shift 2
            ;;
        --fail-on-autofix)
            FAIL_ON_AUTOFIX=true
            shift
            ;;
        --test-scope)
            TEST_SCOPE="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
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

# Track overall success and auto-fixes
OVERALL_FAILED=0
AUTOFIXES_APPLIED=0

# Create temporary directory for storing results
tmpdir=$(mktemp -d)

# Store process IDs for parallel execution
declare -a pids=()

# Cleanup function
cleanup() {
    # Gracefully terminate any remaining background processes
    if [ ${#pids[@]} -gt 0 ]; then
        # First try SIGTERM for graceful shutdown
        for pid in "${pids[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        # Give processes a moment to exit cleanly
        sleep 0.05
        # Force kill any that are still running
        for pid in "${pids[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        # Wait for all to finish
        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    fi
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
        # Handle SIGTERM gracefully in subshell
        trap 'exit 143' TERM
        
        echo "[$check_name] Starting..." > "$output_file"
        if eval "$check_cmd" >> "$output_file" 2>&1; then
            echo 0 > "$result_file"
            # Don't add generic "Passed!" message - let each check provide its own
            if ! grep -q "✅\|✨" "$output_file"; then
                echo "[$check_name] ✅ Passed!" >> "$output_file"
            fi
        else
            echo 1 > "$result_file"
            # Don't add generic "Failed!" message if check already provided one
            if ! grep -q "❌\|⚠️" "$output_file"; then
                echo "[$check_name] ❌ Failed!" >> "$output_file"
            fi
        fi
    ) &
    local pid=$!
    pids+=("$pid")
}

# Start all checks in parallel

# Tests
if ! should_skip "tests"; then
    if command -v pytest &> /dev/null; then
        if [ "$INCLUDE_SLOW_TESTS" = true ] || [ "$TEST_SCOPE" = "all" ]; then
            # Ensure PostgreSQL is available for integration tests
            if [ -z "$RAGZOOM_DATABASE_URL" ]; then
                echo "[PostgreSQL] Ensuring PostgreSQL is running for integration tests..."
                python -c "from ragzoom.docker_postgres import DockerPostgres; dp = DockerPostgres(); dp.ensure_running()" 2>/dev/null || {
                    echo "[PostgreSQL] Warning: Could not start PostgreSQL, integration tests may fail"
                }
            else
                echo "[PostgreSQL] Using provided database URL: $RAGZOOM_DATABASE_URL"
            fi
            # Run all tests including slow and integration
            run_check_background "Tests" "pytest tests/ -q --tb=short -m 'not benchmark' -n 8 --no-header"
        elif [ "$TEST_SCOPE" = "smoke" ]; then
            # Run a minimal, fast smoke subset: SQLite-backed tests only
            # These exercise core retrieval, assembly, and tree logic without external services
            # Note: avoid xdist overhead for small suites
            run_check_background "Tests" "pytest tests/ -q --tb=short -k 'sqlite' --no-header"
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
        run_check_background "Mypy" "dmypy run -- ragzoom tests --no-error-summary --check-untyped-defs"
    else
        echo "[Mypy] ❌ dmypy not installed - cannot run type checks" >&2
        exit 2
    fi
fi

# ruff
if ! should_skip "ruff"; then
    if command -v ruff &> /dev/null; then
        # Prefer running on changed files during git commits for speed
        if [ -n "$modified_files" ]; then
            files="$modified_files"
        else
            files="$TARGETS"
        fi
            # Enhanced ruff command that detects auto-fixes
            ruff_cmd="(
                # Capture initial state
                before_hash=\$(echo $files | xargs -I {} sh -c 'if [ -d {} ]; then find {} -name \"*.py\"; else echo {}; fi' 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
                
                # Run ruff with auto-fix
                if ruff check $files --fix --output-format concise"
            if [ -n "$IGNORE_LINT_RULES" ]; then
                ruff_cmd="$ruff_cmd --ignore $IGNORE_LINT_RULES"
            fi
            ruff_cmd="$ruff_cmd; then
                    # Check if files were modified
                    after_hash=\$(echo $files | xargs -I {} sh -c 'if [ -d {} ]; then find {} -name \"*.py\"; else echo {}; fi' 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
                    if [ \"\$before_hash\" != \"\$after_hash\" ]; then
                        echo '[Ruff] ✨ Auto-fixed all issues!'
                        echo 'AUTOFIX_OCCURRED' > $tmpdir/ruff_autofix
                    else
                        echo '[Ruff] ✅ No issues found!'
                    fi
                    exit 0
                else
                    exit_code=\$?
                    # Check if files were modified (partial fixes)
                    after_hash=\$(echo $files | xargs -I {} sh -c 'if [ -d {} ]; then find {} -name \"*.py\"; else echo {}; fi' 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
                if [ \"\$before_hash\" != \"\$after_hash\" ]; then
                    echo '[Ruff] ⚠️ Auto-fixed some issues, but manual fixes needed above'
                    echo 'AUTOFIX_OCCURRED' > $tmpdir/ruff_autofix
                else
                    echo '[Ruff] ❌ Issues found that need manual fixes (see above)'
                fi
                exit \$exit_code
                fi
            )"
            run_check_background "Ruff" "$ruff_cmd"
    else
        echo "[Ruff] Skipped (not installed)"
    fi
fi

# black
if ! should_skip "black"; then
    if command -v black &> /dev/null; then
        # Prefer running on changed files during git commits for speed
        if [ -n "$modified_files" ]; then
            files="$modified_files"
        else
            files="$TARGETS"
        fi
            # Enhanced black command that detects formatting changes
            black_cmd="(
                # Capture initial state
                before_hash=\$(echo $files | xargs -I {} sh -c 'if [ -d {} ]; then find {} -name \"*.py\"; else echo {}; fi' 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
                
                # Run black (it auto-formats by default)
                if black $files --quiet; then
                    # Check if files were modified
                    after_hash=\$(echo $files | xargs -I {} sh -c 'if [ -d {} ]; then find {} -name \"*.py\"; else echo {}; fi' 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
                    if [ \"\$before_hash\" != \"\$after_hash\" ]; then
                        echo '[Black] ✨ Reformatted files!'
                        echo 'AUTOFIX_OCCURRED' > $tmpdir/black_autofix
                    else
                        echo '[Black] ✅ All files already formatted!'
                    fi
                    exit 0
                else
                    echo '[Black] ❌ Error during formatting'
                    exit 1
                fi
            )"
            run_check_background "Black" "$black_cmd"
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
        # Filter out the "Test in comment" warnings which are just noise from nosec comments
        # Using grep -E with ? to make the pattern optional, avoiding exit code 1 when all lines match
        run_check_background "Bandit" "bandit -r ragzoom/ -ll --quiet 2>&1 | { grep -v 'WARNING.*Test in comment' || test \$? -eq 1; }"
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
                        # Found a failure, gracefully terminate all other processes
                        {
                            # Send SIGTERM to all processes
                            for other_pid in "${pids[@]}"; do
                                if [ "$other_pid" != "$pid" ] && kill -0 "$other_pid" 2>/dev/null; then
                                    kill -TERM "$other_pid" 2>/dev/null || true
                                fi
                            done
                            # Give processes a moment to exit cleanly
                            sleep 0.1
                            # Force kill any stragglers
                            for other_pid in "${pids[@]}"; do
                                if [ "$other_pid" != "$pid" ] && kill -0 "$other_pid" 2>/dev/null; then
                                    kill -KILL "$other_pid" 2>/dev/null || true
                                fi
                            done
                            # Wait for all processes to exit
                            for other_pid in "${pids[@]}"; do
                                wait "$other_pid" 2>/dev/null || true
                            done
                        } &>/dev/null
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

# Check for auto-fixes after all background processes have completed
for check in Ruff Black; do
    check_lower=$(echo "$check" | tr '[:upper:]' '[:lower:]')
    autofix_file="$tmpdir/${check_lower}_autofix"
    if [ -f "$autofix_file" ]; then
        AUTOFIXES_APPLIED=1
    fi
done

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

# Exit with code 2 if any check failed or auto-fixes were applied in strict mode
if [ $OVERALL_FAILED -ne 0 ]; then
    exit 2
elif [ $AUTOFIXES_APPLIED -ne 0 ] && [ "$FAIL_ON_AUTOFIX" = true ]; then
    echo "" >&2
    echo "⚠️  Auto-fixes were applied. Please review and re-commit." >&2
    exit 2
else
    exit 0
fi
