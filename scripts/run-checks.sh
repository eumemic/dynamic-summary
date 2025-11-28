#!/bin/bash
# Common script for running Python quality checks
# Used by both git hooks and Claude hooks
#
# Usage: 
#   run-checks.sh [OPTIONS] [file_or_directory ...]
#
# Options:
#   --skip CHECKS              Skip specific checks (comma-separated): tests,dmypy,ruff,black,jscpd,bandit
#   --fail-fast                Stop at first failure (useful for debugging)
#   --include-integration-tests  Include integration tests (benchmarks still excluded)
#   --impacted-only FILES...   Run only tests downstream of the provided files (required)
#   --ignore-lint-rules RULES   Ignore specific lint rules (comma-separated): F401,E402,etc.
#   --fail-on-autofix          Exit with failure if any auto-fixes were applied
#   --per-test-timeout SECONDS Run tests with a hard per-test timeout by enumerating tests
#   --help                     Show this help message
#
# Exit codes:
#   0 - All checks passed
#   2 - One or more checks failed (Claude-compatible)

set -uo pipefail  # Don't use -e, we handle errors explicitly

# Parse command line arguments
SKIP_CHECKS=""
TARGETS=""
FAIL_FAST=false
INCLUDE_INTEGRATION=false
IGNORE_LINT_RULES=""
FAIL_ON_AUTOFIX=false
TEST_SCOPE="fast"  # deprecated; kept for backward-compatibility
IMPACTED_ONLY=false
IMPACTED_FILES=()
PER_TEST_TIMEOUT=""

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
        --include-integration-tests)
            INCLUDE_INTEGRATION=true
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
        --per-test-timeout)
            PER_TEST_TIMEOUT="$2"
            shift 2
            ;;
        --impacted-only)
            IMPACTED_ONLY=true
            shift
            # Collect file arguments until next option or end
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                IMPACTED_FILES+=("$1")
                shift
            done
            ;;
        --test-scope)
            # Deprecated: use --include-integration-tests or default fast path
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

# Validate impacted-only usage
if [ "$IMPACTED_ONLY" = true ] && [ ${#IMPACTED_FILES[@]} -eq 0 ]; then
    echo "Error: --impacted-only requires at least one FILE argument" >&2
    echo "Usage: $0 --impacted-only FILE [FILE ...]" >&2
    exit 1
fi

# Note: --include-integration-tests with --skip tests is a no-op for tests; keep running other checks
if [ "$INCLUDE_INTEGRATION" = true ] && [[ "$SKIP_CHECKS" == *"tests"* ]]; then
    echo "[Tests] Skipped: --include-integration-tests has no effect when tests are skipped" >&2
fi

# Get repository root (works in main repo and worktrees)
GIT_ROOT="$(git rev-parse --show-toplevel)"

# Optional guard: fail if legacy mock store is referenced in tests
# Enable by setting RZ_GUARD_NO_MOCKS=1 in the environment (disabled by default)
if [[ "${RZ_GUARD_NO_MOCKS:-}" = "1" ]]; then
    if command -v rg &> /dev/null; then
        if rg -n "from tests\\.mock_store import|SimpleMockStore\\b" tests >/tmp/mock_guard_hits 2>/dev/null; then
            echo "[MockGuard] ❌ SimpleMockStore references detected in tests:" >&2
            cat /tmp/mock_guard_hits >&2 || true
            exit 2
        else
            echo "[MockGuard] ✅ No SimpleMockStore references detected"
        fi
    else
        if grep -REn "from tests\\.mock_store import|SimpleMockStore\\b" tests >/tmp/mock_guard_hits 2>/dev/null; then
            echo "[MockGuard] ❌ SimpleMockStore references detected in tests:" >&2
            cat /tmp/mock_guard_hits >&2 || true
            exit 2
        else
            echo "[MockGuard] ✅ No SimpleMockStore references detected"
        fi
    fi
fi

# Backend-agnostic migration guards - prevent regressions
if command -v rg &> /dev/null; then
    # Guard against SQLiteStorageBackend imports in non-sqlite files (except conftest.py)
    if rg -l "SQLiteStorageBackend" tests --type py | grep -v sqlite | grep -v conftest.py >/tmp/sqlite_guard_hits 2>/dev/null; then
        echo "❌ Found SQLiteStorageBackend imports in non-*_sqlite*.py files:" >&2
        cat /tmp/sqlite_guard_hits >&2 || true
        exit 2
    fi
    
    # Guard against SessionLocal usage in tests
    if rg -l "SessionLocal\(" tests --type py >/tmp/session_guard_hits 2>/dev/null; then
        echo "❌ Found SessionLocal() usage in test files:" >&2
        cat /tmp/session_guard_hits >&2 || true
        exit 2
    fi
    
    # Guard against LocalStoreAdapter references
    if rg -l "LocalStoreAdapter" . --type py >/tmp/local_store_guard_hits 2>/dev/null; then
        echo "❌ Found LocalStoreAdapter references (should be fully migrated):" >&2
        cat /tmp/local_store_guard_hits >&2 || true
        exit 2
    fi
    
    echo "✅ Backend-agnostic migration guards passed"
else
    # Fallback to find/grep if rg not available
    if find tests -name "*.py" -not -name "*sqlite*" -not -name "conftest.py" -exec grep -l "SQLiteStorageBackend" {} \; 2>/dev/null | head -1 | grep -q .; then
        echo "❌ Found SQLiteStorageBackend imports in non-*_sqlite*.py files:" >&2
        find tests -name "*.py" -not -name "*sqlite*" -not -name "conftest.py" -exec grep -l "SQLiteStorageBackend" {} \; 2>/dev/null >&2
        exit 2
    fi
    
    if find tests -name "*.py" -exec grep -l "SessionLocal(" {} \; 2>/dev/null | head -1 | grep -q .; then
        echo "❌ Found SessionLocal() usage in test files:" >&2
        find tests -name "*.py" -exec grep -l "SessionLocal(" {} \; 2>/dev/null >&2
        exit 2
    fi
    
    if find . -name "*.py" -exec grep -l "LocalStoreAdapter" {} \; 2>/dev/null | head -1 | grep -q .; then
        echo "❌ Found LocalStoreAdapter references (should be fully migrated):" >&2
        find . -name "*.py" -exec grep -l "LocalStoreAdapter" {} \; 2>/dev/null >&2
        exit 2
    fi
    
    echo "✅ Backend-agnostic migration guards passed"
fi

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

# Default targets if none specified (always lint/typecheck code and tests)
if [[ -z "$TARGETS" ]]; then
    # Always lint library, tests, and developer scripts/experiments.
    # Mypy uses its own explicit targets (ragzoom tests), so this only widens
    # Ruff/Black scope without changing type-check coverage.
    TARGETS="ragzoom tests scripts prompt-experiments"
fi

# Track overall success and auto-fixes
OVERALL_FAILED=0
AUTOFIXES_APPLIED=0

# Create temporary directory for storing results
tmpdir=$(mktemp -d)

# Store process IDs for parallel execution
declare -a pids=()
ANY_PIDS=0

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
    
    # Choose a portable time command
    local TIME_BIN="/usr/bin/time"
    if command -v gtime >/dev/null 2>&1; then
        TIME_BIN="gtime"
    fi

    (
        # Handle SIGTERM gracefully in subshell
        trap 'exit 143' TERM
        
        echo "[$check_name] Starting..." > "$output_file"
        # Measure wall time for the check; append timing to output
        if $TIME_BIN -p bash -lc "$check_cmd" >> "$output_file" 2>&1; then
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
    ANY_PIDS=1
}

# Start non-test checks in parallel; tests will run after type checking passes

# Guard: workflows should not use unpinned pip installs
if [ -x scripts/check-workflow-installs.sh ]; then
    run_check_background "WorkflowPins" "bash scripts/check-workflow-installs.sh"
fi

# dmypy
if ! should_skip "dmypy"; then
    if command -v dmypy &> /dev/null; then
        # Use existing daemon for speed; do not stop/restart here.
        # Always typecheck both library and tests regardless of --skip tests.
        run_check_background "Mypy" "dmypy run -- ragzoom tests --no-error-summary --check-untyped-defs"
    else
        echo "[Mypy] ❌ dmypy not installed - cannot run type checks" >&2
        exit 2
    fi
fi

# ruff
if ! should_skip "ruff"; then
    if command -v ruff &> /dev/null; then
        # Run on full targets for reliability
        # Enhanced ruff command that detects auto-fixes
        ruff_cmd="(
            # Capture initial state
            before_hash=\$(find $TARGETS -name '*.py' -print0 2>/dev/null | xargs -0 md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
            
            # Run ruff with auto-fix
            if ruff check $TARGETS --fix --output-format concise"
            if [ -n "$IGNORE_LINT_RULES" ]; then
                ruff_cmd="$ruff_cmd --ignore $IGNORE_LINT_RULES"
            fi
            ruff_cmd="$ruff_cmd; then
                    # Check if files were modified
                    after_hash=\$(find $TARGETS -name '*.py' -print0 2>/dev/null | xargs -0 md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
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
                    after_hash=\$(find $TARGETS -name '*.py' -print0 2>/dev/null | xargs -0 md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
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
        # Run on full targets for reliability
        # Enhanced black command that detects formatting changes
        black_cmd="(
            # Capture initial state
            before_hash=\$(find $TARGETS -name '*.py' -print0 2>/dev/null | xargs -0 md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
            
            # Run black (it auto-formats by default)
            if black $TARGETS --quiet; then
                # Check if files were modified
                after_hash=\$(find $TARGETS -name '*.py' -print0 2>/dev/null | xargs -0 md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
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
    # Prefer global jscpd, then local node_modules, then npx fallback
    if command -v jscpd &> /dev/null; then
        JSCPD_BIN="$(command -v jscpd)"
    elif [ -x "$GIT_ROOT/node_modules/.bin/jscpd" ]; then
        JSCPD_BIN="$GIT_ROOT/node_modules/.bin/jscpd"
    elif command -v npx &> /dev/null; then
        JSCPD_BIN="npx jscpd@latest"
        echo "[JSCPD] Using npx fallback (consider: npm install -g jscpd)"
    else
        JSCPD_BIN=""
    fi

    if [ -n "$JSCPD_BIN" ]; then
        if [ "$IMPACTED_ONLY" = true ]; then
            # Limit jscpd scan to impacted source files under ragzoom/
            impacted_src=()
            for f in "${IMPACTED_FILES[@]}"; do
                case "$f" in
                    *.py)
                        if [[ "$f" == ragzoom/* || "$f" == */ragzoom/* ]]; then
                            impacted_src+=("$f")
                        fi
                        ;;
                esac
            done
            if [ ${#impacted_src[@]} -gt 0 ]; then
                jscpd_targets="${impacted_src[*]}"
                run_check_background "JSCPD" "$JSCPD_BIN $jscpd_targets --config $GIT_ROOT/.jscpd.json"
            else
                echo "[JSCPD] Skipped (no impacted source files)"
            fi
        else
            run_check_background "JSCPD" "$JSCPD_BIN ragzoom/ --config $GIT_ROOT/.jscpd.json"
        fi
    else
        echo "[JSCPD] Skipped (jscpd not available)"
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

######## Phase 1: wait for non-test checks ########
if [ "$ANY_PIDS" -eq 1 ]; then
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
fi

# Determine if Mypy passed; if not, skip tests
MYPY_RESULT_FILE="$tmpdir/Mypy.result"
RUN_TESTS=true
if should_skip "tests"; then
    RUN_TESTS=false
elif [ -f "$MYPY_RESULT_FILE" ] && [ "$(cat "$MYPY_RESULT_FILE")" != "0" ]; then
    echo "[Tests] Skipped (type checking failed)" > "$tmpdir/Tests.output"
    echo 1 > "$tmpdir/Tests.result"
    RUN_TESTS=false
fi

######## Phase 2: run tests (after Mypy passes) ########
if [ "$RUN_TESTS" = true ]; then
    if command -v pytest &> /dev/null; then
        # Marker expression: always exclude benchmarks; include integration only when requested
        if [ "$INCLUDE_INTEGRATION" = true ] || [ "$TEST_SCOPE" = "all" ]; then
            marker_expr="not benchmark"
        else
            marker_expr="not benchmark and not integration"
        fi

        # Paths that always require integration tests when impacted-only is used
        integration_test_paths=(
            "$GIT_ROOT/tests/test_integration.py"
            "$GIT_ROOT/tests/test_concurrency.py"
        )

        # Ensure PostgreSQL only if integration tests are requested and backend is postgres
        if [ "$INCLUDE_INTEGRATION" = true ] || [ "$TEST_SCOPE" = "all" ]; then
            BACKEND="${RAGZOOM_BACKEND:-sqlite}"
            DB_URL="${RAGZOOM_DATABASE_URL:-}"
            if [ "$BACKEND" = "postgres" ] || [[ "$DB_URL" =~ ^postgres ]]; then
                if [ -z "$DB_URL" ]; then
                    echo "[PostgreSQL] Ensuring PostgreSQL is running for integration tests..."
                    python -c "from ragzoom.docker_postgres import DockerPostgres; dp = DockerPostgres(); dp.ensure_running()" 2>/dev/null || {
                        echo "[PostgreSQL] Warning: Could not start PostgreSQL, integration tests may fail"
                    }
                else
                    echo "[PostgreSQL] Using provided database URL: $DB_URL"
                fi
            else
                echo "[PostgreSQL] Skipping: backend is '$BACKEND' (using SQLite)"
            fi
        fi

        # Ensure across-the-board 1s per-test timeout unless explicitly overridden via env
        export RZ_MAX_TEST_DURATION="${RZ_MAX_TEST_DURATION:-1.0}"

        dur_flag=""
        if [ -n "${PYTEST_DURATIONS:-}" ]; then
            dur_flag="--durations=$PYTEST_DURATIONS"
        fi

        if [ -n "$PER_TEST_TIMEOUT" ]; then
            runner_cmd="python $GIT_ROOT/scripts/run_tests_with_timeouts.py --per-test-seconds $PER_TEST_TIMEOUT"
            if [ "$INCLUDE_INTEGRATION" = true ] || [ "$TEST_SCOPE" = "all" ]; then
                runner_cmd="$runner_cmd --include-integration"
            fi
            run_check_background "Tests" "$runner_cmd"
        elif [ "$IMPACTED_ONLY" = true ]; then
            impacted="$(python "$GIT_ROOT/scripts/find-impacted-tests.py" ${IMPACTED_FILES[@]} || true)"
            if [ -n "$impacted" ]; then
                impacted_marker="$marker_expr"
                if [ "$INCLUDE_INTEGRATION" != true ]; then
                    for path in "${integration_test_paths[@]}"; do
                        if [[ "$impacted" == *"$path"* ]]; then
                            impacted_marker="not benchmark"
                            break
                        fi
                    done
                fi
                # Wrap pytest to treat exit code 5 (no tests collected) as success
                # This happens when impacted files have no matching tests after marker filtering
                run_check_background "Tests" "pytest $impacted -q --tb=short -m '$impacted_marker' -n \${PYTEST_XDIST_WORKERS:-8} --dist=worksteal --no-header --max-test-duration \${RZ_MAX_TEST_DURATION} \${dur_flag}; ret=\$?; if [ \$ret -eq 5 ]; then echo '[Tests] ✅ Passed (no matching tests)'; exit 0; else exit \$ret; fi"
            else
                echo "[Tests] Skipped (no impacted tests)" > "$tmpdir/Tests.output"
                echo 0 > "$tmpdir/Tests.result"
            fi
        else
            run_check_background "Tests" "pytest tests/ -q --tb=short -m '$marker_expr' -n \${PYTEST_XDIST_WORKERS:-8} --dist=worksteal --no-header --max-test-duration \${RZ_MAX_TEST_DURATION} \${dur_flag}"
        fi
    else
        echo "[Tests] Skipped (pytest not installed)" > "$tmpdir/Tests.output"
        echo 0 > "$tmpdir/Tests.result"
    fi
fi

# Wait for tests (if any) to complete
if [ ${#pids[@]} -gt 0 ]; then
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
for check in WorkflowPins Tests Mypy Ruff Black JSCPD Bandit; do
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
