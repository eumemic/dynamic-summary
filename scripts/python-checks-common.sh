#!/bin/bash
# Common functions for Python quality checks used by both pre-commit and Claude hooks

# Function to run black
run_black() {
    local targets="$1"  # Now accepts space-separated directories
    local modified_check="$2"
    
    if command -v black &> /dev/null; then
        if [ -z "$modified_check" ] || [ -n "$modified_check" ]; then
            echo "[Black] Starting..."
            # Run with quiet flag, capture output
            local tmpfile=$(mktemp)
            # Use eval to properly expand space-separated targets
            eval "black $targets --quiet" > "$tmpfile" 2>&1
            local result=$?
            if [ $result -ne 0 ]; then
                echo "[Black] ❌ Formatting failed."
                # Show full output on failure
                cat "$tmpfile"
                rm -f "$tmpfile"
                return 1
            else
                echo "[Black] ✅ Passed!"
                rm -f "$tmpfile"
                return 0
            fi
        else
            echo "[Black] Skipped (no Python files)"
            return 0
        fi
    else
        echo "[Black] Skipped (not installed)"
        return 0
    fi
}

# Function to run ruff
run_ruff() {
    local targets="$1"  # Now accepts space-separated directories
    local modified_check="$2"
    
    if command -v ruff &> /dev/null; then
        if [ -z "$modified_check" ] || [ -n "$modified_check" ]; then
            echo "[Ruff] Starting..."
            # Run with quiet flag, capture output
            local tmpfile=$(mktemp)
            # Use eval to properly expand space-separated targets
            eval "ruff check $targets --fix --quiet" > "$tmpfile" 2>&1
            local result=$?
            if [ $result -ne 0 ]; then
                echo "[Ruff] ❌ Found issues that could not be auto-fixed."
                # Show full output on failure
                cat "$tmpfile"
                rm -f "$tmpfile"
                return 1
            else
                echo "[Ruff] ✅ Passed!"
                rm -f "$tmpfile"
                return 0
            fi
        else
            echo "[Ruff] Skipped (no Python files)"
            return 0
        fi
    else
        echo "[Ruff] Skipped (not installed)"
        return 0
    fi
}

# Function to run mypy
run_mypy() {
    local targets="$1"  # Now accepts space-separated directories
    local modified_check="$2"
    
    if command -v dmypy &> /dev/null || command -v mypy &> /dev/null; then
        if [ -z "$modified_check" ] || [ -n "$modified_check" ]; then
            echo "[Mypy] Starting..."
            # Only check ragzoom directory - tests don't need strict type checking
            if command -v dmypy &> /dev/null; then
                # Use dmypy run which starts daemon if needed and runs check
                dmypy run -- ragzoom --ignore-missing-imports --no-error-summary --check-untyped-defs
                local result=$?
            else
                # Fallback to regular mypy if dmypy not available
                mypy ragzoom --ignore-missing-imports --no-error-summary --check-untyped-defs
                local result=$?
            fi
            if [ $result -ne 0 ]; then
                echo "[Mypy] ❌ Failed! Type errors found"
                return $result
            else
                echo "[Mypy] ✅ Passed!"
                return 0
            fi
        else
            echo "[Mypy] Skipped (no Python files)"
            return 0
        fi
    else
        echo "[Mypy] Skipped (not installed)"
        return 0
    fi
}

# Function to run Python checks in parallel
# Args: $1 = targets (space-separated files or directories), $2 = modified files check (optional), $3 = exit_on_failure (default true)
run_python_checks_parallel() {
    local targets="$1"  # Now accepts space-separated directories
    local modified_check="$2"
    local exit_on_failure="${3:-true}"
    
    # Create temporary directory for storing results
    local tmpdir=$(mktemp -d)
    
    # Set up trap to clean up temp files and kill any background processes on exit
    cleanup() {
        # Kill any background processes that might still be running
        kill $pid_mypy $pid_ruff $pid_black 2>/dev/null || true
        # Clean up temp directory
        rm -rf "$tmpdir"
    }
    trap cleanup EXIT INT TERM
    
    # Run tools in parallel, capturing output - pass targets without quotes to allow expansion
    (run_mypy "$targets" "$modified_check" > "$tmpdir/mypy.output" 2>&1; echo $? > "$tmpdir/mypy.result") &
    pid_mypy=$!
    
    (run_ruff "$targets" "$modified_check" > "$tmpdir/ruff.output" 2>&1; echo $? > "$tmpdir/ruff.result") &
    pid_ruff=$!
    
    (run_black "$targets" "$modified_check" > "$tmpdir/black.output" 2>&1; echo $? > "$tmpdir/black.result") &
    pid_black=$!
    
    # Wait for all background jobs to complete
    wait $pid_mypy $pid_ruff $pid_black
    
    # Display output in predictable order: mypy, ruff, black
    cat "$tmpdir/mypy.output" 2>/dev/null
    cat "$tmpdir/ruff.output" 2>/dev/null
    cat "$tmpdir/black.output" 2>/dev/null
    
    # Check results
    local failed=0
    
    # Check mypy results
    if [ -f "$tmpdir/mypy.result" ] && [ "$(cat "$tmpdir/mypy.result")" -ne 0 ]; then
        failed=1
    fi
    
    # Check ruff results
    if [ -f "$tmpdir/ruff.result" ] && [ "$(cat "$tmpdir/ruff.result")" -ne 0 ]; then
        failed=1
    fi
    
    # Check black results
    if [ -f "$tmpdir/black.result" ] && [ "$(cat "$tmpdir/black.result")" -ne 0 ]; then
        failed=1
    fi
    
    # Clean up trap
    trap - EXIT INT TERM
    cleanup
    
    # Return based on exit_on_failure setting
    if [ "$exit_on_failure" = "true" ]; then
        return $failed
    else
        return 0  # Always succeed for Claude hook
    fi
}