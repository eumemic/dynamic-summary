#!/bin/bash
# Claude Code hook script for running relevant tests after edits
# This script is called by the post_edit hook

# Check if CLAUDE_EDITED_FILES environment variable is set
if [ -z "$CLAUDE_EDITED_FILES" ]; then
    echo "No edited files detected"
    exit 0
fi

# Parse the edited files (they come as a comma-separated list)
IFS=',' read -ra EDITED_FILES <<< "$CLAUDE_EDITED_FILES"

# Track which tests to run
declare -A tests_to_run

# Analyze each edited file
for file in "${EDITED_FILES[@]}"; do
    # Trim whitespace
    file=$(echo "$file" | xargs)
    
    case "$file" in
        */splitter.py)
            tests_to_run["tests/test_splitter.py"]=1
            echo "📝 Changed: splitter.py → Will test: test_splitter.py"
            ;;
        */store.py)
            tests_to_run["tests/test_store.py"]=1
            echo "📝 Changed: store.py → Will test: test_store.py"
            ;;
        */index.py|*/retrieve.py|*/assemble.py)
            tests_to_run["tests/test_integration.py"]=1
            echo "📝 Changed: $(basename "$file") → Will test: test_integration.py"
            ;;
        */api.py)
            tests_to_run["tests/test_concurrency.py"]=1
            echo "📝 Changed: api.py → Will test: test_concurrency.py"
            ;;
        */config.py)
            # Config changes affect everything
            echo "📝 Changed: config.py → Will run ALL tests"
            pytest tests/ --tb=short -q
            exit $?
            ;;
        */progress.py|*/cli.py|*/utils.py)
            echo "⚠️  Changed: $(basename "$file") → No specific tests available yet"
            ;;
        */test_*.py)
            # If a test file was edited, run it
            tests_to_run["$file"]=1
            echo "📝 Changed test file: $(basename "$file")"
            ;;
    esac
done

# If no tests were selected, exit
if [ ${#tests_to_run[@]} -eq 0 ]; then
    echo "✓ No tests needed for these changes"
    exit 0
fi

# Run the selected tests
echo ""
echo "🧪 Running ${#tests_to_run[@]} test file(s)..."
echo ""

# Convert associative array keys to space-separated string
test_files="${!tests_to_run[*]}"

# Run pytest with concise output
pytest $test_files --tb=short -q

# Capture result
result=$?

if [ $result -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "❌ Tests failed! Run 'pytest $test_files -v' for details"
fi

exit $result