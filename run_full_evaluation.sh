#!/bin/bash
# Run full evaluation of Claude's duplication detection

set -e

# Parameters
MIN_LINES=${1:-6}
MIN_TOKENS=${2:-10}
NUM_TRIAGERS=${3:-4}
MAX_DUPLICATES=${4:-10}  # Limit for faster testing

echo "Running evaluation with min-lines=$MIN_LINES, min-tokens=$MIN_TOKENS, num-triagers=$NUM_TRIAGERS, max=$MAX_DUPLICATES"

# Generate jscpd report
echo "Generating duplication report..."
npx jscpd ragzoom/ --min-lines $MIN_LINES --min-tokens $MIN_TOKENS -r json --silent --threshold 100 2>/dev/null

# Optionally limit duplicates for faster testing
if [ "$MAX_DUPLICATES" != "all" ]; then
    echo "Limiting to first $MAX_DUPLICATES duplicates..."
    python3 -c "
import json
with open('jscpd-report.json', 'r') as f:
    data = json.load(f)
data['duplicates'] = data['duplicates'][:$MAX_DUPLICATES]
with open('jscpd-report.json', 'w') as f:
    json.dump(data, f)

# Also limit answer sheet
with open('answer_sheet_rated.json', 'r') as f:
    answers = json.load(f)
limited = [a for a in answers if a['id'] <= $MAX_DUPLICATES]
with open('answer_sheet_rated_temp.json', 'w') as f:
    json.dump(limited, f)
"
    mv answer_sheet_rated.json answer_sheet_rated.json.backup
    mv answer_sheet_rated_temp.json answer_sheet_rated.json
fi

# Make a backup of jscpd report for evaluation
cp jscpd-report.json jscpd-report-backup.json

# Run Claude analysis (suppress output)
echo "Running Claude analysis..."
python3 find_duplicated_code.py ragzoom/ --min-lines $MIN_LINES --min-tokens $MIN_TOKENS --num-triagers $NUM_TRIAGERS > claude_output.txt 2>&1

# Restore jscpd report for evaluation
cp jscpd-report-backup.json jscpd-report.json

# Run evaluation
echo -e "\nRunning evaluation..."
python3 evaluate_claude_ratings.py

# Clean up backup
rm -f jscpd-report-backup.json

# Restore original answer sheet if we limited it
if [ -f answer_sheet_rated.json.backup ]; then
    mv answer_sheet_rated.json.backup answer_sheet_rated.json
fi

# Show Claude's output
echo -e "\n=== CLAUDE'S OUTPUT ==="
cat claude_output.txt

# Clean up
rm -f claude_output.txt