#!/bin/bash
set -euo pipefail

# Generate clean commit message for squash merge
# Usage: ./scripts/generate-merge-message.sh [PR_NUMBER]

# Get PR number from argument or detect from current branch
if [ $# -eq 1 ]; then
    PR_NUMBER="$1"
    PR_TITLE=$(gh pr view "$PR_NUMBER" --json title -q .title)
    PR_BODY=$(gh pr view "$PR_NUMBER" --json body -q .body)
else
    # Auto-detect from current branch using worktree workflow
    CURRENT_BRANCH=$(git branch --show-current)
    PR_DATA=$(gh pr list --head "$CURRENT_BRANCH" --state open --json title,body,number)
    
    if [ "$(echo "$PR_DATA" | jq length)" -eq 0 ]; then
        echo "Error: No open PR found for branch $CURRENT_BRANCH" >&2
        exit 1
    fi
    
    PR_TITLE=$(echo "$PR_DATA" | jq -r '.[0].title')
    PR_NUMBER=$(echo "$PR_DATA" | jq -r '.[0].number')
    PR_BODY=$(echo "$PR_DATA" | jq -r '.[0].body')
fi

# Construct clean commit message
cat <<EOF
# ${PR_TITLE} (#${PR_NUMBER})

${PR_BODY}
EOF