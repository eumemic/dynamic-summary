#!/bin/bash
set -euo pipefail

# Rewrite git history with clean PR commit messages
# Usage: ./scripts/rewrite-commit-history.sh [--execute] [--show-all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=true
SHOW_ALL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --execute)
            DRY_RUN=false
            shift
            ;;
        --show-all)
            SHOW_ALL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--execute] [--show-all]"
            echo ""
            echo "Rewrite git history to use clean PR commit messages instead of"
            echo "GitHub's verbose concatenated squash merge messages."
            echo ""
            echo "Options:"
            echo "  --execute    Actually perform the rewrite (default: dry-run only)"
            echo "  --show-all   Show all commits in dry-run (default: first 3 only)"
            echo "  --help       Show this help message"
            echo ""
            echo "By default, runs in dry-run mode showing first 3 proposed changes."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Extract PR number from commit message (last parenthesized number)
extract_pr_number() {
    local commit_msg="$1"
    echo "$commit_msg" | grep -oE '\(#[0-9]+\)' | tail -1 | grep -oE '[0-9]+' || echo ""
}

# Generate new commit message with preserved subject and PR body
generate_new_message() {
    local pr_number="$1"
    local commit_sha="$2"
    
    # Keep first line (subject) which already has title + PR number
    local subject=$(git log --format="%s" -1 "$commit_sha")
    
    # Get PR body from GitHub
    local pr_body
    if ! pr_body=$(gh pr view "$pr_number" --json body -q .body 2>/dev/null); then
        echo "Failed to fetch PR #$pr_number body (commit $commit_sha)" >&2
        return 1
    fi
    
    # Combine subject with PR body
    echo "${subject}

${pr_body}"
}

# Find all commits with PR numbers
echo "🔍 Scanning git history for PR commits..."
commits=()
while IFS= read -r line; do
    commits+=("$line")
done < <(git log --format="%H %s" | grep '(#[0-9]')

if [ ${#commits[@]} -eq 0 ]; then
    echo "No commits with PR numbers found."
    exit 0
fi

echo "Found ${#commits[@]} commits with PR references"
echo ""

# Process each commit
declare -a changes=()
declare -a failed_prs=()

for commit_line in "${commits[@]}"; do
    commit_sha=$(echo "$commit_line" | cut -d' ' -f1)
    commit_msg=$(echo "$commit_line" | cut -d' ' -f2-)
    
    pr_number=$(extract_pr_number "$commit_msg")
    
    if [ -z "$pr_number" ]; then
        echo "⚠️  Skipping $commit_sha: Could not extract PR number from '$commit_msg'"
        continue
    fi
    
    echo -n "📝 Processing commit $commit_sha (PR #$pr_number)... "
    
    if generate_new_message "$pr_number" "$commit_sha" >/dev/null 2>&1; then
        changes+=("$commit_sha|$pr_number")
        echo "✅"
    else
        echo "⚠️  (skipping - likely issue reference)"
        failed_prs+=("PR #$pr_number (commit $commit_sha) - skipped")
    fi
done

echo ""

if [ ${#failed_prs[@]} -gt 0 ]; then
    echo "⚠️  Skipped ${#failed_prs[@]} commits (likely issue references, not PRs):"
    printf '   %s\n' "${failed_prs[@]}"
    echo ""
fi

if [ ${#changes[@]} -eq 0 ]; then
    echo "No commits can be processed. Exiting."
    exit 1
fi

echo "📋 Summary: ${#changes[@]} commits will be updated"
echo ""

if [ "$DRY_RUN" = true ]; then
    if [ "$SHOW_ALL" = true ]; then
        echo "🔍 DRY RUN - Showing all proposed changes:"
    else
        echo "🔍 DRY RUN - Showing proposed changes (first 3 commits):"
    fi
    echo "========================================"
    echo ""
    
    count=0
    for change in "${changes[@]}"; do
        if [ "$SHOW_ALL" = false ] && [ $count -ge 3 ]; then
            echo "... and $((${#changes[@]} - 3)) more commits"
            echo ""
            echo "To see all changes, run: $0 --show-all"
            break
        fi
        IFS='|' read -r commit_sha pr_number <<< "$change"
        
        current_message=$(git log --format="%B" -1 "$commit_sha")
        new_message=$(generate_new_message "$pr_number" "$commit_sha")
        
        echo "Commit: $commit_sha (PR #$pr_number)"
        echo "Current message:"
        echo "$current_message" | sed 's/^/  /'
        echo ""
        echo "New message:"
        echo "$new_message" | sed 's/^/  /'
        echo ""
        echo "----------------------------------------"
        echo ""
        
        ((count++))
    done
    
    echo "To execute these changes, run:"
    echo "  $0 --execute"
    echo ""
    echo "⚠️  WARNING: This will rewrite git history and break GitHub PR references!"
    
else
    echo "⚠️  DANGER ZONE: About to rewrite git history!"
    echo ""
    echo "This will:"
    echo "  • Rewrite ${#changes[@]} commits with new messages"
    echo "  • Change all commit SHAs from the first modified commit onward"
    echo "  • Break GitHub PR page references to these commits"
    echo "  • Require force-push to update remote branches"
    echo ""
    echo "Benefits:"
    echo "  • Clean, scannable git history"
    echo "  • Consistent commit message format"
    echo "  • Better context for future code archaeology"
    echo ""
    
    read -p "Are you absolutely sure you want to proceed? (type 'YES' to confirm): " confirmation
    
    if [ "$confirmation" != "YES" ]; then
        echo "Aborted."
        exit 1
    fi
    
    echo ""
    echo "🔄 Creating backup branch..."
    backup_branch="backup-before-rewrite-$(date +%Y%m%d-%H%M%S)"
    git branch "$backup_branch"
    echo "Created backup branch: $backup_branch"
    echo ""
    
    echo "🔄 Rewriting git history..."
    
    # Create a temporary script for git filter-branch
    filter_script=$(mktemp)
    cat > "$filter_script" << 'EOF'
#!/bin/bash
commit_sha="$GIT_COMMIT"

# Check if this commit needs rewriting
EOF
    
    # Add commit rewriting logic to the filter script
    for change in "${changes[@]}"; do
        IFS='|' read -r commit_sha pr_number <<< "$change"
        new_message=$(generate_new_message "$pr_number" "$commit_sha")
        # Escape the new message for bash
        escaped_message=$(printf '%q' "$new_message")
        echo "if [ \"\$commit_sha\" = \"$commit_sha\" ]; then" >> "$filter_script"
        echo "    echo $escaped_message" >> "$filter_script"
        echo "    exit 0" >> "$filter_script"
        echo "fi" >> "$filter_script"
    done
    
    # Default case - keep original message
    echo 'cat' >> "$filter_script"
    
    chmod +x "$filter_script"
    
    # Use git filter-branch to rewrite history
    if git filter-branch --msg-filter "$filter_script" -- --all; then
        echo "✅ Git history rewritten successfully!"
        echo ""
        echo "Next steps:"
        echo "  1. Review the changes: git log --oneline -10"
        echo "  2. Force-push to remote: git push --force-with-lease origin master"
        echo "  3. Update worktree branches to sync with new master"
        echo ""
        echo "Backup branch available: $backup_branch"
    else
        echo "❌ Git history rewrite failed!"
        echo "Your repository is unchanged."
        echo "Backup branch: $backup_branch"
        exit 1
    fi
    
    # Cleanup
    rm -f "$filter_script"
fi