# Push Reference

## Safety Check

Before pushing:
```bash
git branch --show-current
```
If on master, stop and ask about intended workflow.

## Push Command

```bash
git push -u origin $(git branch --show-current)
```

If push fails due to divergent branches, report the issue - never force push without explicit permission.

## Check for Existing PR

```bash
gh pr list --head $(git branch --show-current) --state open --json number,title,body -q '.[0]'
```

If no PR exists, inform user they can create one with the pr-create operation.

## PR Update Criteria

Compare current PR title/description against what was pushed.

**Check what's changed:**
```bash
# Commits in this PR
git log $(git merge-base HEAD origin/master)..HEAD --oneline

# PR's current state
gh pr view --json title,body
```

**Update PR title if:**
- Original scope has grown significantly (e.g., "Fix login bug" → now includes session management)
- Title no longer accurately describes the changes

**Update PR description if:**
- New features added that aren't mentioned
- Significant implementation details changed
- Test plan needs updating

**Skip updates if:**
- Just fixing tests or CI
- Addressing review feedback
- Minor tweaks within original scope

## Update Commands

```bash
# Update title
gh pr edit --title "New title here"

# Update body (HEREDOC for multi-line)
gh pr edit --body "$(cat <<'EOF'
## Summary
- Updated summary here

## Test plan
- [ ] Updated test plan

Generated with Claude
EOF
)"
```

## Examples

**Simple push, no PR:**
```
User: "push these commits"
→ Push to remote, note that no PR exists yet
```

**Push with PR update needed:**
```
User: "push"
→ Push, check PR, notice new feature added, update PR title and description
```

**Push after review fixes:**
```
User: "push the review fixes"
→ Push, check PR, determine changes are within scope, skip PR update
```
