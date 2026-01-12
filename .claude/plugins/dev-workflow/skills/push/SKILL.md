---
name: push
description: This skill should be used when the user asks to "push", "push changes", "push to remote", "push commits", or mentions pushing code to the remote repository. Handles pushing and updating PR title/description if needed.
---

# Git Push

Push commits to remote and update PR metadata if a PR exists.

## Safety Check

Before pushing, verify not on master:
```bash
git branch --show-current
```
If on master, stop and ask the user about their intended workflow.

## Process

### 1. Push to Remote

```bash
git push -u origin $(git branch --show-current)
```

If the push fails due to divergent branches, report the issue - never force push without explicit user permission.

### 2. Check for Existing PR

```bash
gh pr list --head $(git branch --show-current) --state open --json number,title,body -q '.[0]'
```

If no PR exists, inform the user they can create one with `dev-workflow:pr-create`.

### 3. Evaluate PR Updates (If PR Exists)

Compare the current PR title/description against what was just pushed:

**Check what's changed since PR creation:**
```bash
# Get commits in this PR
git log $(git merge-base HEAD origin/master)..HEAD --oneline

# Get the PR's current state
gh pr view --json title,body
```

**Update PR title if:**
- Original scope has grown significantly (e.g., "Fix login bug" → now includes session management)
- The title no longer accurately describes the changes

**Update PR description if:**
- New features were added that aren't mentioned
- Significant implementation details changed
- The test plan needs updating

**Skip updates if:**
- Just fixing tests or CI
- Addressing review feedback
- Minor tweaks within original scope

### 4. Update PR (If Needed)

```bash
# Update title
gh pr edit --title "New title here"

# Update body (use HEREDOC for multi-line)
gh pr edit --body "$(cat <<'EOF'
## Summary
- Updated summary here

## Test plan
- [ ] Updated test plan

🤖 Generated with Claude
EOF
)"
```

## What This Skill Does NOT Do

- **Commit**: Use `dev-workflow:commit` first to create commits
- **Create PR**: Use `dev-workflow:pr-create` if no PR exists
- **Monitor CI**: Use `dev-workflow:pr-monitor` to watch CI status

## Related Skills

- If no commits to push: "Use `dev-workflow:commit` to create commits first"
- If no PR exists after pushing: "Use `dev-workflow:pr-create` to open a pull request"

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
