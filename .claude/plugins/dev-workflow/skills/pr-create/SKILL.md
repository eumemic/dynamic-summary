---
name: pr-create
description: This skill should be used when the user asks to "create a PR", "open a PR", "create a pull request", "open a pull request", or mentions creating/opening a PR for their changes.
---

# Create Pull Request

Create a pull request from the current branch.

## Prerequisites

Before creating a PR:
1. Commits should exist on the branch
2. Branch should be pushed to remote (use `dev-workflow:push` if not)

Check current state:
```bash
git branch --show-current
git log origin/master..HEAD --oneline  # Commits to include
gh pr list --head $(git branch --show-current) --state open  # Existing PR?
```

If a PR already exists, inform the user and offer to open it in browser.

## Process

### 1. Gather Context

Understand what the PR contains:
```bash
# All commits in this branch
git log $(git merge-base HEAD origin/master)..HEAD --oneline

# Full diff for summary
git diff $(git merge-base HEAD origin/master)..HEAD --stat
```

### 2. Identify Related Issues

Look for issue references in:
- Commit messages
- Branch name (e.g., `fix-123-login-bug`)
- Recent conversation context

Use `Fixes #123` or `Closes #123` syntax to auto-close issues on merge.

### 3. Create PR

```bash
gh pr create --title "Title here" --body "$(cat <<'EOF'
## Summary
- Key change 1
- Key change 2

## Test plan
- [ ] Test case 1
- [ ] Test case 2

Fixes #123

🤖 Generated with Claude
EOF
)"
```

**Title guidelines:**
- Start with type if appropriate: `feat:`, `fix:`, `refactor:`
- Be specific about what changed
- Keep under 72 characters

**Body guidelines:**
- Summary: 1-3 bullet points explaining the "what" and "why"
- Test plan: Concrete steps to verify the changes work
- Issue references: Link related issues
- Keep it concise - reviewers can read the code

### 4. Report Success

After creation, output:
```
✅ PR created: https://github.com/owner/repo/pull/N
```

## What This Skill Does NOT Do

- **Push**: Assumes branch is already pushed. Use `dev-workflow:push` first if needed.
- **Monitor CI**: Use `dev-workflow:pr-monitor` to watch build status
- **Request reviews**: Use `dev-workflow:pr-review` when ready for review

## Related Skills

- Before creating: "Use `dev-workflow:push` to push commits to remote"
- After creating: "Use `dev-workflow:pr-monitor` to watch CI status"

## Examples

**Create PR for feature:**
```
User: "create a PR for this feature"
→ Gather commits, create PR with summary and test plan
```

**PR already exists:**
```
User: "open a PR"
→ Check, find existing PR #42, inform user and offer to open in browser
```

**Branch not pushed:**
```
User: "create PR"
→ Notice branch not on remote, suggest using dev-workflow:push first
```
