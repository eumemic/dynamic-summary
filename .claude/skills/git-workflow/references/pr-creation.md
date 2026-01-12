# PR Create Reference

## Prerequisites

Before creating a PR:
1. Commits should exist on the branch
2. Branch should be pushed to remote

**Check current state:**
```bash
git branch --show-current
git log origin/master..HEAD --oneline  # Commits to include
gh pr list --head $(git branch --show-current) --state open  # Existing PR?
```

If PR already exists, inform user and offer to open in browser.

## Gather Context

Understand what the PR contains:
```bash
# All commits in this branch
git log $(git merge-base HEAD origin/master)..HEAD --oneline

# Full diff for summary
git diff $(git merge-base HEAD origin/master)..HEAD --stat
```

## Issue References

Look for issue references in:
- Commit messages
- Branch name (e.g., `fix-123-login-bug`)
- Recent conversation context

Use `Fixes #123` or `Closes #123` syntax to auto-close issues on merge.

## Title Guidelines

- Start with type if appropriate: `feat:`, `fix:`, `refactor:`
- Be specific about what changed
- Keep under 72 characters

**Good titles:**
- `feat: add user authentication with JWT`
- `fix: resolve race condition in session handler`
- `refactor: extract validation logic to separate module`

**Bad titles:**
- `Update code` (too vague)
- `Fix bug` (which bug?)
- `WIP` (not descriptive)

## Body Guidelines

```markdown
## Summary
- Key change 1 (the "what" and "why")
- Key change 2

## Test plan
- [ ] Concrete step to verify change 1
- [ ] Concrete step to verify change 2

Fixes #123

Generated with Claude
```

**Summary section:**
- 1-3 bullet points
- Focus on "what" and "why"
- Keep it concise - reviewers can read the code

**Test plan section:**
- Concrete, actionable steps
- Checkboxes for reviewer to follow
- Include edge cases if relevant

## Create Command

```bash
gh pr create --title "Title here" --body "$(cat <<'EOF'
## Summary
- Key change 1
- Key change 2

## Test plan
- [ ] Test case 1
- [ ] Test case 2

Fixes #123

Generated with Claude
EOF
)"
```

## Report Success

After creation:
```
PR created: https://github.com/owner/repo/pull/N
```

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
→ Notice branch not on remote, push first then create PR
```
