---
name: git-workflow
description: This skill should be used when the user asks to "commit", "push", "create a PR", "open a pull request", "monitor CI", "fix CI", "fix the build", "request review", "merge", "merge the PR", or mentions any git-based development workflow operations.
---

# Git Workflow

Orchestrate git-based development workflows: commit, push, PR creation, CI monitoring, code review, and merging.

## Safety Checks (Non-Negotiable)

Before any git operation:

1. **Never operate on master**: Run `git branch --show-current`. If on master, stop and ask about switching to a feature branch.

2. **Never use --no-verify**: Pre-commit hooks are guardians. Never bypass without explicit permission.

3. **Never force push**: Report divergent branches; never force push without explicit permission.

## Workflow Overview

```
commit → push → pr-create → pr-monitor → pr-review → merge
```

Each step can be invoked independently or chained together based on user request.

## Operations

### Commit

Create clean, atomic commits with safety checks and debug code cleanup.

**Process:**
1. Review changes: `git status`, `git diff`, `git diff --cached`
2. Clean debug code (print statements, console.log, debugger) - use judgment
3. Stage logically by feature: `git add path/to/related/files`
4. Create atomic commit with conventional message format

**Message format**: `type: description` (50 chars max)
- `feat:` new feature
- `fix:` bug fix
- `refactor:` code restructuring
- `docs:` documentation
- `test:` test changes
- `chore:` maintenance

**HEREDOC for commits:**
```bash
git commit -m "$(cat <<'EOF'
feat: add user authentication

- Add login/logout endpoints
- Implement session management

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

For multiple logical changes, create separate commits.

See `references/committing.md` for detailed guidance.

### Push

Push commits to remote and update PR metadata if needed.

**Process:**
1. Push: `git push -u origin $(git branch --show-current)`
2. Check for existing PR: `gh pr list --head $(git branch --show-current) --state open`
3. If PR exists, evaluate if title/description needs updating based on new commits
4. Update PR only if scope has grown significantly

See `references/pushing.md` for PR update criteria.

### PR Create

Create a pull request from the current branch.

**Prerequisites:** Commits exist, branch pushed to remote.

**Process:**
1. Gather context: `git log $(git merge-base HEAD origin/master)..HEAD --oneline`
2. Identify related issues from commits, branch name, or conversation
3. Create PR with summary and test plan

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

See `references/pr-creation.md` for title/body guidelines.

### PR Monitor

Monitor CI status, fix failures, loop until green.

**Process:**
1. Check status: `gh pr checks --json name,state,link`
2. Poll until complete (avoid `--watch` - excessive output)
3. On failure: identify issue, fix, commit, push, resume monitoring
4. On success: assess if changes warrant independent review

**Request review if:** Multiple files with significant logic, new features, security-sensitive code, complex algorithms.

**Skip review if:** Documentation-only, trivial fixes, test-only changes.

See `references/pr-monitoring.md` for CI fix strategies.

### PR Review

Request code review and handle review dialogue.

**No preconditions** - can be invoked anytime regardless of CI status.

**Process:**
1. Gather context: `gh pr view --json number,title,body,additions,deletions,changedFiles`
2. Identify review focus areas (complex logic, security, performance)
3. Post review request with context:

```bash
gh pr comment --body "$(cat <<'EOF'
@reviewer please review this PR.

**Key changes:**
- [Summary of main changes]

**Areas of concern:**
- [Specific areas needing review]
EOF
)"
```

4. Handle feedback: categorize as critical/important/minor, discuss with user, implement fixes
5. Respond to reviewer with summary of changes

See `references/requesting-review.md` for feedback handling.

### Merge

Squash merge the PR and sync branch with master.

**Prerequisites:** Not on master, no uncommitted changes, PR exists and mergeable, CI passing.

**Process:**
```bash
./scripts/squash-merge.sh
```

The script handles validation, squash merge, and branch sync.

**On failure:**
- "Not on feature branch" → switch to correct branch
- "CI not passing" → use pr-monitor to fix
- "No open PR" → use pr-create first

See `references/merging.md` for error handling.

## Chaining Operations

Parse user requests to identify which operations to perform:

**"commit and push"** → commit → push

**"create a PR"** → (commit if needed) → (push if needed) → pr-create

**"commit, push, create PR, and merge when build passes"** → commit → push → pr-create → pr-monitor (wait for green) → merge

**"fix CI and merge"** → pr-monitor (fix failures) → merge

## Key Principles

- **Atomic commits**: Each commit is a complete, revertable change
- **Fail fast**: Exit on failure to fix immediately
- **Minimal commits**: One fix commit beats many tiny ones
- **Guide reviewers**: Provide context, not just "please review"
- **Discuss before fixing**: Check with user before implementing review feedback

## Additional Resources

### Reference Files

For detailed guidance on each operation:
- **`references/committing.md`** - Debug cleanup, staging strategies, multiple commits
- **`references/pushing.md`** - PR update criteria, divergent branch handling
- **`references/pr-creation.md`** - Title/body guidelines, issue linking
- **`references/pr-monitoring.md`** - CI polling, failure diagnosis, fix strategies
- **`references/requesting-review.md`** - Review request templates, feedback categorization
- **`references/merging.md`** - Script details, error recovery
