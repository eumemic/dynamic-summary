---
name: commit
description: This skill should be used when the user asks to "commit", "create a commit", "commit these changes", "stage and commit", or mentions committing code changes. Handles staging, cleanup, safety checks, and atomic commit creation.
---

# Git Commit

Create clean, atomic commits with automatic safety checks and debug code cleanup.

## Safety Checks (Non-Negotiable)

Before any commit operation:

1. **Never commit to master**: Run `git branch --show-current`. If on master, stop and ask the user to switch to a feature branch or worktree.

2. **Never use --no-verify**: Pre-commit hooks are guardians of code quality. Never bypass them without explicit user permission.

3. **Check for secrets**: Scan staged changes for potential secrets:
   ```bash
   git diff --cached | grep -iE '(password|secret|api_key|token|credential).*='
   ```
   If matches found, warn the user and confirm before proceeding.

## Process

### 1. Review Changes

```bash
git status
git diff
git diff --cached  # Already staged
```

### 2. Clean Up Debug Code

Before staging, automatically remove debug statements:
- Python: `print(` statements that look like debugging
- JavaScript/TypeScript: `console.log(`, `console.debug(`
- Generic: `debugger`, `TODO: remove`, `FIXME: temp`

Use judgment - preserve intentional logging (error handling, user-facing output). When in doubt, ask.

### 3. Stage Changes

Stage files logically by feature, not by file type:

```bash
git add path/to/related/files
```

### 4. Create Atomic Commits

Each commit should:
- Be a complete, working change
- Could be reverted independently
- Tell part of the story

**Message format**: `type: description` (50 chars max for subject)
- `feat:` new feature
- `fix:` bug fix
- `refactor:` code restructuring
- `docs:` documentation
- `test:` test changes
- `chore:` maintenance

Use a HEREDOC for multi-line messages:
```bash
git commit -m "$(cat <<'EOF'
feat: add user authentication

- Add login/logout endpoints
- Implement session management
- Add auth middleware

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 5. Multiple Commits if Needed

If changes span multiple features, create separate commits:

```bash
# First feature
git add src/auth/*
git commit -m "feat: add authentication module"

# Second feature
git add src/api/*
git commit -m "feat: add API endpoints"
```

## What This Skill Does NOT Do

- **Push**: Use `dev-workflow:push` to push commits to remote
- **Create PR**: Use `dev-workflow:pr-create` after pushing
- **Update PR**: The push skill handles PR updates

## Examples

**Single feature commit:**
```
User: "commit these changes"
→ Review diff, clean debug code, stage all, create one atomic commit
```

**Multiple logical changes:**
```
User: "commit"
→ Identify 2 features in changes, create 2 separate commits
```

**On master branch:**
```
User: "commit this fix"
→ "You're on master. What feature branch should I create?"
```
