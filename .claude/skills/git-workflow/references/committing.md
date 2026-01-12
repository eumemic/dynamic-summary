# Commit Reference

## Safety Checks

Before any commit:

1. **Check branch**: `git branch --show-current` - if on master, stop and ask user
2. **Never --no-verify**: Pre-commit hooks are non-negotiable
3. **Scan for secrets**:
   ```bash
   git diff --cached | grep -iE '(password|secret|api_key|token|credential).*='
   ```
   Warn user if matches found.

## Debug Code Cleanup

Before staging, remove debug statements:

**Python:**
- `print(` statements that look like debugging (not user-facing output)

**JavaScript/TypeScript:**
- `console.log(`, `console.debug(`

**Generic:**
- `debugger`
- `TODO: remove`
- `FIXME: temp`

Use judgment - preserve intentional logging (error handling, user-facing output). When in doubt, ask.

## Staging Strategies

Stage files logically by feature, not by file type:

```bash
# Good: grouped by feature
git add src/auth/* tests/auth/*

# Bad: grouped by type
git add src/*.py
git add tests/*.py
```

## Atomic Commits

Each commit should:
- Be a complete, working change
- Could be reverted independently
- Tell part of the story

## Message Format

**Subject line**: `type: description` (50 chars max)

Types:
- `feat:` new feature
- `fix:` bug fix
- `refactor:` code restructuring
- `docs:` documentation
- `test:` test changes
- `chore:` maintenance

**Multi-line messages** with HEREDOC:
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

## Multiple Commits

If changes span multiple features, create separate commits:

```bash
# First feature
git add src/auth/*
git commit -m "feat: add authentication module"

# Second feature
git add src/api/*
git commit -m "feat: add API endpoints"
```

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
