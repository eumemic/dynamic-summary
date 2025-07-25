# /branch

Arguments: "$ARGUMENTS"

_Note: if no arguments provided above, the branch name will be inferred from recent conversation context._

Create a new git worktree and branch for feature development, then change to that directory.

## Instructions:

1. **Determine Branch Name**:
   - If arguments provided: Use them to generate a descriptive branch name
   - If no arguments: Analyze the conversation context to infer what feature/fix is being worked on
   - Follow naming conventions:
     - `feature/` for new features
     - `fix/` for bug fixes  
     - `refactor/` for code refactoring
     - `docs/` for documentation updates

2. **Prepare Branch Name**:
   - Convert to lowercase
   - Replace spaces with hyphens
   - Remove special characters
   - Keep it concise but descriptive

3. **Check for Conflicts**:
   - Run `git worktree list` to see existing worktrees
   - Run `git branch -a` to check existing branches
   - If proposed name conflicts, append a number or modify slightly

4. **Confirm with User**:
   - Present the proposed worktree and branch name
   - Ask: "I'll create a worktree at `worktrees/[name]` with branch `[branch-name]`. Is this correct?"
   - Wait for user confirmation before proceeding
   - If user suggests changes, use their preferred name

5. **Create Worktree**:
   - Ensure currently on master and up to date
   - Create the worktree: `git worktree add worktrees/[name] -b [branch-name]`
   - Change to the new worktree directory
   - Copy environment configuration: `cp .env worktrees/[name]/.env` (if .env exists)

6. **Confirm Ready**:
   - Show current directory and branch
   - Confirm ready for development

## Example Flow:

```bash
# Ensure on master and updated
git checkout master
git pull

# Check for existing worktrees and branches
git worktree list
git branch -a | grep [proposed-name]

# After user confirmation
git worktree add worktrees/authentication -b feature/authentication
cd worktrees/authentication

# Copy environment configuration if it exists
if [ -f ../../.env ]; then
    cp ../../.env .env
fi

# Confirm status
pwd
git branch --show-current
```

## Error Handling:

- If not on master, switch to master first
- If proposed branch already exists, suggest alternative with number suffix
- If worktree already exists, inform user and modify name
- If inference unclear, ask user for clarification

## Argument Examples:

- "authentication system" → Propose "feature/authentication-system" for worktree/branch name
- "fix memory leak" → Propose "fix/memory-leak"
- "refactor database queries" → Propose "refactor/database-queries"
- "" → Infer from context or ask what we'll be working on to come up with a proposal