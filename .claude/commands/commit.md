# /commit

Clean up any temporary files, remove ephemeral debugging code, and create well-organized git commits.

## Instructions:

1. **Check Current Branch**:
   - Run `git branch --show-current` to see the current branch
   - **CRITICAL**: Never commit directly to `master` or `main` branches
   - If on master/main, create and switch to a feature branch first:
     - Use descriptive branch names like `fix-mypy-errors`, `add-user-auth`, etc.
     - If unsure about the branch name, ask the user what feature you're about to work on
     - Run `git checkout -b <branch-name>` to create and switch to the new branch
   - Only proceed with commits after confirming you're on a feature branch

2. **Update CLAUDE.md Documentation**:
   - Check if any TODO items in CLAUDE.md have been completed
   - Remove completed items from the TODO list entirely
   - **CRITICAL**: Document any new development utilities created (Makefiles, scripts, commands)
   - Update Essential Commands section if new commands were added
   - Ensure any new patterns, fixes, or important discoveries are documented
   - This keeps the documentation accurate for future sessions

3. **Pre-commit cleanup**:
   - Remove any temporary debugging print statements or console.log calls that were added for troubleshooting
   - Delete any temporary test files or scratch files created during development
   - Clean up any commented-out code that was used for testing
   - Ensure no sensitive information (passwords, API keys, etc.) is being committed

4. **Review changes**:
   - Run `git status` to see all modified and untracked files
   - Run `git diff` to review the actual changes
   - Only commit code that you have personally worked on in this session
   - Do NOT commit changes that were already present when the session started

5. **Organize commits**:
   - **IMPORTANT**: Group commits by FEATURES, not by files!
   - Each commit should leave the application in a fully functional state
   - A single feature often requires changes across multiple files - these should be ONE commit
   - Break changes into logical commits based on:
     - Self-contained features (e.g., "Add pricing breakdown modal" - includes UI, backend, dependencies)
     - Complete bug fixes (e.g., "Fix checkout flow" - includes all related changes)
     - Full refactoring (e.g., "Refactor authentication" - includes all affected files)
     - Documentation updates (can be separate if truly independent)
   - BAD example: Separate commits for "Add Dialog component", "Add dialog dependency", "Use dialog in UI"
   - GOOD example: Single commit "Add pricing breakdown modal" that includes all the above
   - Ask yourself: "If someone checks out this commit, will the app work correctly?"
   - If the answer is no, you're probably splitting too granularly

6. **Commit message format**:
   - Use clear, concise commit messages
   - Start with a verb in present tense (Add, Fix, Update, Refactor, etc.)
   - Keep the first line under 50 characters if possible
   - Add a blank line and more details if needed for complex changes

7. **Final checks**:
   - Run `make check` in claims-buddy-ui if UI code was modified
   - Ensure all tests pass (if applicable)
   - Verify the code still runs correctly
   - Make sure no merge conflicts exist
   - Double-check that CLAUDE.md is updated with any new dev utilities or important information

8. **Push to remote**:
   - After all commits are created, push to the remote repository
   - Run `git push origin <branch-name>` (push your feature branch, NOT master/main)
   - If this is the first push of a new branch, use `git push -u origin <branch-name>`
   - If the push fails due to diverged branches, DO NOT force push without checking with the user first
   - Report any push errors or conflicts to the user
   - After successful push, remind the user to create a Pull Request if appropriate

Remember: Each commit should represent a complete, working feature or fix. Group related changes together, even if they span multiple files. The app should be fully functional at every commit in the history.