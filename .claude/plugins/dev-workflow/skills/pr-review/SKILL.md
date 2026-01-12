---
name: pr-review
description: This skill should be used when the user asks to "request a review", "get a review", "ask for review", "request code review", or mentions wanting someone to review their PR. Can be invoked at any point in the PR lifecycle.
---

# Request Code Review

Request an independent code review and handle the review dialogue. This skill can be invoked at any point - there are no preconditions on CI status or clean state.

## Process

### 1. Gather PR Context

```bash
# Get PR details
gh pr view --json number,title,body,additions,deletions,changedFiles

# Get list of changed files
gh pr diff --name-only

# Get recent commits
git log $(git merge-base HEAD origin/master)..HEAD --oneline
```

### 2. Identify Review Focus Areas

Analyze the changes to guide the reviewer:

**Areas to highlight:**
- Complex logic that needs careful review
- Security-sensitive changes (auth, permissions, crypto)
- Performance-critical code paths
- Architectural decisions
- Edge cases that might be missed

### 3. Post Review Request

Post a comment requesting review with context:

```bash
gh pr comment --body "$(cat <<'EOF'
@claude please review this PR.

**Key changes:**
- [Summary of main changes]

**Areas of concern:**
- [Specific areas needing careful review]

**Questions:**
- [Any specific questions for the reviewer]
EOF
)"
```

**Good review requests include:**
- Summary of what changed and why
- Specific areas where review is most valuable
- Any trade-offs or decisions that need validation
- Questions about approach or implementation

### 4. Handle Review Dialogue

When review comments arrive:

1. **Read all feedback:**
   ```bash
   gh pr view --comments
   ```

2. **Categorize feedback:**
   - **Critical**: Must fix - bugs, security issues, breaking changes
   - **Important**: Should fix - code quality, maintainability
   - **Minor**: Nice to fix - style, naming suggestions
   - **Discussion**: Needs conversation - architectural debates

3. **Discuss with user:**
   "The reviewer identified [issues]. Should I fix [X]?"

4. **Implement agreed fixes:**
   - Use `dev-workflow:commit` to commit changes
   - Use `dev-workflow:push` to push (which will update PR if needed)

5. **Respond to reviewer:**
   ```bash
   gh pr comment --body "$(cat <<'EOF'
   @claude I've addressed the following:
   - [Fixed issue 1]
   - [Fixed issue 2]

   Regarding [other issue], keeping as-is because [reason].
   EOF
   )"
   ```

6. **Continue dialogue** until consensus reached

### 5. Review Complete

When the reviewer approves or no more feedback:

```
✅ Review complete
- [Summary of what was addressed]
- [Any items left as-is with reasoning]
```

## Key Principles

- **No preconditions**: This skill can be invoked anytime, regardless of CI status
- **Guide the reviewer**: Don't just ask for review - provide context
- **Discuss before fixing**: Check with user before implementing review feedback
- **Respond to all comments**: Even if declining a suggestion, explain why
- **Iterate until consensus**: Continue dialogue until agreement reached

## What This Skill Does NOT Do

- **Monitor CI**: Use `dev-workflow:pr-monitor` for that
- **Merge**: Use `dev-workflow:merge` after approval

## Related Skills

- To commit review fixes: Use `dev-workflow:commit`
- To push fixes: Use `dev-workflow:push`
- After approval: Use `dev-workflow:merge` to merge the PR

## Examples

**Request review for complex changes:**
```
User: "request a review"
→ Analyze changes, post detailed review request with areas of concern
```

**Handle review feedback:**
```
Reviewer: "This could cause a race condition"
→ Discuss with user, fix if agreed, respond to reviewer
```

**Invoked during CI failure:**
```
User: "get a review anyway"
→ Post review request (no precondition on CI status)
```
