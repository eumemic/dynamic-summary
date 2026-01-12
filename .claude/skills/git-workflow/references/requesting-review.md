# PR Review Reference

## No Preconditions

This operation can be invoked anytime - no requirements on CI status or clean state.

## Gather PR Context

```bash
# Get PR details
gh pr view --json number,title,body,additions,deletions,changedFiles

# List changed files
gh pr diff --name-only

# Recent commits
git log $(git merge-base HEAD origin/master)..HEAD --oneline
```

## Identify Review Focus Areas

Analyze changes to guide the reviewer:

**Highlight these areas:**
- Complex logic needing careful review
- Security-sensitive changes (auth, permissions, crypto)
- Performance-critical code paths
- Architectural decisions
- Edge cases that might be missed

## Post Review Request

```bash
gh pr comment --body "$(cat <<'EOF'
@reviewer please review this PR.

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
- Trade-offs or decisions needing validation
- Questions about approach or implementation

## Handle Review Feedback

**Read all feedback:**
```bash
gh pr view --comments
```

**Categorize feedback:**
- **Critical**: Must fix - bugs, security issues, breaking changes
- **Important**: Should fix - code quality, maintainability
- **Minor**: Nice to fix - style, naming suggestions
- **Discussion**: Needs conversation - architectural debates

## Discuss Before Fixing

Before implementing changes:
- "The reviewer identified [issues]. Should I fix [X]?"
- Get user agreement on which feedback to address
- Discuss any disagreements with reviewer's suggestions

## Implement Fixes

For agreed-upon changes:
1. Use commit operation to create fix commit
2. Use push operation (which updates PR if needed)

## Respond to Reviewer

```bash
gh pr comment --body "$(cat <<'EOF'
@reviewer I've addressed the following:
- [Fixed issue 1]
- [Fixed issue 2]

Regarding [other issue], keeping as-is because [reason].
EOF
)"
```

**Response guidelines:**
- Acknowledge all comments
- Explain what was fixed
- Justify any pushback with reasoning
- Be respectful and constructive

## Review Complete

When reviewer approves or no more feedback:

```
Review complete
- [Summary of what was addressed]
- [Any items left as-is with reasoning]
```

## Continue Dialogue

Iterate until consensus:
- Reviewer may have follow-up comments
- Continue fixing and responding
- Escalate disagreements to user if needed

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
