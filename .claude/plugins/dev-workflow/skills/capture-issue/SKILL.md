---
name: capture-issue
description: This skill should be used when the user asks to "create an issue", "open an issue", "file a bug", "file an issue", "track this", or mentions creating a GitHub issue from the current context.
---

# Create GitHub Issue

Capture issues in the moment of discovery while context is fresh.

## Process

### 1. Gather Context

```bash
# Available labels
gh label list --json name -q '.[].name' | head -20

# Recent issues for style reference
gh issue list --limit 3 --json number,title -q '.[] | "#" + (.number|tostring) + ": " + .title'
```

### 2. Determine Issue Scope

**Minimal issue** (quick TODO, user's off-the-cuff thought):
- Brief title
- One-line description
- Skip labels if uncertain

**Rich issue** (something discussed/designed extensively):
- Descriptive title
- Full context from conversation
- Code locations, error messages
- Relevant labels

### 3. Create Issue

```bash
gh issue create --title "Title here" --body "$(cat <<'EOF'
## Description
[What's the problem or feature?]

## Context
[Relevant details, code locations, error messages]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
EOF
)" --label "bug,enhancement"
```

### 4. Report Success

```
✅ Issue created: #123 - Title here
   https://github.com/owner/repo/issues/123
```

## Key Principles

- **Capture while fresh**: Best bug reports are written in the moment
- **Don't speculate**: Only include details that have been worked out
- **Match depth to context**: Barebones for quick TODOs, comprehensive for designed features
- **Trust future readers**: Include pointers, let them investigate details

## What This Skill Does NOT Do

- **Assign issues**: Let the user or team triage
- **Set milestones**: Leave for project planning
- **Link PRs**: Use PR description for that

## Examples

**Quick TODO:**
```
User: "create an issue for that edge case we noticed"
→ Brief title, one-line description from context
```

**After design discussion:**
```
User: "file an issue for the caching feature we just designed"
→ Full context, acceptance criteria, relevant labels
```
