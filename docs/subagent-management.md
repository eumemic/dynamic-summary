# Subagent Management

This document describes how Claude Code subagents are managed in this repository.

## Overview

Subagents in `.claude/agents/` can come from two sources:
1. **External agents**: Symlinks to agents from the git submodule (stay up-to-date with upstream)
2. **Local agents**: Custom `.md` files defined specifically for this project

## Structure

- **Agent directory**: `.claude/agents/` contains both symlinks and local agent definitions
- **Submodule**: `vendor/awesome-claude-code-subagents/` provides external agents
- **Upstream repository**: https://github.com/VoltAgent/awesome-claude-code-subagents

## Common Tasks

### Creating a Local Agent

Create a new `.md` file directly in `.claude/agents/`:

```bash
cat > .claude/agents/my-custom-agent.md << 'EOF'
---
name: my-custom-agent
description: Description of what this agent specializes in
tools: Read, Write, Bash
---

You are a specialized agent for...
EOF
```

### Adding an External Agent

1. Browse available agents in `vendor/awesome-claude-code-subagents/categories/`
2. Create a symlink in `.claude/agents/`:

```bash
cd .claude/agents/
ln -s ../../vendor/awesome-claude-code-subagents/categories/<category>/<agent-name>.md .
```

### Updating External Agents

To update all external agents to the latest upstream version:

```bash
git submodule update --remote vendor/awesome-claude-code-subagents
```

Note: This only updates the submodule content; symlinks will automatically point to the updated versions.

### Viewing Agents

To see which agents are currently enabled (both local and symlinked):

```bash
ls -la .claude/agents/
```

To browse all available external agents in the submodule:

```bash
ls vendor/awesome-claude-code-subagents/categories/*/
```

## Worktree Support

The `scripts/create-worktree` script automatically initializes submodules for new worktrees. If you need to manually initialize a submodule in an existing worktree:

```bash
git submodule init
git submodule update
```

## Technical Details

- External agents are read-only symlinks - modifications should be made upstream
- Local agents are regular files that can be modified directly
- Each worktree maintains its own copy of the submodule
- The `.gitmodules` file tracks the submodule configuration
- Both symlinks and local agent files are tracked in git