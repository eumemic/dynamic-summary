---
name: ralph
description: This skill should be used when the user asks to "write a spec", "create a spec", "define requirements", "use ralph", "ralph methodology", "autonomous development", "sync specs to code", or mentions the Ralph workflow for AI-assisted development.
---

# Ralph: Spec-to-Code Synchronization

Ralph is a methodology for autonomous AI development where humans write specs and the machine syncs code to match. The human stays in the problem space (defining what to build), while the machine handles the solution space (implementing it).

Based on [Geoffrey Huntley's Ralph pattern](https://ghuntley.com/ralph/) with adaptations for Claude Code's skill system.

## Core Concept

```
specs/ ──► [planning loop] ──► IMPLEMENTATION_PLAN.md ──► [building loop] ──► code
```

- **Human's job:** Write and refine specs (the "what")
- **Machine's job:** Gap analysis + implementation (the "how")

Specs are the source of truth. Code is the derived artifact.

## Architecture

### Project Structure

```
project-root/
├── specs/                      # Source of truth (project artifact)
│   ├── user-auth.md
│   └── data-export.md
├── ralph/                      # Sync engine (tooling)
│   ├── loop.sh                 # Bash loop that cranks the engine
│   ├── PROMPT_plan.md          # Planning mode instructions
│   ├── PROMPT_build.md         # Building mode instructions
│   └── IMPLEMENTATION_PLAN.md  # Tasks + learnings (generated)
├── src/
└── ...
```

### Two Loops

**Planning Loop** (`./ralph/loop.sh plan`):
1. Reads specs/
2. Studies existing codebase
3. Performs gap analysis (spec vs implementation)
4. Generates/updates IMPLEMENTATION_PLAN.md
5. No implementation, no commits

**Building Loop** (`./ralph/loop.sh`):
1. Reads specs + plan
2. Picks most important task
3. Implements, tests, commits
4. Deletes completed task from plan
5. Loop restarts with fresh context

Each iteration gets a fresh context window. The bash loop is intentionally dumb - it just keeps restarting the agent. Intelligence lives in the prompts and specs.

### Backpressure

Work is validated through:
- Pre-commit hooks (run-checks.sh)
- Tests encoding acceptance criteria
- code-simplifier and code-review skills before commit

If validation fails, the agent fixes and retries. Bad work doesn't escape.

## Spec Development

This is the human's primary activity. Collaborate with Claude to turn fuzzy ideas into clear specs.

### Jobs to Be Done (JTBD)

Start with user outcomes, not implementation tasks:

- **JTBD (user's job):** "Export my data to share with finance"
- **NOT:** "Implement CSV export function"

JTBD answers: What is the *user* trying to accomplish?

### Topics of Concern

Break each JTBD into topics. Each topic becomes one spec file.

**Test:** Can you describe the topic in one sentence without "and"?

- "The color extraction system analyzes images to identify dominant colors" (one topic)
- "The user system handles authentication, profiles, and billing" (three topics - split it)

### Writing Specs

Format is flexible - let the spec take whatever shape captures the requirements clearly. Focus on:

- **What success looks like** - observable outcomes
- **Acceptance criteria** - how to verify it works
- **Edge cases** - what could go wrong
- **Constraints** - performance, security, compatibility

Specs should be detailed enough that a planning agent can do gap analysis against the codebase.

### Conversation Flow

When developing a spec (assume `specs/` directory already exists):

1. **Explore** - Discuss the JTBD, understand the user need
2. **Scope** - Identify topics of concern, decide what's in/out
3. **Detail** - Elaborate acceptance criteria, edge cases
4. **Decide** - Use AskUserQuestion for specific choices when needed
5. **Write** - Write the spec file directly to `specs/filename.md`

Do not check if directories exist or set up infrastructure - that's already done.

## Operations

### When to Run Planning

- No plan exists yet
- Specs have changed significantly
- Plan feels stale or doesn't match reality
- Confused about what's actually done

### When to Run Building

- Plan exists and looks correct
- Ready to implement

### When to Regenerate Plan

- Agent is going in circles (implementing wrong things, duplicating work)
- Too much clutter from completed items
- Trajectory feels wrong

The plan is disposable. Regeneration costs one planning loop - cheap compared to wasted building loops.

### Operational Knowledge

When implementing agents discover operational patterns (how to run tests, gotchas about the build system), they should update relevant skills or create new ones using skill-development. This keeps operational knowledge in the skill system rather than project-specific files.

## Additional Resources

For detailed methodology and prompt structure:

- **`references/methodology.md`** - Deep dive on Ralph principles, context optimization, subagent architecture
- **`references/prompt-anatomy.md`** - Prompt structure, guardrail numbering, language patterns

## Credits

- Original methodology: [Geoffrey Huntley](https://ghuntley.com/ralph/)
- Playbook synthesis: [Clayton Farr](https://github.com/ClaytonFarr/ralph-playbook)
