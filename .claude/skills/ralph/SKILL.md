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

**CRITICAL: Do NOT write the spec file until Phase 3.** The conversation must reach closure first.

### Three-Phase Process

#### Phase 1: Freeform Discussion

Let the user talk through the idea. This is a *conversation*, not an interrogation:

- Let the user explain in their own words - don't interrupt with structured questions yet
- Understand the JTBD (Job to Be Done) - what is the *user* trying to accomplish?
- Note design issues the user raises
- **Investigate the codebase** - read relevant code to understand what exists and how it works
- **Consult skills** - use relevant skills (e.g., `ragzoom-development`) to understand the system
- Ask clarifying questions naturally as they arise, but keep it conversational
- Do NOT create any files yet
- Do NOT jump to Phase 2 until the user has fully expressed their initial thoughts

#### Phase 2: Structured Interrogation

Once the user has shared their initial thoughts and the conversation naturally slows, systematically probe for gaps:

- **Use the AskUserQuestion tool** (not bullet points in prose) to clarify specific design decisions
- AskUserQuestion presents multiple-choice options - use it for decisions with clear alternatives
- Continue investigating code as new areas come up
- Identify edge cases, constraints, acceptance criteria
- Surface tradeoffs and get the user's preference
- Keep probing until both parties feel the design is complete

**Signs you're not done:** Open questions remain, user seems uncertain, key decisions are deferred with "we'll figure it out later", you haven't looked at the code that will be affected.

#### Phase 3: Draft the Spec

Only when the user confirms readiness, write the spec:

- Write directly to `specs/filename.md` (directory already exists)
- Format is flexible - capture what matters clearly
- Include: JTBD, acceptance criteria, edge cases, constraints, key design decisions
- No "open questions" section - those should be resolved in Phase 2

If the user isn't satisfied with the draft, return to Phase 1 for another round.

### Topics of Concern

Break each JTBD into topics. Each topic becomes one spec file.

**Test:** Can you describe the topic in one sentence without "and"?

- "The color extraction system analyzes images to identify dominant colors" (one topic)
- "The user system handles authentication, profiles, and billing" (three topics - split it)

### Spec Content

Specs should be detailed enough that a planning agent can do gap analysis against the codebase. Focus on:

- **What success looks like** - observable outcomes
- **Acceptance criteria** - how to verify it works
- **Edge cases** - what could go wrong
- **Constraints** - performance, security, compatibility
- **Key design decisions** - choices made during Phase 2 and their rationale

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
