---
name: ralph
description: This skill should be used when the user asks to "write a spec", "create a spec", "define requirements", "use ralph", "ralph methodology", "autonomous development", "sync specs to code", "run the planning loop", "run the building loop", or mentions the Ralph workflow for AI-assisted development.
---

# Ralph: Spec-to-Code Synchronization

**IMPORTANT: If user asks to "run the planning loop" or "run the building loop", just run `./ralph/loop.sh plan` or `./ralph/loop.sh`. Do NOT try to act out the loop by reading specs and doing gap analysis yourself - that defeats the purpose of fresh context.**

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

- **Even if a starting point exists** (e.g., a GitHub issue), don't assume it's complete
- Summarize your understanding, then ask: "Is there anything else you want to add or discuss before I start asking specific questions?"
- Let the user explain in their own words - don't interrupt with structured questions yet
- Understand the JTBD (Job to Be Done) - what is the *user* trying to accomplish?
- Note design issues the user raises
- **Investigate the codebase** - read relevant code to understand what exists and how it works
- **Consult skills** - use relevant skills (e.g., `ragzoom-development`) to understand the system
- Ask clarifying questions naturally as they arise, but keep it conversational
- Do NOT create any files yet
- Do NOT jump to Phase 2 until the user explicitly indicates they're ready

#### Phase 2: Structured Interrogation

Once the user confirms they're ready for questions, systematically probe for gaps using the AskUserQuestion tool:

- **You MUST use the AskUserQuestion tool** - not bullet points, not numbered lists, not prose questions
- AskUserQuestion is a tool that presents multiple-choice options to the user
- Keep asking questions until every design decision is resolved with zero ambiguity
- Continue investigating code as new areas come up
- Identify edge cases, constraints, acceptance criteria
- Surface tradeoffs and get the user's preference

**The goal of Phase 2 is a completely unambiguous spec.** No "open questions", no "TBD", no "we'll figure it out later". If something is unclear, ask another question.

**Signs you're not done:** Any decision could go multiple ways, user seems uncertain, you're tempted to write "open questions" in the spec, you haven't looked at all the code that will be affected.

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
- **Acceptance criteria** - testable conditions that prove the feature works. Each criterion should be concrete enough to become a test case (e.g., "`append_batch(['A', 'B', 'C'])` creates exactly 3 leaves" not "batch append works correctly")
- **Edge cases** - what could go wrong, with expected behavior for each
- **Constraints** - performance, security, compatibility
- **Key design decisions** - choices made during Phase 2 and their rationale

**Testability principle:** If you can't describe how to test a requirement, the requirement isn't clear enough. Every acceptance criterion should map to a test.

## Operations

**Infrastructure assumption:** The `specs/` and `ralph/` directories already exist and are set up. Do not check if they exist or try to create them.

### Running the Loops

The loops are **separate processes** run from the terminal, not something the current agent executes inline:

```bash
./ralph/loop.sh plan    # Run planning loop
./ralph/loop.sh         # Run building loop
./ralph/loop.sh 10      # Run building loop, max 10 iterations
```

When the user asks to "run the planning loop" or "run the building loop", run the command in the background (`run_in_background: true`). **Do NOT attempt to act out the loop yourself** by reading the prompts and doing gap analysis - that defeats the purpose of fresh context each iteration.

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
