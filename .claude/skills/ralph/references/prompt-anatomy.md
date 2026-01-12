# Prompt Anatomy

This document explains the structure and language patterns used in Ralph's prompts.

## Prompt Structure

Both planning and building prompts follow a similar structure:

### Phase 0: Orientation (0a, 0b, 0c, ...)

Load context before doing work:

```
0a. Study `specs/*` with up to 500 parallel Sonnet subagents...
0b. Study @IMPLEMENTATION_PLAN.md...
0c. For reference, the application source code is in `src/*`.
```

The agent orients itself to:
- What should be built (specs)
- What's planned (implementation plan)
- What exists (source code)

### Phases 1-4: Main Instructions

The core work of the iteration:

**Planning mode:**
1. Gap analysis (specs vs code)
2. Generate/update implementation plan
3. No implementation

**Building mode:**
1. Choose most important task
2. Implement with validation
3. Update plan with findings
4. Commit when tests pass

### Guardrails (99999... numbering)

Higher numbers = more critical. This unusual numbering ensures guardrails sort to the end and signal priority:

```
99999. Important: When authoring documentation, capture the why...
999999. Important: Single sources of truth, no migrations/adapters...
9999999. As soon as there are no build errors create a git tag...
...
999999999999999. IMPORTANT: Keep operational notes brief...
```

The escalating 9s create visual hierarchy and (possibly) influence model attention.

## Key Language Patterns

These specific phrases have been discovered to work effectively:

### "Study" (not "read" or "look at")

```
Study `specs/*` with up to 500 parallel Sonnet subagents...
```

"Study" implies deeper comprehension than "read".

### "Don't assume not implemented"

```
Before making changes, search the codebase (don't assume not implemented)...
```

This is critical - prevents agents from reimplementing existing functionality. Forces investigation before creation.

### "Using parallel subagents" / "Up to N subagents"

```
You may use up to 500 parallel Sonnet subagents for searches/reads
and only 1 Sonnet subagent for build/tests.
```

Explicit subagent limits:
- High parallelism for read operations (cheap, fast)
- Single threading for mutations (backpressure)

### "Ultrathink"

```
Use an Opus subagent to analyze findings, prioritize tasks... Ultrathink.
```

Triggers extended thinking mode for complex reasoning.

### "Capture the why"

```
Important: When authoring documentation, capture the why — tests and implementation importance.
```

Don't just document what, explain why it matters.

### "Keep it up to date"

```
Keep @IMPLEMENTATION_PLAN.md current with learnings using a subagent —
future work depends on this to avoid duplicating efforts.
```

The plan is living documentation, not a static checklist.

### "If functionality is missing then it's your job to add it"

```
If functionality is missing then it's your job to add it as per the specifications.
```

Agents should be self-sufficient, not blocked by missing pieces.

### "Resolve them or document them"

```
For any bugs you notice, resolve them or document them in @IMPLEMENTATION_PLAN.md
```

Nothing gets ignored - either fix it now or ensure it's tracked.

## Planning Prompt Template

```markdown
0a. Study `specs/*` with up to 250 parallel Sonnet subagents...
0b. Study @IMPLEMENTATION_PLAN.md (if present)...
0c. Study `src/lib/*` with up to 250 parallel Sonnet subagents...
0d. For reference, the application source code is in `src/*`.

1. Study @IMPLEMENTATION_PLAN.md (if present; it may be incorrect) and use
   up to 500 Sonnet subagents to study existing source code in `src/*` and
   compare it against `specs/*`. Use an Opus subagent to analyze findings,
   prioritize tasks, and create/update @IMPLEMENTATION_PLAN.md...

IMPORTANT: Plan only. Do NOT implement anything. Do NOT assume functionality
is missing; confirm with code search first.

ULTIMATE GOAL: We want to achieve [project-specific goal]...
```

Key elements:
- Orientation before analysis
- Gap analysis with subagents
- Opus for prioritization
- Explicit "plan only" constraint
- Project goal for context

## Building Prompt Template

```markdown
0a. Study `specs/*` with up to 500 parallel Sonnet subagents...
0b. Study @IMPLEMENTATION_PLAN.md.
0c. For reference, the application source code is in `src/*`.

1. Your task is to implement functionality per the specifications using
   parallel subagents. Follow @IMPLEMENTATION_PLAN.md and choose the most
   important item to address. Before making changes, search the codebase
   (don't assume not implemented)...

2. After implementing functionality or resolving problems, run the tests...

3. When you discover issues, immediately update @IMPLEMENTATION_PLAN.md...

4. When the tests pass, update @IMPLEMENTATION_PLAN.md, then `git add -A`
   then `git commit`...

99999. Important: When authoring documentation, capture the why...
[more guardrails with escalating 9s]
```

Key elements:
- Same orientation pattern
- Single task selection
- Investigation before implementation
- Validation before commit
- Guardrails for quality

## Our Adaptations

### Pre-commit as Backpressure

Instead of explicit "run tests" instructions, our pre-commit hook runs `run-checks.sh` automatically. The agent just tries to commit - if checks fail, the commit is rejected.

### Skills Before Commit

Before attempting commit, use:
1. `code-simplifier` - Clean up the implementation
2. `code-review` - Check for issues

This adds quality gates before the pre-commit hook.

### Skills for Operational Knowledge

Instead of updating a project-specific AGENTS.md, operational discoveries should:
1. Update relevant existing skills
2. Create new skills via `skill-development` if no existing skill fits

This keeps operational knowledge in the reusable skill system.

### Task Deletion

When a task is complete, delete it from the plan rather than marking it done. The planning loop will re-detect any gaps if the work was incomplete.
