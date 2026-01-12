# Ralph Methodology Deep Dive

This document covers the principles and mechanics behind the Ralph pattern for autonomous AI development.

## Context Is Everything

When working with LLMs, context window management is critical:

- 200K tokens advertised ≈ 176K truly usable
- 40-60% context utilization for "smart zone" (where model performs best)
- **Tight tasks + 1 task per loop = 100% smart zone utilization**

This drives the entire architecture:

### Main Agent as Scheduler

Don't allocate expensive work to the main context. Use subagents for heavy lifting:

- **Up to 500 parallel subagents** for search/read operations
- **Only 1 subagent** for build/tests (creates backpressure)
- **Opus subagents** for complex reasoning (debugging, architectural decisions)

### Subagents as Memory Extension

Each subagent gets ~156KB that's garbage collected when done. Fan out to avoid polluting the main context with search results and intermediate work.

### Simplicity Wins

- Fewer parts in the system
- Terse prompt content
- Verbose inputs degrade determinism
- Prefer Markdown over JSON for token efficiency

## Steering Ralph

Create signals and gates to guide successful output. Steer from two directions:

### Steer Upstream (Patterns)

**Deterministic setup:**
- First ~5,000 tokens reserved for specs
- Every loop iteration loads identical files (PROMPT.md + any operational context)
- Model starts from known state despite non-deterministic outputs

**Code as guidance:**
- Existing code shapes what gets generated
- If Ralph generates wrong patterns, add utilities and patterns to steer it right
- The codebase itself is a form of prompt

### Steer Downstream (Backpressure)

**Programmatic validation:**
- Tests, typechecks, lints, builds
- Reject invalid/unacceptable work automatically
- Pre-commit hooks as the gate

**Subjective validation:**
- Some criteria resist programmatic checks (creative quality, UX feel)
- LLM-as-judge can provide backpressure for subjective criteria
- Binary pass/fail reviews that iterate until passing

### Skills as Operational Knowledge

Our adaptation: Instead of project-specific AGENTS.md files, operational knowledge lives in skills:

- `ragzoom-development` - how to run tests, lint, typecheck
- `git-workflow` - how to commit, push, create PRs
- New skills created via `skill-development` when patterns emerge

Skills are reusable across projects and maintained in one place.

## Let Ralph Ralph

Ralph's effectiveness comes from trusting it to self-correct through iteration:

### Trust the Loop

- Lean into LLM's ability to self-identify, self-correct, self-improve
- Applies to implementation plan, task definition, prioritization
- Eventual consistency achieved through iteration, not upfront perfection

### The Plan is Disposable

- Wrong plan? Throw it out, regenerate
- Regeneration cost = one planning loop
- Much cheaper than Ralph going in circles on a bad trajectory

**Regenerate when:**
- Going off track (wrong things, duplicate work)
- Plan feels stale or doesn't match current state
- Too much clutter from completed items
- Significant spec changes
- Confused about what's actually done

### Move Outside the Loop

To get the most from Ralph, get out of his way:

- Ralph does ALL the work, including deciding what to implement next
- Your job: engineer the setup and environment for success
- Sit ON the loop, not IN it

**Observe and course correct:**
- Watch patterns emerge, especially early on
- Where does Ralph go wrong?
- What signs does he need?
- Prompts evolve through observed failure patterns

**Tune like a guitar:**
- Don't prescribe everything upfront
- Observe and adjust reactively
- When Ralph fails a specific way, add a sign to help next time

### Signs Aren't Just Prompts

Signs are anything Ralph can discover:

- Prompt guardrails ("don't assume not implemented")
- Operational knowledge in skills
- Utilities and patterns in the codebase
- Test fixtures that demonstrate correct behavior

## Three Phases, Two Prompts, One Loop

### Phase 1: Define Requirements (Human + LLM)

- Discuss project ideas → identify Jobs to Be Done
- Break JTBD into topics of concern
- Write specs for each topic
- This is creative, collaborative, human-driven

### Phase 2: Planning Loop

- Gap analysis: specs vs existing code
- Generate prioritized task list
- No implementation, just planning
- Can run multiple iterations if needed

### Phase 3: Building Loop

- Pick most important task from plan
- Implement, test, validate
- Update plan, commit
- Loop restarts with fresh context

Same loop mechanism, different prompts. The bash script is intentionally dumb:

```bash
while true; do
  cat PROMPT.md | claude -p --dangerously-skip-permissions
done
```

## Task Lifecycle in Building Loop

Each iteration:

1. **Orient** - Study specs (requirements)
2. **Read plan** - Study IMPLEMENTATION_PLAN.md
3. **Select** - Pick most important task
4. **Investigate** - Search codebase ("don't assume not implemented")
5. **Implement** - Make changes
6. **Validate** - Run tests (backpressure)
7. **Update plan** - Mark done, note discoveries
8. **Commit** - If validation passes
9. **Exit** - Context cleared, next iteration starts fresh

The fresh context each iteration is key - no accumulated context debt.

## Terminology

| Term | Definition |
|------|------------|
| **JTBD** | High-level user need or outcome (Job to Be Done) |
| **Topic of Concern** | A distinct aspect/component within a JTBD |
| **Spec** | Requirements doc for one topic (`specs/*.md`) |
| **Task** | Unit of work derived from comparing specs to code |

**Relationships:**
- 1 JTBD → multiple topics of concern
- 1 topic → 1 spec
- 1 spec → multiple tasks

## Further Reading

- [Geoffrey Huntley's original Ralph post](https://ghuntley.com/ralph/)
- [Clayton Farr's Ralph Playbook](https://github.com/ClaytonFarr/ralph-playbook)
- [Snarktank's Ralph implementation](https://github.com/snarktank/ralph)
