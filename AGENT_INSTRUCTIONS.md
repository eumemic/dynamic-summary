# Agent Instructions for RagZoom

This file contains comprehensive instructions for any AI agent working on this repository.

## Project Documentation

Note: Documentation that applies to both human developers and AI agents belongs in README.md,
which is included below. This file (CLAUDE.md) contains only agent-specific instructions.

@include README.md

## Agent-Specific Instructions

### 1. Agent Philosophy & Collaboration

- **Be Zen, Not Flustered:** When you get stuck in a loop or a series of failures, stop. Take a step back, breathe, and rethink the problem from first principles. There is always a logical solution.
- **Don't Be Long-winded:** Keep it concise unless elaboration is warranted. Don't repeat yourself or summarize your own messages at the end. You're not writing an essay, we're having a conversation.
- **Use the Scientific Method:** For complex problems, form a hypothesis, propose a test to validate it, and discuss it with the user before implementing.
- **Raise Blockers:** If you are instructed to do something and discover an insurmountable roadblock or a fundamental inconsistency, do not switch gears. Bring the issue to the user's attention and decide on a new course of action together.
- **Leave the Codebase Better:** As a master craftsman, always improve the code you touch. Look for opportunities to refactor, clarify, or simplify. When you notice unrelated problems—poor naming, unclear logic, missing error handling—fix them opportunistically. Every interaction should leave the codebase more elegant than you found it. This is your signature as a professional.
- **No Fallback Code:** NEVER write fallback code that papers over issues. This includes: silent skipping (`if x: continue`), dummy values (`x = y or 0`), "sensible defaults" (`get(key, default_value)`), or any form of error suppression. If an invariant is violated or expected data is missing, fail hard with a clear error message. We need to be able to rely on invariants. Every assumption should be validated with explicit assertions or exceptions.
- **Update Documentation:** If you discover that a document is out of date or missing information in the course of your work, update it as part of your task.
- **Update These Rules:** If you discover a new principle or best practice during your work, add it to this file.

### 2. Design Philosophy

- **Design First:** Before implementing any large initiative, work with the user to create a well-thought-out design proposal, including rationale and pseudocode.
- **Clarity Before Code:** Do not start implementing until you have a design with no major gaps or open questions. Ask the user to clarify any ambiguities.
- **"Correct-by-Construction":** The central architectural principle of this system is to be "correct-by-construction". Avoid multi-stage, corrective pipelines that patch up errors. Design algorithms that produce a valid final state in a single, principled pass. The tiling algorithm in `ragzoom/greedy_tiling.py` exemplifies this approach.
- **Design Reflects Craft:** Great design enables great code. A well-designed system makes correct implementation natural and incorrect implementation difficult. Poor design forces good developers to write bad code. Always question whether complexity in implementation signals a design problem.

### 3. Code Craftsmanship & Pride

> **"Write code as if it will be etched on your tombstone, or presented to you at the pearly gates. Be a master craftsman, proud of every single line."**

Every agent must approach code as a master craftsperson. Each line you write is a reflection of your skill and dedication to the craft. Never write code you wouldn't be proud to show to the greatest programmer in history.

- **Clean:** Your code should be pristine. No dead code, no commented-out blocks, no temporary hacks left behind. Every line has a purpose and earns its place. If something exists, it should belong there.

- **Simple:** Favor clarity over cleverness. The most elegant solution is often the simplest one that correctly solves the problem. Complex problems require simple, composable solutions. YAGNI (You Ain't Gonna Need It) is your friend.

- **Comprehensible:** A stranger should be able to read your code like prose. Use intention-revealing names. Minimize cognitive load. If you need to explain what your code does, the code probably isn't clear enough.

- **Well-Factored:** Each piece of code should do one thing and do it well. Functions should fit comfortably on a screen. Classes should have cohesive responsibilities. Abstractions should hide complexity, not create it.

- **Single Responsibility:** Every class, function, and module should have one reason to change. If you find yourself writing "and" in a description of what your code does, consider splitting it.

- **Testable:** Write code that invites testing. Prefer pure functions over stateful ones. Use dependency injection. Isolate side effects. If your code is hard to test, it's probably hard to use and maintain.

- **DRY (Don't Repeat Yourself):** Extract common patterns, but don't over-abstract. Three instances of similar code might warrant extraction; two might not. Good abstraction reduces complexity; poor abstraction increases it.

**Quality Gates:**
- Before considering any code complete, ask: "Would I be proud to have this reviewed by the world's best developers?"
- If you wouldn't put this code in your portfolio, rewrite it.
- Remember: bad code is not just technical debt—it's an insult to everyone who will read it after you.

### 4. Version Control & Collaboration Rules

- **Never Commit to Master:** Always ensure you're on a feature branch before committing. Check with `git branch --show-current`. If on master/main, ALWAYS ask the user what feature you're about to work on before creating a branch - don't assume the scope.
- **NEVER Use `--no-verify`:** ABSOLUTELY NEVER use `--no-verify` to bypass pre-commit hooks unless given EXPLICIT PERMISSION by the user. Pre-commit hooks are the guardians of code quality and must not be bypassed. They prevent broken code, failed tests, and style violations from entering the repository.
- **No Unauthorized Commits:** Never commit code unless explicitly directed to by the user.
- **Atomic Commits:** When asked to commit, group changes into small, logical, atomic commits with clear messages. Do not lump unrelated changes together.
- **Don't Deprecate, Delete:** Do not leave old code paths behind a feature flag or comment them out. Remove them. The git history will preserve them if we ever need to look back.
- **Zero Code Duplication:** This codebase maintains a strict zero-duplication policy. Always refactor duplicated code. Only mark legitimate false positives (like async/sync wrappers) with `jscpd:ignore` comments and clear justification. See `docs/developer-guide.md` section 4.5 for detailed guidelines.

### 5. System Architecture & Technical Documentation

@include docs/architecture.md

### 6. Development Practices & Testing

@include docs/developer-guide.md

### 7. Quality Checks & Testing for Agents

**IMPORTANT: Most quality checks happen automatically - you rarely need to run them manually.**

#### Automatic Checks
- **On every Python edit**: `dmypy`, `ruff`, and `black` run automatically (~750ms)
- **On every commit**: All checks run via pre-commit hook (tests, linting, formatting, security, duplication)

#### When to Run Checks Manually
1. **Before committing** (if you want to verify tests pass): Use `./scripts/run-checks.sh`
2. **To test specific functionality**: Use `./scripts/run-checks.sh` with appropriate options
3. **NEVER use `pytest` directly unless given explicit approval by the user** - in general you should use `run-checks.sh` which ensures proper environment setup

#### Common Commands for Agents
```bash
# Preferred: Run quality checks (excludes integration/benchmarks by default)
./scripts/run-checks.sh

# Include integration tests
./scripts/run-checks.sh --include-integration-tests

# Run only tests impacted by specific files
./scripts/run-checks.sh --impacted-only path/to/changed1.py path/to/changed2.py

# Stop at first error (useful for debugging)
./scripts/run-checks.sh --fail-fast
```

**Key Points:**
- If checks pass on edit → They'll likely pass on commit
- If pre-commit fails → It will show you exactly what's wrong
- The system is designed to give you immediate feedback without manual intervention
- **VERY IMPORTANT**: When you have completed a task, run `./scripts/run-checks.sh` to ensure your code is correct (NOT `pytest`)

### 8. Type Safety Requirements

- **Strict Type Checking**: The codebase enforces `strict = true` in mypy configuration
- **No Explicit Any Types**: The codebase has eliminated all explicit `Any` types with `disallow_any_explicit = true`
- **Complete Annotations**: All functions, methods, and class attributes must have type hints
- **NEVER Add Type Ignores Without Permission**: NEVER add `# type: ignore` comments without explicit approval from the user. If you encounter a type error that seems to require suppression, stop and discuss the issue with the user first. There's usually a better solution.
- **Type Ignore Documentation**: Every existing `# type: ignore` must have an explanatory comment
- **Test Type Coverage**: Tests are type-checked as strictly as production code
- **Note on disallow_any_expr**: This flag is not enabled as it would require extensive workarounds for Pydantic and external libraries (~2000+ errors). The current configuration provides excellent type safety while remaining pragmatic.

### 9. Algorithm Deep Dive

For detailed understanding of the core tiling algorithm:

@include docs/deep-dives/tiling-algorithm.md

### 9. Custom Claude Commands

Custom slash commands are stored in `.claude/commands/` as markdown files:
- `/commit` - Create atomic commits and push to origin
- `/pr` - Create PR if needed and monitor CI (formerly `/push`)
- `/merge` - Merge PR and sync with master
- `/test` - Run tests
- `/review` - Code review
To modify: edit `.claude/commands/<command-name>.md`

### 10. Subagent Management

Claude Code subagents can be either local custom agents or symlinks to external agents via git submodule. See `docs/subagent-management.md` for details.

### 11. Agent Slots for Parallel Development

This repository uses persistent worktree "agent slots" for isolated Claude sessions. Each slot provides:
- Separate conversation history (stored in ~/.claude/projects/)
- Isolated git workspace
- Ability to work on multiple features in sequence

### Creating a New Agent Slot
```bash
./scripts/create-worktree
cd worktrees/worktree-N && claude
```

### Workflow Within Slots (Sequential PR Model)
- Work directly on the worktree branch (e.g., `worktree-2`)
- Use `/commit` to commit and push changes
- Use `/pr` to create PR and monitor CI
- Use `/merge` to merge PR and sync with master
- The same branch is reused for multiple sequential PRs

### 13. Agent-Specific Troubleshooting

- **Pre-commit Hook Issues:** If pre-commit hooks fail, they'll automatically fix most issues. Just re-commit after they run.
- **Testing Issues:** Always use `./scripts/run-checks.sh` instead of `pytest` directly - it handles environment setup correctly.
