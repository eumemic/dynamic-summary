# Agent Instructions for RagZoom

## Agent Philosophy

- **Be Zen, Not Flustered:** When stuck in a loop or series of failures, stop. Rethink from first principles.
- **Don't Be Long-winded:** Keep it concise. Don't repeat yourself or summarize your own messages.
- **Use the Scientific Method:** For complex problems, form a hypothesis, propose a test, discuss before implementing.
- **Investigate Before Fixing:** When encountering strange phenomena (unexpected errors, processes respawning, state inconsistencies), use the `/investigate` skill to understand root cause before attempting fixes. Don't just kill processes or delete files - understand *why* they're in that state first.
- **Raise Blockers:** If you hit an insurmountable roadblock, bring it to the user's attention. Don't switch gears silently.
- **Leave the Codebase Better:** As a master craftsman, always improve code you touch. Fix poor naming, unclear logic, missing error handling opportunistically.
- **No Fallback Code:** NEVER write fallback code that papers over issues. No silent skipping, dummy values, or error suppression. Fail hard with clear error messages.
- **Update Documentation:** If you discover outdated or missing information, update it as part of your task.

## Design Philosophy

- **Design First:** Before implementing large initiatives, create a design proposal with rationale and pseudocode.
- **Clarity Before Code:** Don't implement until the design has no major gaps or open questions.
- **"Correct-by-Construction":** Avoid multi-stage corrective pipelines. Design algorithms that produce valid final state in a single pass.

## Code Craftsmanship

> "Write code as if it will be etched on your tombstone. Be a master craftsman, proud of every single line."

- **Clean:** No dead code, no commented-out blocks, no temporary hacks.
- **Simple:** Favor clarity over cleverness. YAGNI is your friend.
- **Comprehensible:** A stranger should read your code like prose.
- **Well-Factored:** Each piece does one thing well. Functions fit on a screen.
- **Testable:** Prefer pure functions, use dependency injection, isolate side effects.
- **DRY:** Extract common patterns, but don't over-abstract.

## Version Control Rules

- **NEVER Use `--no-verify`:** Pre-commit hooks are guardians of code quality. Never bypass them without explicit permission.
- **No Unauthorized Commits:** Never commit unless explicitly directed by the user.
- **Atomic Commits:** Group changes into small, logical commits. Don't lump unrelated changes.
- **Don't Deprecate, Delete:** Remove old code paths. Git history preserves them.
- **Zero Code Duplication:** Always refactor duplicated code. Mark legitimate false positives with `jscpd:ignore`.

## Type Safety

- Codebase enforces `strict = true` in mypy with `disallow_any_explicit = true`
- All functions, methods, and class attributes must have type hints
- **Never add `# type: ignore`** without explicit user permission
- Tests are type-checked as strictly as production code

## Quality Checks

Most checks run automatically:
- **On every Python edit**: `dmypy`, `ruff`, and `black` (~750ms)
- **On every commit**: Pre-commit hook runs all checks

**Just commit when done** - don't run checks first, let pre-commit do its job. If it fails, fix and re-commit.

## Custom Commands

- `/commit` - Create atomic commits and push
- `/pr` - Create PR and monitor CI
- `/merge` - Merge PR and sync with master
- `/test` - Run tests
- `/review` - Code review

## Worktree Path Resolution

When in a worktree, the current working directory IS the project root. Run commands directly (e.g., `scripts/memory-admin`), never `cd` to the parent repo first.

## Memory Tool

When the `recall` memory tool is available:

1. **Load the skill first**: Before your first memory query, load the `memory-tool-usage` skill to get effective retrieval patterns
2. **Use it proactively**: Query whenever you need to recall details - after compaction, when resuming sessions, when details feel fuzzy, or when specifics matter
3. **Zoom aggressively**: For specific recall, use tight time windows (minutes, not hours) to get verbatim content instead of summaries

## Dev/Prod Server Separation

The RagZoom server automatically uses different ports and state directories based on how it's invoked:

| Invocation | Mode | Port | State Directory |
|------------|------|------|-----------------|
| `ragzoom server start` | Production | 50051 | `~/.local/state/ragzoom/` |
| `python -m ragzoom.cli server start` | Development | 50052 | `~/.local/state/ragzoom-dev/` |

**Key behaviors:**
- **Production mode** (`ragzoom`): Auto-starts server if not running when CLI commands need it
- **Development mode** (`python -m ragzoom.cli`): Fails fast if server not running (no auto-start)
- Explicit `--port` flag always overrides the default
- `RAGZOOM_STATE_DIR` env var overrides the state directory

**Code separation:** Production must be a **non-editable install** (`pip install .`, not `pip install -e .`). This ensures code changes don't affect the running production daemon until explicitly reinstalled. After merging changes that affect the daemon, reinstall production: `pip uninstall ragzoom -y && pip install /Users/tom/code/dynamic-summary`

This separation prevents dev testing from interfering with production data.

## Integration Packages

Client-specific integrations live in `integrations/`. Each is an independent pip-installable package:

- `integrations/claude-code/` - Claude Code transcript sync and MCP server
- `integrations/clawdbot/` - Clawdbot transcript sync

See `integrations/CLAUDE.md` for architecture and development instructions.
