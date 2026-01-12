# dev-workflow

Git-based development workflow plugin for Claude Code. Provides composable skills for the commit → push → PR → merge lifecycle.

## Skills

| Skill | Invocation | Purpose |
|-------|------------|---------|
| **commit** | `dev-workflow:commit` | Stage, clean up debug code, check for secrets, create atomic commits |
| **push** | `dev-workflow:push` | Push to remote, update PR title/description if needed |
| **pr-create** | `dev-workflow:pr-create` | Create PR from current branch |
| **pr-monitor** | `dev-workflow:pr-monitor` | Monitor CI, fix failures, loop until green |
| **pr-review** | `dev-workflow:pr-review` | Request code review, handle review dialogue |
| **merge** | `dev-workflow:merge` | Squash merge via script, sync with master |

## Typical Workflow

1. Make changes to code
2. `/dev-workflow:commit` - creates atomic commit(s)
3. `/dev-workflow:push` - pushes and updates PR if exists
4. `/dev-workflow:pr-create` - opens PR (if not already open)
5. `/dev-workflow:pr-monitor` - watches CI, fixes issues
6. `/dev-workflow:pr-review` - requests review when ready
7. `/dev-workflow:merge` - merges when approved

Skills auto-trigger on natural language too:
- "commit these changes" → commit
- "push to remote" → push
- "create a PR" → pr-create
- "fix the CI" → pr-monitor
- "request a review" → pr-review
- "merge this PR" → merge

## Safety

- Never commits to master
- Never uses --no-verify
- Checks for secrets before committing
- Auto-cleans debug code (console.logs, print statements)

## Dependencies

- `gh` CLI for GitHub operations
- `./scripts/squash-merge.sh` for merge operations
