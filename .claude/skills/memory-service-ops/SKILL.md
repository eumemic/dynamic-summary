---
name: Memory Service Operations
description: This skill should be used when the user asks to "check memory service status", "debug memory service", "check PR environment", "check production environment", "get database URL", "run admin commands", or mentions memory service, hosted service, or Railway operations.
---

# Memory Service Operations

Guidance for operating the hosted RagZoom memory service (currently deployed on Railway).

## Critical: Environment and Database Mapping

The hosted service has **two separate databases** for different environments:

| Environment | Database Service | Public Proxy |
|-------------|------------------|--------------|
| Production (`production`) | `pgvector` | `tramway.proxy.rlwy.net:48318` |
| PR environments (`dynamic-summary-pr-XXX`) | `pgvector-rW-f` | `nozomi.proxy.rlwy.net:30284` |

**Never assume** which database an environment uses. Always verify by checking the service variables.

## Workflow: Check Memory Service Status

### Step 1: Determine the Target Environment

Ask the user or determine from context:
- **Production**: Use environment name `production`
- **PR environment**: Use environment name `dynamic-summary-pr-{NUMBER}`

To find the PR number for the current branch:
```bash
gh pr list --state open --head $(git branch --show-current) --json number,title
```

### Step 2: Link to the Environment

```bash
railway link -p 9d168ba6-ac78-4739-a53c-7ca04e211678 -e {ENVIRONMENT_NAME}
```

### Step 3: Discover the Correct Database

**Do not skip this step.** Check which database the `dynamic-summary` service connects to:

```bash
railway variables --service dynamic-summary --json | grep RAGZOOM_DATABASE_URL
```

The internal hostname reveals which database service to use:
- `pgvector.railway.internal` → Use `pgvector` service
- `pgvector-rw-f.railway.internal` → Use `pgvector-rW-f` service

### Step 4: Get the Public Database URL

Use the **correct service name** from Step 3:

```bash
# For production (pgvector)
railway variables --service pgvector --kv | grep DATABASE_PUBLIC_URL

# For PR environments (pgvector-rW-f)
railway variables --service pgvector-rW-f --kv | grep DATABASE_PUBLIC_URL
```

### Step 5: Run Admin Commands

Set the database URL and run the command:

```bash
RAGZOOM_DATABASE_URL="{URL_FROM_STEP_4}" python -m memory_service.admin status
```

**Important**: Copy-paste the URL directly. Do not use command substitution like `$(railway ...)` as it can mangle special characters in passwords.

## Common Admin Commands

```bash
# Service status and session inventory
python -m memory_service.admin status

# Reset a session for full re-index
python -m memory_service.admin reset {session-id}

# Validate indexed content matches transcript
python -m memory_service.admin validate {session-id}

# Transcribe stored JSONL to text
python -m memory_service.admin transcribe {session-id} [-o output.txt]
```

## Deployment Notes

- The service auto-deploys when pushing to PR branches
- **Never use** `railway deployment redeploy` - it redeploys old code
- To deploy new code, push to the branch: `git push origin {branch}`

## Additional Resources

For detailed operational procedures:
- **`references/detailed-ops.md`** - Complete operational procedures
- **`scripts/check-status.sh`** - Automated status check script
