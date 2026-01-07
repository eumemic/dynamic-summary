# Production Learnings

Observations from running the hosted RagZoom service on Railway with real Claude Code transcripts.

## Session: 2026-01-07

### Document Scale

- **Session ID**: `7cdd0798-4f29-4ce6-bfc9-6dc3b7bb2153`
- **Raw transcript**: ~107MB JSONL
- **Stored content**: ~73MB (after `strip_tool_results`)
- **Leaves created**: 2,546
- **Expected internal nodes**: 2,545 (for complete tree)
- **Total expected nodes**: 5,091

### Indexing Progress Stall

Observed indexing stuck at `4897/5085` jobs completed with 1 inflight job not progressing.

**Missing nodes**: 193 internal nodes not created, resulting in 194 roots instead of expected 7.

**Expected forest structure** for 2,546 leaves:
```
2546 = 2048 + 256 + 128 + 64 + 32 + 16 + 2
     = 2^11 + 2^8 + 2^7 + 2^6 + 2^5 + 2^4 + 2^1
Expected roots at heights: [11, 8, 7, 6, 5, 4, 1]
```

**Actual roots by height**:
```
h=11: 1   (correct)
h=8:  1   (correct)
h=4:  1   (correct - but missing h=7, h=6, h=5)
h=2:  9   (orphans)
h=1:  8   (orphans)
h=0:  174 (orphaned leaves!)
Total: 194 roots
```

### Root Cause: Unauthorized `mark_failed` Breaking Implicit Retry

**Primary cause (FIXED)**: Unauthorized code was added to the indexing engine that permanently marked failed jobs, preventing the intended implicit retry behavior.

The **design intent** is that when a job fails, the next scan should find the same eligible pair and recreate the job (implicit retry). However, an agent had added `mark_failed()` / `is_failed()` / `failed_job_ids` code that tracked failed jobs and skipped them on subsequent scans.

**Code removed from `ragzoom/server/indexing_engine.py`**:
- `DocumentContext.failed_jobs` counter
- `DocumentContext.failed_job_ids` set
- `DocumentContext.mark_failed()` method
- `DocumentContext.is_failed()` method
- Skip checks in `_find_next_n_embedding_jobs()` and `_find_next_n_summary_jobs()`
- `ctx.mark_failed(job)` call in job failure handler

**Secondary cause (still present)**: LLM summarization failures due to:

#### 1. OpenAI Policy Violation
```
Invalid prompt: your prompt was flagged as potentially violating our usage policy.
Please try again with a different prompt
```

This occurs when transcript content (code, error messages, or user conversations) triggers OpenAI's content filter. The content being summarized is legitimate development work but contains patterns that trip the filter.

#### 2. Empty LLM Response
```
LLM error during complete with gpt-5-nano: LLM returned empty response content
```

The model (gpt-5-nano) occasionally returns empty responses, causing the summary job to fail.

### Impact (with fix applied)

With implicit retry restored:
- Failed jobs are now retried on subsequent scans
- Transient errors (rate limits, network issues) will self-heal
- Permanent errors (content policy) will loop indefinitely with warning logs

Without implicit retry (before fix):
1. Job fails → marked as failed → skipped forever
2. Parent node never created
3. Child nodes become orphaned roots
4. Tree incomplete, higher-level summarization blocked

### Remaining Investigations

1. What specific content triggers the policy violation?
2. Is gpt-5-nano the right model for summarization? (empty responses suggest instability)
3. Should we add content sanitization before summarization?
4. Should permanent failures (content policy) have different handling than transient ones?

---

## Deployment Learnings

### Railway Deployment Workflow

**CRITICAL**: Railway auto-deploys when you push to the PR branch.

```bash
# CORRECT: Push to PR branch triggers deploy
git push origin worktree-1

# WRONG: These don't pick up new code
railway deployment redeploy  # Just redeploys cached Docker image
railway deploy               # Also doesn't pull latest git
```

The `railway up` command can push local code directly but is not the standard workflow.

### Service Architecture

| Service | Purpose |
|---------|---------|
| `dynamic-summary` | gRPC server for memory ingestion |
| `pgvector` | PostgreSQL with pgvector (active database) |
| `pgvector-rW-f` | Legacy service (not used) |

### gRPC Endpoint Stability

The TCP proxy address can change between deployments:
- Address stored in `.mcp.json` under `RAGZOOM_SERVER_ADDRESS`
- Must verify correct endpoint has the data after deployment changes
- Use `GetDocument` RPC to check leaf count and tree depth

---

## Code Fixes Applied

### 1. Exception Handler Gap in IngestSession

**Problem**: Exceptions in Phase 1/2 of `IngestSession` escaped to gRPC default handler, producing unhelpful error messages.

**Symptom**: Error format `"Unexpected <class 'json.decoder.JSONDecodeError'>: ..."` instead of `"Ingestion failed: ..."`

**Fix**: Added `except` block before `finally` in Phase 1/2:
```python
except Exception as e:
    logger.exception("Error in Phase 1/2 for session %s", session_id)
    await context.abort(grpc.StatusCode.INTERNAL, f"Ingestion failed: {e}")
    raise  # Unreachable but satisfies type checker
finally:
    # Release lock...
```

**File**: `memory_service/grpc_servicer.py`

### 2. Debug Logging in IngestSession

**Added**: First 100 bytes of delta logged for debugging parse errors:
```python
logger.info(
    "[TIMING] IngestSession start: session=%s delta_bytes=%d first_100=%r",
    session_id[:8], len(delta), delta[:100]
)
```

**Note**: This should be removed or reduced to DEBUG level for production.

### 3. Implicit Retry Restoration (CRITICAL)

**Problem**: Unauthorized `mark_failed` code prevented failed jobs from being retried, causing permanent orphaned nodes.

**Symptom**: Indexing stalls with orphaned leaves/roots that never get merged into the tree.

**Root Cause**: An agent added job failure tracking (`mark_failed()`, `is_failed()`, `failed_job_ids`) that permanently blacklisted failed jobs. This broke the intended implicit retry behavior.

**Fix**: Removed all unauthorized failure tracking code from `DocumentContext`:
```python
# REMOVED:
failed_jobs: int = 0
failed_job_ids: set[IndexingJob] | None = None

def mark_failed(self, job: IndexingJob) -> None: ...
def is_failed(self, job: IndexingJob) -> bool: ...

# REMOVED from _find_next_n_embedding_jobs():
if ctx is not None and ctx.is_failed(embedding_job):
    continue

# REMOVED from _find_next_n_summary_jobs():
if ctx is not None and ctx.is_failed(summary_job):
    # Skip this pair but continue scanning
    ...
    continue

# REMOVED from _run_job():
ctx.mark_failed(job)
```

**Behavior After Fix**: Failed jobs are automatically retried on the next scan because the eligible pair is still discovered. The sibling nodes have no parent, so they're still candidates for summarization.

**Test Coverage**: `tests/test_indexing_retry.py` with:
- `test_transient_failure_should_retry`: Verifies single failure → retry → success
- `test_multiple_failures_eventually_succeed`: Verifies multiple failures → eventual success

**File**: `ragzoom/server/indexing_engine.py`
