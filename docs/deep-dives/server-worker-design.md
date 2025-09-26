# RagZoom Server-Managed Worker Model

## Purpose

This document defines the architecture for running RagZoom indexing and summarization inside a single, long-lived gRPC server. It supersedes the prior CLI-driven, process-local indexing flow by making the server authoritative for database and vector-store mutations, queue coordination, and background summarization. The spec is self-contained so any engineer can implement or extend the system without additional context.

## Goals

- **Single source of truth:** The gRPC server exclusively mutates the relational database and vector store. All CLI and API clients interact with the server via RPC, never by touching the stores directly.
- **Correct-by-construction appends:** After an append RPC returns, the document’s leaf layer fully reflects the new text span and can be queried immediately.
- **Eventual summarization:** Background workers continuously build parents above the leaves. They may lag behind append throughput, but converge to a complete tree when load subsides.
- **Crash resilience:** A server restart can discover the current tree state and resume work without manual intervention or data loss.
- **Telemetry parity:** Existing telemetry and CLI experiences (`ragzoom index`, `--telemetry`, etc.) continue to function via server-managed telemetry runs and the `GetTelemetry` RPC.

## Non-Goals

- Implementing multi-process or distributed workers. The server runs workers in-process using asyncio.
- Retrying LLM or embedding calls beyond the existing summarization retry logic (e.g., retries triggered by non-200 responses).
- Introducing a persistent job queue table. The database nodes themselves encode pending work.

## High-Level Architecture

```
+--------------------+       append/query gRPC       +------------------------+
| CLI / API clients  |  <--------------------------> | RagZoom gRPC Server    |
| (ragzoom cli, etc) |                               |                        |
+--------------------+                               |  +------------------+  |
                                                     |  | WorkerCoordinator|  |
                                                     |  +------------------+  |
                                                     |     ^          ^       |
                                                     |     | poke()   | pop() |
                                                     |  +--------+ +-------+  |
                                                     |  |Queue   | |Workers|  |
                                                     |  +--------+ +-------+  |
                                                     +------------------------+
                                                            |          |
                                                            v          v
                                             +---------------------+ +-------------------+
                                             | Relational Store    | | Vector Store      |
                                             +---------------------+ +-------------------+
```

Key principles:

- **Tree-as-truth:** Leaf and interior nodes persist in the configured relational storage backend (file-backed SQLite by default, Postgres in production deployments). Work readiness is derived from the current nodes, not from a separate queue table.
- **In-memory coordination:** An asyncio queue distributes work to worker tasks. A coordinator coroutine populates the queue by inspecting the database, mirroring today’s `poke()` semantics.
- **Idempotent workers:** Each worker validates that its target nodes still exist before mutating anything, so appends that delete a path cannot produce stale summaries.

## Data Model Updates

The design supports any `StorageBackend` implementation. In development the server
runs against file-backed SQLite; production deployments typically switch to
Postgres. Likewise, the default `VectorIndex` remains Chroma, but the flow only
assumes the ability to upsert and delete embeddings by node identifier.

No new tables are required. We rely on existing `nodes` schema fields:

- `span_start`, `span_end`
- `left_child_id`, `right_child_id`
- `preceding_neighbor_id`, `following_neighbor_id`
- `parent_id`

### Document Span Tracking

We extend `documents` (or equivalent metadata table) with `span_end` to store the current terminal token offset. Append updates this value so workers know whether a node should have a following neighbor.

## Readiness Predicate

A node qualifies as the **left child** of a missing parent when all of the following are true:

1. `parent_id IS NULL`
2. `span_end - span_start > 0` (non-empty range)
3. `span_start > 0` implies `preceding_neighbor_id IS NOT NULL`
4. `span_end < document_span_end` implies `following_neighbor_id IS NOT NULL`

Symmetrically, the **right child** predicate mirrors the neighbor checks. These invariants ensure both children are present and linked before we build the parent. The dispatcher queries for nodes that satisfy the left-child predicate; the parent-building worker then loads both siblings.

## Append Flow

1. **Receive RPC:** `AppendText(document_id, content, collect_telemetry)` on `IndexerService`.
2. **Load trailing leaf:** Fetch the rightmost leaf (`span_end == document_span_end`). Extract its text and delete it along with all ancestors up to the root.
3. **Chunking:** Prepend the trailing leaf text to the incoming payload and split using the existing chunker.
4. **Transactional mutation:** Within a single DB transaction:
   - Delete the previous rightmost leaf and ancestors.
   - Insert new leaf nodes with accurate neighbor pointers to existing leaves.
   - Update document span metadata.
5. **Synchronous embeddings:** Batch embeddings for all new leaves before committing. Handle provider limits by slicing into multiple sequential batches. Insertion order:
   - Write all leaf nodes.
   - Generate embeddings for those leaf IDs.
   - Insert embeddings into the vector store.
   - On any failure, delete any embeddings already written, roll back the DB transaction, and surface an error.
6. **Telemetry capture:** Collect append-phase telemetry (e.g., chunk count, tokens ingested) for CLI reporting.
7. **Wake workers:** `poke()` the coordinator so it enqueues parents of the new leaves.
8. **Response:** Return `AppendTextResponse` containing leaf stats and telemetry identifiers. At this point queries see the extended document because the leaf layer and embeddings are committed.

## WorkerCoordinator & Queue

- Maintains an asyncio `PriorityQueue` keyed by `(span_depth, span_start)` to favor lower levels first (bottom-up build).
- Tracks `in_flight` node IDs to avoid duplicate enqueue.
- On startup:
  1. Query readiness predicate for all documents.
  2. Enqueue all qualifying left-child nodes.
  3. Spawn worker tasks (configurable count).
- `poke(document_id, span_range)` re-runs the readiness query for the affected region and enqueues new candidates.
- Workers listen on the queue, validate readiness again, perform the work, then call `poke()` for the parent’s parent.

Pseudo-code for enqueue:

```python
def enqueue_ready_nodes(doc_id: str, span_start: int, span_end: int) -> None:
    rows = repo.fetch_ready_left_children(doc_id, span_start, span_end)
    for row in rows:
        if row.node_id not in in_flight:
            queue.put_nowait(row)
            in_flight.add(row.node_id)
```

When a worker finishes or aborts, it removes the node from `in_flight`.

## Worker Lifecycle

Per worker task:

1. Pop candidate from queue.
2. Re-validate readiness: ensure both children exist, still share the expected span, and each child’s `parent_id` is `NULL`.
3. Generate summary via existing summarization pipeline (with its internal retry loop for target token mismatches).
4. Confirm children still exist and share the expected generation (detecting append deletions).
5. Insert the parent node within a transaction:
   - Write `parent` row with `span_start`, `span_end`, neighbor pointers.
   - Update `left_child.parent_id` and `right_child.parent_id`.
   - Update neighboring parents’ `preceding_neighbor_id`/`following_neighbor_id` if applicable.
6. Embed the parent text (batch size 1 is acceptable) and insert into vector store. On failure roll back the transaction and delete any vectors written in this step.
7. Commit transaction.
8. Emit telemetry event (nodes processed, depth, duration) for streaming consumers.
9. `poke()` the coordinator for the newly created parent’s parent span.

### Failure Modes

- **Child deleted mid-flight:** DB update fails or readiness check fails. Worker logs and returns. No requeue—append will enqueue the new path when needed.
- **Embedding failure:** Transaction rolls back, worker raises. Coordinator leaves node off the queue; the next append or readiness scan will reintroduce it once the system stabilizes.
- **LLM refusal / invalid summary:** Existing summarization retry logic applies. Persistent failures bubble up to telemetry and logs for manual intervention.

## CLI Semantics

- `ragzoom index [FILE]`: uploads content via append RPC, then waits for the worker queue to drain.
  1. Call `AppendText` for the full document (reusing streaming upload if needed).
  2. Subscribe to `RunWorkers(mode=UNTIL_IDLE)` to receive progress events until `queue_depth == 0`.
  3. If `--telemetry` flagged, capture the `telemetry_run_id` from the response and, once `RunWorkers` reports idle, call `GetTelemetry(wait=true)` to persist the returned JSON.
- `ragzoom index --async`: only runs the append RPC, prints basic stats, and exits. `--telemetry` is invalid with `--async` (CLI should error and explain).

## gRPC Service Adjustments

- `IndexerService.AppendText`: Implements the new flow and returns `DocumentStats` that include counts for inserted leaves and queued parents.
- `WorkerService.RunWorkers`:
  - `UNTIL_IDLE`: Streams periodic `RunWorkersResponse` messages containing queue depth, active worker count, processed node IDs, latest error message, and optional telemetry chunk references. Completes when queue empty for the target document(s).
  - `CONTINUOUS`: Streams indefinitely until client cancels.
- `WorkerService.GetDocument`: Returns `DocumentStatus` with:
  - `leaf_count`
  - `pending_work_count` (derived from readiness query)
  - `tree_depth`
  - Last failure (if any)
  - `document_span_end`

## Telemetry

- Append emits ingestion telemetry (chunks created, tokens, cost estimates).
- Workers publish node-level telemetry events (depth, duration, retries) to the existing telemetry sink.
- The server stores telemetry per run; the CLI waits for `RunWorkers` to go idle and then retrieves the finalized JSON via `GetTelemetry`.

## Error Handling & Atomicity

- **Append compensation:** Embeddings are written after the DB transaction commits. On vector-store failure we issue explicit deletes for the newly inserted vector IDs, then raise an RPC error. The database transaction is rolled back so the tree remains on the previous span.
- **Worker compensation:** All DB mutations and vector writes occur inside one transaction. On any exception, we delete vectors written in this attempt (if any) and roll back the transaction. The candidate will be rediscovered once conditions are satisfied again.
- **Crash recovery:** Because node state is authoritative, a restart only needs to rerun the readiness query. No persistent queue cleanup is necessary.

## Implementation Plan (Suggested Order)

1. **Schema & metadata updates**
   - Add `document_span_end` (if not already present).
   - Ensure neighbor pointers are indexed for readiness queries.
2. **Server coordinator scaffolding**
   - Implement dispatcher queue + worker lifecycle with placeholders.
   - Wire coordinator into server startup/shutdown.
3. **Append RPC refactor**
   - Implement transactional leaf replacement + synchronous embedding writes.
   - Ensure compensation for vector-store failures.
4. **Parent builder workers**
   - Implement readiness queries, generation checks, neighbor wiring.
   - Integrate existing summarization pipeline.
5. **gRPC API updates**
   - Finalize `RunWorkers` streaming payloads.
   - Update CLI client to block on queue drain, add `--async`, and enforce telemetry rules.
6. **Telemetry alignment**
   - Ensure append + worker events feed the telemetry pipeline used by CLI and benchmarks.
7. **Testing**
   - Unit tests for append splitting, neighbor wiring, and worker readiness checks.
   - Integration test covering append → query → worker completion (with simulated worker crash/restart).
   - Regression tests for CLI synchronous flow and telemetry output.

## Open Questions

- **Dispatcher triggers:** We rely on `poke()` calls from append and worker completion. If we find missed pokes in practice, we may add a periodic reconciliation scan.
- **Backpressure:** Append currently has no throttle. If queue depth grows unbounded we may need to expose warnings or reject appends.
- **Embedding concurrency:** For now we batch sequentially per append. Future work could parallelize across embedding API limits without changing the architecture.

## Summary

This design keeps RagZoom’s database and vector store consistent under continuous appends, replaces the previous single-shot indexing run with a server-managed worker loop, and preserves existing CLI semantics and telemetry. Because workers derive their backlog from the persisted node state, the system recovers automatically from crashes or missed wake-ups. The spec aims to be fully actionable for engineering teams implementing the new flow.
