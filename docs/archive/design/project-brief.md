# Project Brief v0.1

## 1 · Goal

Build an **incremental, hierarchical RAG** memory that returns **one chronologically ordered “dynamic summary”**:

* Global context always present (root synopsis).
* Resolution “zooms in” only where the latest query is relevant.
* Optional features: slope-capped resolution jumps, pinned nodes, sliding working-set eviction, smoothing pass.

---

## 2 · Core algorithm (MVP)

```text
A. Index-build  (append-only)
   1. Split source into leaf chunks (≈ L tokens, boundary-aware splitter).
   2. Build forest of perfect binary trees; summarise parent when both children exist.
   3. Embed every node (leaf + internal)  → 1 vector DB.

B. Runtime   (per query)
   1. Retrieve top-k = 2·N_max hits (any depth) with optional MMR diversity.
   2. Trim to N_max (derived from budget B): N_max = ⌊B / (2·L)⌋.
   3. covered ← hits ∪ ancestors(hits); OR-propagate upward.
   4. Walk frontier left→right; enforce slope_cap (±1 depth step).
   5. Concatenate frontier chunks → prompt.
   6. If smoothing_pass.enabled → inject <<UP/DOWN>> tags, run cheap model to polish joins.
```

---

## 3 · Config knobs

| Key                       | Default        | Purpose                                               |
| ------------------------- | -------------- | ----------------------------------------------------- |
| `B`                       | 8 000 t        | hard budget for stitched summary                      |
| `leaf_tokens`             | 200            | target leaf size                                      |
| `mmr_k`                   | auto (`N_max`) | post-retrieval diversification size                   |
| `slope_cap`               | `true`         | forbid jumps > 1 level                                |
| `adjacent_context_tokens` | `75` (≤ `L`)   | prev/next context fed to summariser                   |
| `smoothing_pass`          | off            | optional polish (model, max\_tokens, boundary\_tags)  |
| `pin_depth_max`           | 2              | deepest level a node may be permanently pinned        |
| `ttl_turns`               | 0 (disabled)   | working-set eviction TTL; 0 ⇒ use score queue instead |

---

## 4 · Incremental updates

* **Append-only** leaves; create parents when both children present.
* Edits/delete → bubble “dirty” flag upward and re-summarise only along that path.
* Tree may end ragged; retrieval uses `{depth, span}` metadata regardless.

---

## 5 · Optional augmentations

| Feature       | One-liner implementation                                                      |
| ------------- | ----------------------------------------------------------------------------- |
| Pinned nodes  | `pinned ∪= {node_ids}`; always marked covered.                                |
| Sliding queue | Priority = (similarity·freshness); evict low-priority nodes until tokens ≤ B. |
| Coverage TTL  | Store `ttl`, decrement each turn; remove when 0 or compress to parent.        |

---

## 6 · Evaluation plan

**Datasets**

* *Moby-Dick* (narrative), Bible subset (high-freq term stress), 100-turn chat logs.

**Baselines**

1. Rolling summary + focus buffer.
2. Leaf-only hierarchical RAG.
3. (Optional) big-window model.

**Metrics**

| Axis            | Metric                                   | Success                 |
| --------------- | ---------------------------------------- | ----------------------- |
| Answer accuracy | F1 / EM (LongRangeQA, NarrativeQA)       | +3 pp vs. best baseline |
| Thematic recall | P\@k, R\@k on hand-labeled theme queries | +10 pp recall           |
| Coherence       | Human 1-5 flow rating                    | ≥ 4.0 avg               |
| Tokens          | mean / p95                               | ≤ 1.5× rolling-summary  |
| Latency         | p95                                      | ≤ 1.2× rolling-summary  |

---

## 7 · Roadmap (4-week sprint)

| Week | Milestone                                                         |
| ---- | ----------------------------------------------------------------- |
| 1    | Chunk splitter, tree builder, embeddings, simple frontier concat  |
| 2    | MMR diversifier, budget guard, slope\_cap                         |
| 3    | Config flags, unit tests, metrics harness, score-queue eviction   |
| 4    | Smoothing pass, adjacent\_context tuning, full benchmark & report |
