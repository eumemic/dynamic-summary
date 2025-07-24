# Implementation Notes v0.1

## 1 · Architecture & tech stack

| Question                  | Pragmatic default                                                                                                   | Why                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Language / framework      | **Python 3.10 + FastAPI** (CLI entry point wraps the same service funcs)                                            | Most RAG libs & embeddings are Python-first; FastAPI is lightweight if you later expose endpoints. |
| Vector DB                 | **Chroma** for dev (zero-ops, pure Python).  Swap to pgvector / Pinecone if you need horizontal scale.              | Start local → upgrade only when perf/ops demand.                                                   |
| Embedding model           | **OpenAI `text-embedding-3-small`** (or `ada-002` fallback).  If offline: `sentence-transformers/all-MiniLM-L6-v2`. | Good recall/price; ecosystem support.                                                              |
| LLM for summarising nodes | **GPT-4o** (best quality); config flag to fall back to GPT-3.5 / local (`mistral-7B-instruct`) for cost.            | Summary quality is downstream-critical; start with highest-Q.                                      |

---

## 2 · Data-structure details

| Topic          | Default                                                                                                                                            |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tree storage   | **SQLite (or Postgres)** table: `id, parent_id, depth, span_start, span_end, text, embed_vector`.  Keep a small LRU in-memory cache for hot nodes. |
| Persistence    | DB handles it; just dump new leaves/parents as you append.                                                                                         |
| Doc size range | Tested up to **\~100 MB** per doc (≈ 50 k leaf chunks). Beyond that you’ll shard per-chapter/file first.                                           |

---

## 3 · API design

| Choice              | Default                                                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Surface             | **Importable Python library** (`ragzoom.*`) **+ optional REST layer** (FastAPI) for remote calls.                           |
| Integration pattern | Can run embedded inside an agent process (direct Python calls) **or** as a micro-service (REST).  Same core code both ways. |

---

## 4 · Algorithm clarifications

| Question                        | Starter value                                                                                                                                                                                                        |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Boundary-aware splitter         | Use **LangChain `RecursiveCharacterTextSplitter`** with separators `["\n\n", "\n", ". ", " "]`, leaf size ≈ 200 tokens, 20-token overlap.                                                                            |
| MMR diversity λ / k             | `λ = 0.7` (relevance 70 %, novelty 30 %); request `2 × N_max` hits, then MMR down to `N_max`.                                                                                                                        |
| Parent-summary prompt           | `Summarise the passage in ≤{target_tokens} tokens, third-person past tense, no pronouns, keep names.`  Prepend last *adjacent\_context\_tokens* of previous chunk & first *adjacent\_context\_tokens* of next chunk. |
| Freshness score (sliding queue) | `priority = sim_at_fetch × 0.9^(turns_since_fetch)` (0.9 ≈ “10 % decay per turn”).  Evict lowest priority when token budget exceeded.                                                                                |

---

## 5 · Development priorities

| Q               | Recommendation                                                                                                                                                                                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MVP vs. options | **Ship MVP first**: tree build, frontier concat, MMR, budget guard.  Flags (`slope_cap`, pinned, smoothing, TTL) can be toggled but implement later.                                                                                                                    |
| Test datasets   | ✅ *Moby-Dick* (Gutenberg), mini-Bible subset, 100-turn chat log (any Slack/Discord export).                                                                                                                                                                             |
| Code structure  | `ragzoom/`<br> • `splitter.py`<br> • `index.py` (tree builder)<br> • `store.py` (DB, vector ops)<br> • `retrieve.py` (MMR, frontier)<br> • `assemble.py` (slope-cap, concat, smoothing)<br> • `config.py` (pydantic settings)<br> • `api.py` (FastAPI routes)<br>tests/ |
