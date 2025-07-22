# Smart Synopsis System – North‑Star Requirements

This document defines the **non‑negotiable outcomes** (“P0”) and the strong but secondary goals (“P1–P2”) for any design of the Smart Synopsis System.  It does **not** prescribe algorithms or internal data structures; it states *what* the system must deliver, regardless of *how*.

---

## 1 · Problem Statement

Given:

* **A large source document** (potentially hundreds of thousands of tokens) that can be treated as a **chronological sequence**.
* **A user query** expressing what aspects of the document are currently relevant (e.g. a keyword, topic, or natural‑language question).
* **A strict token budget *B*** – the maximum number of tokens the synopsis may consume (to fit into a downstream LLM context window).

Produce, on demand, **one synopsis** that satisfies all requirements below.

---

## 2 · Primary (“P0”) Requirements — *Must Always Hold*

| ID       | Requirement                                                                                                                                                                                                                                                            | Rationale                                                                                 |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **P0‑1** | **Complete Coverage** – Every part of the source document is represented in the synopsis, in the correct chronological order, with **no overlaps or gaps** (though low‑relevance spans may be summarised very coarsely).                                               | Ensures global context is never lost; enables the synopsis to stand in for the full text. |
| **P0‑2** | **Detail ∝ Relevance** – The number of tokens devoted to any span is proportional to that span’s **semantic relevance density** with respect to the query.  Highly relevant passages receive more space (even down to raw text); low‑relevance regions are compressed. | Delivers focused depth where the user cares, while preserving completeness elsewhere.     |
| **P0‑3** | **Budget Guarantee** – The synopsis, including any connective tissue or metadata, **must not exceed the caller’s budget *B***.  If necessary, coarser summarisation or truncation is applied.                                                                          | Enables predictable use inside fixed‑size LLM contexts.                                   |

---

## 3 · Secondary Requirements

| Priority | ID                                | Requirement                                                                                                                                                                                                      | Notes |
| -------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| **P1**   | **R‑4 Smooth Readability**        | The synopsis should flow naturally, avoiding abrupt subject or tense shifts.  Brief connective phrases may be inserted, but they count against the same budget *B*.                                              |       |
| **P2**   | **R‑5 Minimal Extraneous Detail** | While complete coverage is mandatory, the synopsis should avoid including information that is both low‑relevance **and** redundant.  Compression techniques may be used, provided P0‑1 to P0‑3 remain satisfied. |       |

---

## 4 · Relevance Distribution Scenarios (Informative)

* **Isolated Spike** – A single, short scene contains the only mention of “gazebo.”
  *Expectation*: That leaf scene appears largely verbatim; the rest of the novel collapses into broad chapter‑level summaries.

* **Broad Band** – A concept like “Jesus” appears throughout large portions of the document.
  *Expectation*: Most of the budget is spent on medium‑resolution synopses of the heavily referenced sections (e.g. the Gospels), while earlier, less relevant spans (Old Testament) are condensed into a very brief preface.

These scenarios illustrate the “detail ∝ relevance” requirement but do not constrain implementation.

---

## 5 · Out‑of‑Scope (for this document)

* Specific data structures (trees, graphs, sliding windows, etc.).
* Index‑update strategies or real‑time streaming requirements.
* User‑tunable concision knobs beyond the single token‑budget parameter *B*.
* Historical compatibility with any prior CLI flags (e.g., `--n-max`).

---

### 6 · Summary

Any future algorithm or system claiming to implement the Smart Synopsis capability **must be judged solely** on whether every produced synopsis:

1. Covers the **entire** source chronologically without overlap,
2. Allocates **more words to more relevant spans**,
3. **Never** breaches the caller’s token budget,
4. **Reads smoothly**, and
5. Minimises low‑value filler where feasible.

All design choices are free—so long as these north‑star requirements are met.
