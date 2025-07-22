## How the “Refine‑Until‑Budget” Algorithm Works

*A narrative walk‑through using two concrete queries*

---

### 1 · Bird’s‑eye view

Think of your document tree as a **topographic map**:

* **Leaves** = fine‑grained tiles (≈ 200 tokens of raw text).
* **Internal nodes** = progressively zoomed‑out mini‑maps (pre‑written synopses).
* Each tile carries a **relevance mass** for the current query—much like rainfall intensity on a weather map.

The algorithm’s job is to carve out one continuous strip of map whose **total ink (tokens) fits the caller’s budget**, using **thicker strokes in stormy areas (high relevance)** and a faint outline where it’s calm.

The flow is always the same:

1. **Measure** how much “storm” (relevance) falls on every tile.
2. **Budget** tokens proportionally to that storm intensity.
3. **Refine** any tile that got more tokens than its current synopsis length, splitting it into its children, and recurse.
4. **Insert pins** (if any) and repaint local neighbors to stay within budget.
5. **Smooth** the seams with a few connective words so the final panorama reads like a narrative, not a collage.

Because the process begins with the **entire root synopsis** and only zooms in, *coverage never breaks* and *the budget is never overspent*.

---

### 2 · Needle‑in‑a‑Haystack (“Gazebo”) Walk‑through

> **Scenario** A 250‑k‑token novel.
> The word **“gazebo”** appears exactly once, deep in Chapter 28 during a two‑paragraph childhood flashback (one leaf chunk).
> Query `gazebo`
> Budget *B* = 1 000 tokens.

| Stage                 | What happens                                                                                                                                                                                 | Outcome                                                                                                                                                                                                                                            |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Leaf scan**         | The lone leaf containing “gazebo” scores **1.00** relevance.  Its siblings (same scene) also get spill‑over relevance (≈ 0.3).  All other 1 249 leaf chunks get near‑zero.                   | Relevance mass of the gazebo leaf ≈ 200; entire rest of book ≈ 150.                                                                                                                                                                                |
| **Mass propagation**  | Parents of the gazebo leaf inherit that 200‑token mass upward to the root.                                                                                                                   | The *gazebo line* gives its ancestral branch much higher mass than the rest of the tree.                                                                                                                                                           |
| **Budget allocation** | Tokens distributed ∝ mass.  Calculation:   *B* × M(branch)/ΣM ≈ 1 000 × 200/350 ≈ 571 tokens to the Chapter 28 branch.  The remaining 429 tokens spread thinly across the other 27 chapters. | Almost 60 % of the budget is earmarked for the gazebo scene’s span.                                                                                                                                                                                |
| **Refine‑to‑Target**  | *Gazebo branch* keeps splitting: root → Part → Book → Act → Chapter → Scene → **leaf**, because each parent’s synopsis (200 tokens) is bigger than its token allotment *except at the leaf*. | The final frontier holds the *exact leaf chunk* describing the gazebo (≈ 200 tokens) plus a medium‑resolution synopsis of the surrounding chapter (≈ 370 tokens) and compressed one‑sentence summaries for every other chapter (≈ 15 tokens each). |
| **Smoothing**         | The algorithm inserts “*Later,*” and “*Meanwhile,*” between era jumps.                                                                                                                       | The text flows as: high‑level intro → fast‑forward through early plot → graceful zoom into Chapter 28 → verbatim gazebo flashback → quick wrap‑up.                                                                                                 |
| **Budget check**      | 200 + 370 + (26 × 15) = 960 tokens ≤ 1 000.                                                                                                                                                  | ✔️ All P0 requirements satisfied; the needle is fully preserved.                                                                                                                                                                                   |

**Reader experience**

> *“…After a whirlwind of early campaigns, the Duke returns home. Later, in the quiet gardens of Duchenne Manor, he pauses by a weather‑worn **gazebo**. Its latticework transports him back to sun‑bleached afternoons of childhood…”*

The gazebo scene appears in full; everything else is contextually skimmed.

---

### 3 · Dense‑Hit (“Jesus”) Walk‑through

> **Scenario** The entire Christian Bible (\~750 k tokens, 1 500 leaves).
> Query `Jesus`
> Budget *B* = 8 000 tokens.

| Stage                 | What happens                                                                                                                                                                                                                                                                                                                                        | Outcome                                                                                                                                                                                                                                                  |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Leaf scan**         | Nearly every New‑Testament leaf (Gospels, Acts, Epistles) contains “Jesus” and scores high.  Old‑Testament leaves score ≈ 0.05 (prophecies) or zero.                                                                                                                                                                                                | New Testament mass ≈ 500 k; Old Testament mass ≈ 40 k.                                                                                                                                                                                                   |
| **Budget allocation** | 8 000 × (500/540) ≈ 7 400 tokens go to NT subtree; \~600 tokens to OT.                                                                                                                                                                                                                                                                              | 92 % of the budget flows to the NT.                                                                                                                                                                                                                      |
| **Refine‑to‑Target**  | *New Testament* keeps refining down to **book‑level** synopses (e.g., one for Matthew, one for Mark…) and then further to **per‑section** synopses inside the Gospels because the allotment (≈ 1 200 tokens per Gospel) exceeds the 200‑token synopsis size.  Leaves are **not** shown because each section synopsis already fits its budget share. | Frontier includes Gospels’ section‑level synopses (say 80 × 120 tokens = 9 600 tokens, but many are collapsed), Acts/Epistles at book‑level, Revelation short, and an ultra‑compressed Old Testament overview (\~600 tokens). Final size ≈ 7 960 tokens. |
| **Smoothing**         | Connectives (“*Centuries earlier,*”, “*In Paul’s letters,*”) bridge OT‑NT and book transitions.                                                                                                                                                                                                                                                     | Reads like a cohesive high‑speed commentary, heavy on Gospel narrative.                                                                                                                                                                                  |
| **Budget check**      | 7 960 ≤ 8 000. Coverage intact.                                                                                                                                                                                                                                                                                                                     | ✔️ Requirements met; level of detail mirrors relevance spread.                                                                                                                                                                                           |

**Reader experience**

> *“…Centuries earlier, prophets alluded to a coming Messiah. Fast‑forward to the Gospel of Matthew: Jesus is born in Bethlehem, escapes to Egypt, and later proclaims the Sermon on the Mount in Galilee (120 tokens)… Mark condenses the same ministry with urgency (110 tokens)… Luke adds childhood anecdotes and parables of mercy (130 tokens)… John focuses on divinity, opening with the cosmic Logos (125 tokens)…*”

No single verse is quoted verbatim, but the New Testament receives thousands of tokens of mid‑level summary, dwarfing the Old Testament’s single paragraph—exactly in proportion to query relevance.

---

### 4 · Why *n‑max* Is Irrelevant in Both Cases

* **Gazebo** – Frontier ends with ≈ 30 nodes; raising or lowering a separate *n‑max* cap would only force pre‑mature merging or unnecessary splits, harming either coverage or budget.
* **Jesus** – Frontier may have \~120 synopses, but all belows the 8 000‑token ceiling and produced in < 150 ms.  A mechanical node limit would cut detail in the very part of the text users care about.

Since latency is already bounded by the token budget and logarithmic steps, controlling the **number** of nodes offers no extra safety but would block legitimate output shapes.

---

### 5 · Take‑aways

1. **Budget obeyed from the first split onward.** No trimming phase sneaks in overflow.
2. **Density drives detail**, not absolute mention counts—hence a single spike is preserved verbatim, while widespread mentions get summarised.
3. **Pinning slots right in.** Had the user pinned, say, Psalm 22, that node would reserve its synopsis inside the 600 OT tokens without breaking anything.
4. **Smoothness costs pennies.** Two‑ or three‑word connectors polish flow while counting toward the same budget.

In both edge cases the reader receives *one fluid narrative* whose granularity feels tailor‑made to the query—yet the algorithm never had to juggle special‑case rules or post‑hoc repairs.
