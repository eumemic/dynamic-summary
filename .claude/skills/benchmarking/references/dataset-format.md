# LoCoMo Dataset Format

## Source

Dataset: https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
Paper: https://arxiv.org/abs/2402.17753 (Snap Research, ACL 2024)

## Format Gotchas

The actual LoCoMo JSON format differs from what documentation and research papers suggest. These were discovered empirically by parsing `locomo10.json`.

### Top-Level Structure

The file is a JSON array of conversation objects:

```json
[
  {
    "sample_id": 26,
    "conversation": {
      "session_1": [...],
      "session_1_date_time": "1:56 pm on 8 May, 2023",
      "session_2": [...],
      "session_2_date_time": "3:22 pm on 15 May, 2023"
    },
    "qa": [...]
  }
]
```

### Sessions Are Nested

Sessions live under a `conversation` sub-dict, NOT at the top level. The parser must extract `raw["conversation"]` before scanning for `session_N` keys.

### Turn Fields Are `speaker`/`text`, Not `role`/`content`

```json
{
  "speaker": "Caroline",
  "text": "Hey Mel! Good to see you!",
  "dia_id": "D26:1:1"
}
```

NOT the `role`/`content` format common in chat APIs.

### Sessions Are 1-Indexed

Session keys start at `session_1`, not `session_0`. The regex `session_(\d+)` captures the index.

### Categories Are 1-Indexed (1-5)

```
1 = SINGLE_HOP
2 = MULTI_HOP
3 = TEMPORAL
4 = OPEN_DOMAIN
5 = ADVERSARIAL
```

### Category 5 Uses `adversarial_answer`

Adversarial QA pairs use the key `adversarial_answer` instead of `answer`:
```json
{
  "question": "...",
  "adversarial_answer": "...",
  "category": 5
}
```

The parser falls back: `qa.get("answer") or qa.get("adversarial_answer")`.

### Timestamps Are Human-Readable, Not ISO 8601

```
"session_1_date_time": "1:56 pm on 8 May, 2023"
```

The ingestion adapter converts these to ISO 8601 using a regex parser that handles:
- 12-hour time with am/pm
- Day-Month-Year ordering
- Month names (January through December)
- Optional comma before year

Regex: `(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})`

### Some QA Pairs Have No Category

A small number of QA pairs have `null` category. The parser skips these.

### Speaker Names Not at Top Level

`speaker_a`/`speaker_b` fields are NOT present in `locomo10.json`. Speaker names are inferred from the first session's turns by extracting unique speakers in order.

## Dataset Statistics

From `locomo10.json`:
- 10 conversations
- 5,882 total turns
- 1,986 QA pairs total
- 1,540 non-adversarial QA pairs (used for evaluation)
- ~300-600 turns per conversation
- Up to 35 sessions per conversation

## Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Open-domain | 841 | 54.6% |
| Multi-hop | 321 | 20.8% |
| Single-hop | 282 | 18.3% |
| Temporal | 96 | 6.2% |
