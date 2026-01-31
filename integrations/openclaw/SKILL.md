---
name: ragzoom-memory
description: Set up persistent hierarchical memory using RagZoom. Use when bootstrapping an OpenClaw agent with long-term memory, configuring transcript sync, or querying historical context. Triggers on "set up memory", "install ragzoom", "persistent memory", "recall past conversations".
---

# RagZoom Memory Integration

Persistent, queryable memory for OpenClaw agents using hierarchical summarization.

## What This Provides

- **Transcript sync**: Index your session history with per-step granularity
- **Thinking preservation**: Internal reasoning (💭) is part of your memory
- **Temporal queries**: Recall what happened at specific times
- **Zoom capability**: Start broad, drill into details

## Quick Start (Already Set Up)

If RagZoom is already installed and running:

```bash
# Sync your session
cd /path/to/dynamic-summary
source .venv/bin/activate
ragzoom-openclaw sync ~/.openclaw/agents/main/sessions/<session-id>.jsonl --document-id <your-doc-id>

# Query your memory (Python)
from ragzoom_claude_code.recall import execute_recall, format_for_cli
result = execute_recall("what did we discuss about X", document_id="<your-doc-id>", server_address="localhost:50052")
print(format_for_cli(result))
```

## Full Setup (From Scratch)

### Prerequisites

- Python 3.12+
- GitHub access to the `dynamic-summary` repo (ask repo owner for access)
- OpenAI API key (for embeddings)

### Step 1: Clone and Install

```bash
# Clone the repo (requires access)
git clone git@github.com:eumemic/dynamic-summary.git
cd dynamic-summary

# Create venv and install
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e integrations/openclaw
```

### Step 2: Configure

Create `~/.config/ragzoom/.env`:

```bash
mkdir -p ~/.config/ragzoom
cat > ~/.config/ragzoom/.env << 'EOF'
OPENAI_API_KEY=sk-your-key-here
EOF
```

### Step 3: Start the Server

```bash
cd /path/to/dynamic-summary
source .venv/bin/activate
ragzoom server start --port 50052
```

For persistent operation, run in a tmux/screen session or set up as a system service.

### Step 4: Sync Your Session

Find your session file:
```bash
ls ~/.openclaw/agents/main/sessions/
```

Sync it:
```bash
ragzoom-openclaw sync ~/.openclaw/agents/main/sessions/<session-id>.jsonl --document-id <your-name>-main
```

### Step 5: Test a Query

```python
import sys
sys.path.insert(0, "/path/to/dynamic-summary/integrations/claude-code/src")
from ragzoom_claude_code.recall import execute_recall, format_for_cli

result = execute_recall(
    "what was I working on yesterday",
    document_id="<your-name>-main",
    token_budget=2000,
    server_address="localhost:50052"
)
print(format_for_cli(result))
```

## When to Use Recall

### DO use recall for:
- Questions about past work, decisions, or conversations ("what did we decide about X?")
- Finding when something happened ("when did we set up the Signal integration?")
- Remembering people, preferences, or context from previous sessions
- Cross-session queries ("what did I discuss in the group chat?")

### DON'T use recall for:
- Every message — it costs tokens and adds latency
- Things you just discussed (it's in your context window)
- General knowledge questions (use web search or your training)
- When the user is clearly asking about something new

### Best practices:
1. **Check context first** — if it's recent, it's probably already in your window
2. **Use time constraints** — if you know roughly when, add `--start`/`--end`
3. **Start with low budget** — 1000-2000 tokens usually enough, increase if needed
4. **Query your own session by default** — cross-session queries when explicitly needed

## Document ID = Session Key

Your document ID is your session key (e.g., `agent:main:main`).

### Getting Your Session Key

Call `session_status` to discover your session key:
```
🧵 Session: agent:main:main
```

### Querying Your Memory

**Always pass your session key explicitly:**
```bash
ragzoom-openclaw recall "topic" --session "agent:main:main"
```

**Pattern for agents:** Before calling recall, get your session key:
1. Call `session_status` tool
2. Extract the session key from the output
3. Pass it via `--session`

This ensures you query YOUR memory, not the default `agent:main:main`.

### Querying Another Session

```bash
ragzoom-openclaw recall "topic" --session "agent:main:signal:group:h4zieSM..."
```

## Understanding Results

Query results include `<Span>` markers:

```xml
<Span time_start="2026-01-31T19:26:38Z" time_end="2026-01-31T19:27:28Z" height=3>
Content here...
</Span>
```

- **height=0**: Verbatim transcript
- **height=1+**: Increasingly summarized
- **time_start/time_end**: Time range covered

To zoom in, query again with a constrained time range.

### The Iterative Zoom Workflow

See `skills/memory-tool-usage/SKILL.md` for the full pattern:

1. **Survey** — broad query, find relevant time ranges
2. **Zoom** — constrain to that time range, get more detail
3. **Zoom aggressively** — sub-hour windows for verbatim content

Key insight: window size determines content type. Broad = summaries, tight = verbatim.

## Ongoing Usage

### Auto-Sync with Cron

Use **system cron** (not OpenClaw's cron tool) for auto-sync. OpenClaw's isolated sessions run in sandboxed containers that can't access host tools like the Python venv.

**Step 1: Create a sync script**

```bash
cat > ~/.openclaw/workspace/bin/ragzoom-sync-all << 'EOF'
#!/bin/bash
set -e
cd /path/to/dynamic-summary
source .venv/bin/activate

SESSIONS_DIR="$HOME/.openclaw/agents/main/sessions"

# Sync your main session (find the session ID in the sessions dir)
ragzoom-openclaw sync "$SESSIONS_DIR/<your-session-id>.jsonl" \
    --document-id "agent:main:main" 2>/dev/null || true

# Add more sessions as needed:
# ragzoom-openclaw sync "$SESSIONS_DIR/<other-session>.jsonl" \
#     --document-id "agent:main:signal:group:..." 2>/dev/null || true
EOF

chmod +x ~/.openclaw/workspace/bin/ragzoom-sync-all
```

**Step 2: Install system crontab**

```bash
crontab -e
```

Add this line (syncs every 2 minutes):
```
*/2 * * * * ~/.openclaw/workspace/bin/ragzoom-sync-all >> /tmp/ragzoom-sync.log 2>&1
```

**Why not OpenClaw cron?**

OpenClaw's `cron` tool with `sessionTarget: "isolated"` spawns sandboxed Docker containers (when sandbox mode is enabled). These containers can't access host paths like `/path/to/dynamic-summary/.venv`. System cron runs on the host with full filesystem access.
```

### Finding Your Session Key

Use `session_status` to discover your session key:
```
🧵 Session: agent:main:main
```

Use this as your document ID for sync and recall.

### CLI Recall

```bash
# Basic query
ragzoom-openclaw recall "what did we discuss about X"

# With time constraints (zoom in)
ragzoom-openclaw recall "topic" --start 2026-01-31T14:00:00Z --end 2026-01-31T15:00:00Z

# Query another session
ragzoom-openclaw recall "topic" --session "agent:main:signal:group:..."

# Higher budget for more detail
ragzoom-openclaw recall "topic" --budget 3000
```

### Server Management

```bash
# Check if running
curl -s localhost:50052/health || echo "not running"

# Start (from dynamic-summary dir with venv active)
ragzoom server start --port 50052

# The server runs in foreground; use tmux/screen for persistence
```

## Troubleshooting

### "Connection refused" on queries
Server isn't running. Start it with `ragzoom server start --port 50052`.

### "Lease already held" error
Another sync process died mid-operation. Clear the lease:
```bash
sqlite3 /path/to/dynamic-summary/data/sqlite.db "DELETE FROM indexer_leases WHERE document_id='<your-doc-id>'"
```

### Sync shows 0 new steps
Already synced. Check state file in `data/openclaw-state/`.

### Missing OpenAI key
Ensure `~/.config/ragzoom/.env` exists with `OPENAI_API_KEY=sk-...`.
