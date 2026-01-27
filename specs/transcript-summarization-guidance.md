---
status: COMPLETE
---

# Transcript Summarization Guidance

## Overview

Add conversation-specific summarization guidance to the Claude Code transcript sync. This improves summary quality by instructing the LLM to preserve the narrative structure, identity/agency, and decision outcomes that matter for conversation recall.

## Problem Statement

The current transcript sync indexes conversation turns without domain-specific guidance. The default summarizer treats conversation transcripts like generic documents, potentially losing:

- **Who said/did what** - Identity and agency in the conversation
- **Cause and effect** - Why decisions were made, not just what was decided
- **Chronological flow** - The temporal narrative of events
- **Decisions and outcomes** - Action items and conclusions vs. discussion details

## Requirements

### 1. Add summarization_guidance to BatchAppendTextRequest

Extend the proto to support guidance on batch appends:

```protobuf
message BatchAppendTextRequest {
  string document_id = 1;
  repeated AppendUnit units = 2;
  bool collect_telemetry = 3;
  optional string summarization_guidance = 4;  // NEW
}
```

This mirrors the existing `summarization_guidance` field in `AppendTextRequest`.

### 2. Thread Through gRPC Client

Add parameter to `GrpcRagzoomClient.batch_append_text()`:

```python
def batch_append_text(
    self,
    *,
    document_id: str,
    units: list[str],
    collect_telemetry: bool = False,
    timestamps: list[str | tuple[str, str]] | None = None,
    summarization_guidance: str | None = None,  # NEW
) -> IndexingResult:
```

### 3. Thread Through Wrapper

Add parameter to `RagZoom.batch_append()`:

```python
def batch_append(
    self,
    document_id: str,
    units: list[str] | list[AppendUnit],
    *,
    collect_telemetry: bool = False,
    timestamps: list[str | tuple[str, str]] | None = None,
    summarization_guidance: str | None = None,  # NEW
) -> IndexingResult:
```

### 4. Hardcoded Guidance for Conversation Transcripts

Add a constant to `transcript_sync.py`:

```python
CONVERSATION_SUMMARIZATION_GUIDANCE = """
This is a conversation transcript between a human and an AI assistant.

When summarizing, preserve:
- **Identity and agency**: Who said what, who performed which actions
- **Decisions and outcomes**: What was decided, what actions were taken
- **Cause and effect**: Why things happened, the reasoning behind decisions
- **Chronological flow**: The temporal sequence of events

Focus on the narrative of what happened and why, not just the facts.
Preserve exact technical terms, file paths, function names, and code references.
"""
```

### 5. Pass Guidance in execute_sync

Update the `batch_append()` call in `execute_sync()`:

```python
batch_append = getattr(client, "batch_append")
batch_append(
    document_id,
    non_empty,
    summarization_guidance=CONVERSATION_SUMMARIZATION_GUIDANCE,
)
```

## Implementation Outline

### Phase 1: Proto and Server

1. Add `summarization_guidance` field to `BatchAppendTextRequest`
2. Regenerate Python protobuf files
3. Update servicer to extract and pass guidance to indexing engine
4. Thread through `IndexingEngine.batch_append_text()`

### Phase 2: Client Stack

1. Add parameter to `GrpcRagzoomClient.batch_append_text()`
2. Add parameter to `RagZoom.batch_append()`
3. Add parameter to runtime `batch_append_text()` if needed

### Phase 3: Transcript Sync

1. Add `CONVERSATION_SUMMARIZATION_GUIDANCE` constant
2. Update `execute_sync()` to pass guidance
3. Integration test verifying guidance reaches summarizer

## Acceptance Criteria

1. ⬚ `BatchAppendTextRequest` proto has `summarization_guidance` field
2. ⬚ `GrpcRagzoomClient.batch_append_text()` accepts `summarization_guidance`
3. ⬚ `RagZoom.batch_append()` accepts `summarization_guidance`
4. ⬚ Guidance is threaded to the summarizer (verify via telemetry or logs)
5. ⬚ `execute_sync()` passes conversation-specific guidance
6. ⬚ Existing tests pass (no regression)

## Non-Goals

- Configurable guidance per-invocation (hardcoded is sufficient)
- Per-document stored guidance (passed per-batch)
- Guidance for non-conversation documents (other integrations can add their own)

## Dependencies

- `custom-prompt-config.md`: Establishes the `summarization_guidance` pattern (COMPLETE)
- `timestamped-transcript-sync.md`: Turn-based sync using `batch_append` (COMPLETE)
