---
status: READY
---

# Custom Prompt Config for Indexing Summaries

## Overview

Allow users to customize the system prompt used during summary generation. This enables better summaries for domain-specific data by providing domain context to the LLM.

## Goals

1. **Domain adaptation** - Users can tune summaries for specific content types (legal, medical, code, etc.)
2. **Simple configuration** - Single field in IndexConfig, no complex prompt templates
3. **Safe defaults** - Works out of the box with current behavior

## Non-Goals

- Customizing user prompt templates (the compression instructions)
- Per-document prompt overrides
- Prompt versioning or A/B testing
- Customizing contextualization prompts (future work)

## Configuration

### IndexConfig Field

Add `summary_system_prompt` to IndexConfig:

```python
@dataclass
class IndexConfig:
    # ... existing fields ...

    summary_system_prompt: str | None = None
    """Custom system prompt for summary generation.

    If None, uses the default: "You are a text compressor. You compress
    sections of documents while preserving their meaning. You output
    ONLY the compressed text, nothing else."

    For domain-specific content, provide context about the domain to
    improve summary quality. Always include the instruction to output
    only compressed text.
    """
```

### Config File

```yaml
# indexing.yaml
summary_model: gpt-4o-mini
summary_system_prompt: |
  You are an expert legal document summarizer. You compress sections of
  legal documents while preserving their precise meaning and legal
  terminology. You output ONLY the compressed text, nothing else.
```

### CLI Override

```bash
ragzoom index document.txt --summary-system-prompt "You are a medical note summarizer..."
```

## Default Prompt

Current hardcoded prompt becomes the default:

```
You are a text compressor. You compress sections of documents while
preserving their meaning. You output ONLY the compressed text, nothing else.
```

## Implementation

### summary_utils.py Changes

Update `prepare_summary_inputs` to accept optional system prompt:

```python
def prepare_summary_inputs(
    *,
    text: str,
    target_tokens: int,
    prev_context: str | None = None,
    text_tokens: int | None = None,
    use_anti_verbatim_vaccine: bool = False,
    system_prompt: str | None = None,  # NEW
) -> SummaryPreparation:
    # ... existing logic ...

    default_system = (
        "You are a text compressor. You compress sections of documents while "
        "preserving their meaning. You output ONLY the compressed text, nothing else."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt or default_system},
        {"role": "user", "content": full_prompt},
    ]
    # ...
```

### Workflow Changes

Thread `summary_system_prompt` from IndexConfig through:
1. `SummaryWorkflowConfig` - add field
2. `run_summary_workflow` - pass to `prepare_summary_inputs`
3. `run_summary_from_config` - extract from IndexConfig

### Telemetry

Update `_get_system_prompts()` in telemetry_collection.py to record the actual prompt used (custom or default) for reproducibility.

## Examples

### Legal Documents

```yaml
summary_system_prompt: |
  You are a legal document summarizer specializing in contracts and agreements.
  Preserve exact legal terminology, party names, dates, and obligations.
  Output ONLY the compressed text, nothing else.
```

### Code Documentation

```yaml
summary_system_prompt: |
  You are a technical documentation summarizer for software projects.
  Preserve function names, API references, and technical terminology exactly.
  Output ONLY the compressed text, nothing else.
```

### Meeting Notes

```yaml
summary_system_prompt: |
  You are a meeting notes summarizer. Preserve action items, decisions,
  and participant names. Focus on outcomes over discussion.
  Output ONLY the compressed text, nothing else.
```

## Guidelines for Custom Prompts

1. **Always include output instruction** - End with "Output ONLY the compressed text, nothing else."
2. **Keep it concise** - System prompt adds to every API call's token count
3. **Focus on domain context** - Describe what makes this content special
4. **Preserve terminology** - Instruct preservation of domain-specific terms

## Testing

### Unit Tests

- Default prompt used when `summary_system_prompt` is None
- Custom prompt passed through to LLM messages
- Custom prompt recorded in telemetry

### Integration Tests

- Index with custom prompt, verify summaries reflect domain knowledge
- Verify telemetry captures custom prompt for reproducibility

## Migration

No migration needed - new optional field with backward-compatible default.
