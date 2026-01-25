---
status: COMPLETE
---

# Custom Prompt Config for Indexing Summaries

## Overview

Allow users to provide additional guidance for summary generation. This enables better summaries for domain-specific data by providing domain context to the LLM, while preserving the essential default behavior.

## Goals

1. **Domain adaptation** - Users can tune summaries for specific content types (legal, medical, code, etc.)
2. **Simple configuration** - Single field in IndexConfig, no complex prompt templates
3. **Safe defaults** - Works out of the box with current behavior
4. **Additive, not replacement** - Custom guidance appends to (not replaces) the default prompt

## Non-Goals

- Customizing user prompt templates (the compression instructions)
- Per-document prompt overrides
- Prompt versioning or A/B testing
- Customizing contextualization prompts (future work)

## Configuration

### IndexConfig Field

Add `summarization_guidance` to IndexConfig:

```python
@dataclass
class IndexConfig:
    # ... existing fields ...

    summarization_guidance: str | None = None
    """Additional guidance for summary generation.

    If provided, this guidance is appended to the default system prompt
    under a "# Summarization Guidance" section. The default prompt's
    essential instructions (output only compressed text) are preserved.

    Use this to provide domain context that improves summary quality
    for specific content types (legal, medical, code, etc.).
    """
```

### Config File

```yaml
# indexing.yaml
summary_model: gpt-4o-mini
summarization_guidance: |
  This document contains legal contracts. Preserve exact legal terminology,
  party names, dates, and obligations. Be especially careful with defined
  terms (capitalized phrases that have specific meanings in the contract).
```

### CLI Override

```bash
ragzoom index document.txt --summarization-guidance "This is medical documentation. Preserve drug names, dosages, and clinical terminology exactly."
```

## System Prompt Structure

The system prompt is constructed as follows:

```
You are a text compressor. You compress sections of documents while
preserving their meaning. You output ONLY the compressed text, nothing else.

# Summarization Guidance
{user's custom guidance here, if provided}
```

When no custom guidance is provided, only the default prompt is used (no empty section).

## Implementation

### constants.py Changes

Define the base system prompt as a template:

```python
DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "You are a text compressor. You compress sections of documents while "
    "preserving their meaning. You output ONLY the compressed text, nothing else."
)
```

### summary_utils.py Changes

Update `prepare_summary_inputs` to build the full system prompt:

```python
def prepare_summary_inputs(
    *,
    text: str,
    target_tokens: int,
    prev_context: str | None = None,
    text_tokens: int | None = None,
    use_anti_verbatim_vaccine: bool = False,
    summarization_guidance: str | None = None,  # NEW
) -> SummaryPreparation:
    # ... existing logic ...

    # Build system prompt with optional guidance section
    if summarization_guidance:
        system_prompt = (
            f"{DEFAULT_SUMMARY_SYSTEM_PROMPT}\n\n"
            f"# Summarization Guidance\n{summarization_guidance}"
        )
    else:
        system_prompt = DEFAULT_SUMMARY_SYSTEM_PROMPT

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt},
    ]
    # ...
```

### Workflow Changes

Thread `summarization_guidance` from IndexConfig through:
1. `SummaryWorkflowConfig` - add field
2. `run_summary_workflow` - pass to `prepare_summary_inputs`
3. `run_summary_from_config` - extract from IndexConfig

### Telemetry

Update `_get_system_prompts()` in telemetry_collection.py to record the full constructed prompt (including any custom guidance) for reproducibility.

## Examples

### Legal Documents

```yaml
summarization_guidance: |
  This document contains legal contracts and agreements. Preserve exact legal
  terminology, party names, dates, and monetary amounts. Pay special attention
  to defined terms (capitalized phrases) and obligation language (shall, must, may).
```

### Code Documentation

```yaml
summarization_guidance: |
  This is technical documentation for a software project. Preserve function names,
  class names, API references, and technical terminology exactly as written.
  Code examples should be kept intact when possible.
```

### Meeting Notes

```yaml
summarization_guidance: |
  These are meeting notes. Prioritize preserving action items, decisions made,
  and participant names. Focus on outcomes and commitments over discussion details.
```

### Medical Records

```yaml
summarization_guidance: |
  This contains medical documentation. Preserve drug names, dosages, lab values,
  and clinical terminology exactly. Patient identifiers and dates are important.
```

## Guidelines for Custom Guidance

1. **Keep it concise** - Guidance adds to every API call's token count
2. **Focus on domain context** - Describe what makes this content special
3. **Specify preservation rules** - Call out terminology that must be kept exact
4. **No need to repeat output rules** - The default prompt already handles that

## Testing

### Unit Tests

- Default prompt used when `summarization_guidance` is None
- Guidance appended under "# Summarization Guidance" section when provided
- Full constructed prompt recorded in telemetry

### Integration Tests

- Index with custom guidance, verify summaries reflect domain knowledge
- Verify telemetry captures full prompt for reproducibility

## Migration

### Current State (Needs Fixing)

The initial implementation used `summary_system_prompt` with **replacement** semantics:
- Field completely replaces the default system prompt
- Users can accidentally break summarization by omitting "output ONLY the compressed text"

This violates Goal #4 (additive, not replacement).

### Required Migration

1. **Field Rename**: `summary_system_prompt` → `summarization_guidance`
   - Database: `ALTER TABLE documents RENAME COLUMN summary_system_prompt TO summarization_guidance`
   - Config file: Accept both names (old name logs deprecation warning)
   - CLI flag: `--summary-system-prompt` → `--summarization-guidance`
   - Protobuf: Rename field in AppendTextRequest

2. **Semantic Change**: Replace → Append
   - Update `prepare_summary_inputs()` to append guidance under "# Summarization Guidance"
   - Existing documents with old prompts may need manual review

3. **Schema Migration**
   - Add migration script to handle both SQLite and PostgreSQL
   - Migration should be idempotent (safe to run multiple times)
   - Detect old schema (column named `summary_system_prompt`) and migrate
   - If column doesn't exist, create as `summarization_guidance`

### Backward Compatibility

- Old configs with `summary_system_prompt` continue working (with warning)
- Old databases migrated automatically on startup
- No data loss - existing prompt content preserved, just renamed
