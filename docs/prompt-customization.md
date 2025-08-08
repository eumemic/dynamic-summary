# Prompt Customization Guide

This guide explains how to customize the prompts used by RagZoom's summarization system.

## Overview

RagZoom uses a template-based prompt system that allows you to customize how the LLM generates summaries. Prompts are loaded from external files and support variable substitution for dynamic values.

## Built-in Prompts

RagZoom includes two default prompts in the `prompts/summarization/` directory:

### System Prompt (`system.txt`)
Defines the overall behavior and instructions for the summarization model. Used at the beginning of each summarization request.

**Available variables:**
- `{target_tokens}` - The target number of tokens for the summary

### Retry Prompt (`retry.txt`)
Used when a summary deviates significantly from the target token count and needs correction.

**Available variables:**
- `{target_tokens}` - The target number of tokens
- `{current_tokens}` - The actual number of tokens in the current summary
- `{deviation_pct}` - The percentage deviation from target

## Using Custom Prompts

### Via Command Line

You can specify custom prompt files when indexing documents:

```bash
# Use a custom system prompt
ragzoom index document.txt --summary-system-prompt /path/to/custom/system.txt

# Use a custom retry prompt
ragzoom index document.txt --summary-retry-prompt /path/to/custom/retry.txt

# Use both custom prompts
ragzoom index document.txt \
  --summary-system-prompt /path/to/custom/system.txt \
  --summary-retry-prompt /path/to/custom/retry.txt
```

### Security Restrictions

For security reasons, custom prompt files must be located within these allowed directories:
- The package's `prompts/` directory
- Your current working directory
- A `prompts/` subdirectory in your current working directory

Attempting to load prompts from outside these directories will result in an error:
```
ValueError: Custom prompt path '/etc/passwd' is outside allowed directories
```

## Creating Custom Prompts

### Template Syntax

Prompts use Python's string format syntax with curly braces for variables:

```text
Your summary should be exactly {target_tokens} tokens.
The current version has {current_tokens} tokens.
```

### Variable Validation

The system validates that:
1. All required variables in the template are provided
2. Variables are properly formatted (alphanumeric with underscores)

Note: Extra variables provided but not used in the template are allowed, giving you flexibility to use the same code with different prompt templates.

### Best Practices

1. **Be specific about requirements**: Clearly state token limits, formatting requirements, and constraints
2. **Use all available context**: Reference variables like `{target_tokens}` to make prompts adaptive
3. **Test with various inputs**: Ensure your prompts work well with different document types
4. **Version control your prompts**: Keep custom prompts in your repository for reproducibility

## Example Custom Prompts

### Academic Summarization System Prompt

```text
You are an academic summarizer specializing in scholarly texts. Create a summary 
of exactly {target_tokens} tokens that:

1. Preserves technical terminology and citations
2. Maintains the logical flow of arguments
3. Highlights key findings and contributions
4. Uses formal academic language

CRITICAL: The summary MUST be {target_tokens} tokens - no more, no less.
```

### Creative Writing Retry Prompt

```text
Your summary was {current_tokens} tokens, but needs to be {target_tokens} tokens 
(off by {deviation_pct:.1%}).

Please revise while:
- Maintaining narrative flow and voice
- Preserving character names and key plot points
- Adjusting descriptive passages to meet the token target

Target: EXACTLY {target_tokens} tokens.
```

## Troubleshooting

### Common Issues

1. **"Prompt file not found" error**
   - Verify the file path exists
   - Check file permissions
   - Ensure you're using absolute paths or paths relative to CWD

2. **"Missing required variables" error**
   - Check that your template uses the correct variable names
   - Ensure variable names match exactly (case-sensitive)

3. **"Custom path outside allowed directories" error**
   - Move your prompt files to the current directory or a `prompts/` subdirectory
   - Use relative paths from your current working directory

### Debug Mode

To see which prompts are being loaded and from where, use the `--debug` flag:

```bash
ragzoom index document.txt --debug --summary-system-prompt custom.txt
# Output: Loaded prompt 'summarization/system' from: /path/to/custom.txt
```

## Advanced Usage

### Programmatic Access

For developers integrating RagZoom as a library:

```python
from ragzoom.prompt import PromptManager
from ragzoom.index import TreeBuilder

# Create prompt manager
prompt_mgr = PromptManager()

# Load and hydrate a prompt
prompt = prompt_mgr.load_and_hydrate(
    "summarization/system",
    {"target_tokens": 100},
    custom_path="/path/to/custom.txt"
)

# Use with TreeBuilder
tree_builder = TreeBuilder(
    config, 
    store,
    summary_system_prompt_path="/path/to/system.txt",
    retry_prompt_path="/path/to/retry.txt"
)
```

### Creating New Prompt Categories

To add new prompt types beyond summarization:

1. Create a new directory under `prompts/` (e.g., `prompts/extraction/`)
2. Add your prompt files
3. Load them using the path relative to `prompts/`:

```python
prompt = prompt_mgr.load_prompt("extraction/entities")
```

## Related Documentation

- [API Reference](api-reference.md) - Complete CLI and API documentation
- [Architecture](architecture.md) - System design and components
- [Configuration](api-reference.md#configuration) - All configuration options