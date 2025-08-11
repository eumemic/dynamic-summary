# Prompt Customization Guide

This guide explains how to customize the prompts used by RagZoom's summarization system.

## Overview

RagZoom uses a template-based prompt system that allows you to customize how the LLM generates summaries. Prompts are loaded from external files and support variable substitution for dynamic values.

## Built-in Prompts

RagZoom includes two default prompts in the `prompts/summarization/` directory:

### Initial Prompt (`initial.txt`)
Defines the initial user prompt for summarization, including instructions and context formatting.

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
# Use a custom initial prompt
ragzoom index document.txt --summary-initial-prompt /path/to/custom/initial.txt

# Use a custom retry prompt
ragzoom index document.txt --summary-retry-prompt /path/to/custom/retry.txt

# Use both custom prompts
ragzoom index document.txt \
  --summary-initial-prompt /path/to/custom/initial.txt \
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

### Academic Summarization Initial Prompt

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
ragzoom index document.txt --debug --summary-initial-prompt custom.txt
# Output: Loaded prompt from: /path/to/custom.txt with variables: {'target_tokens'}
```

## Advanced Usage

### Programmatic Access

For developers integrating RagZoom as a library:

```python
from ragzoom.index import TreeBuilder

# TreeBuilder with custom prompts
tree_builder = TreeBuilder(
    config, 
    store,
    initial_prompt_path="/path/to/initial.txt",
    retry_prompt_path="/path/to/retry.txt"
)

# Or use defaults (loads from prompts/summarization/)
tree_builder = TreeBuilder(config, store)
```

### Prompt Architecture

The new `PromptManager` class:
- Loads a single prompt template at construction time
- Extracts required variables once for fast validation
- Provides a `hydrate()` method for efficient variable substitution
- No file I/O or regex parsing during hydration for optimal performance

## Related Documentation

- [API Reference](api-reference.md) - Complete CLI and API documentation
- [Architecture](architecture.md) - System design and components
- [Configuration](api-reference.md#configuration) - All configuration options