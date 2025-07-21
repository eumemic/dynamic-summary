# RagZoom Documentation

Welcome to the RagZoom documentation! This guide will help you understand, use, and contribute to RagZoom.

## 📚 Documentation Overview

### Core Documentation

- **[Architecture](./architecture.md)** - System design, components, and data flow
- **[Developer Guide](./developer-guide.md)** - Setup, development workflow, testing, and debugging
- **[API Reference](./api-reference.md)** - CLI commands, REST endpoints, and configuration
- **[Agent Handoff](./agent-handoff.md)** - AI agent collaboration guide and session history
- **[Vision](./vision.md)** - Future plans and long-term project vision

### Technical Deep-Dives

- **[Tiling Algorithm](./deep-dives/tiling-algorithm.md)** - Detailed explanation of the Dynamic Programming algorithm
- **[Tree Visualization](./deep-dives/tree-visualization.md)** - How the ASCII tree visualization system works

### Historical Archive

The `archive/` directory contains historical documents that provide context about the project's evolution but do not describe the current system. These include:

- **Legacy Algorithm** - Original "Zoom-Lens" algorithm (deprecated)
- **Refactoring Logs** - Completed refactoring work
- **Investigations** - Bug reports and analyses
- **V2 Planning** - Aspirational designs not yet implemented

## 🚀 Quick Start

1. **New to RagZoom?** Start with the [Architecture](./architecture.md) overview
2. **Setting up development?** Follow the [Developer Guide](./developer-guide.md)
3. **Using RagZoom?** Check the [API Reference](./api-reference.md)
4. **Understanding the algorithm?** Read the [Tiling Algorithm](./deep-dives/tiling-algorithm.md) deep-dive

## 📖 Reading Order

For a comprehensive understanding:

1. [Architecture](./architecture.md) - Get the big picture
2. [API Reference](./api-reference.md) - Learn how to use RagZoom
3. [Developer Guide](./developer-guide.md) - Set up your environment
4. [Tiling Algorithm](./deep-dives/tiling-algorithm.md) - Understand the core algorithm
5. [Vision](./vision.md) - See where we're heading

## 🔍 Finding Information

| If you need to know about... | Look in... |
|------------------------------|------------|
| System components and design | [Architecture](./architecture.md) |
| CLI commands and options | [API Reference](./api-reference.md) |
| Configuration parameters | [API Reference - Configuration](./api-reference.md#configuration-parameters) |
| Running tests | [Developer Guide](./developer-guide.md) |
| Testing strategy and markers | [Developer Guide - Testing](./developer-guide.md#testing-strategy) |
| DP algorithm details | [Tiling Algorithm](./deep-dives/tiling-algorithm.md) |
| Implementation status | [Tiling Algorithm - What's NOT Implemented](./deep-dives/tiling-algorithm.md#whats-not-implemented) |
| Tree visualization | [Tree Visualization](./deep-dives/tree-visualization.md) |
| Future features | [Vision](./vision.md) |
| AI agent collaboration | [Agent Handoff](./agent-handoff.md) |

## 📝 Documentation Standards

When updating documentation:

1. **Accuracy**: Verify all claims against the current implementation
2. **Status Tags**: Use `STATUS: IMPLEMENTED` or `STATUS: PLANNED` for features
3. **No Redundancy**: Each topic should be covered in exactly one place
4. **Cross-References**: Link to other docs rather than duplicating content
5. **Code Examples**: Include working examples from the actual codebase

## 🗓️ Last Updated

This documentation structure was created in January 2025. Individual documents show their own last-updated dates.