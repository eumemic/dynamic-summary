# Integration Packages

This directory contains client-specific integrations that sit on top of the core RagZoom platform. Each integration is an independent pip-installable package.

## Available Integrations

| Package | Description | Install |
|---------|-------------|---------|
| `ragzoom-claude-code` | Claude Code integration with MCP server | `pip install -e integrations/claude-code` |
| `ragzoom-clawdbot` | Clawdbot transcript sync | `pip install -e integrations/clawdbot` |

## Architecture

```
integrations/
├── claude-code/           # Claude Code integration
│   ├── pyproject.toml     # pip install ragzoom-claude-code
│   ├── src/ragzoom_claude_code/
│   └── tests/
└── clawdbot/              # Clawdbot integration
    ├── pyproject.toml     # pip install ragzoom-clawdbot
    ├── src/ragzoom_clawdbot/
    └── tests/
```

## Design Principles

1. **Depend on ragzoom core**: Integrations use the `RagZoom` wrapper and gRPC client from the core package
2. **Add client-specific logic**: Each integration handles transcript parsing, turn grouping, and state tracking for its specific client
3. **Independent packages**: Each integration can be installed separately without pulling in other integrations
4. **Own CLI entry points**: Each integration exposes its own CLI commands via `project.scripts`

## Development

### Installing for Development

```bash
# Install all integrations in editable mode
pip install -e integrations/claude-code
pip install -e integrations/clawdbot
```

### Running Tests

```bash
# Run integration-specific tests
pytest integrations/claude-code/tests/
pytest integrations/clawdbot/tests/

# Run all tests including core
pytest
```

### Type Checking

Integrations follow the same strict type checking as the core package:

```bash
dmypy check integrations/claude-code/src/
dmypy check integrations/clawdbot/src/
```
