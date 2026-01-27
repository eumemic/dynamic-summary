---
name: ragzoom-usage
description: This skill should be used when the user asks "how do I index a document", "how do I query", "CLI commands", "REST API", "Python API", "configuration options", or mentions using RagZoom as an end user rather than developing it.
---

# RagZoom Usage

Guidance for using RagZoom to index documents and query them.

## Quick Start

```bash
# Start the gRPC server (leave running)
ragzoom server start

# Index a document
ragzoom index document.txt

# Query the document
ragzoom query "What is this document about?" -d document.txt
```

## CLI Commands

### Indexing

```bash
# Index with filename as document ID (default)
ragzoom index document.txt

# Index with custom document ID
ragzoom index document.txt --document-id my-doc

# Index without waiting for background summarization
ragzoom index document.txt --no-await-workers

# Re-index (automatically clears existing data)
ragzoom index document.txt

# Append to existing document (incremental)
ragzoom index newcontent.txt --append --document-id existing-doc
```

### Querying

```bash
# Query a specific document (required)
ragzoom query "What happens to the main character?" -d document.txt

# Query with custom token budget
ragzoom query "summarize" -d document.txt --token-budget 4000

# Query with MMR diversity tuning
ragzoom query "key themes" -d document.txt --mmr-lambda 0.8
```

### Document Management

```bash
# List all indexed documents
ragzoom documents

# Clear a specific document
ragzoom clear -d document.txt --confirm

# Clear all documents
ragzoom clear --confirm

# Validate document structure
ragzoom validate document.txt

# Pin important nodes
ragzoom pin <node-id>
```

### System Commands

```bash
# Show system status
ragzoom status

# Check system health
ragzoom doctor

# Start REST API server
ragzoom serve

# Start gRPC server
ragzoom server start
```

## REST API

Start server: `ragzoom serve`

### Index Document

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text...", "document_id": "my-doc"}'
```

### Query Document

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here", "document_id": "my-doc"}'
```

### List Documents

```bash
curl http://localhost:8000/documents
```

### Check Status

```bash
curl http://localhost:8000/status
```

## Python API

```python
from ragzoom import IndexConfig, QueryConfig, OperationalConfig, create_store
from ragzoom.indexing import IndexerRuntime
from ragzoom.retrieve import Retriever
from ragzoom.assemble import Assembler

# Initialize
index_config = IndexConfig.load()
query_config = QueryConfig()
operational_config = OperationalConfig()  # defaults to SQLite
store = create_store(operational_config)

# Create runtime for indexing
runtime = IndexerRuntime(
    index_config, store,
    operational_config.openai_api_key.get_secret_value()
)

# Index a document
document_id = "my-doc-id"
await runtime.append_text(
    document_id,
    "Your document text here...",
    replace_existing=True
)

# Query
document_store = store.for_document(document_id)
retriever = Retriever(query_config, document_store, ...)
result = await retriever.retrieve_async("Your query", document_id=document_id)

assembler = Assembler(document_store)
summary = assembler.assemble(result)
```

## Configuration

### CLI Options (highest priority)

```bash
ragzoom index document.txt \
  --target-chunk-tokens 300 \
  --embedding-model text-embedding-3-large \
  --max-retries 2

ragzoom query "question" -d doc.txt \
  --token-budget 4000 \
  --mmr-lambda 0.8
```

### Config Files

```json
{
  "target_chunk_tokens": 300,
  "embedding_model": "text-embedding-3-large",
  "retry_threshold": 0.15,
  "max_retries": 2
}
```

Use with `--config my-config.json`.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_chunk_tokens` | 200 | Target size for leaf chunks |
| `token_budget` | 8000 | Maximum tokens in query result |
| `mmr_lambda` | 0.7 | MMR relevance vs diversity (0-1) |
| `embedding_model` | text-embedding-3-small | Model for embeddings |

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-api-key"

# Backend selection (default: SQLite)
export RAGZOOM_BACKEND=postgres
export RAGZOOM_DATABASE_URL="postgresql://..."
```

## Document Isolation

Each document is completely isolated:
- Queries only search within the specified document
- Document IDs default to filename when indexing files
- Re-indexing automatically clears existing data first

## Common Patterns

### Index Multiple Documents

```bash
ragzoom index report-2023.pdf
ragzoom index report-2024.pdf

# Query each separately
ragzoom query "key findings" -d report-2023.pdf
ragzoom query "key findings" -d report-2024.pdf
```

### Development vs Production

```bash
# Development: Fast indexing
ragzoom index doc.txt --target-chunk-tokens 150 --max-retries 0

# Production: High quality
ragzoom index doc.txt \
  --target-chunk-tokens 300 \
  --max-retries 2 \
  --embedding-model text-embedding-3-large
```

## Integration Packages

Client-specific integrations are separate packages in `integrations/`:

### Claude Code Integration

```bash
# Install
pip install -e integrations/claude-code

# Sync a transcript
ragzoom-claude-code sync ~/.claude/projects/.../session.jsonl

# Reset (clears both state file AND document, then re-syncs)
ragzoom-claude-code reset session.jsonl

# Reset without re-sync
ragzoom-claude-code reset session.jsonl --no-resync

# Start MCP server
ragzoom-claude-code mcp-server
```

**Important:** The `reset` command clears the RagZoom document and re-syncs from scratch. The sync algorithm is stateless - it derives all state from the transcript file and RagZoom document status API.

### Clawdbot Integration

```bash
pip install -e integrations/clawdbot
ragzoom-clawdbot sync <transcript-file>
```

See `integrations/CLAUDE.md` for architecture details.
