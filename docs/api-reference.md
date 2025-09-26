# RagZoom API Reference

This document provides a comprehensive reference for all RagZoom interfaces: CLI commands, REST API endpoints, Python API, and configuration options.

**Last Verified**: September 2025

## Table of Contents

- [CLI Commands](#cli-commands)
- [REST API](#rest-api)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

## CLI Commands

### `ragzoom index`

Index a document into the RagZoom system.

```bash
ragzoom index <input> [OPTIONS]
```

**Arguments:**
- `input` - File path or text to index (use `-` for stdin)

**Options:**
- `--document-id, -d` - Explicit document identifier (defaults to filename when indexing from a file)
- `--config PATH` - Load indexing configuration from JSON
- `--target-chunk-tokens INT` - Override target leaf chunk size
- `--preceding-context-tokens INT` - Override contextual tokens carried forward
- `--summary-model, -m TEXT` - Summarization model override
- `--embedding-model TEXT` - Embedding model override
- `--retry-threshold FLOAT` - Deviation threshold before summary retry
- `--max-retries INT` - Maximum summary retries
- `--embedding-batch-size INT` - Override embedding batch size
- `--data-dir PATH` - Base directory for databases and vector stores
- `--database URL` - Explicit database URL (sqlite:///... or postgresql+psycopg://...)
- `--no-progress` - Disable progress bars
- `--validate` - Enable validation checks during indexing
- `--debug` - Emit detailed debug logging (includes token usage)
- `--telemetry [PATH]` - Request telemetry collection. The CLI waits for server workers to finish, persists the telemetry JSON (default `telemetry.json` if no path is provided), and prints the telemetry run ID. Use `ragzoom telemetry` to fetch the same run later if needed.
- `--append` - Append the file's contents to an existing document (requires `--document-id`)

The command clears the target document before indexing unless `--append` is provided. Both paths run through the incremental patch engine, so fresh indexes and appends share the same invariants and telemetry.

**Examples:**
```bash
# Index a file using filename as document ID
ragzoom index document.txt  # defaults to SQLite + local vector index

# Index with custom document ID
ragzoom index document.txt --document-id my-doc

# Re-index a document (automatically clears existing data)
ragzoom index document.txt

# Index with validation
ragzoom index document.txt --validate

# Index with debug logging
ragzoom index document.txt --debug

# Index with telemetry collection
ragzoom index document.txt --telemetry

# Append new content to an existing document
ragzoom index delta.txt --document-id my-doc --append
```

### `ragzoom query`

Query an indexed document.

```bash
ragzoom query <query_text> [OPTIONS]
```

**Arguments:**
- `query_text` - The search query

**Required Options:**
- `--document-id, -d` - Document to query (REQUIRED)

**Optional Parameters:**
- `--num-seeds` - Number of seed nodes to retrieve
- `--token-budget` - Token budget for summary
- `--debug` - Show debug information and tree visualization
- `--validate` - Enable validation checks
- `--viz-width` - Tree visualization width (defaults to terminal width)
- `--viz-coords` - Coordinate system: `source-chars` or `output-tokens` (default: output-tokens)

**Examples:**
```bash
# Basic query
ragzoom query "What is machine learning?" -d my-doc

# Query with custom parameters
ragzoom query "neural networks" -d my-doc -n 10 -b 4000

# Debug mode with visualization
ragzoom query "transformer architecture" -d my-doc --debug

# Validate tiling coverage
ragzoom query "attention mechanism" -d my-doc --validate
```

### `ragzoom documents`

List all indexed documents.

```bash
ragzoom documents
```

**Output Format:**
```
Documents in database:
  - my-doc (1234 nodes, indexed: 2025-01-15 10:30:00)
  - paper.pdf (567 nodes, indexed: 2025-01-14 15:45:00)
```

### `ragzoom telemetry`

Fetch telemetry produced by a previous indexing run. The command is helpful when `--telemetry` was used with `ragzoom index` but the results need to be retrieved later.

```bash
ragzoom telemetry --document-id my-doc --run-id <run_id> [--wait] [--output telemetry.json]
```

If `--wait` is provided the command blocks until the run finishes collecting telemetry. Otherwise it returns immediately when the run is still active. Passing `--output` writes the telemetry JSON to disk; without it the payload is printed to stdout.

### `ragzoom pin`

Pin nodes to always include them in results.

```bash
ragzoom pin <node_ids> [OPTIONS]
```

**Arguments:**
- `node_ids` - Comma-separated list of node IDs

**Options:**
- `--document-id, -d` - Document ID (required)

**Example:**
```bash
ragzoom pin "3_100_200_abc123,4_200_300_def456" -d my-doc
```

### `ragzoom status`

Show system status and configuration.

```bash
ragzoom status
```

**Output includes:**
- Current configuration values
- Database statistics
- Cache statistics
- Model information

### `ragzoom serve`

Start the REST API server.

```bash
ragzoom serve [OPTIONS]
```

**Options:**
- `--host` - Host to bind to (default: 127.0.0.1) 
- `--port` - Port to bind to (default: 8000)

**Example:**
```bash
# Start on default localhost
ragzoom serve

# Start on all interfaces
ragzoom serve --host 0.0.0.0
```

### `ragzoom clear`

Clear data from the database.

```bash
ragzoom clear [OPTIONS]
```

**Options:**
- `--document-id, -d` - Clear specific document only
- `--confirm` - Skip confirmation prompt

**Examples:**
```bash
# Clear all data (with confirmation)
ragzoom clear

# Clear specific document
ragzoom clear -d my-doc --confirm
```

### `ragzoom export`

Export tree structure to file.

```bash
ragzoom export <output_file> [OPTIONS]
```

**Options:**
- `--document-id, -d` - Document to export (required)
- `--format` - Export format: `json` or `dot` (default: json)

## REST API

Base URL: `http://localhost:8000`

### Authentication

Currently no authentication is required. In production, implement appropriate auth.

### Endpoints

#### `POST /index`

Index a new document.

**Request Body:**
```json
{
  "text": "Document content to index",
  "document_id": "my-doc",
  "file_path": "path/used/for/metadata.json"
}
```

**Response:**
```json
{
  "document_id": "my-doc",
  "chunks_created": 127,
  "tree_depth": 7
}
```

The endpoint clears any existing nodes for the document before invoking the same patch
engine used by CLI indexing. Append-style updates are currently surfaced via the CLI and
service layer (`IndexingService.append_to_document`).

#### `GET /documents`

List all documents.

**Response:**
```json
{
  "documents": [
    {
      "document_id": "my-doc",
      "file_path": "reports/2024.txt",
      "indexed_at": "2025-01-15T10:30:00Z",
      "node_count": 127
    }
  ]
}
```

#### `POST /query`

Query a document.

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "document_id": "my-doc",
  "n_nodes": 20,
  "budget_tokens": 8000,
  "validate": false
}
```

**Response:**
```json
{
  "summary": "Machine learning is...",
  "nodes": [
    {
      "node_id": "3_100_200_abc123",
      "text": "...",
      "span_start": 100,
      "span_end": 200
    }
  ],
  "tokens_used": 3456,
  "query_time": 1.2
}
```

#### `POST /pin`

Pin nodes.

**Request Body:**
```json
{
  "node_ids": ["3_100_200_abc123", "4_200_300_def456"],
  "document_id": "my-doc"
}
```

#### `PATCH /config`

Update configuration.

**Request Body:**
```json
{
  "budget_tokens": 10000,
  "mmr_lambda": 0.8
}
```

#### `GET /status`

Get system status.

**Response:**
```json
{
  "config": {
    "budget_tokens": 8000,
    "leaf_tokens": 200,
    "mmr_lambda": 0.7
  },
  "stats": {
    "total_documents": 5,
    "total_nodes": 1523,
    "cache_size": 234
  }
}
```


#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Python API

### Basic Usage

```python
from ragzoom import RagZoom

# Initialize
rz = RagZoom()

# Index a document
rz.index("This is my document content...", document_id="my-doc")

# Query
result = rz.query("What is this about?", document_id="my-doc")
print(result.summary)
```

### Async Usage

```python
import asyncio
from ragzoom import AsyncRagZoom

async def main():
    rz = AsyncRagZoom()
    
    # Index asynchronously
    await rz.index_async("Document content...", document_id="my-doc")
    
    # Query asynchronously
    result = await rz.query_async("Query text", document_id="my-doc")
    print(result.summary)

asyncio.run(main())
```

### Advanced Usage

```python
from ragzoom import Store, TreeBuilder, Retriever, IndexConfig, QueryConfig, OperationalConfig

# Custom configuration
index_config = IndexConfig(
    target_chunk_tokens=300
)
query_config = QueryConfig(
    budget_tokens=10000,
    mmr_lambda=0.8
)
operational_config = OperationalConfig()

# Initialize components
store = Store(operational_config)
builder = TreeBuilder(index_config, store, operational_config.openai_api_key)
retriever = Retriever(query_config, index_config, store, operational_config.openai_api_key)

# Direct component usage
nodes = store.get_leaf_nodes(document_id="my-doc")
result = retriever.retrieve("query", document_id="my-doc")
```

## Configuration

### Configuration Parameters

All parameters can be set via CLI options or config files. For an overview of how these parameters affect system behavior, see [Architecture - Configuration](architecture.md#configuration).

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `budget_tokens` | int | 8000 | Maximum tokens in final summary |
| `mmr_lambda` | float | 0.7 | MMR diversity parameter (0=diverse, 1=relevant) |
| `mmr_k_multiplier` | float | 2.0 | Multiplier for MMR candidate selection |
| `embedding_model` | str | "text-embedding-3-small" | OpenAI embedding model |

#### Indexing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_chunk_tokens` | int | 200 | Target tokens per leaf chunk |
| `preceding_context_tokens` | int | 75 | Context tokens before each chunk |
| `summary_model` | str | "gpt-5-nano" | Model for summarization |
| `retry_threshold` | float | 0.2 | Retry threshold for failed operations |
| `max_retries` | int | 3 | Maximum number of retries |
| `embedding_batch_size` | int | 100 | Batch size for embeddings |
| `use_anti_verbatim_vaccine` | bool | true | Enable anti-verbatim processing |

#### Tiling Parameters

| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|---------|
| `enable_slope_cap` | bool | True | Enable slope capping | **STATUS: NOT IMPLEMENTED** |
| `slope_cap_size` | int | 1 | Maximum depth difference | **STATUS: NOT IMPLEMENTED** |
| `enable_smoothing` | bool | False | Enable smoothing pass | **STATUS: NOT IMPLEMENTED** |

#### Operational Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `openai_api_key` | str | "" | OpenAI API key (from env: OPENAI_API_KEY) |
| `database_url` | str | "postgresql+psycopg://localhost/ragzoom" | PostgreSQL database URL |
| `cache_size` | int | 1000 | LRU cache size |
| `log_level` | str | "INFO" | Logging level |
| `validate_pipeline` | bool | false | Enable pipeline validation |

### Configuration Files

#### CLI Options

Use CLI options for configuration:

```bash
# Indexing with configuration
ragzoom index document.txt \
  --target-chunk-tokens 300 \
  --summary-model gpt-4o-mini \
  --debug

# Querying with configuration
ragzoom query "your question" -d document.txt \
  --token-budget 10000 \
  --mmr-lambda 0.8
```

#### Config Files

Create a JSON config file:

```json
{
  "target_chunk_tokens": 300,
  "summary_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-large"
}
```

Then use it:

```bash
ragzoom index document.txt --config myconfig.json
```

#### Python Configuration

```python
from ragzoom import IndexConfig, QueryConfig, OperationalConfig

index_config = IndexConfig(
    target_chunk_tokens=300,
    summary_model="gpt-4o-mini"
)
query_config = QueryConfig(
    budget_tokens=10000,
    mmr_lambda=0.8
)
operational_config = OperationalConfig(
    log_level="DEBUG"
)
```

## Error Handling

### Error Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| 400 | Bad Request | Invalid parameters, missing document ID |
| 404 | Not Found | Document or node not found |
| 422 | Validation Error | Invalid input format |
| 429 | Rate Limited | Too many requests |
| 500 | Server Error | Internal error, check logs |

### Common Errors

```python
# Document not found
{
  "error": "Document 'my-doc' not found",
  "code": "DOCUMENT_NOT_FOUND"
}

# Invalid configuration
{
  "error": "budget_tokens must be positive",
  "code": "INVALID_CONFIG"
}

# API rate limit
{
  "error": "OpenAI API rate limit exceeded",
  "code": "RATE_LIMITED"
}
```

## Best Practices

1. **Document IDs**: Use meaningful, URL-safe identifiers
2. **Token Budgets**: Start with defaults, adjust based on output quality
3. **Validation**: Use `--validate` flag during development
4. **Caching**: Enable caching for repeated queries
5. **Batch Operations**: Use async API for bulk indexing
6. **Error Handling**: Always handle potential errors in production

## Rate Limiting

- OpenAI API limits apply (varies by tier)
- Default: 10,000 requests per minute
- Automatic retry with exponential backoff
- Consider batching for large documents
If another indexing run is already in progress for the same document, the CLI reports a friendly lock message and exits. Locks are per‑document (you can index different documents concurrently).
