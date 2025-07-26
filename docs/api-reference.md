# RagZoom API Reference

This document provides a comprehensive reference for all RagZoom interfaces: CLI commands, REST API endpoints, Python API, and configuration options.

**Last Verified**: January 2025

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
- `--document-id, -d` - Unique document identifier (default: filename)
- `--clear` - Clear existing document before indexing
- `--no-progress` - Disable progress bar
- `--max-concurrent` - Maximum concurrent API calls (default: 10)
- `--validate` - Enable validation checks during indexing

**Examples:**
```bash
# Index a file using filename as document ID
ragzoom index document.txt

# Index with custom document ID
ragzoom index document.txt --document-id my-doc

# Re-index a document (clear then index)
ragzoom index document.txt --clear

# Index with validation
ragzoom index document.txt --validate
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
- `--n-nodes, -n` - Number of seed nodes (default: 20)
- `--n-max` - Maximum leaf nodes in result
- `--budget-tokens, -b` - Token budget for summary
- `--no-cache` - Disable caching
- `--debug` - Show debug information and tree visualization
- `--viz-width` - Tree visualization width (default: 120)
- `--viz-coords` - Coordinate system: `source-chars` or `output-tokens` (default: output-tokens)
- `--validate` - Enable validation checks

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
  "clear_existing": false,
  "validate": false
}
```

**Response:**
```json
{
  "document_id": "my-doc",
  "nodes_created": 127,
  "tree_depth": 7,
  "indexing_time": 12.5
}
```

#### `GET /documents`

List all documents.

**Response:**
```json
{
  "documents": [
    {
      "document_id": "my-doc",
      "node_count": 127,
      "indexed_at": "2025-01-15T10:30:00Z"
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
from ragzoom import Store, TreeBuilder, Retriever, RagZoomConfig

# Custom configuration
config = RagZoomConfig(
    budget_tokens=10000,
    leaf_tokens=300,
    mmr_lambda=0.8
)

# Initialize components
store = Store(config)
builder = TreeBuilder(store, config)
retriever = Retriever(store, config)

# Direct component usage
nodes = store.get_leaf_nodes(document_id="my-doc")
result = retriever.retrieve("query", document_id="my-doc")
```

## Configuration

### Configuration Parameters

All parameters can be set via environment variables with the `RAGZOOM_` prefix. For an overview of how these parameters affect system behavior, see [Architecture - Configuration](architecture.md#configuration).

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `budget_tokens` | int | 8000 | Maximum tokens in final summary |
| `leaf_tokens` | int | 200 | Target tokens per leaf chunk |
| `split_threshold` | float | 0.3 | Threshold for text splitting |

#### Retrieval Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_rerank` | int | 20 | Number of nodes for reranking |
| `mmr_lambda` | float | 0.7 | MMR diversity parameter (0=diverse, 1=relevant) |
| `mmr_consideration_penalty` | float | 0.8 | Penalty for already selected content |

#### Tiling Parameters

| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|---------|
| `enable_slope_cap` | bool | True | Enable slope capping | **STATUS: NOT IMPLEMENTED** |
| `slope_cap_size` | int | 1 | Maximum depth difference | **STATUS: NOT IMPLEMENTED** |
| `enable_smoothing` | bool | False | Enable smoothing pass | **STATUS: NOT IMPLEMENTED** |

#### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | str | "text-embedding-3-small" | OpenAI embedding model |
| `embedding_dimensions` | int | 1536 | Embedding vector dimensions |
| `summary_model` | str | "gpt-4o" | Model for summarization |
| `summary_max_tokens` | int | None | Max tokens for summaries |
| `temperature` | float | 0.2 | Temperature for generation |
| `timeout` | int | 120 | API timeout in seconds |

#### Storage Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `database_url` | str | "ragzoom.db" | SQLite database path |
| `chroma_path` | str | "./chroma_db" | ChromaDB storage path |
| `cache_size` | int | 1000 | LRU cache size |

#### Operational Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | str | "INFO" | Logging level |
| `rate_limit_rpm` | int | 10000 | Requests per minute limit |
| `embedding_batch_size` | int | 100 | Batch size for embedding calls |
| `pin_depth_max` | int | 2 | Deepest level for permanent pinning |

### Configuration Files

#### Environment Variables

Create a `.env` file:

```bash
RAGZOOM_BUDGET_TOKENS=10000
RAGZOOM_LEAF_TOKENS=300
RAGZOOM_MMR_LAMBDA=0.8
RAGZOOM_SUMMARY_MODEL=gpt-4o-mini
RAGZOOM_LOG_LEVEL=DEBUG
```

#### Python Configuration

```python
from ragzoom import RagZoomConfig

config = RagZoomConfig(
    budget_tokens=10000,
    leaf_tokens=300,
    mmr_lambda=0.8,
    summary_model="gpt-4o-mini",
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