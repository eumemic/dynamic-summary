# RagZoom

[![Code Validation](https://github.com/eumemic/dynamic-summary/actions/workflows/code-validation.yml/badge.svg)](https://github.com/eumemic/dynamic-summary/actions/workflows/code-validation.yml)

Incremental, hierarchical RAG (Retrieval-Augmented Generation) memory system that creates dynamic summaries with intelligent resolution control.

## Features

- **Hierarchical Tree Structure**: Binary tree organization with automatic summarization
- **Dynamic Resolution**: "Zooms in" on relevant content based on queries
- **Document Isolation**: Complete namespace separation between indexed documents
- **MMR Diversity**: Maximal Marginal Relevance for diverse, comprehensive results
- **Slope-Capped Transitions**: Smooth depth transitions (±1 level) for coherent summaries
- **Token Budget Management**: Strict adherence to configurable token limits
- **Incremental Updates**: Append-only design with efficient dirty node tracking
- **Optional Features**:
  - Node pinning for always-included content
  - Sliding queue eviction with freshness decay
  - Smoothing pass for enhanced coherence

## Installation

### Requirements

- **Python 3.10+**
- **Docker** (for PostgreSQL database)
- **OpenAI API key**

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd dynamic-summary

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the automated setup script
./scripts/setup-dev.sh

# Add your OpenAI API key
echo "OPENAI_API_KEY=your-key-here" >> .env
```

The setup script automatically:
- Installs all Python dependencies
- **Starts PostgreSQL in Docker** (with pgvector extension)
- Sets up git hooks for automated testing
- Configures your development environment
- Verifies everything is working

### Verify Setup

```bash
# Check system status
ragzoom doctor

# Should show all green checkmarks ✅
```

### Alternative: pip install

```bash
pip install ragzoom
# PostgreSQL will auto-start on first use (requires Docker)
```

### Database Configuration

RagZoom automatically manages PostgreSQL for you:

- **Automatic**: PostgreSQL starts in Docker on first use
- **No configuration needed** for development
- **Persistent data** across restarts

#### Advanced Database Setup

Using existing PostgreSQL:
```bash
export RAGZOOM_DATABASE_URL="postgresql+psycopg://user:pass@host/db"
ragzoom index document.txt
```

Using Docker Compose (optional):
```bash
docker-compose up -d    # Start PostgreSQL
ragzoom index document.txt
```

Disable auto-Docker (use manual setup):
```bash
export RAGZOOM_NO_DOCKER=1
export RAGZOOM_DATABASE_URL="postgresql+psycopg://localhost/ragzoom"
```

#### Troubleshooting

```bash
# System diagnostics
ragzoom doctor

# Common fixes:
docker start ragzoom-postgres    # Start stopped container
docker logs ragzoom-postgres     # Check database logs
```

### Optional: Telemetry Tools

The telemetry analysis commands (`analyze`, `compare`, `visualize`) use optional dependencies:

```bash
# Core package only
pip install ragzoom

# With telemetry tools
pip install ragzoom[telemetry]
```

Usage: `ragzoom-telemetry analyze|compare|visualize` (separate CLI entry point)

This approach provides:
- **Avoid heavy deps in main package**: Matplotlib, seaborn, pandas only installed when needed
- **Clean separation**: Developer tools vs end-user features 
- **Single package maintenance**: No circular dependencies, simpler versioning
- **Idiomatic Python**: Follows PEP 517/518 standards

For comprehensive telemetry documentation, see [docs/telemetry.md](docs/telemetry.md)

## Quick Start

### First Time Use

```bash
# Index your first document (PostgreSQL starts automatically)
ragzoom index document.txt
# ✨ PostgreSQL container created and started automatically

# Query the document
ragzoom query "What is this document about?" -d document.txt
```

That's it! No database setup, no configuration files needed.

### CLI Usage

```bash
# Index a document (uses filename as document ID by default)
ragzoom index document.txt

# Index with custom document ID
ragzoom index document.txt --document-id my-doc

# Re-index a document (automatically clears existing data)
ragzoom index document.txt

# List all indexed documents
ragzoom documents

# Query a specific document (required)
ragzoom query "What happens to the main character?" -d document.txt

# Clear a specific document
ragzoom clear -d document.txt --confirm

# Clear all documents
ragzoom clear --confirm

# Show system status
ragzoom status

# Check system health
ragzoom doctor

# Pin important nodes
ragzoom pin <node-id>

# Start API server
ragzoom serve
```

### Python API

```python
from ragzoom import IndexConfig, QueryConfig, OperationalConfig, TreeBuilder, Retriever, Assembler, Store

# Initialize
index_config = IndexConfig()
query_config = QueryConfig()
operational_config = OperationalConfig()
store = Store(operational_config)
tree_builder = TreeBuilder(index_config, store, operational_config.openai_api_key)
retriever = Retriever(query_config, index_config, store, operational_config.openai_api_key)
assembler = Assembler(store)

# Index a document with explicit ID
doc_id = tree_builder.add_document(
    "Your document text here...",
    document_id="my-doc-id"
)

# Index from file (uses filename as ID)
doc_id = tree_builder.add_document(
    text,
    file_path="/path/to/document.txt"
)

# Query within a specific document
result = retriever.retrieve("Your query here", document_id="my-doc-id")
summary = assembler.assemble(result)
print(summary)

# List all documents
with store.SessionLocal() as session:
    from ragzoom.store import Document
    docs = session.query(Document).all()
    for doc in docs:
        print(f"Document: {doc.id}, indexed at: {doc.indexed_at}")
```

### REST API

```bash
# Start server
ragzoom serve

# Index document with custom ID
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text...", "document_id": "my-doc"}'

# Index document from file path (uses filename as ID)
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text...", "file_path": "/path/to/doc.txt"}'

# List all documents
curl http://localhost:8000/documents

# Query within a specific document (required)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here", "document_id": "my-doc"}'

# Check status
curl http://localhost:8000/status
```

## Document Management

RagZoom maintains complete isolation between indexed documents:

- **Document IDs**: Each document has a unique identifier
  - Defaults to filename when indexing files
  - Can be explicitly set with `--document-id`
- **Namespace Isolation**: Queries only search within the specified document
- **Re-indexing**: Documents are automatically cleared before indexing
- **Bulk Operations**: Clear individual documents or all data

## Advanced Usage Examples

```bash
# Index multiple documents
ragzoom index report-2023.pdf
ragzoom index report-2024.pdf

# Each query targets a specific document
ragzoom query "What were the key findings?" -d report-2023.pdf
ragzoom query "What were the key findings?" -d report-2024.pdf

# Re-index with updated content
ragzoom index report-2024.pdf

# Remove old documents
ragzoom clear -d report-2023.pdf --confirm
```

## Configuration

RagZoom provides flexible configuration through three methods (in order of precedence):

1. **CLI Options** (highest priority)
2. **Config Files**
3. **Default Values** (lowest priority)

### CLI Options

All configuration parameters can be passed as command-line options:

```bash
# Indexing with custom settings
ragzoom index document.txt \
  --target-chunk-tokens 300 \
  --embedding-model text-embedding-3-large \
  --max-retries 2

# Query with custom parameters
ragzoom query "your question" -d document.txt \
  --token-budget 4000 \
  --mmr-lambda 0.8
```

### Config Files

Create JSON config files for reusable settings:

```json
{
  "target_chunk_tokens": 300,
  "preceding_context_tokens": 100,
  "embedding_model": "text-embedding-3-large",
  "retry_threshold": 0.15,
  "max_retries": 2,
  "embedding_batch_size": 50
}
```

Use with the `--config` option:

```bash
# Save config to file
echo '{
  "target_chunk_tokens": 300,
  "embedding_model": "text-embedding-3-large"
}' > my-config.json

# Use config file
ragzoom index document.txt --config my-config.json

# Override specific settings from config
ragzoom index document.txt --config my-config.json --max-retries 3
```

### Configuration Parameters

#### Indexing Parameters
- `target_chunk_tokens`: Target size for leaf chunks (default: 200)
- `preceding_context_tokens`: Context from adjacent chunks (default: 75)
- `summary_model`: Model for summarization
- `embedding_model`: Model for embeddings (default: "text-embedding-3-small")
- `retry_threshold`: Max deviation before retry, 0.2 = 20% (default: 0.2)
- `max_retries`: Maximum summary retries (default: 0)
- `embedding_batch_size`: Batch size for embeddings (default: 100)

#### Query Parameters
- `token_budget`: Maximum tokens for summary (default: 8000)
- `mmr_lambda`: MMR relevance vs diversity, 0-1 (default: 0.7)
- `mmr_k_multiplier`: Retrieve k_multiplier * N_max candidates (default: 2.0)

#### Operational Parameters
- `database_url`: PostgreSQL database URL (default: "postgresql://localhost/ragzoom")
- `cache_size`: LRU cache size (default: 1000)
- `log_level`: Logging level: DEBUG, INFO, WARNING, ERROR (default: "INFO")

### Environment Variables

The OpenAI API key is still configured via environment variable:

```bash
# Required: Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
# Or add to .env file
echo "OPENAI_API_KEY=your-api-key-here" >> .env
```

### Common Configuration Patterns

```bash
# Development: Fast indexing with smaller chunks
ragzoom index doc.txt --target-chunk-tokens 150 --max-retries 0

# Production: High quality with retries
ragzoom index doc.txt \
  --target-chunk-tokens 300 \
  --max-retries 2 \
  --embedding-model text-embedding-3-large

# Memory-constrained: Smaller batches and cache
ragzoom index doc.txt \
  --embedding-batch-size 50 \
  --cache-size 500

# Debugging: Verbose output with validation
ragzoom index doc.txt --debug --validate --log-level DEBUG
```

## Architecture

```
ragzoom/
├── splitter.py      # Text chunking with boundary awareness
├── store.py         # PostgreSQL storage layer with pgvector
├── index.py         # Tree building and summarization
├── retrieve.py      # MMR-based retrieval logic
├── assemble.py      # Tiling assembly
├── config.py        # Pydantic configuration
├── api.py           # FastAPI REST endpoints
└── cli.py           # Click CLI interface
```

## Development

```bash
# Set up development environment (first time only)
./scripts/setup-dev.sh

# Run tests
pytest                      # All tests
./scripts/test_quick.sh            # Quick test runner
./scripts/test_quick.sh splitter   # Test specific module

# Format code
black ragzoom/ tests/
ruff check ragzoom/ tests/

# Type checking
mypy ragzoom/

# Performance benchmarking
./scripts/run-indexing-benchmarks --baseline baseline.json document.txt

# Telemetry analysis
ragzoom-telemetry analyze telemetry.json
ragzoom-telemetry compare baseline.json current.json
ragzoom-telemetry visualize baseline.json current.json -o comparison.png

# Git hooks (automatically installed by setup script)
# - pre-commit: Runs relevant tests for changed files
# - pre-push: Runs full test suite
```

## License

MIT License - see LICENSE file for details.