# RagZoom

[![Code Validation](https://github.com/eumemic/dynamic-summary/actions/workflows/code-validation.yml/badge.svg)](https://github.com/eumemic/dynamic-summary/actions/workflows/code-validation.yml)

Incremental, hierarchical RAG (Retrieval-Augmented Generation) memory system that creates dynamic summaries with intelligent resolution control.

## Features

- **Hierarchical Tree Structure**: Binary tree organization with automatic summarization
- **Dynamic Resolution**: "Zooms in" on relevant content based on queries
- **Document Isolation**: Complete namespace separation between indexed documents
- **MMR Diversity**: Maximal Marginal Relevance for diverse, comprehensive results
- **Slope-Capped Transitions**: Smooth depth transitions (±1 level) for coherent summaries *(planned feature)*
- **Token Budget Management**: Strict adherence to configurable token limits
- **Incremental Appends (beta)**: Default patch-based indexing that reuses existing tree structure while appending new content under strict invariants
- **Optional Features**:
  - Node pinning for always-included content
  - Sliding queue eviction with freshness decay
  - Smoothing pass for enhanced coherence
  - **Temporal documents** with timestamp queries (see [docs/temporal-documents.md](docs/temporal-documents.md))

## Incremental Append (Beta)

Incremental append is now the default indexing pipeline. A fresh `ragzoom index`
run still clears the document and writes version 1, but the work happens through the
same patch engine that powers append-only updates. Before using it, make sure the
storage migrations have introduced the required columns:

- PostgreSQL: migrations run automatically on startup.
- SQLite: the bundled migrations add the same columns on first access.

The gRPC `AppendText` endpoint—and, by proxy, the `ragzoom index --append` CLI path—reuse
the existing tree, resummarize only the affected rightmost path, and rely on storage
consistency checks (missing nodes are filtered after vector search) so queries see a
coherent snapshot. Telemetry files produced during append
runs contain an `append_metadata` block describing the patch (document version, span, and
node counts).

When you need to extend an existing document, pass `--append --document-id <id>` to the
CLI (or call `append_to_document`). For one-off rebuilds, omit `--append`; the service
clears the document first and then seeds a brand-new patch through the same pipeline.

If the schema is out of date, append requests fail fast with a clear error so you can
migrate or re-run in full-reindex mode.

## Installation

### Requirements

- **Python 3.10+**
- **OpenAI API key**

Docker is optional. By default RagZoom uses a local SQLite database and a file‑backed vector index. You can opt into PostgreSQL later if needed.

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd dynamic-summary

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e .
echo "OPENAI_API_KEY=your-key-here" >> .env
```

Defaults and layout:
- Uses SQLite at `data/sqlite.db` (created on first use)
- Uses Chroma vector index at `data/chroma/` (if installed)
- Ignores `data/` directory in Git (.gitignore)

### Verify Setup

```bash
# Check system status
ragzoom doctor

# Should show all green checkmarks ✅
```

### Install from PyPI

Recommended (includes Chroma for the default CLI vector index):

```bash
pip install "ragzoom[chroma]"
```

Base install (no optional backends):

```bash
pip install ragzoom
```

PostgreSQL extras (when using a Postgres storage backend):

```bash
pip install "ragzoom[postgres]"
```

### Backend Selection

By default RagZoom uses SQLite (file‑backed) and a local vector index:

- DB: `sqlite:///data/sqlite.db`
- Vector index (CLI): Chroma in `data/chroma/` (requires `pip install ragzoom[chroma]` or `pip install chromadb`). The CLI fails loudly if Chroma is not available.
- Vector index (programmatic/tests): You may explicitly set `OperationalConfig(vector_backend="python")` to use an in‑memory index for tests. This adapter never persists and is not used by the CLI.

Switch to PostgreSQL:
```bash
export RAGZOOM_BACKEND=postgres
export RAGZOOM_DATABASE_URL="postgresql+psycopg://user:pass@host/db"
ragzoom index document.txt
```
Docker auto-start is used only for Postgres if you use the default local URL. For SQLite, Docker is not required.

### Backend Matrix

- Storage: SQLite by default. PostgreSQL optional for multi-user or external DB setups.
- Vector index: Chroma by default for CLI; Python in-memory adapter for tests/dev.

- SQLite + Chroma:
  - Default for CLI and typical local runs
  - Requires `pip install ragzoom[chroma]` (or `pip install chromadb`)
  - Persists vectors under `data/chroma/`
- SQLite + Python (in-memory):
  - Tests/dev only; set `RAGZOOM_VECTOR_BACKEND=python` or `OperationalConfig(vector_backend="python")`
  - Non-persistent; fastest path; no extra dependencies
- PostgreSQL storage:
  - Enable with `RAGZOOM_BACKEND=postgres` and `RAGZOOM_DATABASE_URL`
  - Install extras: `pip install ragzoom[postgres]`
  - Vector index still selected via `RAGZOOM_VECTOR_BACKEND` (`chroma` recommended for persistence)

Policy: No hidden fallbacks. If `chroma` is selected but `chromadb` is not installed, the system raises `ImportError` with guidance to install the dependency or switch to the Python in-memory adapter.

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

## Development

### Reproducible installs (lockfiles)

We use pip-tools to provide npm-style lockfiles for reproducible installs.

- Lock files:
  - `requirements/app.in` → application entry (`-e .[chroma]`)
  - `requirements/dev.in` → `-r app.in` plus dev tools (pytest, xdist, mypy[dmypy], ruff, black, bandit, etc.)
  - `requirements/app.lock` and `requirements/dev.lock` are generated from the above and committed

Install (locked):
```
python -m pip install --upgrade "pip<24.1"
python -m pip install pip-tools
pip-sync requirements/dev.lock
```

Update locks to latest (including chromadb):
```
python -m pip install --upgrade "pip<24.1"
python -m pip install pip-tools
pip-compile -o requirements/app.lock requirements/app.in
pip-compile -o requirements/dev.lock requirements/dev.in
```
> pip-tools still relies on the legacy `InstallRequirement.use_pep517` attribute that was
> removed in pip 24.1. Pinning pip to `<24.1` keeps `pip-sync` functional until upstream
> releases a compatible version.
Commit the updated lock files. CI installs from locks via `pip-sync` to guarantee parity with local.

Workflow guard: `scripts/run-checks.sh` contains a check that fails if workflows include unpinned `pip install` lines. Only pip-tools/pip-sync or specific tooling installs (e.g. pytest-cov, awscli) are allowed.

### Inspector UI

An interactive web UI for browsing document trees lives under `ui/inspector/`.

```
cd ui/inspector
npm install
npm run dev               # proxies to FastAPI on http://localhost:8000 by default
```

Building for production writes static assets to `ui/inspector/dist`:

```
npm run build
```

The Vite dev server proxies API calls to `http://localhost:8000`. Override by setting
`RAGZOOM_API_URL` before running `npm run dev` if your FastAPI instance listens on a
different origin. When you run the Docker dev stack (below) the dev server is already
started for you inside the `ui` container.

### Docker Dev Stack

To avoid clashing ports across worktrees and run the full system (gRPC server, REST
API, inspector UI) in a single command, use the bundled docker-compose stack:

```
# Set your OpenAI key for summarisation before starting
export OPENAI_API_KEY="sk-..."

./scripts/devstack start
# UI available at http://localhost:${RAGZOOM_UI_PORT:-55300}
# CLI now talks to the stack automatically (gRPC is exposed on 127.0.0.1:50051)

# Tail logs
./scripts/devstack logs

# Run CLI commands inside the stack
./scripts/devstack exec-cli index README.md --document-id readme --collect-telemetry --await-workers

# Rebuild + restart individual services when needed
./scripts/devstack restart api ui

# Tear everything down
./scripts/devstack stop
```

Helpful commands:

```
./scripts/devstack start            # build & start gRPC, API, UI
./scripts/devstack status           # show container status
./scripts/devstack logs             # tail all services
./scripts/devstack exec-cli -- --help    # run ragzoom CLI inside the grpc container
./scripts/devstack restart api ui   # rebuild/restart specific services
./scripts/devstack stop             # tear everything down
```

For hot reload during development, run the stack in foreground watching mode:

```
./scripts/devstack watch
```

This uses `docker compose watch` so Python services restart automatically when
files change, FastAPI reloads in place, and the Vite dev server serves the UI
with HMR.

Values from `.env` are loaded automatically. You can customise exposed ports via
environment variables before calling `start` (either exported or listed in
`.env`):

```
export RAGZOOM_GRPC_PORT=56100   # only change this if you need to avoid clashes
export RAGZOOM_API_PORT=56200
export RAGZOOM_UI_PORT=56300
./scripts/devstack start
```

If Docker listens on a non-default socket, set `DEVSTACK_DOCKER_HOST` (or
`DOCKER_HOST`) before running the script.

Each worktree mounting its own `data/` directory keeps the SQLite database,
Chroma vector store (`data/chroma/`), and telemetry files isolated, so parallel
stacks never collide. Because the stack loads your `.env` before starting,
the containers inherit the same configuration (and defaults) as a local
`ragzoom server` run: embeddings go to `data/chroma/` by default and are
immediately visible to host-side tools like `ragzoom-telemetry`. The inspector
UI served from the container already points at the REST API port you expose on
the host. When the stack is running the regular `python -m ragzoom.cli ...`
commands will hit the containerised gRPC server automatically.

### gRPC API code generation

Generated protobuf stubs live in `ragzoom/rpc/`. When you update files in `proto/`, regenerate the Python code with:

```
python -m grpc_tools.protoc -I proto --python_out=ragzoom/rpc --grpc_python_out=ragzoom/rpc proto/dynamic_summary.proto
```

Install `grpcio-tools` via `pip-sync requirements/dev.lock` to make sure the compiler is available.

### Running checks

```
./scripts/run-checks.sh
```

- Runs ruff/black/mypy/js-cpd/bandit and the test suite.
- Enforces per-test 2s timeout by default (override via `RZ_MAX_TEST_DURATION`).
- Lists slowest tests when `PYTEST_DURATIONS` is set (e.g., `PYTEST_DURATIONS=25`).
- Runs tests after type checks pass to fail fast and reduce contention.

## Quick Start

### First Time Use

```bash
# Start the gRPC server (leave running in its own terminal)
ragzoom server start

# Index your first document
ragzoom index document.txt

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

# Index without waiting for background summarization
ragzoom index document.txt --no-await-workers

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

# Start gRPC server
ragzoom server start
```

### Python API

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
runtime = IndexerRuntime(index_config, store, operational_config.openai_api_key.get_secret_value())

# Index a document
document_id = "my-doc-id"
await runtime.append_text(document_id, "Your document text here...", replace_existing=True)

# Query within a specific document
document_store = store.for_document(document_id)
retriever = Retriever(query_config, document_store, embedding_service, budget_planner, vector_index)
result = await retriever.retrieve_async("Your query here", document_id=document_id)

assembler = Assembler(document_store)
summary = assembler.assemble(result)
print(summary)
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

# Debugging: Verbose output followed by structural validation
ragzoom index doc.txt --debug --log-level DEBUG
ragzoom validate doc.txt
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

### Quality Checks

```bash
# Run all checks (fast tests, linting, formatting, type checking, security, duplication)
./scripts/run-checks.sh

# Include slow and integration tests (auto-starts PostgreSQL if needed)
./scripts/run-checks.sh --include-integration-tests

# Run checks without tests
./scripts/run-checks.sh --skip tests

# Skip specific checks
./scripts/run-checks.sh --skip tests,jscpd,bandit

# Stop at first error (useful for debugging)
./scripts/run-checks.sh --fail-fast
```

**Note:** Claude automatically runs `dmypy`, `ruff`, and `black` on every Python file edit for immediate feedback.

### Testing

```bash
# Run specific test patterns
pytest tests/ -k "test_name"     # Run tests matching pattern
pytest tests/test_file.py        # Run specific test file

# Run by test category
pytest -m "not benchmark and not integration"  # Default marker selection
pytest -m "not benchmark"                       # Include integration tests
pytest -m integration                      # Integration tests only
```

### Other Development Tools

```bash
# Set up development environment (first time only)
./scripts/setup-dev.sh

# Performance benchmarking
./scripts/run-indexing-benchmarks --baseline baseline.json document.txt

# Telemetry analysis
ragzoom-telemetry analyze telemetry.json
ragzoom-telemetry compare baseline.json current.json
ragzoom-telemetry visualize baseline.json current.json -o comparison.png

# Summarization diagnostics
ragzoom-telemetry analyze telemetry.json
ragzoom-telemetry visualize telemetry.json

When telemetry is enabled, RagZoom stores scalar fidelity metrics for every
merge. The telemetry analysis and visualization commands report aggregate
statistics plus low-fidelity merges, and the fidelity scatter plot shows drift
hotspots over the document without re-embedding anything.

# Git hooks (automatically installed by setup script)
# - pre-commit: Runs all quality checks in parallel
# - Claude hooks: Runs Python checks on every edit
```

## License

MIT License - see LICENSE file for details.
