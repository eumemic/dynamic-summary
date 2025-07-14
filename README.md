# RagZoom

Incremental, hierarchical RAG (Retrieval-Augmented Generation) memory system that creates dynamic summaries with intelligent resolution control.

## Features

- **Hierarchical Tree Structure**: Binary tree organization with automatic summarization
- **Dynamic Resolution**: "Zooms in" on relevant content based on queries
- **MMR Diversity**: Maximal Marginal Relevance for diverse, comprehensive results
- **Slope-Capped Transitions**: Smooth depth transitions (±1 level) for coherent summaries
- **Token Budget Management**: Strict adherence to configurable token limits
- **Incremental Updates**: Append-only design with efficient dirty node tracking
- **Optional Features**:
  - Node pinning for always-included content
  - Sliding queue eviction with freshness decay
  - Smoothing pass for enhanced coherence

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd dynamic-summary

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the development setup script
./scripts/setup-dev.sh

# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=your-key-here" >> .env
```

The setup script will:
- Install all dependencies
- Set up git hooks for automated testing
- Configure your development environment
- Verify everything is working

## Quick Start

### CLI Usage

```bash
# Index a document
ragzoom index document.txt

# Query the system
ragzoom query "What happens to the main character?"

# Show system status
ragzoom status

# Pin important nodes
ragzoom pin <node-id>

# Start API server
ragzoom serve
```

### Python API

```python
from ragzoom import RagZoomConfig, TreeBuilder, Retriever, Assembler, Store

# Initialize
config = RagZoomConfig()
store = Store(config)
tree_builder = TreeBuilder(config, store)
retriever = Retriever(config, store)
assembler = Assembler(config, store)

# Index a document
doc_id = tree_builder.add_document("Your document text here...")

# Query
result = retriever.retrieve("Your query here")
summary = assembler.assemble(result)
print(summary)
```

### REST API

```bash
# Start server
ragzoom serve

# Index document
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text..."}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}'

# Check status
curl http://localhost:8000/status
```

## Configuration

Key configuration options (via environment variables or `.env`):

- `RAGZOOM_BUDGET_TOKENS`: Maximum tokens for summary (default: 8000)
- `RAGZOOM_LEAF_TOKENS`: Target size for leaf chunks (default: 200)
- `RAGZOOM_MMR_LAMBDA`: MMR relevance vs diversity (default: 0.7)
- `RAGZOOM_SLOPE_CAP`: Enable slope capping (default: true)
- `RAGZOOM_SMOOTHING_PASS_ENABLED`: Enable smoothing (default: false)

## Architecture

```
ragzoom/
├── splitter.py      # Text chunking with boundary awareness
├── store.py         # SQLite + Chroma storage layer
├── index.py         # Tree building and summarization
├── retrieve.py      # MMR-based retrieval logic
├── assemble.py      # Frontier assembly with slope-cap
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
./test_quick.sh            # Quick test runner
./test_quick.sh splitter   # Test specific module

# Format code
black ragzoom/ tests/
ruff check ragzoom/ tests/

# Type checking
mypy ragzoom/

# Git hooks (automatically installed by setup script)
# - pre-commit: Runs relevant tests for changed files
# - pre-push: Runs full test suite
# Skip with: git commit/push --no-verify
```

## License

MIT License - see LICENSE file for details.