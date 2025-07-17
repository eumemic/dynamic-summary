# Developer Onboarding Guide

Welcome to the RagZoom project! This guide provides a practical overview of the technology stack, development process, and key conventions to help you get started.

## 1. Technology Stack

-   **Core Language:** Python 3.11+
-   **API Framework:** FastAPI
-   **CLI Framework:** `click`
-   **Vector Database:** `chromadb` for storing embeddings and performing semantic search.
-   **Metadata Store:** SQLite with `sqlalchemy` for storing the node tree structure and all other metadata.
-   **LLM Interaction:** `openai` for generating summaries and embeddings.
-   **Tokenization:** `tiktoken` for accurately counting tokens.

## 2. Development Environment Setup

Getting your environment set up correctly is the most important first step.

1.  **Create a Virtual Environment:** It is strongly recommended to use a Python virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
2.  **Run the Setup Script:** The `scripts/setup-dev.sh` script is the one-stop shop for getting everything installed and configured.
    ```bash
    ./scripts/setup-dev.sh
    ```
    This script will:
    - Install all base and development requirements from `requirements.txt` and `requirements-dev.txt`.
    - Install the `ragzoom` package in editable mode (`pip install -e .`).
    - Create a `.env` file from the example for you to add your API keys.
    - **Crucially, it sets up the Git pre-commit hooks.**

## 3. Development Process & Tooling

We use a suite of tools to ensure code quality and consistency. These are run automatically by the pre-commit hook, so it's important to understand what they do.

### 3.1. Pre-Commit Hook

The pre-commit hook is defined in `scripts/git-hooks/pre-commit` and is the guardian of our codebase quality. Before any commit is finalized, it runs the following checks in parallel:

-   **Tests (`pytest`):** Runs the fast unit tests.
-   **Formatting (`black`):** Automatically reformats your code to be consistent.
-   **Linting (`ruff`):** Checks for common errors, style issues, and automatically fixes what it can.
-   **Type Checking (`mypy`):** Statically analyzes type hints to catch potential bugs.

**Important:** The hook is configured to **auto-fix** formatting and simple linting errors. After it runs, it will re-stage any files it modified. If there are still errors (e.g., a failing test or a complex `mypy` error), the commit will be aborted, and you will need to fix the issues manually.

### 3.2. Running Checks Manually

You can and should run these checks yourself as you code:
-   **Run all fast tests:** `./test_quick.sh`
-   **Run the full test suite (including slow/integration tests):** `pytest`
-   **Auto-format your code:** `black ragzoom/ tests/`
-   **Auto-fix linting issues:** `ruff check ragzoom/ tests/ --fix`
-   **Run the type checker:** `mypy ragzoom`

### 3.3. Debugging Type Errors

The `mypy` check can sometimes be noisy, flagging pre-existing issues in files you haven't touched. If you're struggling with a persistent type error, the following command can be very helpful, as it provides a clean, stateless check on just the `ragzoom` directory:
```bash
mypy ragzoom --ignore-missing-imports --no-error-summary --check-untyped-defs
```

## 4. What I Wish I Had Known

-   The `SimpleMockStore` in `tests/mock_store.py` is your best friend for writing fast, reliable unit tests. The real `Store` and `TreeBuilder` can be slow as they involve database I/O and LLM calls. Always use the mock store for testing algorithmic logic.
-   The `pytest` runner can sometimes crash with a `Segmentation fault`. This is almost always a sign that the local `chroma_db` directory has become corrupted. The fix is to delete it (`rm -rf chroma_db/`) and re-index your test documents.
-   The pre-commit hook is your ally, not your enemy. If it's failing, running `black .` and `ruff check . --fix` will often solve the problem automatically. Don't bypass it unless you have a very good reason and have received permission. 