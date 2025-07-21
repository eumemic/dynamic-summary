# Agent Instructions for RagZoom

This file contains comprehensive instructions for any AI agent working on this repository. These instructions should be followed whether you're using Claude Code (claude.ai/code) or Cursor IDE.

## 0. First Thing's First

**Your first action in any new session must be to read the `docs/agent-handoff.md` file.** It contains critical context, a summary of previous work, and guidance on the collaborative process for this project. Do not update this file until the user explicitly signals that the session is over. This is a critical step to ensure a clean handoff to your successor.

## 1. Key Documents

Before starting any work, review the following documents to understand the system architecture and development process:

- **Project Brief:** `docs/architecture.md`
- **Developer Onboarding Guide:** `docs/developer-guide.md`
- **Architecture Overview:** `docs/architecture.md`
- **V2 Algorithm Design:** `docs/v2/dynamic-frontier-design.md`, `docs/tiling-algorithm.md`
- **Implementation Notes:** `docs/implementation-notes.md`
- **Testing Strategy:** `docs/testing-strategy.md`

## 2. General Philosophy & Collaboration

- **Be Zen, Not Flustered:** When you get stuck in a loop or a series of failures, stop. Take a step back, breathe, and rethink the problem from first principles. There is always a logical solution.
- **Don't Be Long-winded:** Keep it concise unless elaboration is warranted. Don't repeat yourself or summarize your own messages at the end. You're not writing an essay, we're having a conversation.
- **Use the Scientific Method:** For complex problems, form a hypothesis, propose a test to validate it, and discuss it with the user before implementing.
- **Raise Blockers:** If you are instructed to do something and discover an insurmountable roadblock or a fundamental inconsistency, do not switch gears. Bring the issue to the user's attention and decide on a new course of action together.
- **Leave the Codebase Better:** Always be looking for opportunities to improve the code you touch, whether it's by refactoring, adding a clarifying comment, or improving a variable name.
- **Update Documentation:** If you discover that a document is out of date or missing information in the course of your work, update it as part of your task.
- **Update These Rules:** If you discover a new principle or best practice during your work, add it to this file.

## 3. Project Overview

RagZoom is a hierarchical RAG system that builds binary trees from documents: leaf nodes contain text chunks, internal nodes contain AI summaries. Retrieval "zooms in" on relevant content while maintaining global context using a Dynamic Programming algorithm.

## 4. Design & Implementation

- **Design First:** Before implementing any large initiative, work with the user to create a well-thought-out design proposal, including rationale and pseudocode.
- **Clarity Before Code:** Do not start implementing until you have a design with no major gaps or open questions. Ask the user to clarify any ambiguities.
- **"Correct-by-Construction":** The central architectural principle of this system is to be "correct-by-construction". Avoid multi-stage, corrective pipelines that patch up errors. Design algorithms that produce a valid final state in a single, principled pass. Refer to the DP implementation in `ragzoom/dynamic_frontier.py` as the canonical example.

### Core Architecture

**Flow**: Index (split → embed → build tree) → Retrieve (search → MMR → DP frontier) → Assemble (segments → summary)

**Key Files**:
- `store.py`: SQLite + ChromaDB + LRU cache
- `index.py`: Async tree building with progress tracking
- `retrieve.py`: MMR diversity + DP frontier extraction  
- `assemble.py`: Segment-based assembly (no budget trimming needed)
- `dynamic_frontier.py`: DP algorithm for correct-by-construction tilings

**Key Config** (`RagZoomConfig`): `budget_tokens=8000`, `leaf_tokens=200`, `slope_cap_size=1`, `mmr_lambda=0.7`

**Critical Details**:
- AsyncOpenAI for indexing, sync for retrieval/assembly
- Node IDs: `{depth}_{span_start}_{span_end}_{hash[:8]}`
- Spans use CHARACTER coordinates (not tokens) for stability
- Left-balanced binary tree, may be ragged during updates

## 5. Testing & Validation

- **Test-Driven Development:** Where possible, practice TDD. Write a failing test that reproduces the bug or demonstrates the new feature before you write the implementation. Then, make the test pass.
- **Test Edge Cases:** Always consider and add tests for edge cases, not just the happy path. This is especially critical for complex algorithmic logic.
- **Trust, but Verify (with Mocks):** The core algorithms should be pure and testable. Do not trust that external systems (databases, LLMs) will always behave as expected. Use the `SimpleMockStore` for fast, reliable, and hermetic unit tests of algorithmic logic.

**Testing**: Use `SimpleMockStore` for fast unit tests, real store only for @integration tests. Add `--validate` flag to index/query commands for comprehensive validation.

## 6. Key Commands

```bash
# Testing
pytest tests/ -m "not slow and not integration" -n 8  # Fast tests only
./test_quick.sh                                        # Quick test runner

# Type checking & linting
dmypy run -- ragzoom/        # Fast type checking with daemon
ruff check ragzoom/ tests/   # Linting

# Core operations
ragzoom index <file> [--document-id ID] [--clear] [--validate]
ragzoom query "text" -d <doc-id> [--debug] [--validate]
ragzoom documents            # List indexed docs
ragzoom serve               # Start API server
```

**Pre-commit hook**: Runs fast tests + linting + type checking automatically

## 7. Version Control & Commits

- **No Unauthorized Commits:** Never commit code unless explicitly directed to by the user.
- **Atomic Commits:** When asked to commit, group changes into small, logical, atomic commits with clear messages. Do not lump unrelated changes together.
- **Pre-commit is Mandatory:** The pre-commit hook (`scripts/git-hooks/pre-commit`) is the guardian of code quality. **You must never bypass it with `--no-verify` without explicit permission.** The hook is configured to auto-fix trivial issues; any remaining errors must be fixed manually.
- **Don't Deprecate, Delete:** Do not leave old code paths behind a feature flag or comment them out. Remove them. The git history will preserve them if we ever need to look back.

## 8. Development Practices

**Type Safety**:
- **ALWAYS write type annotations** for all new functions and methods
- Include parameter types and return types: `def func(x: str, y: int) -> bool:`
- Use `from typing import` imports for complex types: `List`, `Dict`, `Optional`, `Union`, etc.
- Type checking runs in pre-commit hook and will warn about missing annotations
- For SQLAlchemy ORM code, focus on business logic types rather than Column types
- When in doubt, `Any` is better than no annotation, but prefer specific types

**Testing & Commits**:
- Always write regression tests when regressions are discovered
- Group related changes into single commits that leave the app in a working state
- Run tests for modified components before committing

**Adding Features**: Config → Core logic → CLI → API → Tests

**Debugging**: Check `~/.ragzoom/ragzoom.log`, use `--no-progress` flag

## 9. Performance

- Batch embeddings (100 texts/call)
- Async summarization with configurable parallelism
- LRU cache for hot paths

## 10. Troubleshooting

- **Segmentation Faults:** If `pytest` crashes with a `Segmentation fault`, the local `chroma_db/` directory is almost certainly corrupted. The first step in debugging should always be to delete it and restart the test run: `rm -rf chroma_db/`
- **Persistent `mypy` Errors:** The `dmypy` daemon can sometimes get into a bad state. If you are struggling with type errors that you believe you have fixed, run a full, stateless `mypy` check to get a reliable result: `mypy ragzoom --ignore-missing-imports --no-error-summary --check-untyped-defs`

