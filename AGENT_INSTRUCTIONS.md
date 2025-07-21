# Agent Instructions for RagZoom

This file contains comprehensive instructions for any AI agent working on this repository.

## 1. Agent Philosophy & Collaboration

- **Be Zen, Not Flustered:** When you get stuck in a loop or a series of failures, stop. Take a step back, breathe, and rethink the problem from first principles. There is always a logical solution.
- **Don't Be Long-winded:** Keep it concise unless elaboration is warranted. Don't repeat yourself or summarize your own messages at the end. You're not writing an essay, we're having a conversation.
- **Use the Scientific Method:** For complex problems, form a hypothesis, propose a test to validate it, and discuss it with the user before implementing.
- **Raise Blockers:** If you are instructed to do something and discover an insurmountable roadblock or a fundamental inconsistency, do not switch gears. Bring the issue to the user's attention and decide on a new course of action together.
- **Leave the Codebase Better:** Always be looking for opportunities to improve the code you touch, whether it's by refactoring, adding a clarifying comment, or improving a variable name.
- **Update Documentation:** If you discover that a document is out of date or missing information in the course of your work, update it as part of your task.
- **Update These Rules:** If you discover a new principle or best practice during your work, add it to this file.

## 2. Design Philosophy

- **Design First:** Before implementing any large initiative, work with the user to create a well-thought-out design proposal, including rationale and pseudocode.
- **Clarity Before Code:** Do not start implementing until you have a design with no major gaps or open questions. Ask the user to clarify any ambiguities.
- **"Correct-by-Construction":** The central architectural principle of this system is to be "correct-by-construction". Avoid multi-stage, corrective pipelines that patch up errors. Design algorithms that produce a valid final state in a single, principled pass. Refer to the DP implementation in `ragzoom/dynamic_tiling.py` as the canonical example.

## 3. Version Control & Collaboration Rules

- **No Unauthorized Commits:** Never commit code unless explicitly directed to by the user.
- **Atomic Commits:** When asked to commit, group changes into small, logical, atomic commits with clear messages. Do not lump unrelated changes together.
- **Don't Deprecate, Delete:** Do not leave old code paths behind a feature flag or comment them out. Remove them. The git history will preserve them if we ever need to look back.

## 4. System Architecture & Technical Documentation

@include docs/architecture.md

## 5. Development Practices & Testing

@include docs/developer-guide.md

## 6. Algorithm Deep Dive

For detailed understanding of the core tiling algorithm:

@include docs/deep-dives/tiling-algorithm.md

## 7. Quick Reference

### Key Commands
```bash
# Testing
pytest tests/ -m "not slow and not integration" -n 8  # Fast tests only
./scripts/test_quick.sh                                # Quick test runner

# Type checking & linting  
dmypy run -- ragzoom/        # Fast type checking with daemon
ruff check ragzoom/ tests/   # Linting

# Core operations
ragzoom index <file> [--document-id ID] [--clear] [--validate]
ragzoom query "text" -d <doc-id> [--debug] [--validate]
ragzoom documents            # List indexed docs
ragzoom serve               # Start API server
```

### Performance Tips
- Batch embeddings (100 texts/call)
- Async summarization with configurable parallelism
- LRU cache for hot paths

### Common Troubleshooting
- **Segmentation Faults:** If `pytest` crashes with a `Segmentation fault`, the local `chroma_db/` directory is almost certainly corrupted. Delete it and restart: `rm -rf chroma_db/`
- **Persistent `mypy` Errors:** The `dmypy` daemon can sometimes get into a bad state. Run a full check: `mypy ragzoom --ignore-missing-imports --no-error-summary --check-untyped-defs`