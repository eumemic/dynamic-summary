# Frontier Design Clarification

This represents the original design of the frontier concept, which should always be both complete and non-overlapping. Implementation may have diverged from it, and if so we need to understand where and correct it.

## Key Concepts

### Frontier vs Frontier Nodes

1. **Frontier**: A sequence of node-halves that:
   - Provides complete coverage of the source document (no gaps)
   - Has no overlapping content
   - Is the actual output we want to produce

2. **Frontier nodes**: The collection of covered nodes from which the frontier halves are drawn
   - In the n_max=1 case, this is the entire path from selected leaf to root
   - Each covered node may contribute zero, half, or all of its content to the frontier

### The Extraction Rule

To obtain a frontier from frontier nodes, start from the root and apply this rule for each node:

- **Both children covered** → Eliminate entire node (contribute nothing)
- **Left child covered** → Eliminate left half (contribute right half only)
- **Right child covered** → Eliminate right half (contribute left half only)  
- **Neither child covered** → Keep entire node (contribute full text)

This rule guarantees:
- **Completeness**: Every position in the document is covered by exactly one node-half
- **No overlap**: Content covered by children is not duplicated in parent output
