# Documentation Archive

This directory contains historical documentation that provides context about the project's evolution but does not describe the current system implementation.

## Directory Structure

### legacy-algorithm/
Contains documentation for the original "Zoom-Lens" algorithm that has been replaced by the Dynamic Programming approach:
- `core-algorithm-design.md` - Original multi-stage algorithm design
- `frontier-design-clarification.md` - Legacy frontier extraction rules
- `project-brief.md` - Original v0.1 project specification
- `implementation-notes.md` - Early implementation planning

### refactoring/
Documentation from completed refactoring efforts:
- `depth-refactor-*.md` - Dynamic depth calculation refactoring
- `dp-transition-analysis.md` - Migration from legacy to DP algorithm
- `legacy-assembly-test-analysis.md` - Test migration planning
- `testing-*.md` - Testing documentation (merged into developer-guide.md)
- `token-position-resolver-*.md` - Visualization feature implementation
- `tree-viz-coordinate-systems.md` - Coordinate system design
- `validate-flag.md` - Validation feature documentation

### investigations/
Bug reports and analysis documents:
- `bug-investigations.md` - Historical bug investigations

### v2-planning/
Aspirational designs for features not yet implemented:
- `core-algorithm-design-v2.md` - Mass-based relevance propagation design
- `design-walkthrough-v2.md` - Examples of mass-based system

## Why Archive?

These documents are preserved because they:
1. Provide historical context for design decisions
2. Document the evolution of the system
3. Contain valuable lessons learned
4. May inform future development

However, they should NOT be used as references for the current system implementation. For current documentation, see the main docs/ directory.